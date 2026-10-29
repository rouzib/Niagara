"""Fixed-program capture followed by tensor-only execution and functional transforms."""

from __future__ import annotations

import contextvars
import importlib.util
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from caskade import Node
from torch.fx.experimental.proxy_tensor import get_proxy_mode, make_fx
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

from .effects import python_effect
from .errors import CaptureError, GuardError
from .guards import StateGuard
from .placement import _scope, graph_paths
from .replay import CudaReplay, SegmentedReplay, torchscript_program


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _signature(value):
    leaves, spec = tree_flatten(value)
    result = []
    for x in leaves:
        if isinstance(x, torch.Tensor):
            result.append(
                (tuple(x.shape), tuple(x.stride()), x.dtype, x.device, x.requires_grad, x.layout)
            )
        elif x is None or isinstance(x, (str, bool, int, float, torch.dtype, torch.device)):
            result.append((type(x), x))
        else:
            raise CaptureError(f"Unsupported input leaf {type(x).__name__}; use tensor pytrees")
    return spec, tuple(result)


def _modes():
    return (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cpu"),
        torch.get_autocast_dtype("cpu"),
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
    )


def _state_value(value):
    if isinstance(value, torch.Tensor):
        return (id(value), _version(value), tuple(value.shape), value.dtype, value.device)
    if isinstance(value, Node):
        return id(value)
    if isinstance(value, (tuple, list)):
        return tuple(_state_value(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _state_value(v)) for k, v in value.items())
    if value is None or isinstance(value, (str, int, float, bool, torch.dtype, torch.device)):
        return value
    return id(value)


def _state(root):
    if root is None:
        return None
    ignored = {"_memos", "_parents", "meta", "_niagara_placement", "_value"}
    return tuple(
        (
            id(node),
            id(type(node)),
            tuple(
                (k, _state_value(v))
                for k, v in vars(node).items()
                if k not in ignored and not k.startswith("_active_cache")
            ),
        )
        for node in graph_paths(root)[1]
    )


def _program_devices(graph, example):
    tensors = [x for x in tree_flatten(example)[0] if isinstance(x, torch.Tensor)]
    tensors += [t for t in graph.buffers() if t.device.type != "cpu" or t.numel() != 1]
    for node in graph.graph.nodes:
        tensors += [
            x
            for x in tree_flatten(node.meta.get("val"))[0]
            if isinstance(x, torch.Tensor)
            and not (node.op == "get_attr" and x.device.type == "cpu" and x.numel() == 1)
        ]
    return {t.device for t in tensors}


class _CaptureMode(TorchDispatchMode):
    def __init__(self, modes, scopes):
        self.modes, self.scopes = modes, scopes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if _modes() != self.modes:
            raise CaptureError(
                "Local grad, inference, or autocast changes during capture are unsupported"
            )
        if self.scopes:
            from torch.fx.traceback import annotate

            with annotate({"niagara_scope": _scope.get()}):
                return func(*args, **(kwargs or {}))
        return func(*args, **(kwargs or {}))


def _fold_constants(graph):
    """Evaluate pure input-independent tensor work once, preserving RNG and detach."""
    from torch.fx.node import map_arg

    constants = {}
    for node in list(graph.graph.nodes):
        if node.op == "get_attr":
            value = graph
            for part in node.target.split("."):
                value = getattr(value, part)
            if not any(
                t.requires_grad for t in tree_flatten(value)[0] if isinstance(t, torch.Tensor)
            ):
                constants[node] = value
        elif node.op == "call_function" and all(n in constants for n in node.all_input_nodes):
            target = node.target
            if not isinstance(target, torch._ops.OpOverload):
                continue
            if target._schema.is_mutable or torch.Tag.nondeterministic_seeded in target.tags:
                continue
            args = map_arg(node.args, constants.__getitem__)
            kwargs = map_arg(node.kwargs, constants.__getitem__)
            value = target(*args, **kwargs)
            if not isinstance(value, torch.Tensor):
                continue
            name = f"_niagara_constant_{len(constants)}"
            graph.register_buffer(name, value)
            with graph.graph.inserting_before(node):
                replacement = graph.graph.get_attr(name)
                replacement.meta = {**node.meta, "val": value}
            constants[replacement] = value
            node.replace_all_uses_with(replacement)
            graph.graph.erase_node(node)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    return graph


def _convolution_backward(
    grad,
    inputs,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    mask,
):
    """Lower 2-D convolution derivatives into traceable tensor operations.

    TorchScript cannot trace convolution_backward's optional tensor outputs on
    some releases. This targeted decomposition leaves explicit detach intact.
    """
    if transposed or inputs.ndim != 4:
        raise CaptureError("Captured reverse AD currently supports non-transposed 2-D convolutions")
    import torch.nn.functional as F

    grad_input = grad_weight = grad_bias = None
    if mask[0]:
        outpad = tuple(
            inputs.shape[2 + i]
            - (
                (grad.shape[2 + i] - 1) * stride[i]
                - 2 * padding[i]
                + dilation[i] * (weight.shape[2 + i] - 1)
                + 1
            )
            for i in range(2)
        )
        grad_input = F.conv_transpose2d(
            grad,
            weight,
            stride=stride,
            padding=padding,
            output_padding=outpad,
            groups=groups,
            dilation=dilation,
        )
    if mask[1]:
        patches = F.unfold(
            inputs, weight.shape[2:], dilation=dilation, padding=padding, stride=stride
        )
        patches = patches.reshape(inputs.shape[0], groups, -1, patches.shape[-1])
        outgoing = grad.reshape(grad.shape[0], groups, -1, patches.shape[-1])
        grad_weight = torch.einsum("ngol,ngil->goi", outgoing, patches).reshape(weight.shape)
    if mask[2]:
        grad_bias = grad.sum((0, 2, 3))
    return grad_input, grad_weight, grad_bias


def _capture(function, example, record_scopes, requires_grad=False):
    def traced(x):
        # make_fx calls TorchDispatchMode beneath its own tracer. Input mutation
        # is checked separately below; intermediate in-place operations are legal.
        with _CaptureMode(_modes(), record_scopes):
            return function(x)

    cloned = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, example)
    if requires_grad:
        if not isinstance(cloned, torch.Tensor) or not cloned.is_floating_point():
            raise CaptureError("Scalar-gradient capture requires one floating-point input tensor")
        cloned = cloned.detach().requires_grad_()
    tensors = [x for x in tree_flatten(cloned)[0] if isinstance(x, torch.Tensor)]
    versions = [_version(x) for x in tensors]
    decompositions = {torch.ops.aten.convolution_backward.default: _convolution_backward}
    try:
        if record_scopes:
            from torch.fx.traceback import preserve_node_meta

            with preserve_node_meta():
                graph = make_fx(traced, tracing_mode="real", decomposition_table=decompositions)(
                    cloned
                )
        else:
            graph = make_fx(traced, tracing_mode="real", decomposition_table=decompositions)(cloned)
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"Tensor capture failed: {exc}") from exc
    if versions != [_version(x) for x in tensors]:
        raise CaptureError("The callable mutates its inputs; use a functional tensor program")
    graph.graph.eliminate_dead_code()
    graph.recompile()
    # CUDA autograd can emit nodes from worker threads after make_fx's local
    # decomposition context has unwound. Rewrite those explicit derivative
    # nodes and recapture the now-pure forward program on the calling thread.
    lowered = False
    for node in graph.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten.convolution_backward.default
        ):
            node.target = _convolution_backward
            lowered = True
    if lowered:
        graph.recompile()
        return _capture(graph, example, record_scopes)
    return _fold_constants(graph)


class CompiledModel:
    """Guard and execute one fixed tensor program derived from a simulator.

    Parameters
    ----------
    graph : torch.fx.GraphModule
        Captured tensor-only callable accepting one input pytree.
    example : object
        Representative pytree defining input metadata and static scalar leaves.
    root : caskade.Node or None, optional
        Simulator whose discoverable state should be guarded. Arbitrary closure
        state cannot be exhaustively discovered.
    backend : str, optional
        ``"auto"`` selects CUDA replay for suitable tape-free CUDA inputs;
        ordinary CUDA autograd selects Inductor when available. CPU uses the
        TorchScript compatibility path. Explicit compiler names are accepted.
    record_scopes : bool, optional
        Retain Caskade scope labels when capturing subsequent transforms.

    Attributes
    ----------
    graph : torch.fx.GraphModule
        Pure captured graph for inspection and composition before compilation.
        Direct calls bypass wrapper guards.
    backend : str
        Actual selected execution backend.
    setup_seconds : float
        Backend preparation wall time, excluding the original capture.

    Notes
    -----
    Prefer :func:`compile_model` to construct this object. Shape, stride, device,
    dtype, requires-grad, scalar leaves and execution modes specialize the
    program. Captured CUDA replay has mutable internal buffers and is
    single-caller; public output tensors are independent between calls.
    """

    def __init__(
        self,
        graph: torch.fx.GraphModule,
        example: Any,
        *,
        root: Node | None = None,
        backend: str = "auto",
        record_scopes: bool = False,
    ) -> None:
        self.graph, self.root = graph, root
        self.signature, self.modes = _signature(example), _modes()
        self.record_scopes = record_scopes
        self._state_guard = StateGuard(root)
        self._base_state_guard = StateGuard(root, include_metadata=False)
        self.backend_requested = backend
        self.setup_seconds = 0.0
        self._skip_state = contextvars.ContextVar("niagara_fixed_session", default=False)
        start = time.perf_counter()
        if backend == "auto":
            leaves = [t for t in tree_flatten(example)[0] if isinstance(t, torch.Tensor)]
            if (
                leaves
                and all(t.is_cuda and not t.requires_grad for t in leaves)
                and not any(t.requires_grad for t in (*graph.buffers(), *graph.parameters()))
            ):
                devices = _program_devices(graph, example)
                backend = (
                    "cuda_graph"
                    if devices and all(d.type == "cuda" for d in devices)
                    else "torchscript"
                )
            else:
                backend = (
                    "inductor"
                    if (
                        leaves
                        and all(t.is_cuda for t in leaves)
                        and importlib.util.find_spec("triton")
                    )
                    else "torchscript"
                )
        self.backend = backend
        if backend == "eager":
            self._run = graph
        elif backend == "cuda_graph":
            devices = _program_devices(graph, example)
            if not devices or any(d.type != "cuda" for d in devices):
                raise CaptureError("CUDA replay requires a CUDA tensor program")
            if len(devices) == 1:
                self._run = CudaReplay(graph, example, device=next(iter(devices)))
            else:
                if any(
                    t.requires_grad for t in tree_flatten(example)[0] if isinstance(t, torch.Tensor)
                ) or any(t.requires_grad for t in (*graph.buffers(), *graph.parameters())):
                    raise CaptureError(
                        "CUDA replay requires explicit derivatives, without an autograd tape"
                    )
                self._run = SegmentedReplay(graph, example)
        elif backend == "torchscript":
            # trace is only applied to the already captured tensor program, never
            # to stateful Caskade parameter plumbing.
            self._run = torchscript_program(graph, example)
        else:
            self._run = torch.compile(graph, backend=backend, fullgraph=True)
            self._run(example)  # Make compile failures visible during preparation.
        self.setup_seconds = time.perf_counter() - start

    def validate(self, x: Any) -> None:
        """Check that inputs, modes and simulator state satisfy the captured contract.

        Parameters
        ----------
        x : object
            Input pytree to be passed to this program.

        Raises
        ------
        GuardError
            If metadata/modes/state changed, the simulator is active, or an
            unsupported outer functional transform is active.

        Notes
        -----
        Model-state traversal is amortized inside :meth:`session`; input and
        mode checks remain enabled there.
        """
        if get_proxy_mode() is not None or torch.jit.is_tracing():
            raise GuardError(
                "Compose the pure graph before compilation; prepared calls cannot be traced again"
            )
        if torch._C._are_functorch_transforms_active():
            raise GuardError("Apply transforms before compilation with .transform() or ng.vmap()")
        if _signature(x) != self.signature:
            raise GuardError(
                "Input tree, shape, stride, dtype, device, or requires_grad changed; recapture"
            )
        if _modes() != self.modes:
            raise GuardError("Grad, inference, or autocast mode changed; recapture")
        if self.root is not None and self.root.active:
            raise GuardError("Compiled calls require an inactive simulator")
        if not self._skip_state.get() and not self._state_guard.unchanged():
            raise GuardError("Simulator configuration, parameters, or placement changed; recapture")

    def __call__(self, x: Any) -> Any:
        """Validate and execute the prepared tensor program.

        Parameters
        ----------
        x : object
            Pytree matching preparation metadata; tensor values may change.

        Returns
        -------
        object
            Program output pytree on its established devices. CUDA replay
            returns independent tensor copies without an ordinary autograd tape.

        Raises
        ------
        GuardError
            If the input or execution contract no longer matches preparation.
        """
        self.validate(x)
        return self._run(x)

    @contextmanager
    def session(self) -> Iterator[CompiledModel]:
        """Amortize simulator-state validation around a fixed production loop.

        Yields
        ------
        CompiledModel
            This prepared program with input and mode guards still enabled.

        Raises
        ------
        GuardError
            If the simulator changed before or during the session.

        Notes
        -----
        Do not mutate simulator state inside the block. The exit check detects
        accidental changes; it is not a lock for concurrent callers.
        """
        if not self._state_guard.unchanged():
            raise GuardError("Simulator changed before session")
        token = self._skip_state.set(True)
        try:
            yield self
        finally:
            self._skip_state.reset(token)
            if not self._state_guard.unchanged():
                raise GuardError("Simulator was modified during a fixed execution session")

    def transform(
        self,
        transform: Callable,
        example: Any,
        *,
        backend: str | None = None,
    ) -> CompiledModel:
        """Transform the pure graph before preparing a new program.

        Parameters
        ----------
        transform : callable
            Function mapping the captured callable to a new callable. Default
            ``torch.func.grad`` and ``grad_and_value`` use Niagara's explicit
            scalar-gradient capture path; other transforms use native Torch.
        example : object
            Representative input for the transformed callable. The default
            scalar-gradient path requires one floating-point tensor.
        backend : str or None, optional
            Override the original requested compiler selection.

        Returns
        -------
        CompiledModel
            New graph and prepared execution program with the same root guards.

        Raises
        ------
        GuardError
            If model state changed before transformation.
        CaptureError
            If the transformed callable is unsupported by fixed-program capture.

        Notes
        -----
        Functional derivatives become explicit tensor outputs and can therefore
        use replay without accumulating an ordinary autograd tape. Not every
        native Torch transform supports every cross-device operator combination.
        """
        if not self._state_guard.unchanged():
            raise GuardError("Simulator changed before transform")
        # Native torch.func reverse AD across physical CUDA devices fails on some
        # Torch releases (TensorWrapper storage access). Trace ordinary autograd
        # into an explicit derivative program instead; production has no AD tape.
        reverse = transform is torch.func.grad_and_value or transform is torch.func.grad
        if reverse:

            def fn(x):
                value = self.graph(x)
                (gradient,) = torch.autograd.grad(value, x, create_graph=True)
                return (gradient, value) if transform is torch.func.grad_and_value else gradient
        else:
            fn = transform(self.graph)
        graph = _capture(fn, example, self.record_scopes, requires_grad=reverse)
        return CompiledModel(
            graph,
            example,
            root=self.root,
            backend=backend or self.backend_requested,
            record_scopes=self.record_scopes,
        )

    def vmap(
        self,
        example: Any,
        *,
        in_dims: Any = 0,
        out_dims: Any = 0,
        chunk_size: int | None = None,
        randomness: str = "error",
        backend: str | None = None,
    ) -> CompiledModel:
        """Capture a batched transform without changing established device homes.

        Parameters
        ----------
        example : object
            Representative batched input pytree.
        in_dims, out_dims : int or pytree, optional
            Native ``torch.vmap`` input/output batch-dimension specifications.
        chunk_size : int or None, optional
            Native vmap chunk size; None processes the full batch together.
        randomness : str, optional
            Native vmap random-operation policy, defaulting to ``"error"``.
        backend : str or None, optional
            Override the compiler used for the batched graph.

        Returns
        -------
        CompiledModel
            Batch-specialized program. This does not shard the batch across GPUs;
            use ReplicaPool for independent per-device models.
        """
        return self.transform(
            lambda f: torch.vmap(
                f, in_dims=in_dims, out_dims=out_dims, chunk_size=chunk_size, randomness=randomness
            ),
            example,
            backend=backend,
        )

    def execution_plan(self) -> dict[str, Any]:
        """Describe captured tensor dependencies and their declared devices.

        Returns
        -------
        dict
            ``kind``, selected ``backend`` and ``nodes``. Node entries contain
            operation, dependencies, scope, device, shape and dependency level.

        Notes
        -----
        Dependency levels are static ready groups. They are not measured
        execution times and do not establish overlap or optimal placement.
        """
        levels, records = {}, []
        for node in self.graph.graph.nodes:
            deps = list(node.all_input_nodes)
            levels[node] = max((levels[d] + 1 for d in deps), default=0)
            val = node.meta.get("val")
            records.append(
                {
                    "name": node.name,
                    "op": str(node.target),
                    "dependencies": [d.name for d in deps],
                    "level": levels[node],
                    "scope": node.meta.get("custom", {}).get("niagara_scope"),
                    "device": str(val.device) if isinstance(val, torch.Tensor) else None,
                    "shape": list(val.shape) if isinstance(val, torch.Tensor) else None,
                }
            )
        return {"kind": "captured_dependencies", "backend": self.backend, "nodes": records}


def compile_model(
    model: Callable | Node,
    example: Any,
    *,
    method: str | None = None,
    backend: str = "auto",
    record_scopes: bool = False,
) -> CompiledModel:
    """Capture and prepare one simulator call with a representative input pytree.

    Parameters
    ----------
    model : callable or caskade.Module
        Callable accepting one pytree, or simulator receiving packed parameters.
        Bound Module methods retain automatic simulator-state guards.
    example : object
        Representative tensor pytree. Static scalar leaves are allowed; shapes,
        strides, dtype, device and requires-grad status specialize execution.
    method : str or None, optional
        Method to select from ``model`` instead of calling it directly.
    backend : str, optional
        ``"auto"``, ``"cuda_graph"``, ``"torchscript"``, ``"inductor"``,
        ``"aot_eager"``, ``"eager"``, or another explicit Torch compiler backend.
    record_scopes : bool, optional
        Preserve simulator scope labels for graph visualizations/profiling.

    Returns
    -------
    CompiledModel
        Guarded prepared program. ``capture_and_setup_seconds`` records total
        capture and backend preparation wall time.

    Raises
    ------
    CaptureError
        If the simulator is active, inputs are mutated during capture, local
        execution modes change, or the callable cannot be captured safely.

    Notes
    -----
    Bind additional fixed arguments with a lambda or partial. Arbitrary closures
    do not have exhaustive external-state guards; their captured state must stay
    fixed. General data-dependent Python is outside the capture contract. Apply
    transforms to the pure graph before compilation rather than wrapping a
    compiled program in an outer transform.

    Examples
    --------
    Prepare an already placed simulator and reuse input metadata::

        compiled = compile_model(sim, params)
        with compiled.session():
            image = compiled(params)
    """
    root = model if isinstance(model, Node) else getattr(model, "__self__", None)
    root = root if isinstance(root, Node) else None
    if root is not None and root.active:
        raise CaptureError("Capture requires an inactive simulator")
    fn = getattr(model, method) if method is not None else model
    if reason := python_effect(fn, root):
        raise CaptureError(f"{reason} cannot be replayed; use dynamic execution")
    start = time.perf_counter()
    graph = _capture(fn, example, record_scopes)
    result = CompiledModel(graph, example, root=root, backend=backend, record_scopes=record_scopes)
    result.capture_and_setup_seconds = time.perf_counter() - start
    return result


def vmap(compiled: CompiledModel, example_batch: Any, **kwargs: Any) -> CompiledModel:
    """Prepare the batched pure graph of a compiled program.

    Parameters
    ----------
    compiled : CompiledModel
        Program to transform before executing its compiled backend.
    example_batch : object
        Representative batched input pytree.
    **kwargs : object
        Arguments forwarded to :meth:`CompiledModel.vmap`, including dimensions,
        chunk size, randomness policy and backend override.

    Returns
    -------
    CompiledModel
        Guarded batch-specialized program with original region placement.

    Raises
    ------
    TypeError
        If ``compiled`` is not a Niagara CompiledModel.
    """
    if not isinstance(compiled, CompiledModel):
        raise TypeError("ng.vmap expects a CompiledModel and a representative batch")
    return compiled.vmap(example_batch, **kwargs)
