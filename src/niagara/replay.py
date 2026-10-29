"""Fused CUDA replay and dependency-ordered device regions."""

from __future__ import annotations

import importlib.util
import types
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_flatten, tree_map

from .errors import CaptureError


def fused_program(graph: torch.nn.Module, example: Any) -> Callable[[Any], Any]:
    """Prepare one local region for fusion before CUDA graph capture.

    Parameters
    ----------
    graph : torch.nn.Module
        Tensor-only graph/adapter accepting one representative input pytree.
    example : object
        Fixed input metadata used by the TorchScript compatibility path.

    Returns
    -------
    callable
        Full-graph Inductor callable when Triton is installed, otherwise a
        frozen TorchScript callable. CUDA capture is performed separately.

    Notes
    -----
    Adapter code objects are isolated to avoid exhausting Dynamo's shared-frame
    specialization cache. No process-wide compiler limits are changed. This is
    an internal preparation primitive; callers must enforce the capture contract.
    """
    if importlib.util.find_spec("triton") is not None:
        if not isinstance(graph, torch.fx.GraphModule):
            # Every fixed region is a distinct program, not a new specialization
            # of one shared Adapter.forward frame. Keep Dynamo's per-frame cache
            # local without changing process-wide recompilation limits.
            method = graph.forward
            fn = method.__func__
            isolated = types.FunctionType(
                fn.__code__.replace(), fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
            )
            graph.forward = types.MethodType(isolated, graph)
        return torch.compile(
            graph, backend="inductor", fullgraph=True, options={"triton.cudagraphs": False}
        )
    return torchscript_program(graph, example, freeze=True)


def torchscript_program(
    graph: torch.nn.Module,
    example: Any,
    *,
    freeze: bool = False,
) -> Callable[[Any], Any]:
    """Trace tensor leaves while reconstructing static input-pytree structure.

    Parameters
    ----------
    graph : torch.nn.Module
        Already captured tensor program, not a stateful Caskade simulator.
    example : object
        Input pytree including tensor leaves, scalar flags or empty containers.
    freeze : bool, optional
        Freeze the traced evaluation module for tape-free replay. Ordinary
        autograd paths leave this False to retain trainable captured state.

    Returns
    -------
    callable
        Callable accepting the original pytree with static leaves reconstructed
        from preparation metadata. Input metadata must be guarded externally.

    Notes
    -----
    TorchScript receives only tensor leaf arguments, avoiding unsupported empty
    keyword dictionaries or static scalar argument types. Trace checking is
    disabled because the graph has already been captured and validated.
    """
    if isinstance(example, torch.Tensor):
        traced = torch.jit.trace(graph, (example,), check_trace=False, strict=False)
        return torch.jit.freeze(traced.eval()) if freeze else traced
    leaves, spec = tree_flatten(example)
    indices = tuple(i for i, t in enumerate(leaves) if isinstance(t, torch.Tensor))

    class Flat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.program = graph

        def forward(self, *tensors):
            values = list(leaves)
            for index, tensor in zip(indices, tensors):
                values[index] = tensor
            return self.program(spec.unflatten(values))

    traced = torch.jit.trace(
        Flat().eval(), tuple(leaves[i] for i in indices), check_trace=False, strict=False
    )
    if freeze:
        traced = torch.jit.freeze(traced)

    def run(inputs):
        values = tree_flatten(inputs)[0]
        return traced(*(values[i] for i in indices))

    return run


class CudaReplay:
    """Capture a local fused program for repeated single-device CUDA replay.

    Parameters
    ----------
    graph : torch.nn.Module
        Local tensor program accepting one input pytree.
    example : object
        Input tensors with fixed metadata and no ordinary requires-grad history.
    device : torch.device
        Normalized CUDA device for this local region.
    borrow_outputs : bool, optional
        Return mutable internal output buffers instead of independent copies.
        Reserved for immediately consumed internal region/packing results.

    Raises
    ------
    CaptureError
        If inputs or captured constants require ordinary autograd.

    Notes
    -----
    Preparation warms the fused runner and synchronizes its preparation stream.
    Production uses the caller's current stream and events when that stream
    changes. This runner is single-caller and does not itself validate metadata;
    CompiledModel supplies public guards. Default outputs are independent copies.
    """

    def __init__(
        self,
        graph: torch.nn.Module,
        example: Any,
        *,
        device: torch.device,
        borrow_outputs: bool = False,
    ) -> None:
        leaves = tree_flatten(example)[0]
        if any(t.requires_grad for t in leaves if isinstance(t, torch.Tensor)) or any(
            t.requires_grad for t in (*graph.buffers(), *graph.parameters())
        ):
            raise CaptureError(
                "CUDA replay requires explicit derivatives, without an autograd tape"
            )
        self.device = device
        self.inputs = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, example)
        self.tensor_input = isinstance(self.inputs, torch.Tensor)
        self.input_leaves = tree_flatten(self.inputs)[0]
        self.borrow_outputs = borrow_outputs
        runner = fused_program(graph, self.inputs)
        self.runner = runner
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.device(device), torch.cuda.stream(self.stream):
            for _ in range(20):
                runner(self.inputs)
        self.stream.synchronize()  # Preparation only.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(device), torch.cuda.graph(self.graph, stream=self.stream):
            self.outputs = runner(self.inputs)
        self.tensor_output = isinstance(self.outputs, torch.Tensor)
        torch.cuda.current_stream(device).wait_stream(self.stream)
        self.last_stream = torch.cuda.current_stream(device)

    def __call__(self, x: Any) -> Any:
        """Copy inputs into fixed buffers and replay on the current device stream.

        Parameters
        ----------
        x : object
            Input pytree whose tensor metadata matches preparation exactly.

        Returns
        -------
        object
            Output pytree. Tensor leaves are independent unless the internal
            ``borrow_outputs`` mode was selected.
        """
        caller = torch.cuda.current_stream(self.device)
        # The usual same-stream path needs no events. On a stream change, record
        # the previous queue's tail and wait before reusing graph buffers.
        if caller != self.last_stream:
            self.barrier = torch.cuda.Event()
            self.barrier.record(self.last_stream)
            caller.wait_event(self.barrier)
            self.last_stream = caller
        with torch.cuda.device(self.device):
            if self.tensor_input:
                x.record_stream(caller)
                self.inputs.copy_(x)
            else:
                for dst, src in zip(self.input_leaves, tree_flatten(x)[0]):
                    if isinstance(dst, torch.Tensor):
                        src.record_stream(caller)
                        dst.copy_(src)
            self.graph.replay()
            if self.borrow_outputs:
                result = self.outputs
            elif self.tensor_output:
                result = self.outputs.clone()
            else:
                result = tree_map(
                    lambda t: t.clone() if isinstance(t, torch.Tensor) else t, self.outputs
                )
        return result


class _Region(torch.nn.Module):
    def __init__(self, graph, args, device):
        super().__init__()
        self.device = device
        groups = {}
        for index, value in enumerate(args):
            if (
                isinstance(value, torch.Tensor)
                and value.is_cuda
                and value.device != device
                and value.layout == torch.strided
            ):
                groups.setdefault((value.device, value.dtype), []).append(index)
        self.packets = []
        for (source, _dtype), indices in groups.items():
            if len(indices) < 2:
                continue
            values = tuple(args[i] for i in indices)

            class Pack(torch.nn.Module):
                def forward(self, values):
                    return torch.cat([t.reshape(-1) for t in values])

            packer = CudaReplay(Pack(), values, device=source, borrow_outputs=True)
            self.packets.append(
                (
                    tuple(indices),
                    tuple(t.shape for t in values),
                    tuple(t.numel() for t in values),
                    packer,
                )
            )
        # Cross-device copies stay outside CUDA capture. The local program keeps
        # the captured copy operators, now operating on same-device buffers.
        local = tree_map(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, args)

        class Adapter(torch.nn.Module):
            def __init__(self, program):
                super().__init__()
                self.program = program

            def forward(self, inputs):
                return self.program(*inputs)

        self.replay = CudaReplay(Adapter(graph), local, device=device, borrow_outputs=True)

    def forward(self, *args):
        if self.packets:
            args = list(args)
            for indices, shapes, counts, packer in self.packets:
                packed = packer(tuple(args[i] for i in indices)).to(self.device, non_blocking=True)
                for index, piece, shape in zip(indices, packed.split(counts), shapes):
                    args[index] = piece.view(shape)
            args = tuple(args)
        local = tree_map(
            lambda t: (
                t.to(self.device, non_blocking=True)
                if isinstance(t, torch.Tensor) and t.device != self.device
                else t
            ),
            args,
        )
        return self.replay(local)


class SegmentedReplay:
    """Partition a fixed CUDA DAG into dependency-ordered device-local replays.

    Parameters
    ----------
    graph : torch.fx.GraphModule
        Captured graph with tensor device metadata and fixed region ownership.
    example : object
        Representative CUDA input pytree without an ordinary autograd tape.

    Attributes
    ----------
    region_devices : dict
        Mapping from generated region names to their CUDA device strings.

    Raises
    ------
    CaptureError
        If an operator output spans devices or a region has no usable CUDA home.

    Notes
    -----
    Regions are grouped by device and cross-device dependency depth. Immutable
    buffers are bound inside regions; compatible boundary inputs are packed to
    reduce transfer count. Cross-device transfers remain outside local capture,
    including on systems without peer access. This executes explicit derivative
    programs, not ordinary autograd. Final output clones protect previous calls.
    Public metadata/state guards are provided by CompiledModel.
    """

    def __init__(self, graph: torch.fx.GraphModule, example: Any) -> None:
        owners, depths, groups = {}, {}, {}
        inputs = tree_flatten(example)[0]
        home = next(t.device for t in inputs if isinstance(t, torch.Tensor))
        for node in graph.graph.nodes:
            tensors = [
                t for t in tree_flatten(node.meta.get("val"))[0] if isinstance(t, torch.Tensor)
            ]
            devices = {t.device for t in tensors if t.device.type == "cuda"}
            if len(devices) > 1:
                raise CaptureError(f"Mixed-device operator output at {node.name}")
            owner = next(iter(devices)) if devices else None
            if owner is None and node.op not in ("get_attr", "output"):
                owner = next((owners[d] for d in node.all_input_nodes if owners.get(d)), home)
            owners[node] = owner
            depths[node] = max(
                (
                    depths[d] + int(owners.get(d) is not None and owners[d] != owner)
                    for d in node.all_input_nodes
                ),
                default=0,
            )
            if node.op not in ("placeholder", "get_attr", "output"):
                groups.setdefault((depths[node], owner), len(groups))
        ids = {
            node: groups[(depths[node], owners[node])]
            for node in graph.graph.nodes
            if node.op not in ("placeholder", "get_attr", "output")
        }
        self.plan = split_module(graph, graph, ids.__getitem__, keep_original_node_name=True)
        self.out_spec = getattr(graph, "_out_spec", None)
        # FX split modules pass every buffer as an argument. Bind immutable
        # buffers inside each region so replay only copies actual dynamic data.
        for call in self.plan.graph.nodes:
            if call.op != "call_module":
                continue
            child = self.plan.get_submodule(call.target)
            placeholders = [n for n in child.graph.nodes if n.op == "placeholder"]
            kept = []
            for index, (argument, placeholder) in enumerate(zip(call.args, placeholders)):
                if isinstance(argument, torch.fx.Node) and argument.op == "get_attr":
                    value = self.plan
                    for part in argument.target.split("."):
                        value = getattr(value, part)
                    name = f"_bound_{index}"
                    if isinstance(value, torch.Tensor):
                        child.register_buffer(name, value)
                    else:
                        setattr(child, name, value)
                    with child.graph.inserting_before(placeholder):
                        constant = child.graph.get_attr(name)
                        constant.meta = placeholder.meta.copy()
                    placeholder.replace_all_uses_with(constant)
                    child.graph.erase_node(placeholder)
                else:
                    kept.append(argument)
            call.args = tuple(kept)
            child.recompile()
        self.plan.graph.eliminate_dead_code()
        self.plan.recompile()
        observed = {}
        hooks = []
        for name, child in self.plan.named_children():

            def record(module, args, name=name):
                observed[name] = args

            hooks.append(child.register_forward_pre_hook(record))
        self.plan(*inputs)
        for hook in hooks:
            hook.remove()
        self.region_devices = {}
        for name, args in observed.items():
            child = getattr(self.plan, name)
            nodes = [
                n for n in child.graph.nodes if n.op not in ("placeholder", "get_attr", "output")
            ]
            device = next(
                (
                    owners_by_name.device
                    for n in nodes
                    if isinstance((owners_by_name := n.meta.get("val")), torch.Tensor)
                    and owners_by_name.device.type == "cuda"
                ),
                None,
            )
            if device is None:
                raise CaptureError(f"No CUDA device for region {name}")
            self.region_devices[name] = str(device)
            setattr(self.plan, name, _Region(child, args, device))

    def __call__(self, x: Any) -> Any:
        """Execute prepared regions and reconstruct independent output pytrees.

        Parameters
        ----------
        x : object
            Input pytree matching preparation metadata and scalar leaves.

        Returns
        -------
        object
            Original output structure with final tensor leaves cloned on their
            established devices. Region dependency order is preserved.
        """
        result = self.plan(*tree_flatten(x)[0])
        if self.out_spec is not None:
            result = self.out_spec.unflatten(tree_flatten(result)[0])
        return tree_map(lambda t: t.clone() if isinstance(t, torch.Tensor) else t, result)
