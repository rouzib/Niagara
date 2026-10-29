"""Opt-in execution underneath existing Caskade and Caustics instances."""

from __future__ import annotations

import contextvars
import functools
import types
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

import torch
from caskade import Module, Param
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .compiler import CompiledModel, _modes, _signature, compile_model
from .effects import python_effect
from .errors import CaptureError, GuardError, PlacementError
from .placement import DevicePlan, _inactive, assign_devices


class Backend:
    """Enable Niagara execution on one existing Caskade simulator.

    Parameters
    ----------
    root : caskade.Module
        Root simulator whose existing methods will use the backend.
    plan : DevicePlan
        Validated device ownership and transfer configuration.
    backend : str
        Tensor compiler selection. ``"auto"`` selects replay or a compatible
        compiler from input metadata and installed dependencies.
    compile : bool
        Whether selected methods prepare fixed tensor programs. False enables
        ordinary routed Caskade calls without capture.
    methods : tuple of str
        Class-method names to intercept when compilation is enabled.
    max_programs : int
        Maximum number of prepared method/input/mode specializations.

    Attributes
    ----------
    placement : Placement
        Applied instance-local placement and transfer diagnostics.
    closed : bool
        Whether native dispatch has been restored.

    Notes
    -----
    Use :func:`enable` rather than constructing this handle directly. Existing
    calls continue to receive their original positional/keyword arguments.
    Programs specialize input metadata and execution modes; mutable model state
    remains guarded. A handle is single-caller and is not a concurrency lock.
    Closing restores native classes and co-locates model state at ``plan.default``.
    """

    def __init__(
        self,
        root: Module,
        plan: DevicePlan,
        *,
        backend: str,
        compile: bool,
        methods: tuple[str, ...],
        max_programs: int,
        _preflight_methods: dict[str, Any] | None = None,
        _requires_example: bool = False,
    ) -> None:
        self.root, self.plan = root, plan
        self.backend, self.compile = backend, compile
        self.max_programs = max_programs
        self.closed = False
        self._programs = []
        self._bypass = contextvars.ContextVar("niagara_backend_bypass", default=False)
        self._fixed = contextvars.ContextVar("niagara_backend_session", default=False)
        # Inspect the unplaced model first. A scanner failure must not leave
        # migrated tensors or an instance-local placement behind.
        preflight_methods = (
            _preflight_methods
            if _preflight_methods is not None
            else {name: getattr(root, name) for name in methods}
        )
        self._dynamic_methods = {
            name: reason
            for name, fn in preflight_methods.items()
            if (reason := python_effect(fn, root)) is not None
        }
        if _requires_example and "__call__" in self._dynamic_methods:
            raise CaptureError(
                f"{self._dynamic_methods['__call__']} cannot be replayed; "
                "this method uses dynamic routed execution"
            )
        self.placement = assign_devices(root, plan)
        self._originals = {name: getattr(root, name) for name in methods}
        object.__setattr__(root, "_niagara_backend", self)
        if compile:
            methods = {name: self._dispatch(name, fn) for name, fn in self._originals.items()}
            cls = type(
                f"Enabled{type(root).__name__}",
                (type(root),),
                {**methods, "__module__": __name__},
            )
            object.__setattr__(root, "__class__", cls)

    def _dispatch(self, name, original):
        @functools.wraps(original)
        def invoke(root, *args, **kwargs):
            # Preserve native online/override semantics inside ActiveContext.
            # Internal calls during capture must go through ordinary routing.
            if self._bypass.get() or root.active:
                return original(*args, **kwargs)
            if get_proxy_mode() is not None or torch.jit.is_tracing():
                raise GuardError(
                    "Compose the pure graph before compilation; enabled dispatch cannot be traced again"
                )
            if name in self._dynamic_methods:
                if self._fixed.get():
                    raise GuardError("Dynamic methods cannot run in a fixed compiled session")
                return original(*args, **kwargs)
            try:
                program = self._prepare(name, args, kwargs)
                return program((args, kwargs))
            except CaptureError:
                # A preflight detected a new effect before user code ran.
                if name in self._dynamic_methods:
                    if self._fixed.get():
                        raise GuardError("Dynamic methods cannot run in a fixed compiled session") from None
                    return original(*args, **kwargs)
                raise
            except GuardError:
                # Recapture only metadata changes, never other state mutations.
                # A fixed session retains its existing no-mutation contract.
                if not self._fixed.get() and self._programs:
                    prior = self._programs[0][3]
                    if not prior._state_guard.unchanged() and prior._base_state_guard.unchanged():
                        self._dynamic_methods = {
                            method: reason
                            for method, fn in self._originals.items()
                            if (reason := python_effect(fn, self.root)) is not None
                        }
                        self.clear()
                        if name in self._dynamic_methods:
                            return original(*args, **kwargs)
                        return self._prepare(name, args, kwargs)((args, kwargs))
                raise

        return invoke

    @property
    def dynamic_methods(self) -> dict[str, str]:
        """List selected methods routed dynamically and their detected effects.

        Returns
        -------
        dict of str to str
            A copy mapping method names to the recognized Python effects that
            prevented capture. Mutating the result does not affect the backend.

        Notes
        -----
        Detection is conservative but incomplete. A method absent from this
        mapping is not guaranteed to be free of Python side effects.
        """
        return self._dynamic_methods.copy()

    @property
    def programs(self) -> tuple[CompiledModel, ...]:
        """Prepared programs, including ordinary-autograd specializations.

        Returns
        -------
        tuple of CompiledModel
            Snapshot of the cache in preparation order. A program accepts the
            original call's ``(args, kwargs)`` pytree, not a bare parameter tensor.
        """
        return tuple(program for _name, _signature_value, _mode, program in self._programs)

    def prepare(self, *args: Any, method: str = "__call__", **kwargs: Any) -> CompiledModel:
        """Prepare a representative call before repeated execution.

        Parameters
        ----------
        *args : object
            Original positional arguments to the selected simulator method.
        method : str, optional
            Selected class method. It must have been named during activation.
        **kwargs : object
            Original method keywords. ``method`` is reserved by this helper;
            call the simulator itself to specialize a same-named keyword.

        Returns
        -------
        CompiledModel
            Existing or newly captured guarded tensor program.

        Raises
        ------
        GuardError
            If the handle is closed, state changed, the cache is full, capture
            is disabled, or an unprepared call is requested inside a session.
        ValueError
            If the method was not selected during activation.
        CaptureError
            If the method cannot satisfy the fixed tensor-program contract.
        """
        return self._prepare(method, args, kwargs)

    def _prepare(self, method, args, kwargs):
        if self.closed:
            raise GuardError("This Niagara backend is closed")
        if not self.compile:
            raise GuardError("Enable compilation to prepare a tensor program")
        if method not in self._originals:
            raise ValueError(f"Method {method!r} was not selected when enabling Niagara")
        if method in self._dynamic_methods:
            raise CaptureError(
                f"{self._dynamic_methods[method]} cannot be replayed; "
                "this method uses dynamic routed execution"
            )
        if self.root.active:
            raise GuardError("Prepare calls with an inactive simulator")
        payload = (args, kwargs)
        signature, mode = _signature(payload), _modes()
        for name, known, modes, program in self._programs:
            if name == method and known == signature and modes == mode:
                return program
        if self._fixed.get():
            raise GuardError("Prepare every call signature before entering a session")
        # A new signature must not silently recapture mutated model state.
        if self._programs and not self._programs[0][3]._state_guard.unchanged():
            raise GuardError("Simulator state changed; clear the backend before recapturing")
        if len(self._programs) >= self.max_programs:
            raise GuardError(
                "Niagara program limit reached; clear the backend or raise max_programs"
            )
        original = self._originals[method]
        if reason := python_effect(original, self.root):
            self._dynamic_methods[method] = reason
            raise CaptureError(f"{reason} cannot be replayed; use dynamic routed execution")

        def entry(root, inputs):
            positional, keywords = inputs
            return original(*positional, **keywords)

        token = self._bypass.set(True)
        try:
            # A bound method retains automatic guards for the actual simulator.
            program = compile_model(
                types.MethodType(entry, self.root), payload, backend=self.backend
            )
        finally:
            self._bypass.reset(token)
        self._programs.append((method, signature, mode, program))
        return program

    @contextmanager
    def session(self) -> Iterator[Backend]:
        """Amortize model-state checks around a fixed loop.

        Yields
        ------
        Backend
            This backend, with already prepared calls available normally.

        Raises
        ------
        GuardError
            If no programs were prepared, the backend is closed, state changes,
            or a new specialization is requested inside the block.

        Notes
        -----
        Input/mode guards remain enabled. Prepare every needed call form before
        entering. Do not mutate the simulator, clear programs, or disable the
        backend until the block exits.
        """
        if self.closed or not self._programs:
            raise GuardError("Prepare calls before opening a Niagara session")
        with ExitStack() as stack:
            for program in self.programs:
                stack.enter_context(program.session())
            token = self._fixed.set(True)
            try:
                yield self
            finally:
                self._fixed.reset(token)

    def clear(self) -> None:
        """Release cached programs so subsequent calls recapture current state.

        Raises
        ------
        GuardError
            If the backend is closed or a fixed session is active.
        PlacementError
            If any owned simulator node is active.

        Notes
        -----
        This does not apply a new ownership plan. Disable and re-enable after
        graph-structure or placement changes.
        """
        if self.closed:
            raise GuardError("This Niagara backend is closed")
        if self._fixed.get():
            raise GuardError("Cannot clear programs inside a Niagara session")
        _inactive(self.placement.homes)
        self._programs.clear()

    def close(self) -> None:
        """Restore native calls and co-locate simulator state at the default home.

        Raises
        ------
        GuardError
            If called inside a fixed session.
        PlacementError
            If simulator nodes are active or uniform placement is unsafe.

        Notes
        -----
        Repeated calls after successful closure do nothing. Separately held
        user inputs/results are not migrated. Unlike ``Placement.close()``,
        this method also co-locates model tensors for native Caskade execution.
        """
        if self.closed:
            return
        if self._fixed.get():
            raise GuardError("Cannot disable Niagara inside a fixed session")
        _inactive(self.placement.homes)
        # Ownership may have changed since activation. Reject an unsafe restore
        # while the original handle, dispatch and prepared programs are intact.
        # Pointer-only dependencies had explicit owners during activation;
        # retain those identities when co-locating for native execution.
        restoration = DevicePlan(
            default=self.plan.default,
            factory_device=self.plan.factory_device,
            modules={
                node: self.plan.default for node in self.placement.homes if isinstance(node, Module)
            },
            parameters={
                node: self.plan.default for node in self.placement.homes if isinstance(node, Param)
            },
        )
        restoration.validate(self.root)
        self.clear()
        self.placement.close()
        delattr(self.root, "_niagara_backend")
        # Native Caskade expects co-located tensors; Placement.close alone leaves
        # split homes intact. Migrate once using the same validated machinery.
        assign_devices(self.root, restoration).close()
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise GuardError("This Niagara backend is closed")
        return self

    def __exit__(self, *exc):
        self.close()


def enable(
    root: Module,
    plan: DevicePlan | None = None,
    *,
    backend: str = "auto",
    compile: bool = True,
    methods: tuple[str, ...] | str = ("__call__",),
    max_programs: int = 8,
    example: Any = None,
    **placement: Any,
) -> Backend:
    """Select Niagara beneath an existing Caskade or Caustics simulator.

    Parameters
    ----------
    root : caskade.Module
        Existing simulator; its imports and construction remain unchanged.
    plan : DevicePlan or None, optional
        Explicit plan. Supply either this or DevicePlan keyword arguments.
    backend : str, optional
        Tensor compiler backend, such as ``"auto"``, ``"inductor"``,
        ``"torchscript"``, ``"cuda_graph"``, or ``"eager"``.
    compile : bool, optional
        Capture selected methods. Set False for routed dynamic-Python execution.
    methods : str or tuple of str, optional
        Class-method entry points to compile. Defaults to ``("__call__",)``.
    max_programs : int, optional
        Positive limit on cached method/input/mode specializations.
    example : object or None, optional
        Packed parameters for eager preparation of a one-argument default call.
        Use ``handle.prepare`` for entry points with additional arguments.
    **placement : object
        DevicePlan arguments, including ``default``, ``modules`` and ``parameters``.

    Returns
    -------
    Backend
        Persistent handle for preparation, sessions, clearing and restoration.

    Raises
    ------
    PlacementError
        If ownership, devices, pointer dependencies or activation are unsafe.
    TypeError
        If root is not a Module or a plan and placement keywords are combined.
    ValueError
        If the cache limit or selected methods are invalid.
    CaptureError
        If supplied example preparation fails.

    Notes
    -----
    Calls prepare lazily unless ``example`` is given. Ordinary requires-grad
    calls receive their own autograd-capable program. Native active contexts
    use routed execution. No installed source, global CUDA default or global
    Caskade class is patched. Use separate simulator instances for concurrent
    callers. Compilation specializes fixed metadata and model configuration.

    Examples
    --------
    Given an existing simulator ``sim`` and packed parameters ``params``::

        engine = enable(sim, default="cuda:0", modules={"source": "cuda:1"})
        image = sim(params)
        engine.close()
    """
    if not isinstance(root, Module):
        raise TypeError("Enable Niagara on an existing Caskade/Caustics Module")
    if getattr(root, "_niagara_backend", None) is not None:
        raise PlacementError("Niagara is already enabled on this simulator")
    if plan is not None and placement:
        raise TypeError("Supply a DevicePlan or placement keywords, not both")
    if not isinstance(max_programs, int) or max_programs < 1:
        raise ValueError("max_programs must be a positive integer")
    methods = (methods,) if isinstance(methods, str) else tuple(methods)
    if not compile:
        methods = ()
    if compile and not methods:
        raise ValueError("Select at least one method to compile")
    preflight_methods = {}
    for name in methods:
        if name in vars(root):
            raise ValueError(f"Select a callable class method, got {name!r}")
        method = getattr(root, name, None)
        if not callable(method):
            raise ValueError(f"Select a callable class method, got {name!r}")
        preflight_methods[name] = method
    handle = Backend(
        root,
        plan or DevicePlan(**placement),
        backend=backend,
        compile=compile,
        methods=methods,
        max_programs=max_programs,
        _preflight_methods=preflight_methods,
        _requires_example=example is not None,
    )
    try:
        if example is not None:
            handle.prepare(example)
    except BaseException:
        handle.close()
        raise
    return handle


def disable(root: Module) -> bool:
    """Restore native execution for one enabled simulator.

    Parameters
    ----------
    root : caskade.Module
        Simulator previously passed to :func:`enable`.

    Returns
    -------
    bool
        True if an enabled handle was closed, False if none was attached.

    Notes
    -----
    Closure co-locates model state on the plan's default device and preserves
    parameter identities and ties. See :meth:`Backend.close` for restrictions.
    """
    handle = getattr(root, "_niagara_backend", None)
    if handle is None:
        return False
    handle.close()
    return True
