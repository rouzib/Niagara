"""Instance-local Caskade boundary routing; no monkeypatches of Caskade classes."""

from __future__ import annotations

import contextvars
import functools
import inspect
import re
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from caskade import ActiveContext, Module, Node, OverrideParam, Param, active_cache
from torch.overrides import TorchFunctionMode
from torch.utils._device import _device_constructors
from torch.utils._pytree import tree_flatten, tree_map

from .errors import PlacementError
from .parameters import parameter_methods
from .persistence import module_persistence_methods, param_persistence_methods

_home = contextvars.ContextVar("niagara_home", default=None)
_scope = contextvars.ContextVar("niagara_scope", default=None)
_factory_home = contextvars.ContextVar("niagara_factory_home", default=None)
_factory_state = threading.local()


class _FactoryMode(TorchFunctionMode):
    """Set missing constructor devices without changing PyTorch's global default."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        home = _factory_home.get()
        if (
            home is not None
            and func in _device_constructors()
            and kwargs.get("device") is None
            and kwargs.get("out") is None
        ):
            kwargs = {**kwargs, "device": home}
        return func(*args, **kwargs)


@contextmanager
def _factory_scope(home, enabled):
    token = _factory_home.set(home if enabled else None)
    try:
        if enabled and not getattr(_factory_state, "active", False):
            # Context copies can outlive this scope on the same or another
            # thread. Track the actual mode lifetime outside ContextVars.
            _factory_state.active = True
            try:
                with _FactoryMode():
                    yield
            finally:
                _factory_state.active = False
        else:
            yield
    finally:
        _factory_home.reset(token)


def _parameter_value(runtime, param, descriptor):
    """Evaluate pointers on their dependency device, preserving native caching."""
    home = runtime.homes[param]

    def value(self):
        if not self.pointer:
            result = descriptor.fget(self)
            if self.active and self._value is not None:
                result = runtime.move(result, home)
                self._value = result
            caller = _home.get()
            return runtime.move(result, caller) if caller is not None else result
        if self._value is not None:
            result = runtime.move(descriptor.fget(self), home)
        else:
            try:
                dependencies = {runtime.homes[child] for child in self.children.values()}
            except KeyError as exc:
                raise PlacementError(
                    "Pointer dependencies changed; disable and re-enable Niagara"
                ) from exc
            if len(dependencies) > 1:
                raise PlacementError("Pointer dependencies must remain on one device")
            calculation_home = next(iter(dependencies), home)
            home_token = _home.set(calculation_home)
            scope_token = _scope.set(f"{runtime.paths[self][0]}.value")
            try:
                with _factory_scope(calculation_home, runtime.plan.factory_device):
                    result = runtime.move(descriptor.fget(self), home)
            finally:
                _scope.reset(scope_token)
                _home.reset(home_token)
        if self.active:
            self._value = result
        caller = _home.get()
        return runtime.move(result, caller) if caller is not None else result

    return property(value, descriptor.fset, descriptor.fdel, descriptor.__doc__)


def device(value: str | torch.device) -> torch.device:
    """Normalize and validate a CPU or explicitly indexed CUDA device.

    Parameters
    ----------
    value : str or torch.device
        CPU device or indexed CUDA device such as ``"cuda:1"``.

    Returns
    -------
    torch.device
        Normalized available home; CPU indices normalize to plain ``cpu``.

    Raises
    ------
    PlacementError
        If the device is invalid, unavailable, unindexed CUDA, or unsupported.
    """
    if (
        isinstance(value, str)
        and (match := re.fullmatch(r"cuda:(\d+)", value))
        and int(match[1]) >= torch.cuda.device_count()
    ):
        raise PlacementError(f"CUDA device {value} is unavailable")
    try:
        result = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise PlacementError(f"Invalid device {value!r}") from exc
    if result.type == "cpu":
        return torch.device("cpu")
    if result.type != "cuda":
        raise PlacementError("Niagara supports explicit CPU and CUDA devices")
    if result.index is None:
        raise PlacementError("Use an explicit CUDA index, for example cuda:0")
    if not torch.cuda.is_available() or not 0 <= result.index < torch.cuda.device_count():
        raise PlacementError(f"CUDA device {result} is unavailable")
    return result


def move(
    value: Any,
    target: torch.device,
    *,
    non_blocking: bool = True,
    record: Callable[[torch.Tensor, torch.device], None] | None = None,
) -> Any:
    """Move tensor leaves of a pytree to a normalized device.

    Parameters
    ----------
    value : object
        Tensor or pytree containing tensor and unchanged non-tensor leaves.
    target : torch.device
        Normalized destination, normally returned by :func:`device`.
    non_blocking : bool, optional
        Request nonblocking copies to CUDA when supported by PyTorch.
    record : callable or None, optional
        Callback receiving each transferred tensor and destination.

    Returns
    -------
    object
        Equivalent pytree. Tensor leaves already on the destination are reused;
        cross-device copies retain ordinary autograd relationships.
    """

    def leaf(x):
        if isinstance(x, torch.Tensor) and x.device != target:
            if record is not None:
                record(x, target)
            return x.to(device=target, non_blocking=non_blocking and target.type == "cuda")
        return x

    if isinstance(value, torch.Tensor):
        return leaf(value)
    if not any(isinstance(x, torch.Tensor) and x.device != target for x in tree_flatten(value)[0]):
        return value
    return tree_map(leaf, value)


def graph_paths(root: Node) -> tuple[dict[str, Node], dict[Node, list[str]]]:
    """Enumerate root-qualified graph paths and aliases for every reachable node.

    Parameters
    ----------
    root : caskade.Node
        Common root of the graph to inspect.

    Returns
    -------
    paths : dict
        Mapping from dotted paths to node identities.
    aliases : dict
        Mapping from each node identity to all of its dotted paths.

    Raises
    ------
    PlacementError
        If traversal encounters a cycle.
    """
    paths, aliases = {}, {}

    def visit(node, path, ancestors):
        if node in ancestors:
            raise PlacementError(f"Cycle at {path}")
        paths[path] = node
        aliases.setdefault(node, []).append(path)
        for key, child in node.children.items():
            visit(child, f"{path}.{key}", ancestors | {node})

    visit(root, root.name, set())
    return paths, aliases


def _inactive(nodes):
    if any(node.active for node in nodes):
        raise PlacementError("Cannot assign, restore, or change placement while a model is active")


@dataclass(frozen=True)
class DevicePlan:
    """Declare device homes for complete simulator regions and parameters.

    Parameters
    ----------
    default : str or torch.device, optional
        Root home, inherited by descendants without explicit assignments.
    modules : mapping, optional
        Module identities or dotted paths mapped to explicit CPU/CUDA homes.
        Paths may be root-relative or start with the root's name.
    parameters : mapping, optional
        Param identities or dotted paths mapped to explicit homes.
    non_blocking : bool, optional
        Request nonblocking CUDA transfers where PyTorch supports them.
    diagnostics : bool, optional
        Record routed calls, transfer counts and transferred tensor bytes.
    factory_device : bool, optional
        Give implicit PyTorch tensor constructors the executing region's device.
        Pointer calculations use their dependencies' device. Explicit device
        arguments, output buffers and tensor-derived constructors retain PyTorch semantics.
        Enabled by default; scoped to this call and thread, never global.

    Notes
    -----
    Shared nodes must inherit one unambiguous home or be assigned explicitly by
    identity. A pointer calculation's dependencies must share a device, while
    its result may cross a boundary. Ties preserve Param identity. The dataclass
    is frozen, but caller-supplied mappings should also remain unchanged.

    Examples
    --------
    Split the source region of an existing image simulator::

        plan = DevicePlan(default="cuda:0", modules={"source": "cuda:1"})
        placement = plan.apply(sim)
    """

    default: str | torch.device = "cpu"
    modules: Mapping[Module | str, str | torch.device] = field(default_factory=dict)
    parameters: Mapping[Param | str, str | torch.device] = field(default_factory=dict)
    non_blocking: bool = True
    diagnostics: bool = False
    factory_device: bool = True

    def validate(self, root: Module) -> dict[Node, torch.device]:
        """Resolve ownership and validate safety without migrating model state.

        Parameters
        ----------
        root : caskade.Module
            Common root containing every owner/dependency in the plan.

        Returns
        -------
        dict
            Resolved node-identity to normalized-device assignments.

        Raises
        ------
        PlacementError
            If devices, active state, aliases, external ownership or pointer
            dependencies make the plan unsafe, or Caskade is not using Torch.
        """
        return _resolve(root, self)

    def apply(self, root: Module) -> Placement:
        """Validate and apply this plan to one simulator.

        Parameters
        ----------
        root : caskade.Module
            Inactive simulator to migrate and route.

        Returns
        -------
        Placement
            Persistent placement handle; equivalent to ``assign_devices(root, self)``.
        """
        return assign_devices(root, self)


def _resolve(root, plan):
    if not isinstance(root, Module):
        raise PlacementError("A DevicePlan requires a Caskade Module")
    import caskade

    if caskade.backend.backend != "torch":
        raise PlacementError("Niagara requires Caskade's torch backend")
    paths, aliases = graph_paths(root)
    _inactive(aliases)
    explicit = {}
    for mapping, cls in ((plan.modules, Module), (plan.parameters, Param)):
        for key, value in mapping.items():
            if isinstance(key, str):
                key = paths.get(key, paths.get(f"{root.name}.{key}"))
            if key not in aliases or not isinstance(key, cls):
                raise PlacementError("Placement key must identify a node in this model")
            home = device(value)
            if key in explicit and explicit[key] != home:
                raise PlacementError(f"Conflicting explicit homes for {key.name}")
            explicit[key] = home
    inherited = {}

    def visit(node, home):
        home = explicit.get(node, home)
        inherited.setdefault(node, set()).add(home)
        # Pointer dependencies are references, not ownership regions.
        if isinstance(node, Param):
            return
        for child in node.children.values():
            visit(child, home)

    visit(root, device(plan.default))
    homes = {}
    for node, candidates in inherited.items():
        if len(candidates) != 1:
            raise PlacementError(
                f"Shared node {node.name} inherits {sorted(map(str, candidates))}; "
                "assign it explicitly by identity"
            )
        homes[node] = next(iter(candidates))
    # A parameter reachable only through a pointer needs a declared home.
    for node in aliases:
        if node not in homes:
            if node not in explicit:
                raise PlacementError(f"Declare a home for pointer-only dependency {node.name}")
            homes[node] = explicit[node]
    for node in homes:
        if isinstance(node, Param) and node.pointer:
            dependency_homes = {homes[c] for c in node.children.values()}
            if len(dependency_homes) > 1:
                raise PlacementError(
                    f"Pointer {node.name} combines dependencies on different devices; "
                    "co-locate its dependencies or express the computation in a routed Module"
                )
    for node in homes:
        for parent in node.parents:
            if parent not in aliases:
                raise PlacementError(
                    f"{node.name} is shared with a model outside this plan; include the common root"
                )
    return homes


def _caskade_method(value):
    """Remove only Caskade's forwarding wrappers, preserving user decorators."""
    found = False
    while hasattr(value, "__wrapped__"):
        filename = getattr(getattr(value, "__code__", None), "co_filename", "")
        if not filename.replace("\\", "/").endswith("caskade/decorators.py"):
            break
        found = True
        value = value.__wrapped__
    return value if found else None


def _contains_forward_wrapper(value):
    """Find a Caskade forward wrapper beneath transparent decorators."""
    seen = set()
    while value is not None and id(value) not in seen:
        seen.add(id(value))
        if _caskade_method(value) is not None:
            return True
        value = getattr(value, "__wrapped__", None)
    return False


def _decorated_boundary(method, runtime, module):
    """Route an outer decorator while leaving Caskade's forward wrapper intact."""
    home = runtime.homes[module]
    path = runtime.paths[module][0]

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        caller = _home.get()
        token = _home.set(home)
        scope_token = _scope.set(f"{path}.{method.__name__}")
        try:
            args, kwargs = runtime.move((args, kwargs), home)
            with _factory_scope(home, runtime.plan.factory_device):
                result = method(self, *args, **kwargs)
            if runtime.observations is not None:
                runtime.observations.append((path, str(home)))
            return runtime.move(result, caller if caller is not None else home)
        finally:
            _scope.reset(scope_token)
            _home.reset(token)

    return wrapped


class _RoutedActiveCache(active_cache):
    """Keep cached values on their region and route each caller's result."""

    def __init__(self, cache, runtime, module):
        super().__init__(cache.func)
        self.cache = cache
        self.runtime = runtime
        self.module = module

    def __set_name__(self, owner, name):
        # Caskade registers cache attributes and its reset method on the class.
        self.cache.__set_name__(owner, name)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        bound = self.cache.__get__(instance, owner)
        home = self.runtime.homes[self.module]

        @functools.wraps(bound)
        def wrapped(*args, **kwargs):
            caller = _home.get()
            token = _home.set(home)
            try:
                result = bound(*args, **kwargs)
            finally:
                _home.reset(token)
            return self.runtime.move(result, caller if caller is not None else home)

        return wrapped


def _boundary(method, runtime, module):
    signature = inspect.signature(method)
    names = tuple(signature.parameters)
    home = runtime.homes[module]
    path = runtime.paths[module][0]
    filled_local = all(
        runtime.homes[p] == home
        and (not p.pointer or all(runtime.homes[c] == home for c in p.children.values()))
        for p in module.children.values()
        if isinstance(p, Param)
    )

    def local(self, args, kwargs):
        kwargs = dict(kwargs)

        def invoke(args, filled=None):
            if filled is None:
                filled = {**self.fill_kwargs(names), **kwargs}
            if not filled_local:
                filled = runtime.move(filled, home)
            return method(self, *args, **filled)

        if self.online:
            with ExitStack() as stack:
                for key, value in tuple(kwargs.items()):
                    if key in self.children and isinstance(self.children[key], Param):
                        stack.enter_context(
                            OverrideParam(self.children[key], runtime.move(value, home))
                        )
                        del kwargs[key]
                return invoke(args)
        supplied = "params" in kwargs or bool(self.dynamic_params and args)
        if "params" in kwargs:
            params = kwargs.pop("params")
        elif self.dynamic_params and args:
            params, args = args[-1], args[:-1]
        with ActiveContext(self):
            if supplied:
                runtime.fill_params(self, params)
            # Native Caskade fills inside activation, once. This matters for
            # pointer caching and for an explicitly supplied invalid None.
            filled = {**self.fill_kwargs(names), **kwargs}
            if not supplied and not self.dynamic_params and args:
                try:
                    signature.bind(self, *args, **filled)
                except TypeError:
                    args = args[:-1]
            return invoke(args, filled)

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        caller = _home.get()
        token = _home.set(home)
        scope_token = _scope.set(f"{path}.{method.__name__}")
        try:
            args, kwargs = runtime.move((args, kwargs), home)
            with _factory_scope(home, runtime.plan.factory_device):
                result = local(self, args, kwargs)
            if runtime.observations is not None:
                runtime.observations.append((path, str(home)))
            return runtime.move(result, caller if caller is not None else home)
        finally:
            _scope.reset(scope_token)
            _home.reset(token)

    return wrapped


class Placement:
    """Own an applied device plan and its instance-local routing adapters.

    Parameters
    ----------
    root : caskade.Module
        Root of the validated simulator DAG.
    plan : DevicePlan
        Applied placement configuration.
    homes : dict
        Resolved node-identity to normalized-device assignments.

    Attributes
    ----------
    homes : dict
        Actual declared homes for all owned nodes.
    paths : dict
        Aliases of each reachable node, including shared identities.
    closed : bool
        Whether original dispatch classes have been restored.

    Notes
    -----
    Construct through :func:`assign_devices` or :meth:`DevicePlan.apply`.
    Closing restores dispatch classes but leaves tensor state at its assigned
    homes. Use a uniform plan before closure when native Caskade needs all
    tensors co-located. :class:`niagara.backend.Backend` performs that restoration
    automatically for its higher-level opt-in API.
    """

    def __init__(self, root: Module, plan: DevicePlan, homes: dict[Node, torch.device]) -> None:
        self.root, self.plan, self.homes = root, plan, homes
        self.paths = graph_paths(root)[1]
        self.originals = {}
        self._native_fill = {node: node.fill_params for node in homes if isinstance(node, Module)}
        self.transfers = Counter()
        self.bytes = Counter()
        self.observations = [] if plan.diagnostics else None
        self.closed = False

    def record(self, tensor: torch.Tensor, target: torch.device) -> None:
        """Accumulate one routed transfer, including underlying vmap storage.

        Parameters
        ----------
        tensor : torch.Tensor
            Source tensor before migration.
        target : torch.device
            Destination of this transfer.

        Notes
        -----
        Counters describe explicit routing, not all compiler-generated or
        implicit backward transfers. Internal vmap wrappers are unwrapped for
        byte accounting.
        """
        key = f"{tensor.device}->{target}"
        self.transfers[key] += 1
        # Caustics uses internal vmap: its logical scalar may represent an
        # entire coordinate grid. Count the underlying transferred storage.
        while torch._C._functorch.is_functorch_wrapped_tensor(tensor):
            tensor = torch._C._functorch.get_unwrapped(tensor)
        self.bytes[key] += tensor.numel() * tensor.element_size()

    def move(self, value: Any, target: torch.device) -> Any:
        """Move tensor leaves using this plan's transfer/diagnostic settings.

        Parameters
        ----------
        value : object
            Tensor or tensor pytree to route.
        target : torch.device
            Normalized destination.

        Returns
        -------
        object
            Equivalent pytree whose tensor leaves reside on the destination.
        """
        return move(
            value,
            target,
            non_blocking=self.plan.non_blocking,
            record=self.record if self.plan.diagnostics else None,
        )

    def _fill(self, module: Module, params: Any, *, dynamic: bool = True) -> None:
        # The instance-local from_valid adapter matches bounds to input devices
        # for both public utilities and this native Caskade fill path.
        self._native_fill[module](params, dynamic=dynamic)

    def fill_params(self, module: Module, params: Any, *, dynamic: bool = True) -> None:
        """Fill dynamic parameters and pack remote tensor parameter transfers.

        Parameters
        ----------
        module : caskade.Module
            Active region whose dynamic values are being filled.
        params : object
            Caskade-compatible packed parameter input.
        dynamic : bool, optional
            Fill dynamic parameters, or static parameters when False, matching
            the existing Caskade method.

        Notes
        -----
        A packed tensor with one parameter group is copied once per remote
        destination, then its parameter views are assigned to their homes.
        Other Caskade packing forms move each filled value to its declared home.
        """
        self._fill(module, params, dynamic=dynamic)
        parameters = module.dynamic_params if dynamic else module.static_params
        # Pack scalar/field parameters into one transfer per destination
        # instead of copying every individual view at each method boundary.
        if isinstance(params, torch.Tensor) and len(module.dynamic_param_groups) == 1:
            homes = {self.homes[p] for p in parameters}
            remote_homes = homes - {params.device}
            if remote_homes:
                values = {p: p._value for p in parameters}
                for home in remote_homes:
                    self._fill(module, self.move(params, home), dynamic=dynamic)
                    values.update({p: p._value for p in parameters if self.homes[p] == home})
                for param, value in values.items():
                    param._value = value
        else:
            # Lists, mappings and multiple parameter groups also need declared
            # homes. A nested call may refill them from a different region.
            for param in parameters:
                param._value = self.move(param._value, self.homes[param])

    def diagnostics(self) -> dict[str, Any]:
        """Return declared homes and accumulated explicit-routing observations.

        Returns
        -------
        dict
            ``homes`` maps paths to devices; ``transfers`` and ``bytes`` map
            source/destination pairs to counts; ``observed_calls`` is the call
            list when enabled, otherwise None. These are not overlap timings.
        """
        return {
            "transfers": dict(self.transfers),
            "bytes": dict(self.bytes),
            "observed_calls": self.observations,
            "homes": {self.paths[n][0]: str(d) for n, d in self.homes.items()},
        }

    def reset_diagnostics(self) -> None:
        """Clear observed calls and transfer counters, preserving device homes."""
        self.transfers.clear()
        self.bytes.clear()
        if self.observations is not None:
            self.observations.clear()

    def close(self) -> None:
        """Restore native instance classes without moving model tensors.

        Raises
        ------
        PlacementError
            If an owned graph node is active.

        Notes
        -----
        A split simulator can require uniform re-placement before its restored
        native Caskade methods are called. This low-level method does not undo
        migration. Repeated calls after successful closure do nothing.
        """
        if self.closed:
            return
        _inactive(self.homes)
        for module, cls in self.originals.items():
            object.__setattr__(module, "__class__", cls)
            delattr(module, "_niagara_placement")
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def assign_devices(root: Module, plan: DevicePlan | None = None, **kwargs: Any) -> Placement:
    """Validate and apply explicit device ownership to a simulator DAG.

    Parameters
    ----------
    root : caskade.Module
        Common inactive root containing every owning graph edge.
    plan : DevicePlan or None, optional
        Explicit placement plan. If omitted, construct one from ``kwargs``.
    **kwargs : object
        DevicePlan constructor arguments when no plan is supplied.

    Returns
    -------
    Placement
        Handle controlling instance-local forwarding and diagnostics.

    Raises
    ------
    PlacementError
        If validation fails or a previous placement has not been closed.

    Notes
    -----
    Module regions inherit explicit ``torch.device`` homes. Parameter identity
    and differentiable transfers are preserved. No global CUDA default,
    installed source or process-wide Caskade class is changed. Closure restores
    dispatch classes while leaving tensors at their current homes. The caller
    must own this mutable simulator exclusively during execution.
    """
    plan = plan or DevicePlan(**kwargs)
    homes = plan.validate(root)
    if any(getattr(n, "_niagara_placement", None) is not None for n in homes):
        raise PlacementError("Close the previous Niagara placement before assigning a new plan")
    runtime = Placement(root, plan, homes)
    # Resolve all methods before migrating or modifying any instance.
    classes = {}
    for node in homes:
        if isinstance(node, Param):
            descriptor = inspect.getattr_static(type(node), "value")
            classes[node] = type(
                f"Niagara{type(node).__name__}",
                (type(node),),
                {
                    "value": _parameter_value(runtime, node, descriptor),
                    **param_persistence_methods(type(node)),
                    "__module__": __name__,
                },
            )
        if not isinstance(node, Module):
            continue
        methods = {}
        for name in dir(type(node)):
            value = inspect.getattr_static(type(node), name)
            raw = _caskade_method(value)
            if raw is not None:
                methods[name] = _boundary(raw, runtime, node)
            elif (
                type(value).__module__ == "caskade.decorators"
                and hasattr(value, "func")
                and (raw := _caskade_method(value.func)) is not None
            ):
                # Preserve the active-cache descriptor outside the routed call.
                methods[name] = _RoutedActiveCache(
                    type(value)(_boundary(raw, runtime, node)), runtime, node
                )
            elif (
                type(value).__module__ == "caskade.decorators"
                and hasattr(value, "func")
                and _contains_forward_wrapper(value.func)
            ):
                # Keep Caskade's cache outside the complete decorated call.
                methods[name] = _RoutedActiveCache(
                    type(value)(_decorated_boundary(value.func, runtime, node)), runtime, node
                )
            elif inspect.isfunction(value) and _contains_forward_wrapper(value):
                # Preserve arbitrary outer decorators and the original Caskade
                # forwarding wrapper; only route the execution boundary here.
                methods[name] = _decorated_boundary(value, runtime, node)

        def fill_params(self, params, dynamic=True):
            return runtime.fill_params(self, params, dynamic=dynamic)

        methods["fill_params"] = fill_params
        methods.update(parameter_methods(runtime, node))
        methods.update(module_persistence_methods(type(node)))
        classes[node] = type(
            f"Niagara{type(node).__name__}", (type(node),), {**methods, "__module__": __name__}
        )
    # Migrate each Param once. Param.to would recursively move pointer dependencies.
    for node, home in homes.items():
        if isinstance(node, Param):
            node._device = home
            for name in ("_Param__value", "_value", "_valid"):
                if hasattr(node, name):
                    object.__setattr__(node, name, move(getattr(node, name), home))
        elif isinstance(node, Module):
            # Include private grids/kernels used by Caustics; exclude Node bookkeeping.
            for name, value in tuple(vars(node).items()):
                if name in ("_children", "_parents", "_subgraphs", "meta") or isinstance(
                    value, Node
                ):
                    continue
                object.__setattr__(node, name, move(value, home))
    for node, cls in classes.items():
        runtime.originals[node] = type(node)
        object.__setattr__(node, "__class__", cls)
        object.__setattr__(node, "_niagara_placement", runtime)
    return runtime
