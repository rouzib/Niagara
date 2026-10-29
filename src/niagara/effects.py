"""Recognize common Python effects that cannot be replayed by a tensor graph.

This is a conservative preflight, not a proof that a callable is pure. It
never executes user code. Unknown closures and opaque objects remain the
caller's responsibility.
"""

from __future__ import annotations

import contextlib
import dis
import inspect
import types

from caskade import Node
from caskade.base import meta as Metadata

from .placement import graph_paths


def _random_object(value):
    """Identify Python/NumPy random modules, functions and generator objects."""
    if isinstance(value, types.ModuleType):
        return value.__name__ == "random" or value.__name__.startswith("numpy.random")
    module = ""
    owner = None
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        module = getattr(value, "__module__", "") or ""
        owner = getattr(value, "__self__", None)
    owner_module = type(owner).__module__ if owner is not None else ""
    value_module = type(value).__module__
    return any(
        name == "random" or name.startswith("numpy.random")
        for name in (module, owner_module, value_module)
    )


def _is_active_cache(value):
    """Recognize the published Caskade cache descriptor without binding it."""
    cls = type(value)
    return cls.__module__ == "caskade.decorators" and cls.__name__ == "active_cache"


def _queue_static_attribute(value, pending):
    """Inspect method/property/descriptor code without binding a descriptor."""
    if _is_active_cache(value):
        pending.append(inspect.getattr_static(value, "func"))
    elif isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    if isinstance(value, property):
        if value.fget is not None:
            pending.append(value.fget)
    elif inspect.isfunction(value):
        pending.append(value)
    elif value is not None:
        getter = inspect.getattr_static(type(value), "__get__", None)
        if inspect.isfunction(getter):
            pending.append(getter)


def python_effect(function, root=None):
    """Return a reason for recognized side effects, without calling the function.

    Inspect the selected method and referenced model methods and user functions.
    References to Python or NumPy RNG and explicit printing, input or file opens
    route through native execution. Tensor random operators remain capturable.
    This cannot inspect all dynamic dispatch, arbitrary object state or effects.
    """
    if not inspect.isroutine(function) and callable(function):
        function = function.__call__
    nodes = list(graph_paths(root)[1]) if isinstance(root, Node) else []
    seen_nodes = {id(node) for node in nodes}
    seen_values = set()
    metadata_objects = []

    def discover(value):
        if isinstance(value, Node):
            for node in graph_paths(value)[1]:
                if id(node) not in seen_nodes:
                    seen_nodes.add(id(node))
                    nodes.append(node)
            return
        if isinstance(value, (Metadata, types.SimpleNamespace, dict, list, tuple)):
            if id(value) in seen_values:
                return
            seen_values.add(id(value))
            if isinstance(value, (Metadata, types.SimpleNamespace)):
                children = vars(value).values()
            elif isinstance(value, dict):
                children = value.values()
            else:
                children = value
            if isinstance(value, (Metadata, types.SimpleNamespace)):
                metadata_objects.append(value)
            for child in children:
                discover(child)
        elif value is not None and not isinstance(value, (str, bytes, int, float, bool)):
            metadata_objects.append(value)

    index = 0
    while index < len(nodes):
        discover(vars(nodes[index]).get("meta"))
        index += 1
    pending = [function]
    seen = set()
    while pending:
        function = pending.pop()
        function = getattr(function, "__func__", function)
        code = getattr(function, "__code__", None)
        if code is None or id(function) in seen:
            continue
        seen.add(id(function))
        wrapped = getattr(function, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)
        globals_ = getattr(function, "__globals__", {})
        module = globals_.get("__name__", "")
        if module.startswith(("caskade.", "niagara.")):
            continue
        names = set(code.co_names)
        if "meta" in names:
            # Follow only referenced metadata names and literal keys. Dynamic
            # indexing or getattr can select anything, so inspect all fields.
            metadata_names = names | {value for value in code.co_consts if isinstance(value, str)}
            dynamic_lookup = "getattr" in names or any(
                op.opname == "BINARY_SUBSCR" for op in dis.get_instructions(code)
            )
            scanned = set()

            def contains_random(value, scanned=scanned, dynamic_lookup=dynamic_lookup, metadata_names=metadata_names):
                if _random_object(value):
                    return True
                if id(value) in scanned:
                    return False
                scanned.add(id(value))
                if isinstance(value, (Metadata, types.SimpleNamespace)):
                    children = vars(value)
                    return any(
                        contains_random(child)
                        for key, child in children.items()
                        if dynamic_lookup or key in metadata_names
                    )
                if isinstance(value, dict):
                    return any(
                        contains_random(child)
                        for key, child in value.items()
                        if dynamic_lookup or key in metadata_names
                    )
                if isinstance(value, (list, tuple)):
                    return any(contains_random(child) for child in value)
                return False

            if any(contains_random(vars(node).get("meta")) for node in nodes):
                return "Python or NumPy randomness"
            for node in nodes:
                if node is not root and callable(node):
                    method = inspect.getattr_static(type(node), "__call__", None)
                    if inspect.isfunction(method) or _is_active_cache(method):
                        _queue_static_attribute(method, pending)
            for obj in metadata_objects:
                if isinstance(obj, (Metadata, types.SimpleNamespace)):
                    for name in names:
                        candidate = vars(obj).get(name)
                        if inspect.isfunction(candidate) or inspect.ismethod(candidate):
                            pending.append(candidate)
                for name in names:
                    candidate = inspect.getattr_static(type(obj), name, None)
                    _queue_static_attribute(candidate, pending)
        if names & {"print", "open", "input"}:
            return "Python I/O"
        instructions = tuple(dis.get_instructions(code))
        if any(
            op.opname == "IMPORT_NAME"
            and (
                op.argval == "random"
                or op.argval.startswith("numpy.random")
                or (op.argval == "numpy" and "random" in names)
            )
            for op in instructions
        ):
            return "Python or NumPy randomness"
        direct_self_attributes = {
            operation.argval
            for previous, operation in zip(instructions, instructions[1:])
            if previous.opname == "LOAD_FAST"
            and previous.argval in {"self", "cls"}
            and operation.opname in {"LOAD_METHOD", "LOAD_ATTR"}
        }
        if any(op.opname in {"STORE_GLOBAL", "DELETE_GLOBAL", "STORE_DEREF", "DELETE_DEREF"} for op in instructions):
            return "Python state mutation"
        if (
            code.co_varnames
            and code.co_varnames[0] in {"self", "cls"}
            and any(op.opname in {"STORE_ATTR", "DELETE_ATTR"} for op in instructions)
        ):
            return "Python state mutation"
        closure = {}
        for name, cell in zip(code.co_freevars, getattr(function, "__closure__", ()) or ()):
            with contextlib.suppress(ValueError):
                closure[name] = cell.cell_contents
        subscripts = [
            index for index, op in enumerate(instructions)
            if op.opname in {"STORE_SUBSCR", "DELETE_SUBSCR"}
        ]
        if subscripts:
            # Identify simple fresh local containers, including list(constants).
            # A copied container may hold aliases, so nested writes stay unsafe.
            assignments = {}
            for index, op in enumerate(instructions):
                if op.opname != "STORE_FAST":
                    continue
                recent = instructions[max(0, index - 6):index]
                fresh = bool(recent) and (
                    recent[-1].opname in {"BUILD_LIST", "BUILD_MAP", "BUILD_SET", "BUILD_CONST_KEY_MAP"}
                    or (
                        recent[-1].opname == "CALL"
                        and any(
                            item.opname == "LOAD_GLOBAL"
                            and item.argval in {"list", "dict", "set"}
                            and item.argval not in globals_
                            for item in recent
                        )
                    )
                )
                assignments.setdefault(op.argval, []).append(fresh)
            fresh_locals = {name for name, choices in assignments.items() if all(choices)}
            unsafe_store = any(
                index < 2
                or instructions[index - 2].opname != "LOAD_FAST"
                or instructions[index - 2].argval not in fresh_locals
                for index in subscripts
            )
            # Only treat externally owned Python containers as state. A local
            # tensor indexed assignment must still reach normal capture checks.
            referenced = [globals_.get(name) for name in names]
            referenced.extend(closure.values())
            seen_metadata = set()

            def inspect_metadata(value, seen_metadata=seen_metadata, referenced=referenced, names=names):
                if id(value) in seen_metadata:
                    return
                seen_metadata.add(id(value))
                if isinstance(value, (Metadata, types.SimpleNamespace)):
                    attributes = vars(value)
                    referenced.extend(attributes.get(name) for name in names)
                    for child in attributes.values():
                        inspect_metadata(child)
                elif isinstance(value, dict):
                    for child in value.values():
                        inspect_metadata(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        inspect_metadata(child)

            for node in nodes:
                data = vars(node)
                referenced.extend(data.get(name) for name in names)
                inspect_metadata(data.get("meta"))
            if unsafe_store and any(isinstance(value, (list, dict, set)) for value in referenced):
                return "Python state mutation"
        mutations = {"append", "extend", "insert", "update", "add", "discard", "remove", "pop", "clear"}
        for index, operation in enumerate(instructions):
            if operation.opname not in {"LOAD_METHOD", "LOAD_ATTR"} or operation.argval not in mutations:
                continue
            previous = instructions[max(0, index - 2):index]
            for owner in previous:
                if owner.opname in {"LOAD_DEREF", "LOAD_GLOBAL"}:
                    value = closure.get(owner.argval, globals_.get(owner.argval))
                    if isinstance(value, (list, dict, set)):
                        return "Python state mutation"
                if owner.opname == "LOAD_ATTR":
                    for node in nodes:
                        if isinstance(vars(node).get(owner.argval), (list, dict, set)):
                            return "Python state mutation"
        for name in names:
            value = globals_.get(name)
            if _random_object(value):
                return "Python or NumPy randomness"
            if isinstance(value, types.ModuleType) and value.__name__ == "numpy" and "random" in names:
                return "NumPy randomness"
            if inspect.isfunction(value):
                module = getattr(value, "__module__", "") or ""
                if not module.startswith(("torch", "numpy", "caskade", "niagara")):
                    pending.append(value)
        for cell in getattr(function, "__closure__", ()) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if _random_object(value):
                return "Python or NumPy randomness"
            if isinstance(value, types.ModuleType) and value.__name__ == "numpy" and "random" in names:
                return "NumPy randomness"
            if inspect.isfunction(value) or inspect.ismethod(value):
                pending.append(value)
        for node in nodes:
            for name in names:
                direct = vars(node).get(name)
                metadata = vars(node).get("meta")
                nested = vars(metadata).get(name) if hasattr(metadata, "__dict__") else None
                if _random_object(direct) or _random_object(nested):
                    return "Python or NumPy randomness"
                child = vars(node).get("children", {}).get(name)
                for candidate in (direct, nested, child):
                    if isinstance(candidate, Node) and callable(candidate):
                        method = inspect.getattr_static(type(candidate), "__call__", None)
                        if inspect.isfunction(method) or _is_active_cache(method):
                            _queue_static_attribute(method, pending)
                value = inspect.getattr_static(type(node), name, None)
                candidate = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
                owner_name = getattr(function, "__qualname__", "").rsplit(".", 1)[0]
                same_owner = any(cls.__qualname__ == owner_name for cls in type(node).__mro__)
                if (
                    inspect.isfunction(candidate)
                    and getattr(candidate, "__wrapped__", None) is not None
                ) or _is_active_cache(candidate) or (same_owner and name in direct_self_attributes):
                    _queue_static_attribute(value, pending)
    return None
