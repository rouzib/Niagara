"""Prepare model-state checks once, rather than rebuilding snapshots per call."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from caskade import Node
from caskade.base import meta as Metadata

from .placement import graph_paths


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _tensor_state(tensor):
    return (
        _version(tensor),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.requires_grad,
        tensor.storage_offset(),
    )


class StateGuard:
    """Specialize checks to known fields while retaining mutation detection.

    Generated expressions only contain repr-quoted field names and numeric
    indices. Model values are held in a closure, never interpolated as code.
    Immutable containers specialize by identity; mutable containers and tensors
    are checked recursively, with shared tensor state checked only once.

    Parameters
    ----------
    root : caskade.Node or None
        Root whose discoverable configuration should remain fixed. None leaves
        external-state management to the caller.

    Notes
    -----
    Bookkeeping caches and active dynamic values are deliberately excluded.
    Arbitrary opaque Python-object internals are not exhaustively inspected.
    Tensor checks use identity, available version counters and metadata; no
    device-to-host tensor-value comparison or global CUDA synchronization occurs.
    """

    def __init__(self, root: Node | None, *, include_metadata: bool = True) -> None:
        self.root = root
        if root is None:
            self._check = lambda: True
            return
        ignored = {"_memos", "_parents", "_niagara_placement", "_value"}
        if not include_metadata:
            ignored.add("meta")
        nodes = list(graph_paths(root)[1])
        if include_metadata:
            # Metadata can hold a separately owned Caskade helper graph. Guard
            # its known state too, without changing its placement or ownership.
            seen_nodes = {id(node) for node in nodes}
            seen_values = set()

            def discover(value):
                if isinstance(value, Node):
                    for node in graph_paths(value)[1]:
                        if id(node) not in seen_nodes:
                            seen_nodes.add(id(node))
                            nodes.append(node)
                    return
                if isinstance(value, (Metadata, SimpleNamespace, dict, list, tuple)):
                    if id(value) in seen_values:
                        return
                    seen_values.add(id(value))
                    if isinstance(value, (Metadata, SimpleNamespace)):
                        values = vars(value).values()
                    elif isinstance(value, dict):
                        values = value.values()
                    else:
                        values = value
                    for child in values:
                        discover(child)

            index = 0
            while index < len(nodes):
                discover(vars(nodes[index]).get("meta"))
                index += 1
        nodes = tuple(nodes)
        dictionaries = tuple(vars(node) for node in nodes)
        constants, tensors, tensor_ids = [], [], set()
        metadata_ids, container_ids = set(), set()
        lines = ["def check():", "    try:"]

        def keep(value):
            constants.append(value)
            return f"c[{len(constants) - 1}]"

        def emit(condition):
            lines.append(f"        if {condition}: return False")

        def mutable(value):
            if isinstance(value, (torch.Tensor, Metadata, SimpleNamespace, list, dict)):
                return True
            return isinstance(value, tuple) and any(mutable(v) for v in value)

        def guard(expression, value):
            expected = keep(value)
            if isinstance(value, torch.Tensor):
                emit(f"{expression} is not {expected}")
                if id(value) not in tensor_ids:
                    tensor_ids.add(id(value))
                    tensors.append((value, _tensor_state(value)))
            elif isinstance(value, (Metadata, SimpleNamespace)):
                emit(f"{expression} is not {expected}")
                if id(value) in metadata_ids:
                    return
                metadata_ids.add(id(value))
                attributes = vars(value)
                keys = keep(frozenset(attributes))
                emit(f"frozenset(vars({expression})) != {keys}")
                for name, child in attributes.items():
                    guard(f"vars({expression})[{name!r}]", child)
            elif isinstance(value, Node) or value is None or isinstance(value, (torch.dtype, bool)):
                emit(f"{expression} is not {expected}")
            elif isinstance(value, (str, int, float, torch.device)):
                expected_type = keep(type(value))
                emit(
                    f"type({expression}) is not {expected_type} or "
                    f"({expression} is not {expected} and {expression} != {expected})"
                )
            elif isinstance(value, tuple) and not mutable(value):
                emit(f"{expression} is not {expected}")
            elif isinstance(value, (tuple, list, dict)):
                if id(value) in container_ids:
                    emit(f"{expression} is not {expected}")
                    return
                container_ids.add(id(value))
                expected_type = keep(type(value))
                emit(
                    f"type({expression}) is not {expected_type} or len({expression}) != {len(value)}"
                )
                if isinstance(value, dict):
                    keys = keep(tuple(value))
                    emit(f"tuple({expression}) != {keys}")
                    for key, child in value.items():
                        guard(f"{expression}[{keep(key)}]", child)
                else:
                    for index, child in enumerate(value):
                        guard(f"{expression}[{index}]", child)
            else:
                emit(f"{expression} is not {expected}")

        for index, node in enumerate(nodes):
            dictionary = dictionaries[index]
            cls = keep(type(node))
            emit(f"type(n[{index}]) is not {cls} or vars(n[{index}]) is not d[{index}]")
            keys = keep(frozenset(dictionary))
            meaningful = keep(
                frozenset(
                    k for k in dictionary if k not in ignored and not k.startswith("_active_cache")
                )
            )
            emit(
                f"d[{index}].keys() != {keys} and "
                f"frozenset(k for k in d[{index}] if k not in ignored "
                f"and not k.startswith('_active_cache')) != {meaningful}"
            )
            for name, value in dictionary.items():
                if name not in ignored and not name.startswith("_active_cache"):
                    guard(f"d[{index}][{name!r}]", value)
        lines += [
            "        for tensor, state in tensors:",
            "            if tensor_state(tensor) != state: return False",
            "        return True",
            "    except (KeyError, TypeError, AttributeError):",
            "        return False",
        ]
        namespace = {
            "c": tuple(constants),
            "d": dictionaries,
            "n": nodes,
            "ignored": ignored,
            "tensors": tensors,
            "tensor_state": _tensor_state,
        }
        exec(compile("\n".join(lines), "<niagara-state-guard>", "exec"), namespace)  # noqa: S102
        self._check = namespace["check"]

    def unchanged(self) -> bool:
        """Check current inactivity and the prepared configuration fingerprint.

        Returns
        -------
        bool
            True when the root is absent or inactive and every specialized
            field/tensor check still matches. False indicates recapture is needed.
        """
        return (self.root is None or not self.root.active) and self._check()
