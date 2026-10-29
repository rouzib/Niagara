"""Instance-local HDF5 adapters for placed Caskade simulators.

Standard parameter append failures roll back recorded dataset extents on a
best-effort basis. This is not an atomic transaction: storage/process failures
or custom writers may leave other changes. Pointer-evaluation errors propagate
instead of Caskade's silent skip, avoiding incomplete sample rows. CUDA tensor
attributes stored directly on an object are copied to the host for
HDF5. Save is rejected before writing when such attributes coexist with custom
lookup, descriptors or custom HDF writers, because their effects may change
snapshot ordering. Loading CUDA tensor attributes through custom setters is
also rejected. CPU-only attributes retain Caskade's load and save behavior.
Public persistence overrides are left to their subclasses, including their GPU
serialization responsibilities.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from warnings import warn

import torch
from caskade import Module, Node, Param, SaveStateWarning

_ABSENT = object()


@contextmanager
def _file(handle, mode):
    if isinstance(handle, str):
        if not (handle.endswith(".h5") or handle.endswith(".hdf5")):
            raise NotImplementedError(
                "Only HDF5 files ('.h5') are currently supported for saving state"
            )
        import h5py

        with h5py.File(handle, mode) as opened:
            yield opened
    else:
        yield handle


def _groups(node, group, seen=None):
    """Walk graph paths with the same first-encounter shared-node behavior as Caskade."""
    if seen is None:
        seen = set()
    if id(node) in seen:
        return
    seen.add(id(node))
    yield node, group
    for key, child in node.children.items():
        if key in group:
            yield from _groups(child, group[key], seen)


def _owner(node, path):
    parts = path.split(".")
    for part in parts[:-1]:
        node = getattr(node, part)
    return node, parts[-1]


class _MaskedAttrs(set):
    """Keep Caskade's attribute iteration order while masking tensor entries."""

    def __init__(self, original, excluded):
        self._order = tuple(name for name in original if name not in excluded)
        super().__init__(self._order)

    def __iter__(self):
        for name in self._order:
            if name in self:
                yield name
        for name in set.__iter__(self):
            if name not in self._order:
                yield name


def _static_tensor(node, path):
    """Find a stored tensor without invoking properties or custom getters."""
    value = node
    for part in path.split("."):
        value = inspect.getattr_static(value, part, _ABSENT)
        if value is _ABSENT or (hasattr(value, "__get__") and not isinstance(value, torch.Tensor)):
            return None
    return value if isinstance(value, torch.Tensor) else None


def _descriptor_backed(node, path):
    """Detect attributes whose evaluation may run user code without invoking it."""
    value = node
    for part in path.split("."):
        if type(value).__getattribute__ is not object.__getattribute__:
            return True
        value = inspect.getattr_static(value, part, _ABSENT)
        if value is _ABSENT or (hasattr(value, "__get__") and not isinstance(value, torch.Tensor)):
            return True
    return False


def _save(original):
    @functools.wraps(original)
    def wrapped(self, saveto, appendable=False):
        tensors = {}
        saved_sets = {}
        groups_before = {}
        nodes = self.topological_ordering()
        # A property can mutate a tensor before native Caskade saves it. Since
        # CUDA values require a host snapshot, reject that ambiguous ordering
        # before the native writer can create or truncate any file.
        descriptors = [
            (node, attr)
            for node in nodes
            for attr in node.saveattrs
            if _descriptor_backed(node, attr)
        ]
        cuda_tensors = [
            (node, attr)
            for node in nodes
            for attr in node.saveattrs
            if (value := _static_tensor(node, attr)) is not None and value.is_cuda
        ]
        custom_writers = [
            node
            for node in nodes
            if inspect.getattr_static(type(node), "_save_state_hdf5")
            is not (Param._save_state_hdf5 if isinstance(node, Param) else Node._save_state_hdf5)
        ]
        if cuda_tensors and (descriptors or custom_writers):
            raise RuntimeError(
                "Cannot safely save CUDA tensor attributes together with descriptor-backed "
                "saveattrs, custom attribute lookup, or custom HDF5 writers: close placement "
                "and save on CPU, or remove these custom behaviors."
            )
        try:
            for node in nodes:
                values = {}
                for attr in node.saveattrs:
                    value = _static_tensor(node, attr)
                    if value is not None and value.is_cuda:
                        values[attr] = value.detach().cpu().numpy()
                if values:
                    tensors[node] = values
                    saved_sets[node] = node.saveattrs
                    object.__setattr__(node, "saveattrs", _MaskedAttrs(node.saveattrs, values))
                groups_before[node] = vars(node).get("_h5group", _ABSENT)
            result = original(self, saveto, appendable=appendable)
        finally:
            for node, attrs in saved_sets.items():
                object.__setattr__(node, "saveattrs", attrs)
            for node, old in groups_before.items():
                if old is _ABSENT:
                    if "_h5group" in vars(node):
                        object.__delattr__(node, "_h5group")
                else:
                    object.__setattr__(node, "_h5group", old)
        if tensors:
            with _file(saveto, "a") as handle:
                for node, group in _groups(self, handle[self.name]):
                    for attr, value in tensors.get(node, {}).items():
                        try:
                            group.attrs[attr] = value
                        except TypeError:
                            warn(
                                SaveStateWarning(
                                    f"attribute '{attr}' of type {type(value)} cannot be saved to HDF5"
                                ),
                                stacklevel=2,
                            )
                        except Exception as error:
                            warn(
                                SaveStateWarning(
                                    f"Unable to save attribute '{attr}' due to: {error}"
                                ),
                                stacklevel=2,
                            )
        return result

    return wrapped


def _append_param(self, h5group):
    # Caskade follows every child edge and appends each shared Param once.
    for key, child in self.children.items():
        child._append_state_hdf5(h5group[key])
    if hasattr(self, "appended"):
        return
    self.appended = True
    # Propagate failed pointer evaluation instead of silently dropping a row.
    value = self.value
    if value is not None:
        # Native Caskade evaluates a non-None pointer twice. Preserve that
        # behavior while moving its second result to host before assignment.
        value = self.value
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        dataset = h5group["value"]
        dataset.resize((dataset.shape[0] + 1,) + dataset.shape[1:])
        dataset[-1] = value


def _append(original):
    @functools.wraps(original)
    def wrapped(self, saveto):
        # Snapshot all existing datasets before invoking Caskade, including
        # aliases. Only newly appended rows can be modified by the native path.
        with _file(saveto, "a") as handle:
            original_markers = {
                node: vars(node).get("appended", _ABSENT)
                for node in self.topological_ordering()
                if isinstance(node, Param)
            }
            snapshots = []
            if self.name in handle:
                for node, group in _groups(self, handle[self.name]):
                    if isinstance(node, Param) and "value" in group:
                        dataset = group["value"]
                        snapshots.append((dataset, dataset.shape))
            try:
                return original(self, handle)
            except BaseException as error:
                # Undo *all* successful earlier appends, not just the failing
                # dataset. Preserve the triggering error if rollback also fails.
                for dataset, shape in reversed(snapshots):
                    try:
                        if dataset.shape != shape:
                            dataset.resize(shape)
                    except Exception as rollback_error:
                        error.add_note(f"HDF5 rollback failed: {rollback_error}")
                raise
            finally:
                for node, marker in original_markers.items():
                    if marker is _ABSENT:
                        if "appended" in vars(node):
                            object.__delattr__(node, "appended")
                    else:
                        object.__setattr__(node, "appended", marker)

    return wrapped


def _load(original):
    @functools.wraps(original)
    def wrapped(self, loadfrom, index=-1, **kwargs):
        # Caskade loads HDF5 attributes as NumPy values. Retain device and
        # tensor type for attributes that were tensors in the placed model.
        tensors = []
        for node in self.topological_ordering():
            for attr in node.saveattrs:
                value = _static_tensor(node, attr)
                if value is not None and value.is_cuda:
                    if _descriptor_backed(node, attr):
                        raise RuntimeError(
                            "Cannot safely load CUDA tensor saveattrs through custom attribute lookup"
                        )
                    obj, _ = _owner(node, attr)
                    if type(obj).__setattr__ not in (
                        object.__setattr__,
                        Node.__setattr__,
                        Module.__setattr__,
                    ):
                        raise RuntimeError(
                            "Cannot safely load CUDA tensor saveattrs through a custom attribute setter"
                        )
                    tensors.append((node, attr, value))
        result = original(self, loadfrom, index=index, **kwargs)
        for node, attr, old in tensors:
            obj, name = _owner(node, attr)
            loaded = getattr(obj, name)
            if not isinstance(loaded, torch.Tensor):
                value = torch.as_tensor(loaded, dtype=old.dtype, device=old.device)
                value.requires_grad_(old.requires_grad)
                object.__setattr__(obj, name, value)
        return result

    return wrapped


def module_persistence_methods(cls):
    """Build persistence wrappers for one placed Module instance class."""
    methods = {}
    # Public overrides can have arbitrary signatures and side effects. Leave
    # those methods to their subclass, rather than wrapping a different API.
    for name, factory in (("save_state", _save), ("append_state", _append), ("load_state", _load)):
        if inspect.getattr_static(cls, name) is inspect.getattr_static(Node, name):
            methods[name] = factory(getattr(cls, name))
    return methods


def param_persistence_methods(cls):
    """Convert Param append values to host arrays without moving the Param."""
    if inspect.getattr_static(cls, "_append_state_hdf5") is not inspect.getattr_static(
        Param, "_append_state_hdf5"
    ):
        # Respect custom subclass implementations; the public wrapper still
        # rolls back dataset extents and markers if the custom method fails.
        return {}
    return {"_append_state_hdf5": _append_param}
