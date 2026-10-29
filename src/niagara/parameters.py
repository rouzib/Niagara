"""Device-aware adapters for Caskade's parameter packing utilities."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import getattr_static

import torch

_gather_home = ContextVar("niagara_parameter_gather_home", default=None)


@contextmanager
def _input_device_bounds(runtime, parameters, name):
    """Put each valid range on the device of the tensor being transformed.

    Caskade retains the original transform and its packing/group semantics.
    The temporary instance attributes and bounds are restored even on error.
    A placed simulator, like Caskade's contexts, remains single-caller.
    """
    absent = object()
    originals = []
    try:
        for param in parameters:
            transform = getattr(param, name)
            descriptor = getattr_static(type(param), name, None)
            data_descriptor = descriptor is not None and hasattr(descriptor, "__set__")
            original = type(param) if data_descriptor else vars(param).get(name, absent)
            originals.append((param, original, data_descriptor))

            def adapt(value, *, param=param, transform=transform):
                bounds = param._valid
                try:
                    if isinstance(value, torch.Tensor):
                        param._valid = runtime.move(bounds, value.device)
                    return transform(value)
                finally:
                    param._valid = bounds

            if data_descriptor:
                # Do not invoke a custom property setter: even setting its
                # original value back can alter unrelated instance state.
                # Intercept this lookup in a temporary instance-local class.
                original_class = type(param)

                def access(self, key, *, cls=original_class, fn=adapt, target=name):
                    return fn if key == target else cls.__getattribute__(self, key)

                cls = type(
                    f"NiagaraValid{original_class.__name__}",
                    (original_class,),
                    {"__getattribute__": access, "__module__": __name__},
                )
                object.__setattr__(param, "__class__", cls)
            else:
                object.__setattr__(param, name, adapt)
        yield
    finally:
        for param, original, data_descriptor in reversed(originals):
            if data_descriptor:
                object.__setattr__(param, "__class__", original)
            elif original is absent:
                if name in vars(param):
                    object.__delattr__(param, name)
            else:
                object.__setattr__(param, name, original)



def parameter_methods(runtime, module):
    """Build instance-local adapters without replacing Caskade's implementation."""
    native_get = module.get_values
    native_attribute = module._getattr_or_callable
    native_to_valid = module.to_valid
    native_from_valid = module.from_valid

    @wraps(getattr(native_get, "__func__", native_get))
    def get_values(self, scheme="array", *args, **kwargs):
        token = _gather_home.set(runtime.homes[module] if scheme.lower() in ("array", "tensor") else None)
        try:
            return native_get(scheme, *args, **kwargs)
        finally:
            _gather_home.reset(token)

    def get_attribute(self, obj, attribute):
        value = native_attribute(obj, attribute)
        target = _gather_home.get()
        return runtime.move(value, target) if target is not None else value

    def selected_parameters(args, kwargs):
        parameters = kwargs.get("param_list", args[1] if len(args) > 1 else None)
        if parameters is None:
            parameters = module.dynamic_params
        group = kwargs.get("group", args[2] if len(args) > 2 else None)
        if group is not None and len(module.dynamic_param_groups) > 1:
            parameters = tuple(param for param in parameters if param.group == group)
        return parameters

    @wraps(getattr(native_to_valid, "__func__", native_to_valid))
    def to_valid(self, *args, **kwargs):
        with _input_device_bounds(runtime, selected_parameters(args, kwargs), "to_valid"):
            return native_to_valid(*args, **kwargs)

    @wraps(getattr(native_from_valid, "__func__", native_from_valid))
    def from_valid(self, *args, **kwargs):
        with _input_device_bounds(runtime, selected_parameters(args, kwargs), "from_valid"):
            return native_from_valid(*args, **kwargs)

    return {
        "get_values": get_values,
        "_getattr_or_callable": get_attribute,
        "to_valid": to_valid,
        "from_valid": from_valid,
    }
