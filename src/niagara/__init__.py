"""Niagara: explicit multi-GPU support as a separate Caskade add-on."""

from caskade import ActiveContext, Module, Param, forward

from .backend import Backend, disable, enable
from .compiler import CompiledModel, compile_model, vmap
from .errors import CaptureError, GuardError, NiagaraError, PlacementError
from .execution import AsyncGraphExecutor, GraphFuture, ReplicaPool
from .placement import DevicePlan, Placement, assign_devices
from .timeline import ExecutionTimeline, draw_timeline
from .visualization import draw_execution, draw_model, draw_schedule, gallery, profile

__version__ = "0.7.0"
__all__ = [
    "ActiveContext",
    "AsyncGraphExecutor",
    "Backend",
    "CaptureError",
    "CompiledModel",
    "DevicePlan",
    "ExecutionTimeline",
    "GraphFuture",
    "GuardError",
    "Module",
    "NiagaraError",
    "Param",
    "Placement",
    "PlacementError",
    "ReplicaPool",
    "assign_devices",
    "compile_model",
    "disable",
    "draw_execution",
    "draw_model",
    "draw_schedule",
    "draw_timeline",
    "enable",
    "forward",
    "gallery",
    "profile",
    "vmap",
]
