"""Exceptions raised by Niagara's ownership, capture and execution contracts."""


class NiagaraError(RuntimeError):
    """Base class for Niagara's own runtime-contract errors.

    Notes
    -----
    Underlying PyTorch/compiler exceptions can still propagate when they are
    outside Niagara's explicit validation. Catch this class for Niagara's
    placement, capture and guard errors rather than every possible backend error.
    """


class PlacementError(NiagaraError):
    """Indicate unsafe devices, graph ownership, active state or pointer homes."""


class CaptureError(NiagaraError):
    """Indicate that a callable cannot satisfy fixed tensor-program capture."""


class GuardError(NiagaraError):
    """Indicate changed input metadata, execution modes or guarded model state."""
