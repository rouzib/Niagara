Errors
======

.. autoexception:: niagara.NiagaraError

.. autoexception:: niagara.PlacementError

.. autoexception:: niagara.CaptureError

.. autoexception:: niagara.GuardError

Placement errors identify invalid ownership/device configurations. Capture
errors report unsupported specialization. Guard errors stop execution when
inputs, modes or known static state no longer match. Keep the original
exception context when reporting a compiler failure.
