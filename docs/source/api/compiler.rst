Capture and transforms
======================

.. autofunction:: niagara.compile_model

.. autoclass:: niagara.CompiledModel
   :members:
   :special-members: __call__

.. autofunction:: niagara.vmap

Capture the simulator first. Add its surrounding target through a transform,
apply derivatives/batching to the pure graph, then compile the tensor program.
An arbitrary outer compiler or vmap around an already prepared callable is
unsupported.
