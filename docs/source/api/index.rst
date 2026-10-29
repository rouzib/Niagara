API reference
=============

Niagara adds execution tools to Caskade's scientific model objects. Private
FX/stream helpers are implementation details. The Caustics workloads and MALA
helpers described here are examples in the source checkout, not installed APIs.

.. toctree::
   :maxdepth: 2

   backend
   placement
   compiler
   execution
   sampling
   visualization
   workloads
   errors
   caskade

Choose an interface
-------------------

.. list-table::
   :header-rows: 1

   * - Need
     - Public entry point
   * - Keep an existing simulator call site
     - :func:`niagara.enable`, :func:`niagara.disable`
   * - Explicit whole-region homes
     - :class:`niagara.DevicePlan`, :func:`niagara.assign_devices`
   * - Prepare a pure numerical program
     - :func:`niagara.compile_model`, :class:`niagara.CompiledModel`
   * - Batch a captured program
     - :func:`niagara.vmap`
   * - Independent simulator batches
     - :class:`niagara.ReplicaPool`
   * - Declared coarse task dependencies
     - :class:`niagara.AsyncGraphExecutor`, :class:`niagara.GraphFuture`
   * - Structural views and actual traces
     - :func:`niagara.draw_model`, :func:`niagara.profile`
