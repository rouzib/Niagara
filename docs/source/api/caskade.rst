Upstream Caskade aliases
========================

Niagara re-exports four Caskade objects. They are the same objects used by the
installed dependency, so existing Caustics imports keep Caskade's behavior.

.. currentmodule:: niagara

.. py:class:: Module

   Alias of ``caskade.Module``. It owns the simulator DAG, parameters and
   native context-dependent evaluation. Niagara's planner accepts these same
   upstream instances.

.. py:class:: Param

   Alias of ``caskade.Param``. Static, dynamic and tied parameters retain
   their upstream identities and metadata when Niagara is enabled.

.. py:class:: ActiveContext

   Alias of ``caskade.ActiveContext``. The high-level backend delegates calls
   inside an active context to native routed parameter handling.

.. py:function:: forward(method)

   Alias of the upstream ``caskade.forward`` decorator. It marks the parameter
   evaluation boundaries that Niagara uses for instance-local routing.

These aliases preserve identity, for example ``ng.Param is ck.Param``. See
Caskade for constructor signatures and parameter semantics. Niagara reuses
those implementations.

See the `upstream Caskade project <https://github.com/Ciela-Institute/caskade>`_
for its graph, collection, backend and context API. Niagara requires Caskade's
Torch array backend for execution.
