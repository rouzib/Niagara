Graphs and profiling
====================

.. autofunction:: niagara.draw_model

With ``parameters=False``, child containers remain visible so that LightStack
components and grouped mass planes retain their connections. Only Param nodes
are hidden.

.. autofunction:: niagara.draw_execution

.. autofunction:: niagara.draw_schedule

.. autofunction:: niagara.draw_timeline

.. autoclass:: niagara.ExecutionTimeline
   :members: pipe

.. autofunction:: niagara.gallery

.. autofunction:: niagara.profile

Ownership/dependency views show structure. Use a measured trace to inspect
runtime copies, waits, durations or physical GPU overlap.

To draw recorded intervals with exact positions on a shared time axis::

    ng.profile(compiled_sim, (params[0],), "trace.json", iterations=1, warmup=3)
    timeline = ng.draw_timeline("trace.json", rankdir="TB")
    display(SVG(data=timeline.pipe(format="svg").decode("utf-8")))

Use ``rankdir="LR"`` for time running left to right. The box height in the
vertical view, or width in the horizontal view, is proportional to recorded
duration. CPU boxes show host operator intervals, including dispatch and nested
calls. GPU boxes show actual kernels and copies on recorded device streams.
Fused or replayed execution need not have one profiler event per captured node.
The trace can also be opened in a Chrome-trace-compatible viewer for navigation.
