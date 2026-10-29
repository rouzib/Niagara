Example Caustics workloads
==========================

These helpers live in the source checkout under ``examples/niagara_examples``.
They build upstream Caustics objects for reproducible integration checks and
are not included in the Niagara wheel. Download the
`example helpers <../_static/niagara-example-helpers.zip>`_ for use with a
downloaded notebook; extract the archive beside the notebook.

.. automodule:: examples.niagara_examples.caustics
   :members: ImageProblem, image_problem, pixelated_scene, multiplane_scene

The multiplane example fixes the known static redshift order and checks it
against the stock eager formulation. Capture does not support general sorting
of arbitrary dynamic redshift tensors.
