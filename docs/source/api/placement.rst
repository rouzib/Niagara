Placement
=========

Validate ownership before moving tensors. The plan checks identity/path
aliases, shared nodes and pointer dependencies; see
:doc:`../user_guide/placement`.

.. autoclass:: niagara.DevicePlan
   :members:

.. autofunction:: niagara.assign_devices

.. autoclass:: niagara.Placement
   :members:
   :special-members: __enter__, __exit__

``Placement.close`` restores the native classes and leaves tensors on their
current homes. The high-level backend also co-locates tensors when it closes.
