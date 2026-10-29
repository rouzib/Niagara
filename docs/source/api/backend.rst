Existing-call backend
=====================

Enable Niagara on selected simulator instances while keeping the upstream
import namespace. See :doc:`../user_guide/backend` for call caching, active
contexts and restoration.

.. autofunction:: niagara.enable

.. autofunction:: niagara.disable

.. autoclass:: niagara.Backend
   :members:
   :special-members: __enter__, __exit__
   :show-inheritance:
