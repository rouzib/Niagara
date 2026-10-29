Tasks and replicas
==================

.. autoclass:: niagara.GraphFuture
   :members:

.. autoclass:: niagara.AsyncGraphExecutor
   :members:
   :special-members: __enter__, __exit__

.. autoclass:: niagara.ReplicaPool
   :members:
   :special-members: __call__, __enter__, __exit__

Give each stateful concurrent task/replica its own simulator. These interfaces
run in one process; they do not implement DDP, RPC or multi-host execution.
