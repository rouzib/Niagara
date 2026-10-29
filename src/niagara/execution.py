"""Explicit coarse tasks and persistent replicas for independent simulations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
from caskade import Module
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.utils._pytree import tree_flatten, tree_map

from .compiler import compile_model
from .errors import GuardError, PlacementError
from .placement import DevicePlan, _inactive, assign_devices, move
from .placement import device as normalize_device


def _record_consumer_streams(value):
    """Keep worker allocations alive until each current consumer stream finishes."""
    for tensor in tree_flatten(value)[0]:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            tensor.record_stream(torch.cuda.current_stream(tensor.device))


@dataclass
class GraphFuture:
    """Represent the result of one explicitly submitted coarse task.

    Parameters
    ----------
    future : concurrent.futures.Future
        Worker future resolving to an output pytree and optional CUDA event.
        Instances are normally returned by :meth:`AsyncGraphExecutor.submit`.

    Notes
    -----
    Passing this object inside another task's arguments declares a dependency.
    CUDA event ordering is separate from waiting for the Python worker itself.
    """

    future: Future[tuple[Any, torch.cuda.Event | None]]

    def result(self) -> Any:
        """Resolve the worker result and order the current CUDA consumer stream.

        Returns
        -------
        object
            Task output pytree. CUDA results remain on their declared device;
            this method does not convert them to CPU.

        Raises
        ------
        Exception
            Propagates exceptions raised by the submitted function.

        Notes
        -----
        Waiting for the Python future ensures the producer has enqueued its
        work. The returned event orders CUDA work on the consumer stream without
        a process-wide CUDA synchronization. CPU inspection still follows
        ordinary PyTorch synchronization semantics.
        CUDA allocations are recorded on the consuming stream so queued reads
        finish before the worker's storage can be reused.
        """
        result, event = self.future.result()
        if event is not None:
            torch.cuda.current_stream(event.device).wait_event(event)
        _record_consumer_streams(result)
        return result


class AsyncGraphExecutor:
    """Execute explicitly submitted coarse tasks with persistent device workers.

    Attributes
    ----------
    trace : list of dict
        Declared task records containing device and dependency count. These
        records are not measured runtime or overlap observations.
    closed : bool
        Whether worker submission has been shut down.

    Notes
    -----
    Future-valued arguments declare dependencies. Tasks on one home share a
    worker and CUDA stream; distinct devices can make independent progress when
    their dependencies permit it. Every returned tensor must reside on the
    declared home. Stateful tasks must own disjoint simulators. Ordinary Caskade
    ``@forward`` calls never implicitly create thread-pool jobs.
    """

    def __init__(self) -> None:
        self.workers, self.streams, self.trace = {}, {}, []
        self.closed = False

    def submit(
        self,
        function: Callable[..., Any],
        *args: Any,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> GraphFuture:
        """Submit a callable and route its resolved tensor inputs to one home.

        Parameters
        ----------
        function : callable
            Function invoked as ``function(*args, **kwargs)`` in a device worker.
        *args, **kwargs : object
            Tensor pytrees and/or GraphFuture dependencies. Future leaves are
            resolved before tensors are routed to the declared home.
        device : str or torch.device, optional
            CPU or explicitly indexed available CUDA home.

        Returns
        -------
        GraphFuture
            Result/dependency handle. Worker exceptions are raised on resolution.

        Raises
        ------
        RuntimeError
            If submission is attempted after closure.
        GuardError
            If submission occurs inside tracing or a functional transform.
        PlacementError
            If the device is invalid, or later if a task returns tensors outside
            its declared home.

        Notes
        -----
        Grad, inference and CPU/CUDA autocast modes are inherited from the submitting thread.
        Device streams/events order producer inputs and declared dependencies.
        The executor does not infer purity or make shared mutable models safe.
        """
        if self.closed:
            raise RuntimeError("Executor is closed")
        if (
            get_proxy_mode() is not None
            or torch.jit.is_tracing()
            or torch._C._are_functorch_transforms_active()
        ):
            raise GuardError(
                "Compile the pure task before submission; executor scheduling cannot be captured"
            )
        home = normalize_device(device)
        if home not in self.workers:
            self.workers[home] = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"niagara-{home}"
            )
            if home.type == "cuda":
                self.streams[home] = torch.cuda.Stream(device=home)
        grad, inference = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
        autocast = {
            kind: (torch.is_autocast_enabled(kind), torch.get_autocast_dtype(kind))
            for kind in ("cpu", "cuda")
        }
        autocast_cache = torch.is_autocast_cache_enabled()
        # Record the submitting stream before dispatch, without a CPU synchronize.
        ready = None
        if home.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(home))
        input_events = []
        for source in {
            t.device
            for t in tree_flatten((args, kwargs))[0]
            if isinstance(t, torch.Tensor) and t.is_cuda
        }:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(source))
            input_events.append(event)
        dependencies = [x for x in tree_flatten((args, kwargs))[0] if isinstance(x, GraphFuture)]
        self.trace.append(
            {
                "id": len(self.trace),
                "device": str(home),
                "dependencies": len(dependencies),
                "kind": "declared_task",
            }
        )

        def run():
            stream = self.streams.get(home)
            context = torch.cuda.stream(stream) if stream is not None else _NullContext()
            with (
                torch.inference_mode(inference),
                torch.set_grad_enabled(grad),
                context,
                torch.autocast(
                    "cpu",
                    enabled=autocast["cpu"][0],
                    dtype=autocast["cpu"][1],
                    cache_enabled=autocast_cache,
                ),
                torch.autocast(
                    "cuda",
                    enabled=autocast["cuda"][0],
                    dtype=autocast["cuda"][1],
                    cache_enabled=autocast_cache,
                ),
            ):
                if ready is not None:
                    stream.wait_event(ready)
                for event in input_events:
                    torch.cuda.current_stream(event.device).wait_event(event)
                    if stream is not None:
                        stream.wait_event(event)
                resolved_args, resolved_kwargs = tree_map(
                    lambda x: x.result() if isinstance(x, GraphFuture) else x, (args, kwargs)
                )
                inputs = move((resolved_args, resolved_kwargs), home)
                for tensor in tree_flatten(inputs)[0]:
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        tensor.record_stream(stream)
                value = function(*inputs[0], **inputs[1])
                if any(
                    t.device != home for t in tree_flatten(value)[0] if isinstance(t, torch.Tensor)
                ):
                    raise PlacementError("Task returned a tensor outside its declared device")
                event = None
                if stream is not None:
                    event = torch.cuda.Event()
                    event.record(stream)
                return value, event

        return GraphFuture(self.workers[home].submit(run))

    def resolve(self, value: Any) -> Any:
        """Replace future leaves of an output pytree with their resolved values.

        Parameters
        ----------
        value : object
            Pytree containing GraphFuture leaves and ordinary values.

        Returns
        -------
        object
            Same structure with future leaves resolved using ``result()``.
        """
        return tree_map(lambda x: x.result() if isinstance(x, GraphFuture) else x, value)

    def close(self) -> None:
        """Wait for submitted Python jobs and shut down persistent workers.

        Notes
        -----
        No process-wide CUDA synchronization is added. Resolve futures before
        consuming their tensors; ordinary CUDA stream ordering still applies.
        """
        for worker in self.workers.values():
            worker.shutdown(wait=True)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class ReplicaPool:
    """Replicate independent simulators and partition batched inputs in order.

    Parameters
    ----------
    factory : callable
        Zero-argument factory returning a fresh Caskade Module for every replica.
        Instances must not share mutable Caskade nodes.
    devices : sequence of str or torch.device, optional
        Nonempty, distinct CPU/CUDA homes. CUDA homes must be explicitly indexed.

    Attributes
    ----------
    models : list of caskade.Module
        Persistent simulator instances, one per home.
    devices : tuple of torch.device
        Normalized replica homes in output-order partitioning order.

    Notes
    -----
    Prepare per-shard vmaps before repeated work. Gathering uses differentiable
    transfers and concatenation, preserving original batch order. Replication
    is distinct from placing regions of one simulator on several GPUs. A pool
    is single-caller; it does not provide data-parallel optimizer synchronization.
    """

    def __init__(
        self,
        factory: Callable[[], Module],
        devices: Sequence[str | torch.device] = ("cuda:0", "cuda:1"),
    ) -> None:
        self.devices = tuple(normalize_device(d) for d in devices)
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise PlacementError("Replica homes must be nonempty and distinct")
        self.models, self.placements, self.streams = [], [], []
        self.compiled = None
        for home in self.devices:
            model = factory()
            self.placements.append(assign_devices(model, DevicePlan(home)))
            self.models.append(model)
            self.streams.append(torch.cuda.Stream(device=home) if home.type == "cuda" else None)

    def prepare(
        self,
        example_batch: torch.Tensor,
        *,
        backend: str = "auto",
        chunk_size: int | None = None,
    ) -> ReplicaPool:
        """Capture and prepare each replica's representative batch shard.

        Parameters
        ----------
        example_batch : torch.Tensor
            Tensor with a leading batch dimension and at least one item per
            replica. Per-shard metadata defines subsequent valid calls.
        backend : str, optional
            Backend used for the per-replica batched programs.
        chunk_size : int or None, optional
            Native vmap chunk size within each replica shard.

        Returns
        -------
        ReplicaPool
            This prepared pool, allowing chained construction/preparation.

        Raises
        ------
        ValueError
            If the batch cannot give every replica an item.
        CaptureError
            If a model/shard cannot be captured safely.
        """
        if example_batch.shape[0] < len(self.devices):
            raise ValueError("Provide at least one batch element per replica")
        shards = example_batch.tensor_split(len(self.devices))
        self.compiled = []
        for model, shard, home in zip(self.models, shards, self.devices):
            batch = shard.to(home)
            base = compile_model(model, batch[0], backend="eager")
            self.compiled.append(base.vmap(batch, backend=backend, chunk_size=chunk_size))
        return self

    def __call__(
        self,
        batch: torch.Tensor,
        *,
        output_device: str | torch.device | None = None,
    ) -> Any:
        """Evaluate prepared replicas and gather tensor pytrees in input order.

        Parameters
        ----------
        batch : torch.Tensor
            Batch matching preparation metadata, including each shard's strides.
        output_device : str or torch.device or None, optional
            Gather destination; None uses ``batch.device``.

        Returns
        -------
        object
            Output tensor pytree concatenated along its leading batch dimension.
            Cross-device copies retain ordinary autograd relationships.

        Raises
        ------
        RuntimeError
            If called before preparation.
        GuardError
            If a shard's metadata, modes or model state differ from preparation.

        Notes
        -----
        Device streams/events order work without a global CUDA synchronization.
        A CPU gather waits for the participating copies using PyTorch semantics.
        """
        if self.compiled is None:
            raise RuntimeError("Call prepare(example_batch) before using a ReplicaPool")
        target = normalize_device(output_device or batch.device)
        pieces = []
        for fn, shard, home, stream in zip(
            self.compiled, batch.tensor_split(len(self.devices)), self.devices, self.streams
        ):
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream(home))
                # The source tensor can be produced on a different GPU stream.
                if shard.is_cuda:
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(shard.device))
                    stream.wait_event(ready)
                with torch.cuda.stream(stream):
                    inputs = shard.to(home, non_blocking=True)
                    inputs.record_stream(stream)
                    result = fn(inputs)
                    done = torch.cuda.Event()
                    done.record(stream)
                torch.cuda.current_stream(home).wait_event(done)
                torch.cuda.current_stream(target).wait_event(
                    done
                ) if target.type == "cuda" else None
            else:
                result = fn(shard.to(home))
            # Default cross-device .to synchronizes the participating CUDA streams
            # using PyTorch's copy implementation; no manual global synchronize.
            piece = move(result, target)
            _record_consumer_streams(piece)
            pieces.append(piece)
        spec = tree_flatten(pieces[0])[1]
        flat = [tree_flatten(piece)[0] for piece in pieces]
        return spec.unflatten([torch.cat(items, dim=0) for items in zip(*flat)])

    def close(self) -> None:
        """Restore every replica's native classes, leaving tensors on its home.

        Raises
        ------
        PlacementError
            If a replica is still active.

        Notes
        -----
        Close once after production, or use the pool as a context manager.
        This does not co-locate distinct replicas or change their tensor homes.
        """
        _inactive(node for placement in self.placements for node in placement.homes)
        for placement in self.placements:
            placement.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
