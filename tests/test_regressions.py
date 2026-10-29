"""Execution, capture and device-lifetime safety regressions."""

import caskade as ck
import pytest
import torch
from test_runtime import GPU, Sim
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode

import niagara as ng


@pytest.mark.parametrize("entry", ["compiled", "enabled", "closure"])
def test_prepared_dispatch_is_rejected_inside_another_capture(entry):
    x = torch.tensor([0.5, 0.7], dtype=torch.float64)
    if entry == "compiled":
        prepared = ng.compile_model(lambda z: z.sin(), x, backend="eager")
        with pytest.raises((ng.CaptureError, ng.GuardError), match="capture|graph|compil"):
            ng.compile_model(prepared, x, backend="eager")
        return
    model = Sim()
    with ng.enable(model, backend="eager", example=x):
        target = model if entry == "enabled" else lambda z: model(z)
        with pytest.raises((ng.CaptureError, ng.GuardError), match="capture|graph|compil"):
            ng.compile_model(target, x, backend="eager")


def test_prepared_dispatch_is_rejected_inside_external_fx():
    x = torch.tensor([0.5, 0.7])
    prepared = ng.compile_model(lambda z: z.square().sin(), x, backend="eager")
    with pytest.raises((ng.CaptureError, ng.GuardError), match="capture|graph|compil"):
        make_fx(lambda z: prepared(z))(x)


@pytest.mark.parametrize("capture", ["compile_model", "make_fx", "torchscript"])
def test_executor_submission_is_rejected_during_capture(capture):
    x = torch.tensor([0.2, 0.7])
    with ng.AsyncGraphExecutor() as executor:

        def function(z):
            # The outer multiplication retains a visible input dependency, so
            # checking only the final graph for constant outputs is insufficient.
            return z * executor.submit(lambda value: value.sin(), z).result()

        with pytest.raises((ng.CaptureError, ng.GuardError), match="task|captur|compil"):
            if capture == "compile_model":
                ng.compile_model(function, x, backend="eager")
            elif capture == "make_fx":
                make_fx(function)(x)
            else:
                torch.jit.trace(function, x, check_trace=False)
        assert not executor.workers  # Reject before any worker is scheduled.


@pytest.mark.parametrize("entry", ["compiled", "enabled"])
def test_prepared_dispatch_is_rejected_inside_external_torchscript(entry):
    x = torch.tensor([0.5, 0.7], dtype=torch.float64)
    if entry == "compiled":
        prepared = ng.compile_model(lambda z: z.sin(), x, backend="eager")
        with pytest.raises(ng.GuardError, match="graph|compil|trac"):
            torch.jit.trace(lambda z: prepared(z), x, check_trace=False)
    else:
        model = Sim()
        with (
            ng.enable(model, backend="eager", example=x),
            pytest.raises(ng.GuardError, match="graph|compil|trac"),
        ):
            torch.jit.trace(lambda z: model(z), x, check_trace=False)


def test_failed_native_restoration_keeps_backend_recoverable():
    model = Sim()
    x = torch.tensor([0.5, 0.7], dtype=torch.float64)
    native_class = type(model)
    expected = model(x)
    handle = ng.enable(model, backend="eager", example=x)
    programs = handle.programs
    outside = ck.Module("outside")
    outside.shared = model.left
    try:
        with pytest.raises(ng.PlacementError, match="outside|shared|owner"):
            handle.close()
        assert not handle.closed
        assert model._niagara_backend is handle
        assert model._niagara_placement is handle.placement
        assert handle.programs == programs
    finally:
        outside.unlink("shared")
        handle.close()
    assert handle.closed
    assert type(model) is native_class
    assert not hasattr(model, "_niagara_backend")
    torch.testing.assert_close(model(x), expected)


def test_placement_close_is_idempotent():
    model = Sim()
    original_class = type(model)
    placement = ng.assign_devices(model)
    placement.close()
    placement.close()
    assert placement.closed
    assert type(model) is original_class
    assert not hasattr(model, "_niagara_placement")


def test_executor_preserves_cpu_autocast_context():
    matrix = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    with ng.AsyncGraphExecutor() as executor, torch.autocast("cpu", dtype=torch.bfloat16):
        expected = matrix @ matrix
        actual = executor.submit(lambda x: x @ x, matrix).result()
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _sleep_on_current_stream():
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("The CUDA build does not expose the bounded stream-delay test helper")
    # Roughly tens of milliseconds on the tested GPUs, well below Windows TDR.
    # Only this stream is delayed; another stream may overwrite freed storage.
    torch.cuda._sleep(30_000_000)


@GPU
@pytest.mark.gpu
def test_cuda_replay_is_rejected_inside_another_capture():
    x = torch.linspace(0.1, 0.9, 16, device="cuda:0")
    prepared = ng.compile_model(lambda z: z.sin().square(), x)
    assert prepared.backend == "cuda_graph"
    with pytest.raises((ng.CaptureError, ng.GuardError), match="capture|graph|compil"):
        ng.compile_model(lambda z: prepared(z), x, backend="eager")
    # A failed outer capture must not damage the original prepared program.
    torch.testing.assert_close(prepared(x + 0.1), (x + 0.1).sin().square())


@GPU
@pytest.mark.gpu
def test_future_result_survives_producer_allocator_reuse():
    device = torch.device("cuda:0")
    consumer = torch.cuda.Stream(device=device)
    size = 2**21  # Eight MiB; use a worker stream with its own allocator blocks.
    with ng.AsyncGraphExecutor() as executor:
        for value in range(1, 9):
            future = executor.submit(
                lambda value=value: torch.full((size,), float(value), device=device),
                device=device,
            )
            with torch.cuda.stream(consumer):
                data = future.result()
                _sleep_on_current_stream()
                total = data.sum()
            # No producer-side Python reference may keep the input alive. A
            # consumer-stream lifetime registration must protect this read.
            del data, future
            overwrite = executor.submit(
                lambda: [torch.full((size,), -100.0, device=device) for _ in range(3)],
                device=device,
            )
            garbage = overwrite.result()
            consumer.synchronize()
            assert total.item() == float(value * size)
            del garbage, overwrite


class _LargeOutput(ck.Module):
    def __init__(self):
        super().__init__("large_output")
        self.amplitude = ck.Param("amplitude", None)
        self._field = torch.ones(2**20)

    @ck.forward
    def __call__(self, amplitude):
        return amplitude * self._field


class _DelayConsumerCat(TorchDispatchMode):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if (
            func == torch.ops.aten.cat.default
            and torch.cuda.current_stream(self.stream.device) == self.stream
        ):
            _sleep_on_current_stream()
        return func(*args, **(kwargs or {}))


@GPU
@pytest.mark.gpu
def test_replica_gather_survives_another_caller_stream_reusing_worker_storage():
    device = torch.device("cuda:0")
    first = torch.tensor([[1.0], [2.0]], device=device)
    second = torch.tensor([[3.0], [4.0]], device=device)
    left, right = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    with ng.ReplicaPool(_LargeOutput, devices=(device,)) as pool:
        pool.prepare(first)
        left.wait_stream(torch.cuda.current_stream(device))
        right.wait_stream(torch.cuda.current_stream(device))
        # Delay only the first caller's gather after its worker output has been
        # produced. A second caller uses a different stream, so worker reuse
        # must not rely on the next call waiting for that first caller.
        with torch.cuda.stream(left), _DelayConsumerCat(left):
            actual_first = pool(first)
        with torch.cuda.stream(right):
            actual_second = pool(second)
        left.synchronize()
        right.synchronize()
        torch.testing.assert_close(actual_first, first.expand_as(actual_first))
        torch.testing.assert_close(actual_second, second.expand_as(actual_second))
