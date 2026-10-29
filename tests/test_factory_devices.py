"""Region-local tensor construction, pointer devices and mode lifetime."""

import contextvars
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import caskade as ck
import pytest
import torch
from torch.utils._device import _device_constructors

import niagara as ng
from niagara.placement import _factory_scope

GPU = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")


def _problem(observed):
    class Child(ck.Module):
        def __init__(self, clock):
            super().__init__("child")
            self.scale = ck.Param("scale", [0.7, 0.9])
            self.offset = ck.Param(
                "offset", lambda param: param.clock.value * torch.tensor([1.0, -1.0])
            )
            self.offset.link("clock", clock)

        @ck.forward
        def __call__(self, x, scale, offset):
            samples = torch.rand(2, dtype=x.dtype)
            ones = torch.ones(2, dtype=x.dtype)
            if observed is not None:
                observed.append((samples.device, ones.device, offset.device, self.offset.value.device))
            return (x * scale + offset).sin() + ones + samples * 0

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.clock = ck.Param("clock", 0.25, dynamic=True)
            self.child = Child(self.clock)

        @ck.forward
        def __call__(self, x):
            return self.child(x)

    return Model().to(dtype=torch.float64)


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("root_home", ["cuda:0", "cuda:1"])
def test_nested_factories_and_remote_pointer_values_gradients(root_home):
    observed = []
    model = _problem(observed)
    x = torch.tensor([0.2, 0.4], dtype=torch.float64, requires_grad=True)
    params = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    expected = model(x, params)
    gradients = torch.autograd.grad(expected.sum(), (x, params))
    observed.clear()
    child_home = "cuda:1" if root_home == "cuda:0" else "cuda:0"
    original_types = {node: type(node) for node in model.topological_ordering()}
    original_default = torch.get_default_device()
    with ng.enable(
        model,
        compile=False,
        default=root_home,
        modules={model.child: child_home},
        diagnostics=True,
    ) as handle:
        actual_x = x.detach().clone().requires_grad_()
        actual_params = params.detach().clone().requires_grad_()
        actual = model(actual_x, actual_params)
        torch.testing.assert_close(actual.cpu(), expected)
        for actual_gradient, reference in zip(
            torch.autograd.grad(actual.sum(), (actual_x, actual_params)), gradients, strict=True
        ):
            torch.testing.assert_close(actual_gradient, reference)
        assert observed == [(torch.device(child_home),) * 4]
        assert model.child.offset.value.device == torch.device(child_home)
        assert handle.placement.diagnostics()["transfers"][root_home + "->" + child_home] > 0
        assert torch.ones(1).device.type == "cpu"
        assert torch.get_default_device() == original_default
    assert all(type(node) is cls for node, cls in original_types.items())
    assert torch.ones(1).device.type == "cpu"


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["eager", "torchscript", "auto"])
def test_factory_policy_is_captured_in_tensor_program(backend, monkeypatch):
    # The capture case is pure. The separate dynamic case below checks the
    # observable list append instead of letting capture swallow it.
    model = _problem(None)
    x = torch.tensor([0.2, 0.4], dtype=torch.float64)
    params = torch.tensor([0.3], dtype=torch.float64)
    expected = model(x, params)
    x, params = x.to("cuda:0"), params.to("cuda:0")
    with ng.enable(
        model, backend=backend, default="cuda:0", modules={model.child: "cuda:1"}
    ) as handle:
        torch.testing.assert_close(model(x, params).cpu(), expected)
        program = handle.programs[0]
        assert program.backend == ("cuda_graph" if backend == "auto" else backend)
        assert {tensor.device.type for tensor in program.graph.buffers()} <= {"cuda"}
        torch.testing.assert_close(
            model(x + 0.1, params + 0.2), program(((x + 0.1, params + 0.2), {}))
        )
        assert torch.ones(1).device.type == "cpu"
        reference = _problem(None)

        def no_factory_dispatch(*args, **kwargs):
            raise AssertionError("A prepared program re-entered the Python factory policy")

        from niagara.placement import _FactoryMode

        with monkeypatch.context() as patch:
            patch.setattr(_FactoryMode, "__torch_function__", no_factory_dispatch)
            for index in range(8):
                row, value = x + index * 0.01, params + index * 0.03
                torch.testing.assert_close(
                    model(row, value).cpu(), reference(row.cpu(), value.cpu())
                )


@GPU
@pytest.mark.gpu
def test_factory_policy_observation_runs_on_every_dynamic_fallback_call():
    observed = []
    model = _problem(observed)
    x = torch.tensor([0.2, 0.4], dtype=torch.float64, device="cuda:0")
    params = torch.tensor([0.3], dtype=torch.float64, device="cuda:0")
    with ng.enable(model, default="cuda:0", modules={model.child: "cuda:1"}) as handle:
        for _ in range(2):
            model(x, params)
        assert len(observed) == 2
        assert all(devices == (torch.device("cuda:1"),) * 4 for devices in observed)
        assert not handle.programs
        assert "mutation" in handle.dynamic_methods["__call__"]


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("factory_device", [False, True])
def test_constructor_policy_explicit_devices_and_opt_out(factory_device):
    observed = []
    original_ones = torch.ones
    original_tensor = torch.tensor

    class Model(ck.Module):
        def __init__(self):
            super().__init__()
            self.a = ck.Param("a", 1.0)

        @ck.forward
        def __call__(self):
            values = {
                "alias": original_ones(2),
                "literal": original_tensor([1.0, 2.0]),
                "explicit": torch.ones(2, device="cpu"),
            }
            values["out"] = torch.ones(2, out=values["explicit"])
            values["like"] = torch.ones_like(values["explicit"])
            values["new"] = values["explicit"].new_tensor([3.0, 4.0])
            observed.append({name: value.device for name, value in values.items()})
            return values

    with ng.enable(
        Model(), compile=False, default="cuda:1", factory_device=factory_device
    ) as handle:
        result = handle.root()
        implicit = torch.device("cuda:1" if factory_device else "cpu")
        assert observed == [
            {
                "alias": implicit,
                "literal": implicit,
                "explicit": torch.device("cpu"),
                "out": torch.device("cpu"),
                "like": torch.device("cpu"),
                "new": torch.device("cpu"),
            }
        ]
        assert all(value.device == torch.device("cuda:1") for value in result.values())
    assert torch.get_default_device() == torch.device("cpu")
    assert torch.ones(1).device == torch.device("cpu")


@GPU
@pytest.mark.gpu
def test_region_factory_mode_thread_context_and_exception_cleanup():
    original_default = torch.get_default_device()
    with ThreadPoolExecutor(max_workers=2) as pool, _factory_scope(torch.device("cuda:1"), True):
        assert torch.zeros(1).device == torch.device("cuda:1")
        assert pool.submit(lambda: torch.zeros(1).device).result() == original_default
        copied = contextvars.copy_context()

        def copied_work():
            with _factory_scope(torch.device("cuda:0"), True):
                return torch.zeros(1).device

        assert pool.submit(copied.run, copied_work).result() == torch.device("cuda:0")
        with _factory_scope(torch.device("cuda:0"), False):
            assert torch.zeros(1).device == original_default
        assert torch.zeros(1).device == torch.device("cuda:1")
        with (
            pytest.raises(RuntimeError, match="body failed"),
            _factory_scope(torch.device("cuda:0"), True),
        ):
            assert torch.zeros(1).device == torch.device("cuda:0")
            raise RuntimeError("body failed")
        assert torch.zeros(1).device == torch.device("cuda:1")
    assert torch.zeros(1).device == original_default
    assert torch.get_default_device() == original_default


def test_factory_policy_cpu_and_known_constructor_registry():
    assert {torch.tensor, torch.ones, torch.rand, torch.asarray} <= _device_constructors()
    with _factory_scope(torch.device("cpu"), True):
        value = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        assert value.dtype == torch.float64 and value.requires_grad
        value.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones(2, dtype=torch.float64))


@GPU
@pytest.mark.gpu
def test_pointer_combines_module_and_parameter_on_dependency_home():
    seen = []

    class Helper(ck.Module):
        def __init__(self):
            super().__init__("helper")
            self.scale = ck.Param("scale", 2.0)

        @ck.forward
        def __call__(self, scale):
            seen.append(torch.ones(2).device)
            return scale * torch.ones(2)

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.helper = Helper()
            self.offset = ck.Param(
                "offset",
                lambda param: param.helper() + param.scale.value * torch.tensor([1.0, 2.0]),
            )
            self.offset.link("helper", self.helper)
            self.offset.link("scale", self.helper.scale)

        @ck.forward
        def __call__(self, offset):
            return offset * 2

    model = Model()
    expected = model()
    seen.clear()
    with ng.enable(
        model,
        compile=False,
        default="cuda:0",
        modules={model.helper: "cuda:1"},
        diagnostics=True,
    ) as handle:
        actual = model()
        assert actual.device == torch.device("cuda:0")
        assert seen == [torch.device("cuda:1")]
        torch.testing.assert_close(actual.cpu(), expected)
        assert handle.placement.diagnostics()["transfers"]["cuda:1->cuda:0"] > 0


@GPU
@pytest.mark.gpu
def test_pointer_factory_exception_restores_regions_and_native_setter():
    observed = []
    model = _problem(observed)
    original_property = ck.Param.value

    def fail(param):
        assert torch.ones(2).device == param.clock.value.device
        raise ValueError("pointer failed")

    model.child.offset = fail
    with ng.enable(model, compile=False, default="cuda:0", modules={model.child: "cuda:1"}):
        with pytest.raises(ValueError, match="pointer failed"):
            model(torch.tensor([0.1, 0.2], dtype=torch.float64), torch.tensor([0.3]))
        assert not any(node.active for node in model.topological_ordering())
        assert torch.ones(2).device == torch.device("cpu")
        # The native property setter still allows a pointer replacement while
        # inactive, and the original callable identity remains inspectable.
        model.child.offset = lambda param: param.clock.value * torch.ones(2)
        actual = model(torch.tensor([0.1, 0.2], dtype=torch.float64), torch.tensor([0.3]))
        assert actual.device == torch.device("cuda:0")
        assert ck.Param.value is original_property
    assert torch.ones(2).device == torch.device("cpu")


def test_delayed_same_thread_context_copy_installs_its_factory_mode():
    class Model(ck.Module):
        def __init__(self):
            super().__init__()
            self.a = ck.Param("a", 1.0)

        @ck.forward
        def __call__(self):
            return contextvars.copy_context(), torch.zeros(2).device

    with ng.enable(Model(), compile=False, default="cpu") as handle:
        copied, original = handle.root()
        assert original == torch.device("cpu")
        with torch.device("meta"):
            _fresh_context, actual = copied.run(handle.root)
            assert actual == torch.device("cpu")
            assert torch.zeros(2).device == torch.device("meta")
    assert torch.zeros(2).device == torch.device("cpu")


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("valid", [False, True])
def test_manual_active_context_fill_routes_every_parameter_and_gradient(valid):
    class Leaf(ck.Module):
        def __init__(self):
            super().__init__("leaf")
            self.scale = ck.Param("scale", [0.3, 0.7], dynamic=True, valid=(0.0, None))

        @ck.forward
        def __call__(self, x, scale):
            assert self.scale._value.device == x.device
            return (x * scale).sin()

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.child = Leaf()
            self.bias = ck.Param("bias", [-0.3, -0.7], dynamic=True, valid=(None, 0.0))

        @ck.forward
        def __call__(self, x, bias):
            assert self.bias._value.device == x.device
            return self.child(x) + bias.square()

    model = Model().to(dtype=torch.float64)
    values = model.get_values()
    params = model.to_valid(values) if valid else values
    reference_params = params.detach().clone().requires_grad_()
    reference_x = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    with ck.ValidContext(model) if valid else nullcontext():
        expected = model(reference_x, reference_params)
    gradients = torch.autograd.grad(expected.sum(), (reference_x, reference_params))
    actual_x = reference_x.detach().clone().to("cuda:0").requires_grad_()
    actual_params = params.detach().clone().requires_grad_()
    with ng.enable(model, compile=False, default="cuda:0", modules={model.child: "cuda:1"}):
        with (
            ck.ValidContext(model) if valid else nullcontext(),
            ck.ActiveContext(model),
        ):
            model.fill_params(actual_params)
            assert model.bias._value.device == torch.device("cuda:0")
            assert model.child.scale._value.device == torch.device("cuda:1")
            actual = model(actual_x)
            assert actual.device == torch.device("cuda:0")
            torch.testing.assert_close(actual.cpu(), expected)
            result_gradients = torch.autograd.grad(actual.sum(), (actual_x, actual_params))
        for result, reference in zip(result_gradients, gradients, strict=True):
            torch.testing.assert_close(result.cpu(), reference)
