"""Regression checks for the published-suite adapter and compiled qualification."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import caskade as ck
import pytest
import torch

GPU = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")


@pytest.fixture
def adapter(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/_upstream_caskade_plugin.py"
    name = "_niagara_upstream_adapter_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_MODE", "niagara")
    monkeypatch.setattr(module, "_DEFAULT", "cuda:0")
    monkeypatch.setattr(module, "_CURRENT", "adapter-regression")
    monkeypatch.setattr(module, "_COMPILE_CHECKS", "off")
    return module


@pytest.mark.parametrize("effect", ["random", "mutation", "cache"])
def test_compiled_eligibility_rejects_python_effects(adapter, effect, monkeypatch):
    class Model(ck.Module):
        def __init__(self):
            super().__init__()
            self.p = ck.Param("p", 2.0)
            self.counter = 0

        @ck.forward
        def random(self, x, p):
            return x * p + torch.rand_like(x)

        @ck.forward
        def mutate(self, x, p):
            self.counter += 1
            return x * p

        @ck.active_cache
        @ck.forward
        def cached(self, x, p):
            return x * p

    # Use separate classes so an unrelated method does not decide the result.
    method = {"random": "random", "mutation": "mutate", "cache": "cached"}[effect]
    original = vars(Model)[method]
    model_type = type("OnlyMethod", (ck.Module,), {"calculate": original})
    model = model_type()
    model.p = ck.Param("p", 2.0)
    model.counter = 0
    monkeypatch.setattr(adapter, "_COMPILE_CHECKS", "eligible")
    scope = SimpleNamespace(root=model, nodes={model})
    reason = adapter._compiled_reason(
        "test_module.py::test_input_methods[array]", scope, ((torch.ones(2),), {})
    )
    assert reason and any(word in reason for word in ("random", "mutates", "active-cache"))


@GPU
@pytest.mark.gpu
def test_explicit_active_context_uses_real_niagara_boundaries(adapter, monkeypatch):
    class Leaf(ck.Module):
        def __init__(self):
            super().__init__("leaf")
            self.bias = ck.Param("bias", 2.0)

        @ck.forward
        def calculate(self, x, bias):
            return x.sin() + bias

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.leaf = Leaf()
            self.scale = ck.Param("scale", 3.0)

        @ck.forward
        def calculate(self, x, scale):
            return self.leaf.calculate(x) * scale

    model = Model()
    original_classes = type(model), type(model.leaf)
    x = torch.tensor([0.3, 0.7], dtype=torch.float64)
    expected = model.calculate(x)
    monkeypatch.setattr(ck.ActiveContext, "__enter__", adapter._context_enter)
    monkeypatch.setattr(ck.ActiveContext, "__exit__", adapter._context_exit)
    with ck.ActiveContext(model):
        assert model.active and model.scale.active
        assert model.scale.value.device == torch.device("cpu")
        result = model.calculate(x)
        torch.testing.assert_close(result, expected)
        assert result.device == torch.device("cpu")
        diagnostic = model._niagara_placement.diagnostics()
        assert {home for _name, home in diagnostic["observed_calls"]} == {"cuda:0", "cuda:1"}
        assert diagnostic["transfers"]["cuda:0->cuda:1"] > 0
    assert (type(model), type(model.leaf)) == original_classes
    assert not model.active and not model.leaf.active
    assert not hasattr(model, "_niagara_backend")
    assert adapter._CALLS[0]["status"] == "routed_active_context"
    assert adapter._CALLS[0]["classes_restored"]
    assert not any("native_bypass" in call["status"] for call in adapter._CALLS)


@GPU
@pytest.mark.gpu
def test_independent_metadata_helper_is_enabled_without_host_return(adapter, monkeypatch):
    devices = []

    class Helper(ck.Module):
        def __init__(self):
            super().__init__("helper")
            self.bias = ck.Param("bias", [0.2, 0.5])

        @adapter._forward
        def calculate(self, x, bias):
            return x * bias

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.scale = ck.Param("scale", 3.0)
            self.meta.helper = Helper()

        @adapter._forward
        def calculate(self, x, scale):
            value = self.meta.helper.calculate(x)
            devices.append(value.device)
            return value.sin() * scale

    model = Model()
    original_classes = type(model), type(model.meta.helper)
    x = torch.tensor([0.3, 0.7], requires_grad=True)
    monkeypatch.setattr(adapter, "_MODE", "native")
    expected = model.calculate(x)
    expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
    devices.clear()
    adapter._CALLS.clear()
    monkeypatch.setattr(adapter, "_MODE", "niagara")
    actual = model.calculate(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], expected_gradient)
    assert devices == [torch.device("cuda:0")]
    assert actual.device == torch.device("cpu")
    nested = [call for call in adapter._CALLS if call.get("nested_owned_scope")]
    assert len(nested) == 1 and nested[0]["status"] == "routed"
    assert all(tensor["device"] == "cuda:0" for tensor in nested[0]["returned_tensors"])
    assert (type(model), type(model.meta.helper)) == original_classes
    assert model.meta.helper.bias.value.device == torch.device("cpu")
    assert not any("native_bypass" in call["status"] for call in adapter._CALLS)


@GPU
@pytest.mark.gpu
def test_eligible_program_executes_compiled_values_gradients_and_guards(adapter, monkeypatch):
    class Leaf(ck.Module):
        def __init__(self):
            super().__init__("leaf")
            self.bias = ck.Param("bias", [0.2, 0.5], dynamic=True)

        @adapter._forward
        def calculate(self, x, bias):
            return x.sin() + bias

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.leaf = Leaf()

        @adapter._forward
        def calculate(self, x):
            return self.leaf.calculate(x).cos()

    monkeypatch.setattr(adapter, "_COMPILE_CHECKS", "eligible")
    monkeypatch.setattr(adapter, "_COMPILE_BACKEND", "auto")
    monkeypatch.setattr(adapter, "_CURRENT", "test_module.py::test_input_methods[array]")
    model = Model()
    x = torch.tensor([0.3, 0.7], dtype=torch.float64)
    result = model.calculate(x, model.get_values())
    assert result.device == torch.device("cpu")
    record = adapter._CALLS[0]["compiled"]
    assert record["status"] == "passed"
    assert record["forward_backend"] == record["gradient_backend"] == "cuda_graph"
    assert record["captured_devices"] == ["cuda:0", "cuda:1"]
    assert record["changed_input_checked"]
    assert record["forward_work"]["input_dependent_arithmetic_operator_count"] > 0
    assert record["gradient_work"]["input_dependent_arithmetic_operator_count"] > 0
    assert record["gradient_components_checked"] == 8
    assert record["metadata_guard_rejections"] == 1


@GPU
@pytest.mark.gpu
def test_integer_inputs_use_compiled_changed_values_without_fake_gradients(adapter, monkeypatch):
    class Leaf(ck.Module):
        def __init__(self):
            super().__init__("leaf")
            self.bias = ck.Param("bias", [2, 5], dynamic=True)

        @adapter._forward
        def calculate(self, x, bias):
            return x + bias

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.leaf = Leaf()

        @adapter._forward
        def calculate(self, x):
            return self.leaf.calculate(x) * 2

    monkeypatch.setattr(adapter, "_COMPILE_CHECKS", "eligible")
    monkeypatch.setattr(adapter, "_COMPILE_BACKEND", "auto")
    monkeypatch.setattr(adapter, "_CURRENT", "test_module.py::test_input_methods[array]")
    model = Model()
    result = model.calculate(torch.tensor([3, 7]), model.get_values())
    torch.testing.assert_close(result, torch.tensor([10, 24]))
    record = adapter._CALLS[0]["compiled"]
    assert record["status"] == "passed"
    assert record["forward_backend"] == "cuda_graph"
    assert record["integer_input_components"] == 4
    assert record["gradient_checks"] == 0 and record["gradient_backend"] is None
    assert "Integer" in record["gradient_reason"]
    assert record["changed_input_checked"] and record["metadata_guard_rejections"] == 1
