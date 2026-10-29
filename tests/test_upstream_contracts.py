"""Native forwarding contracts and grouped parameter device ownership."""

import caskade as ck
import pytest
import torch
from test_runtime import GPU

import niagara as ng


@pytest.mark.parametrize("dynamic", [False, True])
def test_explicit_none_parameters_preserve_native_type_error(dynamic):
    class Model(ck.Module):
        def __init__(self):
            super().__init__()
            self.a = ck.Param("a", None if dynamic else 1.0)

        @ck.forward
        def evaluate(self, x, a):
            return x + a

    model = Model()
    with pytest.raises(TypeError):
        model.evaluate(2.0, params=None)
    with ng.enable(model, compile=False):
        with pytest.raises(TypeError):
            model.evaluate(2.0, params=None)
        assert not model.active
        if dynamic:
            with pytest.raises(TypeError):
                model.evaluate(2.0, None)


def test_static_pointer_is_evaluated_once_inside_activation():
    calls = []

    class Model(ck.Module):
        def __init__(self):
            super().__init__()
            self.p = ck.Param("p", lambda _: calls.append(self.active) or torch.tensor(2.0))

        @ck.forward
        def evaluate(self, x, p):
            return x + p

    model = Model()
    calls.clear()
    expected = model.evaluate(3.0)
    assert calls == [True]
    calls.clear()
    with ng.enable(model, compile=False):
        torch.testing.assert_close(model.evaluate(3.0), expected)
        assert calls == [True]
        calls.clear()
        torch.testing.assert_close(model.evaluate(3.0, []), expected)
        assert calls == [True]


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("packing", ["array", "list", "dict"])
@pytest.mark.parametrize("grouped", [False, True])
def test_refilled_nested_parameters_keep_their_declared_homes(packing, grouped):
    seen = []

    class Child(ck.Module):
        def __init__(self):
            super().__init__("child")
            self.a = ck.Param("a", [0.3, 0.7], group=0)
            self.b = ck.Param("b", [0.2, 0.8], group=int(grouped))
            self.to_dynamic(False)

        @ck.forward
        def calculate(self, x, a, b):
            seen.append((a.device, b.device, self.a.value.device, self.b.value.device))
            return (x * a + b).sin()

    class Middle(ck.Module):
        def __init__(self):
            super().__init__("middle")
            self.hierarchical_link("child", Child())

        @ck.forward
        def calculate(self, x, child_params):
            # A hierarchical child is refilled by its own entry point.
            return self.child.calculate(x, params=self.child.get_values(packing))

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.middle = Middle()

        @ck.forward
        def calculate(self, x):
            return self.middle.calculate(x)

    model = Model()
    x = torch.tensor([0.4, 0.6], requires_grad=True)
    expected = model.calculate(x, model.get_values(packing))
    gradient = torch.autograd.grad(expected.sum(), x)[0]
    seen.clear()
    with ng.enable(
        model,
        compile=False,
        default="cuda:0",
        modules={model.middle: "cuda:1", model.middle.child: "cuda:0"},
        diagnostics=True,
    ) as handle:
        result = model.calculate(x, model.get_values(packing))
        assert result.device == torch.device("cuda:0")
        assert seen == [(torch.device("cuda:0"),) * 4]
        torch.testing.assert_close(result.cpu(), expected)
        torch.testing.assert_close(torch.autograd.grad(result.sum(), x)[0], gradient)
        diagnostic = handle.placement.diagnostics()
        assert diagnostic["transfers"]["cuda:0->cuda:1"] > 0
        assert diagnostic["transfers"]["cuda:1->cuda:0"] > 0


def test_top_level_live_tensor_outputs_are_routed_to_the_region_home():
    class Model(ck.Module):
        def __init__(self):
            super().__init__()
            self.bias = ck.Param("bias", 0.0)

        @ck.forward
        def evaluate(self):
            return {"created": torch.tensor([2.0, 3.0]), "unchanged": None}

    model = Model()
    with ng.enable(model, compile=False, default="cpu"):
        result = model.evaluate()
        assert result["created"].device == torch.device("cpu")
    if torch.cuda.device_count() < 2:
        return
    with ng.enable(model, compile=False, default="cuda:1"):
        result = model.evaluate()
        assert result["created"].device == torch.device("cuda:1")
        torch.testing.assert_close(result["created"].cpu(), torch.tensor([2.0, 3.0]))
        assert result["unchanged"] is None


def test_backend_close_preserves_pointer_only_dependency_ownership():
    class Dependency(ck.Module):
        def __init__(self):
            super().__init__("dependency")
            self.scale = ck.Param("scale", 2.0)

        @ck.forward
        def calculate(self, scale):
            return scale * 3

    class Model(ck.Module):
        def __init__(self, dependency):
            super().__init__("model")
            self.result = ck.Param("result", lambda param: param.dependency.calculate())
            self.result.link("dependency", dependency)

        @ck.forward
        def calculate(self, result):
            return result + 1

    dependency = Dependency()
    model = Model(dependency)
    classes = type(model), type(dependency)
    expected = model.calculate()
    handle = ng.enable(
        model,
        compile=False,
        default="cpu",
        modules={dependency: "cpu"},
        parameters={dependency.scale: "cpu"},
    )
    torch.testing.assert_close(model.calculate(), expected)
    handle.close()
    assert handle.closed
    assert (type(model), type(dependency)) == classes
    torch.testing.assert_close(model.calculate(), expected)
    assert dependency.scale.value.device == torch.device("cpu")
