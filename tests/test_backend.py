import caskade as ck
import pytest
import torch
from test_runtime import GPU, Sim
from torch._dynamo.exc import InvalidBackend

import niagara as ng


def test_existing_calls_opt_in_modes_guards_sessions_and_restoration():
    model, untouched = Sim(), Sim()
    native_class, forward = type(model), ck.forward
    x = torch.tensor([0.5, 0.7], dtype=torch.float64)
    expected = model(x)
    handle = ng.enable(model, example=x)
    assert isinstance(model, native_class)
    assert type(untouched) is native_class and ck.forward is forward
    torch.testing.assert_close(model(x), expected)
    torch.testing.assert_close(model(params=x), expected)
    assert len(handle.programs) == 2
    with handle.session():
        torch.testing.assert_close(model(x + 0.1), untouched(x + 0.1))
        with pytest.raises(ng.GuardError, match="Prepare"):
            model(x.float())
        with pytest.raises(ng.GuardError, match="disable"):
            ng.disable(model)
    q = x.clone().requires_grad_()
    actual, reference = model(q), untouched(q)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), q)[0], torch.autograd.grad(reference.sum(), q)[0]
    )
    with torch.no_grad():
        torch.testing.assert_close(model(x), expected)
    assert len(handle.programs) == 4
    model.offset += 1
    with pytest.raises(ng.GuardError, match="changed"):
        model(x)
    with pytest.raises(ng.GuardError, match="changed"):
        model(x.float())
    handle.clear()
    torch.testing.assert_close(model(x), expected + 1)
    assert ng.disable(model) and not ng.disable(model)
    assert type(model) is native_class
    torch.testing.assert_close(model(x), expected + 1)


def test_backend_preserves_active_context_and_failure_cleanup():
    model = Sim()
    x = torch.tensor([0.5, 0.7], dtype=torch.float64)
    expected = model(x)
    handle = ng.enable(model, backend="eager")
    with ck.ActiveContext(model):
        model.fill_params(x)
        torch.testing.assert_close(model(), expected)
        with pytest.raises(ng.PlacementError, match="active"):
            handle.close()
    assert not handle.closed
    handle.close()
    native_class = type(model)
    with pytest.raises(InvalidBackend):
        ng.enable(model, backend="not-a-backend", example=x)
    assert type(model) is native_class
    assert not hasattr(model, "_niagara_backend")
    torch.testing.assert_close(model(x), expected)


def test_backend_limits_eager_routing_and_selected_methods():
    model = Sim()
    x = torch.tensor([0.5, 0.7], dtype=torch.float64)
    handle = ng.enable(model, backend="eager", max_programs=1)
    model(x)
    with pytest.raises(ng.GuardError, match="limit"):
        model(params=x)
    with pytest.raises(ng.PlacementError, match="already"):
        ng.enable(model)
    handle.close()
    with ng.enable(model, compile=False) as routing:
        torch.testing.assert_close(model(x), Sim()(x))
        assert not routing.programs
    with pytest.raises(ValueError, match="class method"):
        ng.enable(model, methods="missing_method")
    with pytest.raises(ValueError, match="positive"):
        ng.enable(model, max_programs=0)
    with pytest.raises(TypeError, match="not both"):
        ng.enable(model, ng.DevicePlan(), default="cpu")


def test_existing_method_keyword_and_selected_class_method():
    class Named(ck.Module):
        def __init__(self):
            super().__init__("named")
            self.scale = ck.Param("scale", 1.0)

        @ck.forward
        def calculate(self, x, scale, method="sin"):
            return (x * scale).sin() if method == "sin" else (x * scale).cos()

    model = Named()
    x = torch.tensor([0.5, 0.7])
    with ng.enable(model, methods="calculate") as handle:
        torch.testing.assert_close(model.calculate(x, method="cos"), x.cos())
        torch.testing.assert_close(model.calculate(x), x.sin())
        assert len(handle.programs) == 2


@GPU
def test_backend_without_triton_accepts_empty_keyword_trees(monkeypatch):
    from niagara import replay

    original = replay.importlib.util.find_spec
    monkeypatch.setattr(
        replay.importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "triton" else original(name, *a, **k),
    )
    model = Sim().to(device="cuda:0")
    model._grid = model._grid.to("cuda:0")
    x = torch.tensor([0.5, 0.7], dtype=torch.float64, device="cuda:0")
    expected = model(x)
    with ng.enable(model, default="cuda:0", example=x):
        torch.testing.assert_close(model(x), expected)
        torch.testing.assert_close(model(params=x), expected)


@GPU
def test_existing_caustics_calls_split_auto_and_native_disable_all_gradients():
    pytest.importorskip("caustics")
    from examples.niagara_examples.caustics import image_problem

    p = image_problem(24, device="cuda:0")
    x = p.truth
    q = x.clone().requires_grad_()
    expected = p.model(q)
    expected_without_lens_light = p.model(x, lens_light=False).detach()
    gradient = torch.autograd.grad(expected.square().sum(), q)[0]
    native_class = type(p.model)
    handle = ng.enable(p.model, default="cuda:0", modules={"source": "cuda:1"}, example=x)
    actual = p.model(x)
    assert actual.device == torch.device("cuda:0")
    assert p.model.source.Re.device == torch.device("cuda:1")
    assert handle.programs[0].backend == "cuda_graph"
    torch.testing.assert_close(actual, expected.detach())
    q = x.clone().requires_grad_()
    result = p.model(q)
    torch.testing.assert_close(result, expected.detach())
    torch.testing.assert_close(torch.autograd.grad(result.square().sum(), q)[0], gradient)
    torch.testing.assert_close(p.model(x, lens_light=False), expected_without_lens_light)
    with handle.session():
        torch.testing.assert_close(p.model(x), expected.detach())
    handle.close()
    assert type(p.model) is native_class
    assert p.model.source.Re.device == torch.device("cuda:0")
    torch.testing.assert_close(p.model(x), expected.detach())
