"""Capture preflight must not change a model when it fails."""

import caskade as ck
import pytest
import torch

import niagara as ng


class Leaf(ck.Module):
    def __init__(self):
        super().__init__("leaf")
        self.scale = ck.Param("scale", torch.tensor([2.0, 3.0]))

    @ck.forward
    def __call__(self, x, scale):
        return x * scale


class Model(ck.Module):
    def __init__(self):
        super().__init__("model")
        self.leaf = Leaf()
        self.offset = ck.Param("offset", torch.tensor([0.1, 0.2]))
        self.grid = torch.tensor([0.4, 0.6])

    @ck.forward
    def __call__(self, x, offset):
        return self.leaf(x) + offset


@pytest.mark.parametrize(
    "initial,default,child",
    [
        ("cpu", "cpu", "cpu"),
        pytest.param(
            "cuda:1",
            "cuda:0",
            "cuda:1",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="requires two physical CUDA devices"
            ),
        ),
    ],
)
def test_preflight_error_leaves_native_model_untouched(monkeypatch, initial, default, child):
    model = Model().to(device=initial)
    model.grid = model.grid.to(initial)
    nodes = tuple(model.topological_ordering())
    classes = {node: type(node) for node in nodes}
    parameters = tuple(node for node in nodes if isinstance(node, ck.Param))
    values = {node: node.value for node in parameters}
    grid = model.grid
    x = torch.tensor([0.5, 0.8], device=initial, requires_grad=True)
    expected = model(x)
    expected_gradient = torch.autograd.grad(expected.sum(), x)[0]

    def fail_preflight(*args, **kwargs):
        raise RuntimeError("injected capture preflight failure")

    with monkeypatch.context() as patch:
        patch.setattr("niagara.backend.python_effect", fail_preflight)
        with pytest.raises(RuntimeError, match="injected capture preflight failure"):
            ng.enable(model, default=default, modules={"leaf": child})

    assert model.grid is grid and model.grid.device == torch.device(initial)
    assert all(type(node) is classes[node] for node in nodes)
    assert all(not node.active for node in nodes)
    assert all(not hasattr(node, "_niagara_placement") for node in nodes)
    assert not hasattr(model, "_niagara_backend")
    for parameter in parameters:
        assert parameter.value is values[parameter]
        assert parameter.value.device == torch.device(initial)
    q = x.detach().clone().requires_grad_()
    actual = model(q)
    torch.testing.assert_close(actual, expected.detach())
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), q)[0], expected_gradient)

    # The model is still usable by Niagara after the injected error is removed.
    with ng.enable(model, default=default, modules={"leaf": child}, compile=False):
        routed = model(x.detach().to(default))
        torch.testing.assert_close(routed.cpu(), expected.detach().cpu())


@pytest.mark.parametrize(
    "initial,default,child",
    [
        ("cpu", "cpu", "cpu"),
        pytest.param(
            "cuda:1", "cuda:0", "cuda:1",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="requires two physical CUDA devices"
            ),
        ),
    ],
)
def test_dynamic_example_rejected_before_any_placement(monkeypatch, initial, default, child):
    import numpy as np

    class StochasticModel(Model):
        @ck.forward
        def __call__(self, x, offset):
            return self.leaf(x) + offset + np.random.poisson(3)

    model = StochasticModel().to(device=initial)
    model.grid = model.grid.to(initial)
    nodes = tuple(model.topological_ordering())
    classes = {node: type(node) for node in nodes}
    parameters = tuple(node for node in nodes if isinstance(node, ck.Param))
    values = {node: node.value for node in parameters}
    grid = model.grid
    x = torch.tensor([0.5, 0.8], device=initial, requires_grad=True)
    expected = x * model.leaf.scale.value + model.offset.value + 3
    calls = []
    monkeypatch.setattr(np.random, "poisson", lambda n: calls.append(n) or 3)

    with pytest.raises(ng.CaptureError, match="dynamic routed execution"):
        ng.enable(model, default=default, modules={"leaf": child}, example=x)

    assert calls == []
    assert model.grid is grid and model.grid.device == torch.device(initial)
    assert all(type(node) is classes[node] for node in nodes)
    assert all(not node.active for node in nodes)
    assert all(not hasattr(node, "_niagara_placement") for node in nodes)
    assert not hasattr(model, "_niagara_backend")
    for parameter in parameters:
        assert parameter.value is values[parameter]
        assert parameter.value.device == torch.device(initial)
    actual = model(x)
    assert calls == [3]
    torch.testing.assert_close(actual, expected)
