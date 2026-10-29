"""Wrapped Caskade forward and cache semantics across Niagara GPU regions."""

import functools
import inspect

import caskade as ck
import pytest
import torch

import niagara as ng

GPU = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")


def tracking_wrapper(events):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(self, x, *args, **kwargs):
            events.append(("outer_enter", x.device))
            try:
                result = fn(self, x + 0.25, *args, **kwargs)
                return result * 2
            finally:
                events.append(("outer_exit", x.device))
        return wrapped
    return decorate


def transparent(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        return fn(self, *args, **kwargs)
    return wrapped


def model_with_wrapper(events, wrapper):
    class Child(ck.Module):
        def __init__(self):
            super().__init__("child")
            self.weight = ck.Param("weight", 0.75, dynamic=True)

        @wrapper
        @ck.forward
        def evaluate(self, x, weight, fail=False):
            if events is not None:
                events.append(("compute", x.device, weight.device))
            if fail:
                raise LookupError("inner failure")
            return (x * weight).sin()

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.child = Child()

        @ck.forward
        def __call__(self, x):
            return self.child.evaluate(x) + 0.3

    return Model()


@GPU
@pytest.mark.gpu
def test_outer_decorator_runs_on_child_and_preserves_overrides_and_cleanup():
    events = []
    reference = model_with_wrapper(events, tracking_wrapper(events)).to(device="cuda:0", dtype=torch.float64)
    model = model_with_wrapper(events, tracking_wrapper(events)).to(device="cuda:0", dtype=torch.float64)
    original_classes = (type(model), type(model.child))
    x0 = torch.tensor([0.4, 0.9], device="cuda:0", dtype=torch.float64, requires_grad=True)
    p0 = reference.get_values().detach().requires_grad_()
    expected = reference(x0, params=p0)
    expected_grad = torch.autograd.grad(expected.sum(), (x0, p0))
    events.clear()

    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1"},
                   diagnostics=True, compile=False) as handle:
        x = x0.detach().clone().requires_grad_()
        params = p0.detach().clone().requires_grad_()
        actual = model(x, params=params)
        assert actual.device == torch.device("cuda:0")
        torch.testing.assert_close(actual, expected)
        for a, b in zip(torch.autograd.grad(actual.sum(), (x, params)), expected_grad, strict=True):
            torch.testing.assert_close(a, b)
        assert events == [("outer_enter", torch.device("cuda:1")),
                          ("compute", torch.device("cuda:1"), torch.device("cuda:1")),
                          ("outer_exit", torch.device("cuda:1"))]
        assert ("model.child", "cuda:1") in handle.placement.diagnostics()["observed_calls"]

        events.clear()
        q = torch.tensor(1.2, device="cuda:0", dtype=torch.float64, requires_grad=True)
        with ck.ActiveContext(model):
            model.fill_params(params)
            result = model.child.evaluate(x, weight=q)
            assert result.device == torch.device("cuda:1")
            expected_override = ((x + 0.25) * q).sin() * 2
            torch.testing.assert_close(result.cpu(), expected_override.cpu())
            actual_grad = torch.autograd.grad(result.sum(), q, retain_graph=True)[0]
            ref_grad = torch.autograd.grad(expected_override.sum(), q)[0]
            torch.testing.assert_close(actual_grad, ref_grad)
            torch.testing.assert_close(model.child.weight.value.cpu(), params[0].detach().cpu())
            with pytest.raises(LookupError, match="inner failure"):
                model.child.evaluate(x, weight=q, fail=True)
            torch.testing.assert_close(model.child.weight.value.cpu(), params[0].detach().cpu())
        assert not model.active and not model.child.active
        assert events == [("outer_enter", torch.device("cuda:1")),
                          ("compute", torch.device("cuda:1"), torch.device("cuda:1")),
                          ("outer_exit", torch.device("cuda:1")),
                          ("outer_enter", torch.device("cuda:1")),
                          ("compute", torch.device("cuda:1"), torch.device("cuda:1")),
                          ("outer_exit", torch.device("cuda:1"))]
    assert (type(model), type(model.child)) == original_classes
    assert not hasattr(model, "_niagara_placement")
    assert not hasattr(model.child, "_niagara_placement")


@GPU
@pytest.mark.gpu
def test_transparent_outer_decorator_is_compiled_with_real_two_gpu_work_and_gradients():
    reference = model_with_wrapper(None, transparent).to(device="cuda:0", dtype=torch.float64)
    model = model_with_wrapper(None, transparent).to(device="cuda:0", dtype=torch.float64)
    x0 = torch.tensor([0.4, 0.9], device="cuda:0", dtype=torch.float64, requires_grad=True)
    p0 = reference.get_values().detach().requires_grad_()
    expected = reference(x0, params=p0)
    expected_grad = torch.autograd.grad(expected.sum(), (x0, p0))
    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1"},
                   diagnostics=True, backend="eager") as handle:
        x = x0.detach().clone().requires_grad_()
        params = p0.detach().clone().requires_grad_()
        actual = model(x, params=params)
        assert len(handle.programs) == 1
        assert ("model.child", "cuda:1") in handle.placement.diagnostics()["observed_calls"]
        torch.testing.assert_close(actual, expected)
        for a, b in zip(torch.autograd.grad(actual.sum(), (x, params)), expected_grad, strict=True):
            torch.testing.assert_close(a, b)


@GPU
@pytest.mark.gpu
def test_active_cache_outside_decorator_routes_each_caller_and_clears():
    events = []

    class Child(ck.Module):
        def __init__(self):
            super().__init__("child")
            self.weight = ck.Param("weight", 0.75, dynamic=True)

        @ck.active_cache
        @tracking_wrapper(events)
        @ck.forward
        def evaluate(self, x, weight, fail=False):
            events.append(("compute", x.device, weight.device))
            if fail:
                raise LookupError("inner failure")
            return (x * weight).sin()

    class Peer(ck.Module):
        def __init__(self, child):
            super().__init__("peer")
            self.child = child

        @ck.forward
        def evaluate(self, x):
            cached = self.child.evaluate(x)
            assert cached.device == x.device
            return x + cached

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.child = Child()
            self.peer = Peer(self.child)

        @ck.forward
        def __call__(self, x):
            return self.child.evaluate(x) + self.peer.evaluate(x)

    model = Model().to(device="cuda:0", dtype=torch.float64)
    native_descriptor = inspect.getattr_static(type(model.child), "evaluate")
    assert isinstance(native_descriptor, ck.active_cache)
    params = model.get_values()
    x = torch.tensor([0.4, 0.9], device="cuda:0", dtype=torch.float64, requires_grad=True)
    expected = 2 * ((x + 0.25) * params[0]).sin() * 2 + x
    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1", "peer": "cuda:1"},
                   diagnostics=True, compile=False) as handle:
        placed_descriptor = inspect.getattr_static(type(model.child), "evaluate")
        assert isinstance(placed_descriptor, ck.active_cache)
        assert placed_descriptor.cache_attr == native_descriptor.cache_attr
        result = model(x, params=params)
        assert result.device == torch.device("cuda:0")
        torch.testing.assert_close(result, expected)
        actual_grad = torch.autograd.grad(result.sum(), x, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), x)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        assert [item[0] for item in events] == ["outer_enter", "compute", "outer_exit"]
        assert events[1][1:] == (torch.device("cuda:1"), torch.device("cuda:1"))
        assert not hasattr(model.child, "_active_cache_evaluate")
        assert ("model.child", "cuda:1") in handle.placement.diagnostics()["observed_calls"]

        events.clear()
        with ck.ActiveContext(model):
            model.fill_params(params)
            with pytest.raises(LookupError, match="inner failure"):
                model.child.evaluate(x, fail=True)
            assert not hasattr(model.child, "_active_cache_evaluate")
            first = model.child.evaluate(x)
            second = model.child.evaluate(x * 2)
            torch.testing.assert_close(first, second)
            assert hasattr(model.child, "_active_cache_evaluate")
        assert not hasattr(model.child, "_active_cache_evaluate")
        assert [item[0] for item in events] == ["outer_enter", "compute", "outer_exit",
                                                  "outer_enter", "compute", "outer_exit"]


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("positional", [False, True])
def test_decorated_direct_child_call_handles_parameter_conventions(positional):
    events = []
    reference = model_with_wrapper(events, tracking_wrapper(events)).to(device="cuda:0", dtype=torch.float64)
    model = model_with_wrapper(events, tracking_wrapper(events)).to(device="cuda:0", dtype=torch.float64)
    p0 = reference.get_values().detach().requires_grad_()
    x0 = torch.tensor([0.4, 0.9], device="cuda:0", dtype=torch.float64, requires_grad=True)
    expected = reference.child.evaluate(x0, p0) if positional else reference.child.evaluate(x0, params=p0)
    expected_grads = torch.autograd.grad(expected.sum(), (x0, p0))
    events.clear()
    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1"},
                   compile=False, diagnostics=True) as handle:
        x = x0.detach().clone().requires_grad_()
        params = p0.detach().clone().requires_grad_()
        actual = model.child.evaluate(x, params) if positional else model.child.evaluate(x, params=params)
        assert actual.device == torch.device("cuda:1")
        torch.testing.assert_close(actual.cpu(), expected.cpu())
        for a, b in zip(torch.autograd.grad(actual.sum(), (x, params)), expected_grads, strict=True):
            torch.testing.assert_close(a, b)
        assert [item[0] for item in events] == ["outer_enter", "compute", "outer_exit"]
        assert ("model.child", "cuda:1") in handle.placement.diagnostics()["observed_calls"]


@GPU
@pytest.mark.gpu
def test_decorated_forward_routes_pytree_and_scopes_implicit_factories():
    def wrapper(fn):
        @functools.wraps(fn)
        def wrapped(self, values, *args, **kwargs):
            offset = torch.tensor(0.25, dtype=values["x"].dtype)
            assert offset.device == values["x"].device == torch.device("cuda:1")
            return fn(self, {"x": values["x"] + offset}, *args, **kwargs)
        return wrapped

    class Child(ck.Module):
        def __init__(self):
            super().__init__("child")
            self.weight = ck.Param("weight", 0.75)

        @wrapper
        @ck.forward
        def evaluate(self, values, weight):
            a = values["x"] * weight
            return {"value": a, "other": (a.square(),)}

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.child = Child()

        @ck.forward
        def __call__(self, x):
            outputs = self.child.evaluate({"x": x})
            assert outputs["value"].device == x.device
            assert outputs["other"][0].device == x.device
            return outputs["value"] + outputs["other"][0]

    model = Model().to(device="cuda:0", dtype=torch.float64)
    x = torch.tensor([0.4, 0.9], device="cuda:0", dtype=torch.float64, requires_grad=True)
    expected_value = (x + 0.25) * 0.75
    expected = expected_value + expected_value.square()
    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1"}, compile=False):
        actual = model(x)
        assert actual.device == torch.device("cuda:0")
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), x)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
