"""Regression tests for state guards and Python effects around capture."""

import functools
import random

import caskade as ck
import numpy as np
import pytest
import torch

import niagara as ng


class Child(ck.Module):
    def __init__(self):
        super().__init__("child")
        self.scale = ck.Param("scale", 2.0)

    @ck.forward
    def __call__(self, x, scale):
        return x * scale


class Population(ck.Module):
    def __init__(self):
        super().__init__("population")
        self.child = Child()

    @ck.forward
    def brightness(self, x, n=8):
        count = np.random.poisson(n)
        return self.child(x[:count])


class MetaModel(ck.Module):
    def __init__(self):
        super().__init__("meta_model")
        self.offset = ck.Param("offset", 1.0)
        self.meta.factor = 2.0
        self.meta.settings = {"shift": [0.0]}

    @ck.forward
    def __call__(self, x, offset):
        return x * self.meta.factor + self.meta.settings["shift"][0] + offset


def audit_wrapper(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        print("one call")
        return function(*args, **kwargs)

    return wrapper


class Wrapped(ck.Module):
    def __init__(self):
        super().__init__("wrapped")
        self.bias = ck.Param("bias", 0.0)

    @ck.forward
    @audit_wrapper
    def __call__(self, x):
        return x.square()


@pytest.mark.parametrize(
    "device,child_home",
    [
        ("cpu", "cpu"),
        pytest.param(
            "cuda:0", "cuda:1", marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="requires two physical CUDA devices"
            )
        ),
    ],
)
def test_numpy_random_runs_once_per_call_and_preserves_gradients(monkeypatch, device, child_home):
    model = Population().to(device=device)
    calls = []

    def poisson(n):
        calls.append(n)
        return (3, 7)[len(calls) - 1]

    monkeypatch.setattr(np.random, "poisson", poisson)
    inputs = torch.arange(10, dtype=torch.float64, device=device, requires_grad=True)
    with ng.enable(
        model, default=device, modules={"child": child_home}, methods="brightness"
    ) as backend:
        assert calls == []
        first = model.brightness(inputs, n=8)
        second = model.brightness(inputs, n=8)
        assert calls == [8, 8]
        assert first.device == second.device == torch.device(device)
        assert (first.numel(), second.numel()) == (3, 7)
        torch.testing.assert_close(first, inputs[:3] * 2)
        torch.testing.assert_close(second, inputs[:7] * 2)
        expected = torch.tensor([4.0] * 3 + [2.0] * 4 + [0.0] * 3, device=device)
        grad = torch.autograd.grad(first.sum() + second.sum(), inputs)[0]
        torch.testing.assert_close(grad, expected.to(dtype=inputs.dtype))
        assert not backend.programs
        with pytest.raises(ng.CaptureError, match="randomness"):
            backend.prepare(inputs, method="brightness")
        with pytest.raises(ng.GuardError, match="Prepare"), backend.session():
            pass
        assert calls == [8, 8]


def test_standalone_capture_rejects_randomness_before_first_draw(monkeypatch):
    calls = []
    monkeypatch.setattr(np.random, "poisson", lambda n: calls.append(n) or 3)

    def fn(x):
        return x[:np.random.poisson(8)]

    with pytest.raises(ng.CaptureError, match="randomness"):
        ng.compile_model(fn, torch.ones(10), backend="torchscript")
    assert calls == []


def test_python_random_global_and_custom_wrapper_side_effects(capsys, monkeypatch):
    class RandomModel(ck.Module):
        def __init__(self):
            super().__init__("random_model")
            self.bias = ck.Param("bias", 0.0)

        @ck.forward
        def __call__(self, x):
            return x + random.random()

    expected_generator = random.Random(1729)
    expected = [expected_generator.random(), expected_generator.random()]
    generator = random.Random(1729)
    with monkeypatch.context() as patch:
        patch.setattr(random, "random", generator.random)
        with ng.enable(RandomModel(), backend="torchscript") as backend:
            assert backend.root(torch.tensor(0.0)).item() == pytest.approx(expected[0])
            assert backend.root(torch.tensor(0.0)).item() == pytest.approx(expected[1])
            assert not backend.programs
    model = Wrapped()
    with ng.enable(model, backend="torchscript") as backend:
        assert model(torch.tensor(2.0)).item() == 4.0
        assert model(torch.tensor(3.0)).item() == 9.0
        assert not backend.programs
    assert capsys.readouterr().out.splitlines() == ["one call", "one call"]


def test_metadata_changes_recapture_only_metadata_and_keep_gradients():
    model = MetaModel()
    x = torch.tensor(3.0, requires_grad=True)
    with ng.enable(model, backend="torchscript") as backend:
        assert model(x).item() == 7
        first = backend.programs[0]
        model.meta.factor = 3.0
        assert model(x).item() == 10
        assert backend.programs[0] is not first
        assert torch.autograd.grad(model(x), x)[0].item() == 3
        model.meta.settings["shift"][0] = 5.0
        assert model(x).item() == 15
        model.new_field = 4
        with pytest.raises(ng.GuardError, match="changed"):
            model(x)


def test_metadata_guards_direct_compiled_and_fixed_session():
    model = MetaModel()
    x = torch.tensor(3.0)
    compiled = ng.compile_model(model, x, backend="eager")
    assert compiled(x).item() == 7
    model.meta.factor = 3.0
    with pytest.raises(ng.GuardError, match="changed"):
        compiled(x)
    model = MetaModel()
    with ng.enable(model, backend="eager") as backend:
        assert model(x).item() == 7
        with pytest.raises(ng.GuardError, match="modified"), backend.session():
            model.meta.factor = 3.0
            model(x)
