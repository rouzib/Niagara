"""Additional Python-effect and metadata capture regressions."""

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


class MetadataHelper(ck.Module):
    def __init__(self):
        super().__init__("metadata_helper")
        self.bias = ck.Param("bias", 1.0)
        self.meta.helper = Child()
        self.meta.options = {"factor": 1.0}
        self.meta.options["cycle"] = self.meta.options

    @ck.forward
    def __call__(self, x, bias):
        return self.meta.helper(x) * self.meta.options["factor"] + bias


def test_metadata_helper_state_and_cycles_trigger_recapture():
    model = MetadataHelper()
    with ng.enable(model, backend="eager") as backend:
        x = torch.tensor(3.0, requires_grad=True)
        assert model(x).item() == 7
        previous = backend.programs[0]
        model.meta.helper.scale.value = 3.0
        assert model(x).item() == 10
        assert backend.programs[0] is not previous
        assert torch.autograd.grad(model(x), x)[0].item() == 3
        model.meta.options["factor"] = 2.0
        value = model(torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(value, torch.tensor([19.0, 25.0]))


def test_closure_rng_and_counter_wrapper_are_detected_before_execution():
    calls = []

    def sample():
        calls.append("draw")
        return np.random.poisson(3)

    def fn(x):
        return x[:sample()]

    with pytest.raises(ng.CaptureError, match="mutation|randomness"):
        ng.compile_model(fn, torch.ones(10), backend="eager")
    assert calls == []

    def wrapper(function):
        @functools.wraps(function)
        def inner(*args, **kwargs):
            calls.append("call")
            return function(*args, **kwargs)

        return inner

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.bias = ck.Param("bias", 0.0)

        @ck.forward
        @wrapper
        def __call__(self, x):
            return x.square()

    with ng.enable(Model(), backend="eager") as backend:
        assert backend.root(torch.tensor(2.0)).item() == 4
        assert backend.root(torch.tensor(3.0)).item() == 9
        assert not backend.programs
    assert calls == ["call", "call"]


def test_stateful_generator_and_counter_each_execute_once():
    class Stateful(ck.Module):
        def __init__(self):
            super().__init__("stateful")
            self.bias = ck.Param("bias", 0.0)
            self.generator = random.Random(9)
            self.calls = 0

        @ck.forward
        def __call__(self, x):
            self.calls += 1
            return x + self.generator.random()

    expected_rng = random.Random(9)
    expected = [expected_rng.random(), expected_rng.random()]
    model = Stateful()
    with ng.enable(model, backend="eager") as backend:
        assert model(torch.tensor(0.0)).item() == pytest.approx(expected[0])
        assert model(torch.tensor(0.0)).item() == pytest.approx(expected[1])
        assert model.calls == 2
        assert not backend.programs


def test_dynamic_method_is_rejected_inside_fixed_session(monkeypatch):
    class Mixed(ck.Module):
        def __init__(self):
            super().__init__("mixed")
            self.bias = ck.Param("bias", 0.0)

        @ck.forward
        def pure(self, x):
            return x * 2

        @ck.forward
        def draw(self, x):
            return x + np.random.poisson(2)

    model = Mixed()
    calls = []
    monkeypatch.setattr(np.random, "poisson", lambda n: calls.append(n) or 1)
    with ng.enable(model, backend="eager", methods=("pure", "draw")) as backend:
        assert model.pure(torch.tensor(2.0)).item() == 4
        with backend.session(), pytest.raises(ng.GuardError, match="Dynamic"):
            model.draw(torch.tensor(2.0))
        assert calls == []
        assert model.draw(torch.tensor(2.0)).item() == 3
        assert calls == [2]


def test_metadata_helper_randomness_falls_back_before_first_draw(monkeypatch):
    class RandomHelper(ck.Module):
        def __init__(self):
            super().__init__("helper")
            self.bias = ck.Param("bias", 0.0)

        @ck.forward
        def __call__(self, x):
            return x + np.random.poisson(3)

    class Parent(ck.Module):
        def __init__(self):
            super().__init__("parent")
            self.bias = ck.Param("bias", 0.0)
            self.meta.helper = RandomHelper()

        @ck.forward
        def __call__(self, x):
            return self.meta.helper(x)

    calls = []
    monkeypatch.setattr(np.random, "poisson", lambda n: calls.append(n) or len(calls))
    model = Parent()
    with ng.enable(model, backend="eager") as backend:
        assert calls == []
        assert model(torch.tensor(1.0)).item() == 2
        assert model(torch.tensor(1.0)).item() == 3
        assert calls == [3, 3]
        assert not backend.programs


_global_counter = 0


def _global_effect(x):
    global _global_counter
    _global_counter += 1
    return x + _global_counter


def test_global_and_nonlocal_mutations_are_rejected_without_executing():
    global _global_counter
    _global_counter = 0
    with pytest.raises(ng.CaptureError, match="mutation"):
        ng.compile_model(_global_effect, torch.tensor(0.0), backend="eager")
    assert _global_counter == 0
    count = 0

    def nonlocal_effect(x):
        nonlocal count
        count += 1
        return x + count

    with pytest.raises(ng.CaptureError, match="mutation"):
        ng.compile_model(nonlocal_effect, torch.tensor(0.0), backend="eager")
    assert count == 0


def test_metadata_subscript_mutation_is_dynamic():
    from types import SimpleNamespace

    class Counter(ck.Module):
        def __init__(self):
            super().__init__("counter")
            self.bias = ck.Param("bias", 0.0)
            self.meta.state = SimpleNamespace(count=[0])

        @ck.forward
        def __call__(self, x):
            self.meta.state.count[0] += 1
            return x + self.meta.state.count[0]

    model = Counter()
    with ng.enable(model, backend="eager") as backend:
        assert model.meta.state.count == [0]
        assert model(torch.tensor(0.0)).item() == 1
        assert model(torch.tensor(0.0)).item() == 2
        assert model.meta.state.count == [2]
        assert not backend.programs


def test_nested_simple_namespace_change_recaptures():
    from types import SimpleNamespace

    model = MetadataHelper()
    model.meta.config = SimpleNamespace(scale=2.0)

    class Config(ck.Module):
        def __init__(self):
            super().__init__("config")
            self.bias = ck.Param("bias", 0.0)
            self.meta.config = SimpleNamespace(scale=2.0)

        @ck.forward
        def __call__(self, x):
            return x * self.meta.config.scale

    model = Config()
    with ng.enable(model, backend="eager") as backend:
        assert model(torch.tensor(3.0)).item() == 6
        prior = backend.programs[0]
        model.meta.config.scale = 4.0
        assert model(torch.tensor(3.0)).item() == 12
        assert backend.programs[0] is not prior


def test_local_tensor_index_assignment_remains_compilable():
    def fn(x):
        copy = x.clone()
        copy[0] = 5
        return copy

    x = torch.tensor([1.0, 2.0])
    program = ng.compile_model(fn, x, backend="eager")
    torch.testing.assert_close(program(x), torch.tensor([5.0, 2.0]))


@pytest.mark.parametrize("replace_before_first_call", [True, False])
def test_new_metadata_helper_effect_is_detected_before_capture(monkeypatch, replace_before_first_call):
    class RandomHelper(ck.Module):
        def __init__(self):
            super().__init__("random_helper")
            self.bias = ck.Param("bias", 0.0)

        @ck.forward
        def __call__(self, x):
            return x + np.random.poisson(3)

    model = MetadataHelper()
    calls = []
    monkeypatch.setattr(np.random, "poisson", lambda n: calls.append(n) or len(calls))
    with ng.enable(model, backend="eager") as backend:
        if not replace_before_first_call:
            assert model(torch.tensor(2.0)).item() == 5
            assert backend.programs
        model.meta.helper = RandomHelper()
        assert model(torch.tensor(2.0)).item() == 4
        assert model(torch.tensor(2.0)).item() == 5
        assert calls == [3, 3]
        assert not backend.programs


def test_plain_model_and_metadata_helper_methods_are_scanned(monkeypatch):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(np.random, "poisson", lambda n: calls.append(n) or len(calls))

    class Plain(ck.Module):
        def __init__(self):
            super().__init__("plain")
            self.bias = ck.Param("bias", 0.0)

        def draw(self):
            return np.random.poisson(3)

        @ck.forward
        def __call__(self, x):
            return x + self.draw()

    class Tool:
        def draw(self):
            return np.random.poisson(3)

    class MetaPlain(ck.Module):
        def __init__(self):
            super().__init__("meta_plain")
            self.bias = ck.Param("bias", 0.0)
            self.meta.config = SimpleNamespace(tool=Tool())

        @ck.forward
        def __call__(self, x):
            return x + self.meta.config.tool.draw()

    for model in (Plain(), MetaPlain()):
        with ng.enable(model, backend="eager") as backend:
            assert "randomness" in backend.dynamic_methods["__call__"]
            detached = backend.dynamic_methods
            detached.clear()
            assert backend.dynamic_methods
            a = model(torch.tensor(0.0))
            b = model(torch.tensor(0.0))
            assert b.item() == a.item() + 1
            assert not backend.programs
    assert calls == [3, 3, 3, 3]


def test_nested_metadata_rng_detected_without_global_random_binding():
    from types import SimpleNamespace

    env = {"ck": ck, "SimpleNamespace": SimpleNamespace, "RNG": random.Random}
    exec(
        '''class Nested(ck.Module):
    def __init__(self):
        super().__init__("nested")
        self.bias = ck.Param("bias", 0.0)
        self.meta.config = SimpleNamespace(rng=RNG(9))
    @ck.forward
    def __call__(self, x):
        return x + self.meta.config.rng.random()
''',
        env,
    )
    model = env["Nested"]()
    reference = random.Random(9)
    with ng.enable(model, backend="eager") as backend:
        assert backend.dynamic_methods
        assert model(torch.tensor(0.0)).item() == pytest.approx(reference.random())
        assert model(torch.tensor(0.0)).item() == pytest.approx(reference.random())
        assert not backend.programs


def test_custom_descriptor_is_not_called_during_preflight():
    class CounterDescriptor:
        def __init__(self):
            self.calls = 0

        def __get__(self, instance, owner):
            self.calls += 1
            return self.calls

    counter = CounterDescriptor()

    class Model(ck.Module):
        value = counter

        def __init__(self):
            super().__init__("descriptor_model")
            self.bias = ck.Param("bias", 0.0)

        @ck.forward
        def __call__(self, x):
            return x + self.value

    with ng.enable(Model(), backend="eager") as backend:
        assert counter.calls == 0
        assert backend.dynamic_methods
        assert backend.root(torch.tensor(0.0)).item() == 1
        assert backend.root(torch.tensor(0.0)).item() == 2
        assert counter.calls == 2


def test_unreferenced_metadata_generator_does_not_disable_compilation():
    class Pure(ck.Module):
        def __init__(self):
            super().__init__("pure_meta")
            self.bias = ck.Param("bias", 0.0)
            self.meta.rng = random.Random(9)
            self.meta.scale = 2.0

        @ck.forward
        def __call__(self, x):
            return x * self.meta.scale

    model = Pure()
    with ng.enable(model, backend="eager") as backend:
        assert model(torch.tensor(3.0)).item() == 6
        assert backend.programs
        assert not backend.dynamic_methods


def test_local_random_imports_fall_back_but_local_torch_still_compiles(monkeypatch):
    class NumpyLocal(ck.Module):
        def __init__(self):
            super().__init__("numpy_local")
            self.bias = ck.Param("bias", 0.0)

        @ck.forward
        def __call__(self, x):
            import numpy as local_numpy
            return x + local_numpy.random.poisson(3)

    class RandomLocal(ck.Module):
        def __init__(self):
            super().__init__("random_local")
            self.bias = ck.Param("bias", 0.0)

        @ck.forward
        def __call__(self, x):
            from random import random as local_draw
            return x + local_draw()

    class TorchLocal(ck.Module):
        def __init__(self):
            super().__init__("torch_local")
            self.bias = ck.Param("bias", 0.0)

        @ck.forward
        def __call__(self, x):
            import torch as local_torch
            return local_torch.sin(x)

    calls = []
    monkeypatch.setattr(np.random, "poisson", lambda n: calls.append(n) or len(calls))
    model = NumpyLocal()
    with ng.enable(model, backend="eager") as backend:
        assert model(torch.tensor(0.0)).item() == 1
        assert model(torch.tensor(0.0)).item() == 2
        assert calls == [3, 3] and backend.dynamic_methods and not backend.programs
    draws = iter([0.25, 0.5])
    monkeypatch.setattr(random, "random", lambda: next(draws))
    model = RandomLocal()
    with ng.enable(model, backend="eager") as backend:
        assert model(torch.tensor(0.0)).item() == 0.25
        assert model(torch.tensor(0.0)).item() == 0.5
        assert backend.dynamic_methods and not backend.programs
    model = TorchLocal()
    with ng.enable(model, backend="eager") as backend:
        x = torch.tensor(0.7)
        torch.testing.assert_close(model(x), torch.sin(x))
        assert backend.programs and not backend.dynamic_methods


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "home,child_home",
    [
        ("cpu", "cpu"),
        pytest.param(
            "cuda:0", "cuda:1",
            marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
        ),
    ],
)
def test_cached_custom_wrapper_effects_are_dynamic_and_cache_once_per_context(
    nested, home, child_home
):
    calls = []

    def observe(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            calls.append("called")
            return function(*args, **kwargs)

        return wrapper

    class Cached(ck.Module):
        def __init__(self):
            super().__init__("cached")
            self.scale = ck.Param("scale", 2.0)

        @ck.active_cache
        @observe
        @ck.forward
        def __call__(self, x, scale):
            return x * scale

    class Root(ck.Module):
        def __init__(self):
            super().__init__("root")
            self.child = Cached()

        @ck.forward
        def __call__(self, x):
            return self.child(x) + self.child(x + 1)

    model = Root() if nested else Cached()
    plan = ng.DevicePlan(home, {"child": child_home} if nested else {})
    with ng.enable(model, plan=plan, backend="eager") as backend:
        assert backend.dynamic_methods
        assert calls == []
        for value in (2.0, 3.0):
            x = torch.tensor(value, device=home, requires_grad=True)
            result = model(x)
            expected = 4 * value if nested else 2 * value
            assert result.device == torch.device(home)
            assert result.item() == expected
            assert torch.autograd.grad(result, x)[0].item() == (4 if nested else 2)
        assert not backend.programs
    assert calls == ["called", "called"]


def test_named_cached_child_wrapper_effect_is_detected_before_capture():
    calls = []

    def observe(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            calls.append("called")
            return function(*args, **kwargs)

        return wrapper

    class Child(ck.Module):
        def __init__(self):
            super().__init__("child")
            self.bias = ck.Param("bias", 1.0)

        @ck.active_cache
        @observe
        @ck.forward
        def value(self, x, bias):
            return x + bias

    class Root(ck.Module):
        def __init__(self):
            super().__init__("root")
            self.child = Child()

        @ck.forward
        def __call__(self, x):
            return self.child.value(x) + self.child.value(x + 1)

    with ng.enable(Root(), backend="eager") as backend:
        assert backend.dynamic_methods and calls == []
        assert backend.root(torch.tensor(2.0)).item() == 6
        assert backend.root(torch.tensor(3.0)).item() == 8
        assert not backend.programs
    assert calls == ["called", "called"]
