"""Caskade packing and direct Param access across persistent GPU placements."""

import inspect

import caskade as ck
import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_map

import niagara as ng

GPU = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")


def tensor_leaves(value):
    return [item for item in tree_flatten(value)[0] if isinstance(item, torch.Tensor)]


def clone_to(value, device, *, grad=False):
    return tree_map(
        lambda item: item.detach().to(device).clone().requires_grad_(grad)
        if isinstance(item, torch.Tensor)
        else item,
        value,
    )


class Child(ck.Module):
    def __init__(self, grouped=False):
        super().__init__("child")
        self.a = ck.Param("a", [0.3, 0.7], shape=(2,), valid=([0.0, 0.1], None),
                          dynamic=True, group=int(grouped))
        self.b = ck.Param("b", [0.2, 0.8], shape=(2,), valid=(None, [1.0, 1.2]),
                          dynamic=True, group=int(grouped))

    @ck.forward
    def __call__(self, x, a, b):
        return (x * a).sin() + b.square()


class Model(ck.Module):
    def __init__(self, grouped=False):
        super().__init__("model")
        self.child = Child(grouped)
        self.bias = ck.Param("bias", [-0.4, -0.6], shape=(2,),
                             valid=([-1.0, -1.0], [0.0, 0.0]), dynamic=True, group=0)

    @ck.forward
    def __call__(self, x, bias):
        return self.child(x) + (x * bias).cos()


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("grouped", [False, True])
def test_split_flat_get_set_and_direct_valid_transforms(grouped):
    reference = Model(grouped).to(device="cuda:0", dtype=torch.float64)
    model = Model(grouped).to(device="cuda:0", dtype=torch.float64)
    original_get = type(model).get_values
    reference_flat = reference.get_values()
    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1"}, compile=False):
        actual_flat = model.get_values()
        assert all(v.device == torch.device("cuda:0") for v in tensor_leaves(actual_flat))
        for a, b in zip(tensor_leaves(actual_flat), tensor_leaves(reference_flat), strict=True):
            torch.testing.assert_close(a, b)
            assert not a.requires_grad  # Caskade's flat get_values detaches.

        for scheme in ("list", "dict"):
            values = model.get_values(scheme)
            devices = {v.device for v in tensor_leaves(values)}
            assert devices == {torch.device("cuda:0"), torch.device("cuda:1")}

        ref_input = clone_to(reference_flat, "cuda:0", grad=True)
        input_cpu = clone_to(reference_flat, "cpu", grad=True)
        expected = reference.from_valid(ref_input)
        actual = model.from_valid(input_cpu)
        for a, b in zip(tensor_leaves(actual), tensor_leaves(expected), strict=True):
            assert a.device.type == "cpu"
            torch.testing.assert_close(a, b.cpu())
        actual_grad = torch.autograd.grad(sum(a.square().sum() for a in tensor_leaves(actual)), tensor_leaves(input_cpu))
        expected_grad = torch.autograd.grad(sum(a.square().sum() for a in tensor_leaves(expected)), tensor_leaves(ref_input))
        for a, b in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(a, b.cpu())

        with ck.ValidContext(reference), ck.ValidContext(model):
            ref_values = reference.get_values()
            values = model.get_values()
            for a, b in zip(tensor_leaves(values), tensor_leaves(ref_values), strict=True):
                torch.testing.assert_close(a, b)
            reference.set_values(clone_to(ref_values, "cuda:0"))
            model.set_values(clone_to(values, "cpu"))
        for a, b in zip(tensor_leaves(model.get_values()), tensor_leaves(reference.get_values()), strict=True):
            torch.testing.assert_close(a, b)
        assert model.bias.device == torch.device("cuda:0")
        assert model.child.a.device == model.child.b.device == torch.device("cuda:1")
        assert model.child.a.valid[0].device == torch.device("cuda:1")
        assert model.child.b.valid[1].device == torch.device("cuda:1")
    assert type(model).get_values is original_get


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(), (2,)])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_direct_child_parameter_read_preserves_value_home_and_gradients(shape, dynamic, compiled):
    class Component(ck.Module):
        def __init__(self):
            super().__init__("child")
            value = torch.tensor(0.7) if shape == () else torch.tensor([0.7, 0.3])
            self.a = ck.Param("a", value, shape=shape, dynamic=dynamic)

    class Direct(ck.Module):
        def __init__(self):
            super().__init__("direct")
            self.child = Component()

        @ck.forward
        def __call__(self, x):
            return (x + self.child.a.value).square()

    reference = Direct().to(device="cuda:0", dtype=torch.float64)
    model = Direct().to(device="cuda:0", dtype=torch.float64)
    input_ref = torch.tensor([1.0, 2.0], device="cuda:0", dtype=torch.float64, requires_grad=True)
    params_ref = reference.get_values().detach().requires_grad_() if dynamic else None
    expected = reference(input_ref, params=params_ref) if dynamic else reference(input_ref)
    refs = (input_ref, params_ref) if dynamic else (input_ref,)
    expected_grads = torch.autograd.grad(expected.sum(), refs)

    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1"},
                   compile=compiled, backend="eager"):
        assert model.child.a.value.device == torch.device("cuda:1")
        x = input_ref.detach().clone().requires_grad_()
        params = params_ref.detach().clone().requires_grad_() if dynamic else None
        result = model(x, params=params) if dynamic else model(x)
        assert result.device == torch.device("cuda:0")
        torch.testing.assert_close(result, expected)
        grads = torch.autograd.grad(result.sum(), (x, params) if dynamic else (x,))
        for a, b in zip(grads, expected_grads, strict=True):
            torch.testing.assert_close(a, b)
        assert model.child.a.value.device == torch.device("cuda:1")


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("scheme", ["list", "dict"])
@pytest.mark.parametrize("grouped", [False, True])
def test_split_valid_list_dict_assignment_from_one_device(scheme, grouped):
    reference = Model(grouped).to(device="cuda:0", dtype=torch.float64)
    model = Model(grouped).to(device="cuda:0", dtype=torch.float64)
    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1"}, compile=False):
        with ck.ValidContext(reference), ck.ValidContext(model):
            original = reference.get_values(scheme)
            reference.set_values(clone_to(original, "cuda:0"))
            model.set_values(clone_to(original, "cpu"))
            expected = reference.get_values(scheme)
            actual = model.get_values(scheme)
            for a, b in zip(tensor_leaves(actual), tensor_leaves(expected), strict=True):
                torch.testing.assert_close(a.cpu(), b.cpu())
        assert model.child.a.value.device == torch.device("cuda:1")
        assert model.bias.value.device == torch.device("cuda:0")


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
def test_direct_child_pointer_read_preserves_dependencies_and_gradients(compiled):
    class Pointed(ck.Module):
        def __init__(self):
            super().__init__("pointed")
            self.child = Child()
            self.child.alias = ck.Param("alias", lambda p: p["a"].value * 2, shape=(2,))
            self.child.alias.link(self.child.a)

        @ck.forward
        def __call__(self, x):
            return (x + self.child.alias.value).square()

    reference = Pointed().to(device="cuda:0", dtype=torch.float64)
    model = Pointed().to(device="cuda:0", dtype=torch.float64)
    x0 = torch.tensor([0.4, 0.9], device="cuda:0", dtype=torch.float64, requires_grad=True)
    p0 = reference.get_values().detach().requires_grad_()
    expected = reference(x0, params=p0)
    expected_grad = torch.autograd.grad(expected.sum(), (x0, p0))
    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1"},
                   compile=compiled, backend="eager"):
        x = x0.detach().clone().requires_grad_()
        p = p0.detach().clone().requires_grad_()
        result = model(x, params=p)
        torch.testing.assert_close(result, expected)
        grads = torch.autograd.grad(result.sum(), (x, p))
        for actual, target in zip(grads, expected_grad, strict=True):
            torch.testing.assert_close(actual, target)
        assert model.child.alias.value.device == torch.device("cuda:1")
        assert model.child.a.value.device == torch.device("cuda:1")


@GPU
@pytest.mark.gpu
def test_class_transform_methods_restored_exactly_and_compiled_guards_hold():
    class ClassTransforms(ck.Param):
        def to_valid(self, value):
            return self._to_valid_leftvalid(value)

        def from_valid(self, value):
            return self._from_valid_leftvalid(value)

    class Simulator(ck.Module):
        def __init__(self):
            super().__init__("sim")
            self.child = ck.Module("child")
            self.child.p = ClassTransforms("p", [0.4, 0.8], shape=(2,),
                                           valid=([0.0, 0.1], None), dynamic=True)
            # Custom class methods replace the instance methods installed by Param.valid.
            object.__delattr__(self.child.p, "to_valid")
            object.__delattr__(self.child.p, "from_valid")

        @ck.forward
        def __call__(self, x):
            return (x + self.child.p.value).sin()

    model = Simulator().to(device="cuda:0", dtype=torch.float64)
    # Param.to restores instance transforms; remove them again before enabling.
    object.__delattr__(model.child.p, "to_valid")
    object.__delattr__(model.child.p, "from_valid")
    x = torch.ones(2, device="cuda:0", dtype=torch.float64)
    params = model.get_values()
    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1"},
                   compile=True, backend="eager") as engine:
        first = model(x, params=params)
        assert len(engine.programs) == 1
        assert "to_valid" not in vars(model.child.p)
        assert "from_valid" not in vars(model.child.p)
        transformed = model.to_valid(params.cpu())
        recovered = model.from_valid(transformed)
        torch.testing.assert_close(recovered, params.cpu())
        assert "to_valid" not in vars(model.child.p)
        assert "from_valid" not in vars(model.child.p)
        second = model(x, params=params)
        torch.testing.assert_close(second, first)
        assert len(engine.programs) == 1


@GPU
@pytest.mark.gpu
def test_valid_transform_exception_restores_instance_attributes():
    class RaisingParam(ck.Param):
        def from_valid(self, value):
            raise ValueError("custom transform failure")

    model = ck.Module("root")
    model.p = RaisingParam("p", 0.5, valid=(0.0, None), dynamic=True)
    object.__delattr__(model.p, "from_valid")
    with ng.enable(model, default="cuda:0", compile=False):
        assert "from_valid" not in vars(model.p)
        with pytest.raises(ValueError, match="custom transform failure"):
            model.from_valid(torch.tensor([0.5]))
        assert "from_valid" not in vars(model.p)
        assert model.p.valid[0].device == torch.device("cuda:0")


@GPU
@pytest.mark.gpu
def test_split_batched_parameter_packing_matches_native_order_and_shape():
    class Batched(ck.Module):
        def __init__(self):
            super().__init__("batched")
            self.child = ck.Module("child")
            self.child.a = ck.Param("a", [[1.0, 2.0], [3.0, 4.0]], shape=(2,),
                                    batch_shape=(2,), dynamic=True)
            self.bias = ck.Param("bias", [[5.0], [6.0]], shape=(1,),
                                 batch_shape=(2,), dynamic=True)

    reference = Batched().to(device="cuda:0", dtype=torch.float64)
    model = Batched().to(device="cuda:0", dtype=torch.float64)
    expected = reference.get_values()
    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1"}, compile=False):
        actual = model.get_values()
        assert actual.device == torch.device("cuda:0")
        assert actual.shape == expected.shape == (2, 3)
        torch.testing.assert_close(actual, expected)
        updated = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=torch.float64)
        model.set_values(updated)
        reference.set_values(updated.to("cuda:0"))
        torch.testing.assert_close(model.get_values(), reference.get_values())
        assert model.child.a.value.device == torch.device("cuda:1")
        assert model.bias.value.device == torch.device("cuda:0")


def test_parameter_utility_signatures_are_preserved_on_cpu():
    model = Model()
    names = ("get_values", "to_valid", "from_valid")
    before = {name: inspect.signature(getattr(model, name)) for name in names}
    with ng.enable(model, default="cpu", compile=False):
        for name in names:
            assert inspect.signature(getattr(model, name)) == before[name]
    for name in names:
        assert inspect.signature(getattr(model, name)) == before[name]


@GPU
@pytest.mark.gpu
def test_valid_transform_property_setter_without_deleter_is_restored():
    class PropertyParam(ck.Param):
        @property
        def from_valid(self):
            return self._custom_from_valid

        @from_valid.setter
        def from_valid(self, value):
            object.__setattr__(self, "_custom_from_valid", value)
            object.__setattr__(self, "_setter_calls", getattr(self, "_setter_calls", 0) + 1)

    class PropertyModel(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.child = ck.Module("child")
            self.child.p = PropertyParam("p", [0.4, 0.8], shape=(2,),
                                         valid=([0.0, 0.1], None), dynamic=True)

        @ck.forward
        def __call__(self, x):
            return (x + self.child.p.value).sin()

    model = PropertyModel().to(device="cuda:0", dtype=torch.float64)
    expected_transform = model.child.p.from_valid
    x = torch.ones(2, device="cuda:0", dtype=torch.float64)
    params = model.get_values()
    with ng.enable(model, default="cuda:0", modules={"child": "cuda:1"},
                   compile=True, backend="eager") as engine:
        first = model(x, params=params)
        assert len(engine.programs) == 1
        before = dict(vars(model.child.p))
        values = torch.tensor([0.0, 0.0], requires_grad=True)
        actual = model.from_valid(values)
        torch.testing.assert_close(actual, torch.tensor([1.0, 1.1], dtype=actual.dtype))
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), values)[0], torch.ones(2))
        assert model.child.p.from_valid is expected_transform
        assert "from_valid" not in vars(model.child.p)
        assert vars(model.child.p).keys() == before.keys()
        assert all(vars(model.child.p)[name] is original for name, original in before.items())
        assert model.child.p.valid[0].device == torch.device("cuda:1")
        torch.testing.assert_close(model(x, params=params), first)
        assert len(engine.programs) == 1


def test_valid_transform_does_not_access_unselected_parameter_properties():
    class CountingParam(ck.Param):
        @property
        def from_valid(self):
            object.__setattr__(self, "accesses", self.accesses + 1)
            return self._custom_from_valid

        @from_valid.setter
        def from_valid(self, value):
            object.__setattr__(self, "_custom_from_valid", value)

        def __init__(self, name, group):
            object.__setattr__(self, "accesses", 0)
            super().__init__(name, 0.5, valid=(0.0, None), dynamic=True, group=group)

    model = ck.Module("model")
    model.a = CountingParam("a", 0)
    model.b = CountingParam("b", 1)
    values = torch.tensor([0.0])
    with ng.enable(model, compile=False):
        previous = (model.a.accesses, model.b.accesses)
        model.from_valid(values, param_list=(model.a,), group=0)
        assert model.a.accesses == previous[0] + 1
        assert model.b.accesses == previous[1]
