"""Actual CPU/GPU placement combinations against pip Caskade on CUDA."""

import caskade as ck
import h5py
import numpy as np
import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_map

import niagara as ng

TOPOLOGIES = [
    pytest.param("cuda:0", "cuda:1", "cuda:0", 2, id="two-gpus"),
    pytest.param("cpu", "cuda:0", "cpu", 1, id="cpu-gpu0"),
    pytest.param("cpu", "cuda:1", "cpu", 2, id="cpu-gpu1"),
    pytest.param("cpu", "cuda:0", "cuda:1", 2, id="cpu-two-gpus"),
]


def setup(root, left, right, count):
    if torch.cuda.device_count() < count:
        pytest.skip(f"requires {count} CUDA devices")
    return root, left, right


class Leaf(ck.Module):
    def __init__(self, name):
        super().__init__(name)
        self.scale = ck.Param("scale", 0.7, dynamic=True, valid=(0.0, None))

    @ck.forward
    def __call__(self, x, scale):
        return (x * scale).sin()


class Model(ck.Module):
    def __init__(self):
        super().__init__("model")
        self.left = Leaf("left")
        self.right = Leaf("right")
        self.bias = ck.Param("bias", -0.3, dynamic=True, valid=(None, 0.0))

    @ck.forward
    def __call__(self, x, bias):
        return self.left(x) + self.right(x, scale=bias) + self.left.scale.value * x + bias.square()


def compare_tensors(a, b):
    aa = [v for v in tree_flatten(a)[0] if isinstance(v, torch.Tensor)]
    bb = [v for v in tree_flatten(b)[0] if isinstance(v, torch.Tensor)]
    assert len(aa) == len(bb)
    for x, y in zip(aa, bb, strict=True):
        torch.testing.assert_close(x.detach().cpu(), y.detach().cpu())


def grad_inputs(model, device):
    x = torch.tensor([0.4, 0.8], dtype=torch.float64, device=device, requires_grad=True)
    params = model.get_values().detach().to(device).clone().requires_grad_()
    return x, params


@pytest.mark.gpu
@pytest.mark.parametrize("root,left,right,count", TOPOLOGIES)
def test_forward_overrides_direct_reads_and_gradients(root, left, right, count):
    setup(root, left, right, count)
    native = Model().to(device="cuda:0", dtype=torch.float64)
    model = Model().to(dtype=torch.float64)
    nx, np_ = grad_inputs(native, "cuda:0")
    expected = native(nx, np_)
    ngs = torch.autograd.grad(expected.square().sum(), (nx, np_))
    with ng.assign_devices(
        model, ng.DevicePlan(root, {"left": left, "right": right}, diagnostics=True)
    ) as plan:
        x, p = grad_inputs(model, root)
        actual = model(x, p)
        grads = torch.autograd.grad(actual.square().sum(), (x, p))
        assert actual.device == torch.device(root)
        compare_tensors((actual, grads), (expected, ngs))
        assert model.left.scale.device == torch.device(left)
        assert model.right.scale.device == torch.device(right)
        assert model.bias.device == torch.device(root)
        calls = plan.diagnostics()["observed_calls"]
        assert ("model.left", left) in calls and ("model.right", right) in calls
        transfers = plan.diagnostics()["transfers"]
        for target in {left, right} - {root}:
            assert transfers[f"{root}->{target}"] > 0
            assert transfers[f"{target}->{root}"] > 0
        assert not model.active


class Batched(ck.Module):
    def __init__(self, name):
        super().__init__(name)
        self.a = ck.Param(
            "a",
            [[0.3, 0.7], [0.4, 0.8]],
            shape=(2,),
            batch_shape=(2,),
            dynamic=True,
            valid=(0.0, None),
            group=1,
        )


class Groups(ck.Module):
    def __init__(self):
        super().__init__("groups")
        self.left = Batched("left")
        self.right = Batched("right")
        self.bias = ck.Param(
            "bias", [-0.4, -0.6], shape=(2,), dynamic=True, valid=(None, 0.0), group=0
        )


def clone(value, device, grad=False):
    return tree_map(
        lambda x: (
            x.detach().to(device).clone().requires_grad_(grad) if isinstance(x, torch.Tensor) else x
        ),
        value,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("root,left,right,count", TOPOLOGIES)
def test_grouped_batched_valid_context(root, left, right, count):
    setup(root, left, right, count)
    native = Groups().to(device="cuda:0", dtype=torch.float64)
    model = Groups().to(dtype=torch.float64)
    with ng.assign_devices(model, ng.DevicePlan(root, {"left": left, "right": right})):
        compare_tensors(model.get_values(), native.get_values())
        with ck.ValidContext(native), ck.ValidContext(model):
            nv, mv = native.get_values(), model.get_values()
            compare_tensors(mv, nv)
            native.set_values(clone(nv, "cuda:0"))
            model.set_values(clone(mv, root))
        compare_tensors(model.get_values(), native.get_values())
        assert model.bias.device == torch.device(root)
        assert model.left.a.device == torch.device(left)
        assert model.right.a.device == torch.device(right)


@pytest.mark.gpu
@pytest.mark.parametrize("root,left,right,count", TOPOLOGIES)
def test_hdf_save_load_values_and_homes(tmp_path, root, left, right, count):
    setup(root, left, right, count)
    native = Groups().to(device="cuda:0", dtype=torch.float64)
    model = Groups().to(dtype=torch.float64)
    # Native HDF serialization requires CPU tensors; values come from native CUDA model.
    reference = [p.value.detach().cpu() for p in (native.bias, native.left.a, native.right.a)]
    filename = str(tmp_path / "state.h5")
    with ng.assign_devices(model, ng.DevicePlan(root, {"left": left, "right": right})):
        model.save_state(filename)
        with h5py.File(filename, "r") as f:
            for name, expected in zip(
                ("groups/bias/value", "groups/left/a/value", "groups/right/a/value"),
                reference,
                strict=True,
            ):
                np.testing.assert_allclose(f[name][()], expected.numpy())
        model.load_state(filename)
        compare_tensors([p.value for p in (model.bias, model.left.a, model.right.a)], reference)
        assert model.bias.device == torch.device(root)
        assert model.left.a.device == torch.device(left)
        assert model.right.a.device == torch.device(right)


class PointerModel(Model):
    def __init__(self):
        super().__init__()
        self.right.scale = self.left.scale

    @ck.forward
    def __call__(self, x, bias):
        return self.left(x) + self.right(x) + bias.square()


@pytest.mark.gpu
@pytest.mark.parametrize("root,left,right,count", TOPOLOGIES)
def test_pointer_forward_and_gradients(root, left, right, count):
    setup(root, left, right, count)
    native = PointerModel().to(device="cuda:0", dtype=torch.float64)
    model = PointerModel().to(dtype=torch.float64)
    nx, np_ = grad_inputs(native, "cuda:0")
    expected = native(nx, np_)
    ngs = torch.autograd.grad(expected.sum(), (nx, np_))
    with ng.assign_devices(
        model, ng.DevicePlan(root, {"left": left, "right": right}, diagnostics=True)
    ) as plan:
        x, p = grad_inputs(model, root)
        actual = model(x, p)
        grads = torch.autograd.grad(actual.sum(), (x, p))
        compare_tensors((actual, grads), (expected, ngs))
        assert actual.device == torch.device(root)
        assert model.left.scale.device == torch.device(left)
        assert model.right.scale.device == torch.device(right)
        assert ("model.right", right) in plan.diagnostics()["observed_calls"]


@pytest.mark.gpu
@pytest.mark.parametrize("root,left,right,count", TOPOLOGIES)
def test_compiled_eager_values_and_gradients(root, left, right, count):
    setup(root, left, right, count)
    native = Model().to(device="cuda:0", dtype=torch.float64)
    model = Model().to(dtype=torch.float64)
    nx, np_ = grad_inputs(native, "cuda:0")
    expected = native(nx, np_)
    ngs = torch.autograd.grad(expected.square().sum(), (nx, np_))
    with ng.enable(
        model, default=root, modules={"left": left, "right": right}, compile=True, backend="eager"
    ) as engine:
        x, p = grad_inputs(model, root)
        actual = model(x, p)
        assert len(engine.programs) == 1
        grads = torch.autograd.grad(actual.square().sum(), (x, p))
        compare_tensors((actual, grads), (expected, ngs))
        assert actual.device == torch.device(root)
        assert model.left.scale.device == torch.device(left)
        assert model.right.scale.device == torch.device(right)
        assert model.bias.device == torch.device(root)


def counted(method):
    from functools import wraps

    @wraps(method)
    def inner(self, *args, **kwargs):
        self.calls += 1
        return method(self, *args, **kwargs)

    return inner


class CachedLeaf(Leaf):
    def __init__(self, name):
        super().__init__(name)
        self.calls = 0

    @ck.active_cache
    @counted
    @ck.forward
    def __call__(self, x, scale):
        return (x * scale).sin()


class CachedModel(ck.Module):
    def __init__(self):
        super().__init__("cached")
        self.left = CachedLeaf("left")
        self.right = Leaf("right")

    @ck.forward
    def __call__(self, x):
        return self.left(x) + self.left(x + 2) + self.right(x)


@pytest.mark.gpu
@pytest.mark.parametrize("root,left,right,count", TOPOLOGIES)
def test_wrapped_forward_and_active_cache(root, left, right, count):
    setup(root, left, right, count)
    native = CachedModel().to(device="cuda:0", dtype=torch.float64)
    model = CachedModel().to(dtype=torch.float64)
    nx, np_ = grad_inputs(native, "cuda:0")
    expected = native(nx, np_)
    ngs = torch.autograd.grad(expected.sum(), (nx, np_))
    assert native.left.calls == 1
    with ng.assign_devices(model, ng.DevicePlan(root, {"left": left, "right": right})):
        x, p = grad_inputs(model, root)
        actual = model(x, p)
        grads = torch.autograd.grad(actual.sum(), (x, p))
        compare_tensors((actual, grads), (expected, ngs))
        assert actual.device == torch.device(root)
        assert model.left.calls == 1
        assert not hasattr(model.left, "_active_cache___call__")


class CyclicRanges(ck.Module):
    def __init__(self):
        super().__init__("cyclic")
        self.left = ck.Module("left")
        self.right = ck.Module("right")
        self.angle = ck.Param("angle", 0.3, valid=(-1.0, 1.0), cyclic=True, dynamic=True)
        self.left.angle = ck.Param("angle", 0.4, valid=(0.0, 2.0), cyclic=True, dynamic=True)
        self.right.angle = ck.Param("angle", -0.1, valid=(-2.0, 3.0), cyclic=True, dynamic=True)


@pytest.mark.gpu
@pytest.mark.parametrize("root,left,right,count", TOPOLOGIES)
@pytest.mark.parametrize("method", ["to_valid", "from_valid"])
def test_split_cyclic_ranges_match_native_values_and_gradients(root, left, right, count, method):
    setup(root, left, right, count)
    native = CyclicRanges().to(device="cuda:0", dtype=torch.float64)
    model = CyclicRanges().to(dtype=torch.float64)
    native_input = torch.tensor([-7.1, 8.9, 19.5], device="cuda:0", dtype=torch.float64,
                                requires_grad=True)
    expected = getattr(native, method)(native_input)
    expected_grad = torch.autograd.grad(expected.square().sum(), native_input)[0]
    with ng.assign_devices(model, ng.DevicePlan(root, {"left": left, "right": right})):
        inputs = native_input.detach().to(root).clone().requires_grad_()
        actual = getattr(model, method)(inputs)
        assert actual.device == torch.device(root)
        compare_tensors(actual, expected)
        actual_grad = torch.autograd.grad(actual.square().sum(), inputs)[0]
        compare_tensors(actual_grad, expected_grad)
        assert model.angle.valid[0].device == torch.device(root)
        assert model.left.angle.valid[0].device == torch.device(left)
        assert model.right.angle.valid[0].device == torch.device(right)
        assert model.left.angle.valid[1].device == torch.device(left)
        assert model.right.angle.valid[1].device == torch.device(right)


@pytest.mark.gpu
@pytest.mark.parametrize("root,left,right,count", TOPOLOGIES)
def test_split_cyclic_valid_context_wraps_and_preserves_homes(root, left, right, count):
    setup(root, left, right, count)
    native = CyclicRanges().to(device="cuda:0", dtype=torch.float64)
    model = CyclicRanges().to(dtype=torch.float64)
    raw = torch.tensor([-7.1, 8.9, 19.5], dtype=torch.float64)
    with ng.assign_devices(model, ng.DevicePlan(root, {"left": left, "right": right})):
        with ck.ValidContext(native), ck.ValidContext(model):
            native.set_values(raw.to("cuda:0"))
            model.set_values(raw.to(root))
            compare_tensors(model.get_values(), native.get_values())
        compare_tensors(model.get_values(), native.get_values())
        for param, device in ((model.angle, root), (model.left.angle, left),
                              (model.right.angle, right)):
            assert param.value.device == torch.device(device)
            assert param.valid[0].device == torch.device(device)
            assert param.valid[1].device == torch.device(device)
            assert (param.value >= param.valid[0]).all()
            assert (param.value < param.valid[1]).all()
