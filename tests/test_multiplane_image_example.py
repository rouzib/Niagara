"""Scientific and ownership checks for the single-image parallel example."""

import pytest
import torch
from caskade import ActiveContext
from test_runtime import GPU

import niagara as ng

pytest.importorskip("caustics")
from examples.niagara_examples.multiplane_image import MultiplaneImage, ParallelMultiplaneImage


def test_fixed_order_image_and_physical_intermediate_deflection():
    model = MultiplaneImage(12, 2, dtype=torch.float64)
    stock = MultiplaneImage(12, 2, dtype=torch.float64, fixed_order=False)
    params = model.get_values().detach().requires_grad_()
    actual = model(params)
    expected = stock(params)
    assert actual.shape == (12, 12)
    assert params.ndim == 1
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    grad = torch.autograd.grad(actual.square().mean(), params)[0]
    reference = torch.autograd.grad(expected.square().mean(), params)[0]
    torch.testing.assert_close(grad, reference, rtol=1e-11, atol=1e-11)

    # The foreground SIE keeps its z=2 normalization in both light branches.
    with ActiveContext(model):
        model.fill_params(params)
        ax, ay = model.foreground.physical_deflection_angle(model._x, model._y)
        z_l, z_m = model.foreground.z_l.value, model.secondary.z_l.value
        factor = model.cosmology.angular_diameter_distance_z1z2(z_l, z_m)
        factor /= model.cosmology.angular_diameter_distance(z_m)
        reduced = model.foreground.reduced_deflection_angle(model._x, model._y)
    torch.testing.assert_close(factor * ax, model._middle_ratio * reduced[0])
    torch.testing.assert_close(factor * ay, model._middle_ratio * reduced[1])


@pytest.mark.parametrize("home", ["cpu", pytest.param("cuda:1", marks=GPU)])
def test_parallel_single_image_values_shared_parameter_gradients_and_repeated_calls(home):
    native = MultiplaneImage(12, 2, dtype=torch.float64, device=home)
    left = MultiplaneImage(12, 2, dtype=torch.float64, device="cuda:0" if home != "cpu" else home)
    right = MultiplaneImage(12, 2, dtype=torch.float64, device=home)
    truth = native.get_values().detach().clone()
    identities = tuple(id(p) for p in left.all_params)
    with ParallelMultiplaneImage(left, right, truth) as parallel:
        for shift in (0.0, 0.001, 0.0):
            params = (truth + shift).requires_grad_()
            expected = native(params)
            actual = parallel(params)
            assert actual.device == torch.device(home)
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
            expected_grad = torch.autograd.grad(expected.square().mean(), params)[0]
            actual_grad = torch.autograd.grad(actual.square().mean(), params)[0]
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)
            assert not left.active and not right.active
    assert tuple(id(p) for p in left.all_params) == identities


@GPU
@pytest.mark.gpu
def test_compiled_single_image_updates_parameters_and_preserves_previous_outputs():
    native = MultiplaneImage(12, 2, device="cuda:1")
    left = MultiplaneImage(12, 2, device="cuda:0")
    right = MultiplaneImage(12, 2, device="cuda:1")
    truth = native.get_values().detach().clone()
    with ParallelMultiplaneImage(left, right, truth, compiled=True) as parallel:
        assert all(program.backend == "cuda_graph" for program in parallel.functions)
        old = parallel(truth)
        saved = old.clone()
        changed = truth * 1.005 + 0.0001
        actual = parallel(changed)
        torch.testing.assert_close(actual, native(changed), rtol=3e-4, atol=3e-4)
        torch.testing.assert_close(parallel(truth), native(truth), rtol=3e-4, atol=3e-4)
        torch.testing.assert_close(old, saved, rtol=0, atol=0)
        assert not torch.equal(actual, old)


def test_parallel_example_rejects_shared_mutable_simulator():
    model = MultiplaneImage(4, 1)
    with pytest.raises(ValueError, match="different simulator instances"):
        ParallelMultiplaneImage(model, model, model.get_values())


def test_full_image_directional_gradient_validation():
    model = MultiplaneImage(8, 2, dtype=torch.float64)
    point = model.get_values().detach().clone()
    program = ng.compile_model(model, point, method="middle", backend="eager")
    weights = torch.ones(8, 8, dtype=point.dtype) / 64

    def scalar(params):
        return (program.graph(params) * weights).sum()

    def directional(inputs):
        params, tangent = inputs
        return torch.func.jvp(scalar, (params,), (tangent,))[1]

    directions = torch.eye(point.numel(), dtype=point.dtype)
    derivative = ng.compile_model(directional, (point, directions[0]), backend="eager")
    actual = torch.stack([derivative((point, tangent)) for tangent in directions])
    params = point.clone().requires_grad_()
    expected = torch.autograd.grad((model.middle(params) * weights).sum(), params)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
