"""A captured intermediate must be fresh for every execution."""

import caskade as ck
import pytest
import torch

import niagara as ng


class InPlaceImage(ck.Module):
    def __init__(self):
        super().__init__("sim")
        self.scale = ck.Param("scale", 2.0)

    @ck.forward
    def __call__(self, scale):
        image = torch.arange(8, dtype=torch.float64) * scale
        image.view(2, 4).add_(3)
        return image


@pytest.mark.parametrize("backend", ["eager", "torchscript"])
def test_noarg_mutated_intermediate_is_not_shared_between_calls(backend):
    sim = InPlaceImage()
    expected = sim().clone()
    with ng.enable(sim, default="cpu", backend=backend) as engine:
        for _ in range(5):
            torch.testing.assert_close(sim(), expected)
        assert engine.programs


@pytest.mark.parametrize("backend", ["eager", "torchscript"])
def test_mutated_constant_intermediate_preserves_input_gradients(backend):
    def objective(x):
        weights = torch.arange(4, dtype=x.dtype, device=x.device) * 2
        weights.add_(3)
        return (x * weights).sum()

    sample = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64, requires_grad=True)
    compiled = ng.compile_model(objective, sample, backend=backend)
    for _ in range(5):
        x = sample.detach().clone().requires_grad_()
        y = sample.detach().clone().requires_grad_()
        actual, expected = compiled(x), objective(y)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, y)[0])
