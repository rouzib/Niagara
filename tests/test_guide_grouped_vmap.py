"""Regression for compiled batching of split grouped simulator parameters."""

import caskade as ck
import pytest
import torch

import niagara as ng


class Gaussian(ck.Module):
    def __init__(self, name, center, width):
        super().__init__(name)
        self.center = ck.Param("center", center, shape=())
        self.width = ck.Param("width", width, shape=())

    @ck.forward
    def brightness(self, x, center, width):
        return (-((x - center) / width).square()).exp()


class Combined(ck.Module):
    def __init__(self):
        super().__init__("combined")
        self.first = Gaussian("first", [-0.7, -0.2, 0.3, 0.8, 1.1],
                              [0.5, 0.7, 0.9, 1.1, 1.3])
        self.first.center.to_dynamic()
        self.first.width.to_dynamic()
        self.first.width.group = 1
        self.second = Gaussian("second", 0.1, 0.8)

    @ck.forward
    def brightness(self, x):
        return self.first.brightness(x) + 0.5 * self.second.brightness(x)


@pytest.mark.gpu
def test_grouped_guide_style_loss_compiled_derivative_batch_across_two_gpus():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    model = Combined().to(device="cuda:0", dtype=torch.float64)
    grid = torch.linspace(-2.0, 2.0, 25, device="cuda:0", dtype=torch.float64)
    first_group, second_group = model.get_values()
    first_width = first_group.shape[-1]
    batch = torch.cat((first_group, second_group), dim=-1).detach().contiguous()

    def loss(z):
        return model.brightness(grid, (z[:first_width], z[first_width:])).sum()

    expected_values, expected_gradients = [], []
    for row in batch:
        inputs = row.detach().clone().requires_grad_()
        value = loss(inputs)
        expected_values.append(value.detach())
        expected_gradients.append(torch.autograd.grad(value, inputs)[0])
    expected_values = torch.stack(expected_values)
    expected_gradients = torch.stack(expected_gradients)

    guide_gradients = torch.vmap(
        torch.func.grad(
            lambda a, p, q: model.brightness(a, (p, q)).sum(),
            argnums=1,
        ),
        in_dims=(None, 0, 0),
    )(grid, first_group, second_group)
    torch.testing.assert_close(guide_gradients, expected_gradients[:, :first_width])

    with ng.assign_devices(model, ng.DevicePlan("cuda:0", {"first": "cuda:1"})):
        example = batch[0].clone()
        compiled = ng.compile_model(loss, example, backend="eager")
        devices = {node["device"] for node in compiled.execution_plan()["nodes"]}
        assert {"cuda:0", "cuda:1"} <= devices
        derivative = compiled.transform(torch.func.grad_and_value, example, backend="eager")
        mapped = ng.vmap(derivative, batch, backend="eager")
        gradients, values = mapped(batch)
        assert gradients.device == values.device == torch.device("cuda:0")
        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(gradients, expected_gradients)
        torch.testing.assert_close(gradients[:, :first_width], guide_gradients)
