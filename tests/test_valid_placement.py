"""Valid transforms and gradients with inherited parameter homes on two GPUs."""

import caskade as ck
import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_map

import niagara as ng

GPU = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")


def _leaves(value):
    return tuple(leaf for leaf in tree_flatten(value)[0] if isinstance(leaf, torch.Tensor))


def _differentiable(value):
    return tree_map(
        lambda leaf: (
            leaf.detach().clone().requires_grad_() if isinstance(leaf, torch.Tensor) else leaf
        ),
        value,
    )


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("packing", ["array", "list", "dict"])
@pytest.mark.parametrize("grouped", [False, True])
def test_valid_context_with_vector_bounds_and_inherited_parameter_homes(packing, grouped):
    seen = []

    class Child(ck.Module):
        def __init__(self):
            super().__init__("child")
            self.a = ck.Param(
                "a",
                [0.3, 0.7],
                shape=(2,),
                valid=(torch.tensor([0.0, 0.1]), None),
                dynamic=True,
                group=int(grouped),
            )
            self.b = ck.Param(
                "b",
                [0.2, 0.8],
                shape=(2,),
                valid=(None, torch.tensor([1.0, 1.2])),
                dynamic=True,
                group=int(grouped),
            )

        @ck.forward
        def calculate(self, x, a, b):
            seen.append(
                {
                    "input": x.device,
                    "arguments": (a.device, b.device),
                    "parameters": (self.a.value.device, self.b.value.device),
                    "bounds": (self.a.valid[0].device, self.b.valid[1].device),
                }
            )
            return (x * a).sin() + b.square()

    class Model(ck.Module):
        def __init__(self):
            super().__init__("model")
            self.child = Child()
            self.bias = ck.Param(
                "bias",
                [-0.4, -0.6],
                shape=(2,),
                valid=(torch.tensor([-1.0, -1.0]), torch.tensor([0.0, 0.0])),
                dynamic=True,
                group=0,
            )

        @ck.forward
        def calculate(self, x, bias):
            return self.child.calculate(x) + (x * bias).cos()

    model = Model().to(dtype=torch.float64)
    # Build all unconstrained input containers before placement. Their device
    # does not change when the simulator's parameters and bounds move to CUDA.
    unconstrained = model.to_valid(model.get_values(packing))
    reference_params = _differentiable(unconstrained)
    reference_x = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    with ck.ValidContext(model):
        expected = model.calculate(reference_x, params=reference_params)
    expected_gradients = torch.autograd.grad(
        expected.square().sum(), (reference_x, *_leaves(reference_params))
    )
    seen.clear()
    actual_params = _differentiable(unconstrained)
    actual_x = reference_x.detach().clone().requires_grad_()
    assert all(leaf.device == torch.device("cpu") for leaf in _leaves(actual_params))

    with ng.enable(
        model,
        compile=False,
        default="cuda:0",
        modules={model.child: "cuda:1"},
        diagnostics=True,
    ) as handle:
        with ck.ValidContext(model):
            result = model.calculate(actual_x, params=actual_params)
        assert result.device == torch.device("cuda:0")
        assert model.bias.device == torch.device("cuda:0")
        assert model.child.a.device == model.child.b.device == torch.device("cuda:1")
        assert model.bias.valid[0].device == model.bias.valid[1].device == torch.device("cuda:0")
        assert seen == [
            {
                "input": torch.device("cuda:1"),
                "arguments": (torch.device("cuda:1"),) * 2,
                "parameters": (torch.device("cuda:1"),) * 2,
                "bounds": (torch.device("cuda:1"),) * 2,
            }
        ]
        torch.testing.assert_close(result.cpu(), expected)
        actual_gradients = torch.autograd.grad(
            result.square().sum(), (actual_x, *_leaves(actual_params))
        )
        for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
            assert actual.device == torch.device("cpu")
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, reference)
        diagnostics = handle.placement.diagnostics()
        assert diagnostics["transfers"]["cuda:0->cuda:1"] > 0
        assert diagnostics["transfers"]["cuda:1->cuda:0"] > 0
        assert {home for _path, home in diagnostics["observed_calls"]} == {"cuda:0", "cuda:1"}

    assert not model.active
    assert not model.child.active
    assert not hasattr(model, "_niagara_backend")
