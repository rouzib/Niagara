import torch

import niagara as ng
from examples.niagara_examples.sampling import mala_transition, sample_mala


def test_mala_standard_normal_stationary_distribution():
    x = torch.zeros(8, 2, dtype=torch.float64)

    def density(x):
        return -x, -0.5 * x.square().sum(-1)

    step = mala_transition(density)
    example = (
        x,
        *density(x),
        torch.ones_like(x),
        torch.full((8,), 0.5, dtype=x.dtype),
        x.new_tensor(0.7),
    )
    compiled = ng.compile_model(step, example, backend="torchscript")
    result = sample_mala(
        compiled, x, *density(x), warmup=200, samples=2000, step_size=0.7, seed=452
    )
    assert result["draws"].grad_fn is None
    assert result["draws"].abs().mean() > 0.5
    torch.testing.assert_close(
        result["draws"].mean((0, 1)), torch.zeros(2, dtype=x.dtype), atol=0.08, rtol=0
    )
    torch.testing.assert_close(
        result["draws"].std((0, 1)), torch.ones(2, dtype=x.dtype), atol=0.08, rtol=0
    )
    assert result["acceptance"].min() > 0.2


def test_nonfinite_proposals_preserve_current_state_and_cache():
    x = torch.ones(2, 3)
    gradient, value = -x, torch.zeros(2)
    step = mala_transition(
        lambda z: (torch.full_like(z, float("nan")), torch.full((2,), float("nan")))
    )
    actual, g, v, accepted = step((x, gradient, value, x, torch.ones(2) * 0.5, torch.tensor(0.1)))
    torch.testing.assert_close(actual, x)
    torch.testing.assert_close(g, gradient)
    torch.testing.assert_close(v, value)
    assert not accepted.any()
