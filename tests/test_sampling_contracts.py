"""Sampling reproducibility, retained-density consistency and timing boundaries."""

import pytest
import torch

from examples.niagara_examples.sampling import mala_transition, sample_mala


@pytest.mark.parametrize("options", [{"warmup": -1}, {"samples": 0}, {"step_size": 0.0}])
def test_invalid_sampling_configuration_is_rejected_before_transition(options):
    def unexpected(_):
        pytest.fail("Invalid configuration reached the transition")

    x = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="Invalid"):
        sample_mala(unexpected, x, -x, torch.zeros(2), **options)


@pytest.mark.parametrize(
    "home",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="requires two CUDA devices"
            ),
        ),
        pytest.param(
            "cuda:1",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="requires two CUDA devices"
            ),
        ),
    ],
)
def test_sampling_seed_density_and_timing_on_each_home(home):
    x = torch.zeros(3, 2, dtype=torch.float64, device=home, requires_grad=True)

    def density(value):
        return -value, -0.5 * value.square().sum(-1)

    transition = mala_transition(density)
    torch.manual_seed(812)
    random_state = torch.get_rng_state().clone()
    result = sample_mala(
        transition, x, *density(x), warmup=7, samples=13, step_size=0.3, seed=42, timing=True
    )
    assert torch.equal(torch.get_rng_state(), random_state)
    torch.manual_seed(983)
    repeated = sample_mala(transition, x, *density(x), warmup=7, samples=13, step_size=0.3, seed=42)
    assert result["draws"].shape == (13, 3, 2)
    assert result["draws"].device == torch.device(home)
    assert result["draws"].grad_fn is None
    assert result["log_probability"].grad_fn is None
    torch.testing.assert_close(result["draws"], repeated["draws"])
    torch.testing.assert_close(result["log_probability"], density(result["draws"])[1])
    torch.testing.assert_close(x.detach(), torch.zeros_like(x))
    assert ((result["acceptance"] >= 0) & (result["acceptance"] <= 1)).all()
    assert result["step_size"].item() > 0
    assert repeated["timings"] is None
    assert all(seconds >= 0 for seconds in result["timings"].values())
