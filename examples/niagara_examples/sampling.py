"""Tensor-only MALA transitions with an explicit Metropolis correction."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch


def mala_transition(
    density: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
    """Construct a tensor-only Metropolis-adjusted Langevin transition.

    Parameters
    ----------
    density : callable
        Batched density returning ``(gradient, log_probability)`` with shapes
        ``(chains, parameters)`` and ``(chains,)``. Explicit derivative programs
        avoid retaining an ordinary autograd tape between transitions.

    Returns
    -------
    callable
        Transition accepting ``(positions, gradient, value, noise, uniform,
        epsilon)`` and returning updated positions, gradient, value and boolean
        acceptance mask. ``noise`` is standard Gaussian with the position shape,
        ``uniform`` has shape ``(chains,)``, and positive ``epsilon`` is scalar.

    Notes
    -----
    The transition contains the forward/reverse proposal correction and rejects
    nonfinite proposed densities, gradients or acceptance ratios. Random inputs
    are supplied explicitly, allowing capture of this deterministic tensor step.
    It does not choose priors, whitening, initial conditions or a convergence gate.
    """

    def step(inputs):
        positions, gradient, value, noise, uniform, epsilon = inputs
        variance = epsilon.square()
        proposal = positions + 0.5 * variance * gradient + epsilon * noise
        proposed_gradient, proposed_value = density(proposal)
        forward = proposal - positions - 0.5 * variance * gradient
        reverse = positions - proposal - 0.5 * variance * proposed_gradient
        log_ratio = (
            proposed_value
            - value
            + 0.5 / variance * (forward.square().sum(-1) - reverse.square().sum(-1))
        )
        finite = (
            torch.isfinite(proposed_value)
            & torch.isfinite(proposed_gradient).all(-1)
            & torch.isfinite(log_ratio)
        )
        accept = finite & (uniform.log() < log_ratio.clamp_max(0))
        return (
            torch.where(accept[:, None], proposal, positions),
            torch.where(accept[:, None], proposed_gradient, gradient),
            torch.where(accept, proposed_value, value),
            accept,
        )

    return step


def sample_mala(
    transition: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
    initial: torch.Tensor,
    initial_gradient: torch.Tensor,
    initial_value: torch.Tensor,
    *,
    warmup: int = 1500,
    samples: int = 4000,
    step_size: float = 0.4,
    seed: int = 123,
    timing: bool = False,
) -> dict[str, Any]:
    """Run batched MALA with warmup-only adaptation of a shared step size.

    Parameters
    ----------
    transition : callable
        Six-input tensor-pytree transition, normally from :func:`mala_transition`
        followed by Niagara capture/compilation.
    initial : torch.Tensor
        Initial chain positions of shape ``(chains, parameters)``.
    initial_gradient : torch.Tensor
        Density gradients at ``initial``, with the same shape/device/dtype.
    initial_value : torch.Tensor
        Log densities at ``initial``, with shape ``(chains,)``.
    warmup : int, optional
        Nonnegative number of adaptation steps, excluded from returned draws.
    samples : int, optional
        Positive number of retained steps per chain.
    step_size : float, optional
        Positive initial Langevin step size. Adaptation stops before production.
    seed : int, optional
        Seed for the dedicated generator on the position tensor's device.
    timing : bool, optional
        Record warmup/production elapsed seconds. CUDA timing introduces an
        explicit synchronization of the final timing event.

    Returns
    -------
    dict
        ``draws`` has shape ``(samples, chains, parameters)``;
        ``log_probability`` has shape ``(samples, chains)``; ``acceptance`` gives
        per-chain production fractions; ``step_size`` is the final scalar tensor.
        ``timings`` contains warmup/production seconds, or None when disabled.

    Raises
    ------
    ValueError
        If warmup/sample counts or the initial step size are invalid.

    Notes
    -----
    Production transitions use independent random draws across chains. Warmup
    adapts one shared scale from mean acceptance. Returned positions/gradients
    are detached each step to prevent growing iteration history. This function
    does not establish convergence; validate all inferred parameters separately.
    """
    if warmup < 0 or samples < 1 or step_size <= 0:
        raise ValueError("Invalid warmup, sample count, or step size")
    x, gradient, value = initial.detach(), initial_gradient.detach(), initial_value.detach()
    generator = torch.Generator(device=x.device).manual_seed(seed)
    log_epsilon = x.new_tensor(step_size).log()
    draws = torch.empty((samples, *x.shape), device=x.device, dtype=x.dtype)
    probabilities = torch.empty((samples, x.shape[0]), device=x.device, dtype=x.dtype)
    accepted = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    boundaries = []

    def mark():
        if x.is_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(x.device))
            boundaries.append(event)
        else:
            boundaries.append(time.perf_counter())

    if timing:
        mark()
    for index in range(warmup + samples):
        if timing and index == warmup:
            mark()
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        uniform = torch.rand(x.shape[0], device=x.device, dtype=x.dtype, generator=generator)
        x, gradient, value, accept = transition(
            (x, gradient, value, noise, uniform, log_epsilon.exp())
        )
        # A captured derivative program returns explicit tensors. Detach also
        # prevents accidental history accumulation in user-supplied transitions.
        x, gradient, value = x.detach(), gradient.detach(), value.detach()
        if index < warmup:
            gain = 0.05 / (1 + index / 100) ** 0.6
            log_epsilon = (log_epsilon + gain * (accept.to(x.dtype).mean() - 0.574)).clamp(-7, 1)
        else:
            draws[index - warmup] = x
            probabilities[index - warmup] = value
            accepted += accept
    times = None
    if timing:
        mark()
        if x.is_cuda:
            boundaries[-1].synchronize()  # Explicit timing boundary only.
            times = {
                "warmup_seconds": boundaries[0].elapsed_time(boundaries[1]) / 1000,
                "production_seconds": boundaries[1].elapsed_time(boundaries[2]) / 1000,
            }
        else:
            times = {
                "warmup_seconds": boundaries[1] - boundaries[0],
                "production_seconds": boundaries[2] - boundaries[1],
            }
    return {
        "draws": draws,
        "log_probability": probabilities,
        "acceptance": accepted / samples,
        "step_size": log_epsilon.exp(),
        "timings": times,
    }
