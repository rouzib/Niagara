"""Full 21-parameter image inference; rebuilds data, fit, chains, and evidence."""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

import niagara as ng

# The examples are distributed with the source checkout, not the wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.niagara_examples.caustics import image_problem  # noqa: E402
from examples.niagara_examples.sampling import mala_transition, sample_mala  # noqa: E402


def synchronize():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def diagnostics(draws, names):
    import arviz as az

    # draws: (draw, independent chain, parameter)
    dataset = az.from_dict(posterior={name: draws[:, :, i].T for i, name in enumerate(names)})
    rhat = az.rhat(dataset, method="rank")
    bulk, tail = az.ess(dataset, method="bulk"), az.ess(dataset, method="tail")
    rows = {
        name: {
            "rhat": float(rhat[name]),
            "bulk_ess": float(bulk[name]),
            "tail_ess": float(tail[name]),
        }
        for name in names
    }
    return {
        "parameters": rows,
        "max_rhat": max(r["rhat"] for r in rows.values()),
        "min_bulk_ess": min(r["bulk_ess"] for r in rows.values()),
        "min_tail_ess": min(r["tail_ess"] for r in rows.values()),
        "accepted": all(
            np.isfinite(list(r.values())).all()
            and r["rhat"] < 1.01
            and min(r["bulk_ess"], r["tail_ess"]) >= 400
            for r in rows.values()
        ),
    }


def summary(draws, names, truth, out):
    flat = draws.reshape(-1, draws.shape[-1])
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "truth", "mean", "std", "q025", "median", "q975"])
        for i, name in enumerate(names):
            writer.writerow(
                [
                    name,
                    truth[i],
                    flat[:, i].mean(),
                    flat[:, i].std(),
                    *np.quantile(flat[:, i], [0.025, 0.5, 0.975]),
                ]
            )


def run(args):
    if args.chains < 2 or args.samples < 4:
        raise ValueError("Use at least two chains and four draws")
    out = args.out
    if (out / "result.json").exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    problem = image_problem(args.size, device=args.device, psf_mode=args.psf_mode)
    model, truth = problem.model, problem.truth
    placement = ng.assign_devices(
        model,
        ng.DevicePlan(args.device, {"source": args.source_device} if args.source_device else {}),
    )
    generator = torch.Generator(device=truth.device).manual_seed(args.seed)
    clean = model(truth).detach()
    variance = clean.abs() / 500 + 0.005**2
    observed = (
        clean
        + (clean.abs() / 500).sqrt()
        * torch.randn(clean.shape, device=truth.device, dtype=truth.dtype, generator=generator)
        + 0.005
        * torch.randn(clean.shape, device=truth.device, dtype=truth.dtype, generator=generator)
    )
    z0 = torch.zeros_like(truth)
    start = time.perf_counter()
    render = ng.compile_model(model, problem.decode(z0), backend=args.backend, record_scopes=True)

    def posterior_transform(f):
        def posterior(z):
            image = f(problem.decode(z))
            return -0.5 * ((image - observed).square() / variance).sum() - 0.5 * z.square().sum()

        return posterior

    posterior = render.transform(posterior_transform, z0, backend=args.backend)
    derivative = posterior.transform(torch.func.grad_and_value, z0, backend=args.backend)
    # Validate every latent gradient against ordinary autograd and finite differences.
    probe = z0.clone().requires_grad_()
    native_value = posterior_transform(model)(probe)
    (native_gradient,) = torch.autograd.grad(native_value, probe)
    gradient, value = derivative(z0)
    torch.testing.assert_close(value, native_value, atol=1e-8, rtol=1e-11)
    torch.testing.assert_close(gradient, native_gradient, atol=1e-8, rtol=1e-10)
    differences = []
    for i in range(len(z0)):
        delta = torch.zeros_like(z0)
        delta[i] = 1e-5
        differences.append((posterior(z0 + delta) - posterior(z0 - delta)) / 2e-5)
    finite_difference = torch.stack(differences)
    finite_error = (gradient - finite_difference).abs()
    if not torch.all(finite_error < 1e-3 + 1e-5 * gradient.abs()):
        raise AssertionError("Finite difference validation failed")

    def objective(z):
        g, v = derivative(torch.as_tensor(z, device=truth.device, dtype=truth.dtype))
        return -float(v.detach().cpu()), -g.detach().cpu().numpy()

    fitted = minimize(
        objective,
        z0.cpu().numpy(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": args.map_steps, "ftol": 1e-12, "gtol": 1e-6},
    )
    zmap = torch.as_tensor(fitted.x, device=truth.device, dtype=truth.dtype)
    # Compute J by ordinary-autograd JVP columns. This also works across physical
    # GPUs on releases whose torch.func reverse AD has TensorWrapper limitations.
    eye = torch.eye(len(zmap), device=truth.device, dtype=truth.dtype)
    jacobian = torch.stack(
        [
            torch.autograd.functional.jvp(lambda z: render.graph(problem.decode(z)), zmap, unit)[
                1
            ].flatten()
            for unit in eye
        ],
        dim=1,
    )
    precision = jacobian.T @ (jacobian / variance.flatten()[:, None]) + eye
    factor = torch.linalg.cholesky(precision)
    whitening = torch.linalg.solve_triangular(factor.T, eye, upper=True)
    white_posterior = posterior.transform(
        lambda f: lambda y: f(zmap + whitening @ y), z0, backend=args.backend
    )
    white_derivative = white_posterior.transform(
        torch.func.grad_and_value, z0, backend=args.backend
    )
    initial = 0.5 * torch.randn(
        (args.chains, len(truth)), device=truth.device, dtype=truth.dtype, generator=generator
    )
    batched = ng.vmap(white_derivative, initial, chunk_size=args.chunk_size, backend=args.backend)
    g0, v0 = batched(initial)
    torch.testing.assert_close(g0, torch.stack([white_derivative(row)[0] for row in initial]))
    synchronize()
    setup = time.perf_counter() - start
    result = {
        "configuration": vars(args) | {"out": str(out)},
        "versions": {
            "torch": torch.__version__,
            "niagara": ng.__version__,
            "caskade": __import__("caskade").__version__,
            "caustics": __import__("caustics").__version__,
        },
        "backend": batched.backend,
        "setup_seconds": setup,
        "map": {
            "success": bool(fitted.success),
            "message": str(fitted.message),
            "iterations": int(fitted.nit),
            "negative_log_probability": float(fitted.fun),
        },
        "validation": {
            "value_abs": float((value - native_value).detach().abs()),
            "all_gradient_abs_max": float((gradient - native_gradient).abs().max()),
            "all_gradient_finite_difference_abs_max": float(finite_error.max()),
        },
        "homes": placement.diagnostics()["homes"],
        "source_sha256": {
            p.relative_to(Path(__file__).resolve().parents[1]).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for folder in ("src", "examples")
            for p in (Path(__file__).resolve().parents[1] / folder).rglob("*.py")
        },
    }
    np.savez(
        out / "data.npz",
        observation=observed.cpu(),
        variance=variance.cpu(),
        clean=clean.cpu(),
        truth=truth.cpu(),
        zmap=zmap.cpu(),
        whitening=whitening.cpu(),
        prior_center=problem.center.cpu(),
        prior_scale=problem.scale.cpu(),
        names=problem.names,
    )
    with (out / "priors.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "transform", "latent_center", "latent_scale"])
        writer.writerows(
            zip(
                problem.names,
                problem.kinds,
                problem.center.cpu().tolist(),
                problem.scale.cpu().tolist(),
            )
        )
    draws_for_plot = None
    if args.sampler in ("mala", "both"):
        transition_example = (
            initial,
            g0,
            v0,
            torch.ones_like(initial),
            torch.full_like(v0, 0.5),
            truth.new_tensor(args.step_size),
        )
        start = time.perf_counter()
        transition = ng.compile_model(
            mala_transition(batched.graph), transition_example, backend=args.backend
        )
        synchronize()
        result["transition_setup_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        with transition.session():
            sampled = sample_mala(
                transition,
                initial,
                g0,
                v0,
                warmup=args.warmup,
                samples=args.samples,
                step_size=args.step_size,
                seed=args.seed + 1,
                timing=True,
            )
        synchronize()
        elapsed = time.perf_counter() - start
        physical = problem.decode(zmap + sampled["draws"] @ whitening.T).cpu().numpy()
        np.savez_compressed(
            out / "mala_samples.npz",
            samples=physical,
            log_probability=sampled["log_probability"].cpu(),
            names=problem.names,
        )
        result["mala"] = {
            "warmup_and_production_seconds": elapsed,
            **sampled["timings"],
            "acceptance": sampled["acceptance"].cpu().tolist(),
            "step_size": float(sampled["step_size"]),
            "diagnostics": diagnostics(physical, problem.names),
        }
        summary(physical, problem.names, truth.cpu().numpy(), out / "mala_summary.csv")
        draws_for_plot = physical
    if args.sampler in ("emcee", "both"):
        import emcee

        if args.walkers < 2 * len(truth) or args.walkers % 2:
            raise ValueError("Emcee needs an even walker count >= 42")
        programs = {}
        for count in (args.walkers, args.walkers // 2):
            example = initial.new_zeros((count, len(truth)))
            programs[count] = ng.vmap(
                white_posterior, example, chunk_size=args.emcee_chunk_size, backend=args.backend
            )

        def density(positions):
            value = programs[len(positions)](
                torch.as_tensor(positions, device=truth.device, dtype=truth.dtype)
            )
            return value.detach().cpu().numpy()

        np.random.seed(args.seed)
        sampler = emcee.EnsembleSampler(args.walkers, len(truth), density, vectorize=True)
        start = time.perf_counter()
        state = sampler.run_mcmc(
            np.random.default_rng(args.seed).normal(scale=0.5, size=(args.walkers, len(truth))),
            args.warmup,
            progress=False,
        )
        sampler.reset()
        sampler.run_mcmc(state, args.samples, progress=False)
        elapsed = time.perf_counter() - start
        chain = torch.as_tensor(sampler.get_chain(), device=truth.device, dtype=truth.dtype)
        physical = problem.decode(zmap + chain @ whitening.T).cpu().numpy()
        np.savez_compressed(
            out / "emcee_samples.npz",
            samples=physical,
            log_probability=sampler.get_log_prob(),
            names=problem.names,
        )
        tau = sampler.get_autocorr_time(tol=0)
        result["emcee"] = {
            "warmup_and_production_seconds": elapsed,
            "mean_acceptance": float(sampler.acceptance_fraction.mean()),
            "tau": tau.tolist(),
            "length_criterion_passed": bool(np.all(args.samples > 50 * tau)),
            "rhat": None,
            "rhat_reason": "Interacting walkers are not independent chains",
        }
        summary(physical, problem.names, truth.cpu().numpy(), out / "emcee_summary.csv")
        if draws_for_plot is None:
            draws_for_plot = physical
    if args.plot and draws_for_plot is not None:
        from mcmc_figures import figures

        flat = draws_for_plot.reshape(-1, 21)
        indices = np.linspace(0, len(flat) - 1, 128, dtype=int)
        theta = torch.as_tensor(flat[indices], device=truth.device, dtype=truth.dtype)
        predictive = ng.vmap(render, theta, chunk_size=args.chunk_size, backend=args.backend)(theta)
        mean, std = predictive.mean(0).cpu().numpy(), predictive.std(0).cpu().numpy()
        residual = (mean - observed.cpu().numpy()) / variance.sqrt().cpu().numpy()
        result["predictive"] = {
            "draws": len(indices),
            "standardized_residual_rms": float(np.sqrt(np.mean(residual**2))),
            "uncertainty": "parameter uncertainty only, excludes measurement noise",
        }
        np.savez(out / "predictive.npz", mean=mean, std=std)
        figures(
            observed.cpu().numpy(),
            mean,
            residual,
            draws_for_plot,
            problem.names,
            truth.cpu().numpy(),
            out,
        )
    ng.draw_model(model).render(str(out / "model"), format="svg", cleanup=True)
    (out / "execution_plan.json").write_text(json.dumps(render.execution_plan(), indent=2))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k in ("map", "mala", "emcee", "validation")},
            indent=2,
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-device")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--psf-mode", choices=("conv2d", "fft"), default="conv2d")
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--map-steps", type=int, default=1000)
    parser.add_argument("--step-size", type=float, default=0.4)
    parser.add_argument("--sampler", choices=("mala", "emcee", "both"), default="mala")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--emcee-chunk-size", type=int)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/mcmc"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
