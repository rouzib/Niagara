"""Render simple/complex split models, placement graphs and a flag gallery."""

import argparse
import json
import sys
from pathlib import Path

import torch

# The example helpers live in the source checkout, outside the Niagara wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import niagara as ng
from examples.niagara_examples.caustics import multiplane_scene, pixelated_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/lensing"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--other-device", default="cpu")
    parser.add_argument("--gallery", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = {}
    for name, builder in (("pixelated", pixelated_scene), ("multiplane", multiplane_scene)):
        model, params = builder()
        model.to(device=args.device)
        params = params.to(args.device).requires_grad_()
        reference = model(params)
        reference_grad = torch.autograd.grad(reference.square().sum(), params)[0]
        if name == "pixelated":
            homes = {"source": args.other_device}
        else:
            homes = {
                model.cosmo: args.device,
                "field": args.other_device,
                "middle": args.other_device,
                "perturber": args.other_device,
            }
        placement = ng.assign_devices(model, ng.DevicePlan(args.device, homes, diagnostics=True))
        actual = model(params)
        gradient = torch.autograd.grad(actual.square().sum(), params)[0]
        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(gradient, reference_grad, rtol=1e-8, atol=1e-9)
        c = ng.compile_model(model, params, backend="torchscript", record_scopes=True)
        torch.testing.assert_close(c(params), reference, rtol=1e-9, atol=1e-10)
        for fmt in ("svg", "pdf"):
            ng.draw_model(model).render(str(args.out / f"{name}_model"), format=fmt, cleanup=True)
            ng.draw_execution(c).render(
                str(args.out / f"{name}_dependencies"), format=fmt, cleanup=True
            )
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(actual.detach().cpu(), origin="lower", cmap="magma")
        ax.set_title("Lensed sources" if name == "multiplane" else "Pixelated lens")
        ax.set_xlabel("Pixel")
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(args.out / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        (args.out / f"{name}_plan.json").write_text(json.dumps(c.execution_plan(), indent=2))
        if args.gallery:
            ng.gallery(model, args.out / f"{name}_gallery", compiled=c)
        schedule = ng.draw_schedule(c)
        for suffix in ("pdf", "svg"):
            schedule.savefig(args.out / f"{name}_schedule.{suffix}")
        plt.close(schedule)
        records[name] = {
            "value_max_abs": float((actual - reference).detach().abs().max()),
            "gradient_max_abs": float((gradient - reference_grad).abs().max()),
            "parameters": params.numel(),
            "placement": placement.diagnostics(),
        }
    (args.out / "validation.json").write_text(json.dumps(records, indent=2))
    (args.out / "captions.txt").write_text(
        "The pixelated scene places the convergence field and source on different devices. "
        "The multiplane scene includes EPL, pixelated convergence, shear, m3/m4, an "
        "intermediate source whose position is tied to the second-plane EPL, and a "
        "background source. Native graph shapes show ownership and ties. Dependency "
        "graphs show captured numerical prerequisites, not measured runtime overlap.\n"
    )
    print(
        json.dumps(
            {k: {n: v for n, v in r.items() if n != "placement"} for k, r in records.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
