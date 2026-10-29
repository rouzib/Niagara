"""Keep native Caustics construction/calls and explicitly select Niagara."""

import argparse
import json
import sys
from pathlib import Path

import caustics
import torch

# The example helpers live in the source checkout, outside the Niagara wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import niagara as ng
from examples.niagara_examples.caustics import image_problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source-device")
    parser.add_argument("--size", type=int, default=60)
    args = parser.parse_args()
    problem = image_problem(args.size, device=args.device)
    sim, params = problem.model, problem.truth
    assert isinstance(sim, caustics.LensSource)
    native_class = type(sim)
    expected = sim(params).detach()
    with ng.enable(
        sim,
        default=args.device,
        modules={"source": args.source_device} if args.source_device else {},
        example=params,
    ) as engine:
        image = sim(params)
        torch.testing.assert_close(image, expected)
        q = params.detach().requires_grad_()
        (gradient,) = torch.autograd.grad(sim(q).square().sum(), q)
        assert torch.isfinite(gradient).all()
        with engine.session():
            torch.testing.assert_close(sim(params), expected)
        print(
            json.dumps(
                {
                    "niagara": ng.__version__,
                    "caustics": caustics.__version__,
                    "backends": [program.backend for program in engine.programs],
                    "image_shape": list(image.shape),
                    "gradient_parameters": gradient.numel(),
                    "homes": engine.placement.diagnostics()["homes"],
                },
                indent=2,
            )
        )
    assert type(sim) is native_class
    torch.testing.assert_close(sim(params), expected)
    print("Native Caustics calls restored.")


if __name__ == "__main__":
    main()
