# One image with parallel light planes

This example renders one double-source-plane lens system. A foreground mass at
redshift 0.4 lenses a luminous galaxy at redshift 0.9. That galaxy's mass also
deflects light from a background galaxy at redshift 2.0. Both galaxies contribute
to the same observed image.

Each galaxy is described by a Caustics `LightStack` of Sersic components. These
are components of a single physical scene. The input is one packed parameter
vector and the output is one two-dimensional image. There is no parameter batch,
ensemble, independent-image likelihood, or interpolated deflection field.

The background contribution uses the complete coupled Caustics `Multiplane`
ray trace. Later mass planes receive positions produced by earlier planes.
The intermediate galaxy's light can be evaluated independently of that
background contribution. Niagara submits those two physical branches to
different GPU workers, then transfers the intermediate contribution and adds
it to the background contribution. The transfer and final addition belong to
the measured forward call.

```text
foreground mass ──► intermediate-galaxy light ────────────┐
                                                        ├──► one observed image
foreground mass ──► second mass ──► background light ─────┘
```

The arrows describe dependencies. Runtime overlap must be established from the
recorded CUDA activity, rather than this diagram.

## Run the example

Use the repository checkout and its `.venv`. The helper lives in `examples`,
outside the Niagara wheel. The executed
[notebook](../notebooks/03_multiplane_parallel.ipynb) includes a direct
Torch-compiled single-GPU control, synchronized timings and a measured timeline.

```python
import torch
from examples.niagara_examples.multiplane_image import (
    MultiplaneImage,
    ParallelMultiplaneImage,
)

options = dict(size=1024, components=8, quad_level=5)
reference = MultiplaneImage(**options, device="cuda:1")
params = reference.get_values().detach().clone()
assert params.ndim == 1

middle = MultiplaneImage(**options, device="cuda:0")
background = MultiplaneImage(**options, device="cuda:1")

with ParallelMultiplaneImage(middle, background, params, compiled=False) as sim:
    image = sim(params)
    assert image.shape == (1024, 1024)
    torch.testing.assert_close(image, reference(params), rtol=3e-4, atol=3e-4)

with ParallelMultiplaneImage(middle, background, params, compiled=True) as sim:
    image = sim(params)
```

Workers own separate Caskade object graphs. Both receive the same physical
parameter vector, so the foreground mass is shared numerically and its
derivatives include both light paths. No mutable Caskade instance is entered
concurrently. Compilation prepares each pure light branch before submitting
work; the Python task scheduler itself is not compiled.

## Exactness and scope

The parallel implementation evaluates the same analytic SIE models, Sersic
components, distance factors and Gaussian quadrature as the serial simulator.
The first SIE's Einstein radius retains its normalization to source redshift
2.0 in both branches. Angular-distance ratios rescale its deflection when
tracing to the intermediate source at redshift 0.9. Moving a source between
redshifts without this rescaling would change the physical foreground mass.

The two fixed lens redshifts are read once to specialize their known ordering
for capture. The eager validation reference retains stock Caustics redshift
sorting. Changing redshifts or cosmology requires rebuilding the example.

This is an intrinsic surface-brightness image without PSF convolution. Subpixel
quadrature and float32 evaluation are identical across the compared execution
paths. Compilation can change floating-point rounding; value and gradient
differences are recorded rather than asserting bitwise equality.

## Reproduce measurements

```console
python benchmarks/multiplane_image.py --size 1024 --components 8 --quad 5 --repeats 20 --trace --out results/multiplane_image
```

The workflow requires two distinct physical CUDA GPUs and reports a clean skip
otherwise. It verifies the published Caskade dependency, records versions and
source hashes, and saves the exact source used alongside the results. The
comparison includes native eager GPU Caskade, directly Torch-compiled GPU
Caskade, eager two-GPU execution and matched Niagara-compiled one- and two-GPU
execution. Timed calls use changing individual parameter vectors, with both
GPUs synchronized only at explicit timing boundaries. Preparation is recorded
separately.

All compared paths render the full single image with the same components and
integration accuracy. Full-image gradients are checked for every packed scalar,
including the shared foreground mass. Spatial chunks limit autograd memory in
the validation reference only. They sum the same pixel objective; timed
simulator calls do not chunk or batch the scene. Explicit compiled derivatives
evaluate the full image without these validation chunks.

Large quadrature grids help amortize Python scheduling and device transfers.
Smaller scenes can lose their advantage. The measured gain belongs to this
physical decomposition and hardware; adding mass planes alone does not create
independent ray-tracing steps.

## Recorded two-GPU result

On 2026-10-29, two physical RTX 3090 cards with PyTorch 2.12.0+cu126,
Caskade 1.1.5 and Caustics 1.7.1 produced these median completed-call times:

| Execution | GPUs | Time per image |
|---|---:|---:|
| Native eager Caskade | 1 | 190.10 ms |
| Niagara eager tasks | 2 | 131.19 ms |
| Direct `torch.compile` of native Caskade | 1 | 18.88 ms |
| Niagara-compiled light branches | 1 | 7.69 ms |
| Niagara-compiled light branches | 2 | 5.29 ms |

The second GPU gave a 1.45x improvement in both eager and matched
Niagara-compiled execution. The compiled two-GPU path was 35.96x faster than
native eager execution and 3.57x faster than the direct Torch-compiled control.
Those larger ratios combine compilation and parallel execution; they do not
measure the benefit of the second GPU alone. Preparation is excluded.

The compiled trace recorded 3.115 ms of simultaneous GPU kernel execution,
99.59% of the shorter branch's kernel-active time. This is a measured overlap
observation from a profiled call, separate from the wall-time benchmark.

Eager images matched native images exactly in this run. The largest compiled
image absolute difference was 1.178e-4, with relative L2 error below 8.48e-7.
The full-image compiled-gradient comparison had maximum absolute error
6.635e-6 and relative L2 error 1.406e-6. Every parameter was checked, with no
gradient sign changes above the recorded reference floor.

The local record is
`results/multiplane_image_20261029/qualified_jvp/validation.json`, with raw
timing samples, source snapshots, gradient error distributions and GPU traces
alongside it. A prior reverse-mode derivative preparation exceeded 24 GB and is
retained separately in `qualified`; the completed run uses exact forward-mode
derivatives for validation. No gradient throughput claim is made here.

The compiled timings use forward-only CUDA replay with parameter inputs that
do not require an ordinary autograd tape. Gradient validation uses separate
explicit derivative programs. Preparing with `requires_grad=True` selects a
different execution path for ordinary autograd and requires its own timings.
