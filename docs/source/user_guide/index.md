# User guide

Choose placement and execution separately. Splitting one simulator, running
independent replicas and submitting coarse tasks each impose different
ownership and communication requirements.

```{figure} ../_static/guide/multi_gpu_flow.svg
:alt: Ray tracing and final image on GPU 0 with source brightness on GPU 1
:width: 100%

A source-region split keeps ray tracing, lens light and PSF convolution on
`cuda:0`. Source evaluation runs on `cuda:1`; tensor copies preserve gradient
flow across the boundary. The arrows describe dependencies, not measured overlap.
```

## Executed examples

The examples include saved cell outputs, simulator images and device checks.
Each notebook builds a complete model. Download it and rerun it with the project
environment; GPU examples state their hardware requirements. Documentation
builds display saved outputs without starting jobs.

```{toctree}
:maxdepth: 2

backend
caskade_compatibility
placement
compilation
derivatives
parallel_work
multiplane_parallel
inference
visualization
performance
troubleshooting
../notebooks/index
```

## A classic lens simulator

```{figure} ../_static/guide/classic_scene.png
:alt: Full image, lensed source, lens light and two-GPU image difference
:width: 100%

A 64 by 64 Caustics simulation with SIE mass, external shear, a Sersic source,
lens light and PSF convolution. The source-only and lens-light panels use their
own intensity scales. The final panel compares source placement on `cuda:1`
with native GPU Caskade on `cuda:0`: maximum image difference is zero; all 21
gradients agree within 2.85e-14 in float64. The lower-right color scale spans
+/-1e-14 to display the zero residual. See the [numerical record](../_static/guide/figures.json).
```
