# Reproducibility

Save the configuration and numerical products needed to repeat each result.
Performance plots need their scene, versions, timing boundary and raw samples
alongside them.

## Record the environment

Save Python, Niagara, Caskade, Caustics, Torch and compiler versions; import
paths; source/distribution hashes; physical GPU names; device visibility and
peer-access capability. CUDA/driver and interconnect differences can change
both performance and capture behavior.

Use the pip-installable Caskade wheel as the reference and label local-fork
compatibility experiments separately. Compare GPU execution with GPU Caskade.
When comparing compilation, include a directly compiled Caskade control that
successfully runs the same workload.

## Preserve the workload

Store all parameter values and their actual packed order, dtype, input layouts,
pixel scale, image extent, PSF configuration, prior/noise model, seeds and batch
settings. State whether the larger image covers a wider field or refines a
fixed field of view. Match scalar losses and all derivative requirements.

For inference, preserve observations, variance, priors, MAP/whitening state,
full samples and per-parameter diagnostics. Predictive uncertainty should say
whether it includes measurement noise.

## Preserve timing and memory definitions

Measure compilation/preparation, warmup, production, synchronized latency and
throughput separately, and keep the raw samples. For memory, distinguish
allocated from reserved bytes and preparation from execution. Include resident
model state, retained outputs and per-device peaks; state which driver or
non-Torch allocations are omitted.

Separate per-device maxima may occur at different times. Label their sum
accordingly. Use a measured trace to assess overlap; task and dependency
diagrams show structure only.

## Keep reports immutable

Write new measurements to a fresh output directory. Preserve older results and
mark which ones have been superseded. Source and wheel hashes should identify
the delivered runtime; historical reports keep their original source versions.
See `docs/reports/README.md` for archived results and the generated validation
summary for the current release.
