# Validation and performance reports

Each report identifies the package version, source hashes, environment,
simulator and hardware used for its measurements.

## Version 0.6.0

`0.6.0/validation.md` records source and installed-wheel qualification.
`0.6.0/runtime_memory.md` describes the classic Caustics resource matrix.
Selected records and figures are stored under:

- `0.6.0/data/forward`: rendering at 60, 128, 256, 512 and 1024 pixels.
- `0.6.0/data/backward`: ordinary forward/loss/backward at 60, 256 and 1024 pixels.

The reference stack is pip Caskade 1.1.5/Caustics 1.7.1, Torch 2.12.0+cu126,
float64, and two physical RTX 3090 GPUs without peer access. The matrix includes
directly Inductor-compiled GPU Caskade, eager GPU Caskade, one-GPU Niagara and a
source-only two-GPU split.

Timing methods were interleaved after warmup. Memory measurements used 32
independent workers that recorded allocated/reserved counters and
per-device/preparation peaks. Summed device maxima are conservative bounds;
they need not occur at the same time.

## Earlier versioned results

The 0.4 and 0.5 reports cover execution and 21-parameter inference. Their
convergence results apply to the recorded releases and gates. Numerical
results and raw-data hashes keep their original version labels.

## Current qualification

Current results are in `results/0.7.0/summary.json`, with a portable mirror at
`docs/source/_static/validation.json`. The documentation's validation page links
the summary and lists the completed numerical and hardware checks.
