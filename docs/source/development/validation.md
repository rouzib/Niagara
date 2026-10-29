# Validation status

The initial validation summary records the environment, source and wheel checks,
CPU and physical-GPU tests, and completed scientific runs. Each result identifies its
package version and hardware configuration.

```{eval-rst}
:download:`Initial 0.7.0 machine-readable validation summary <../_static/validation.json>`
```

The source record is `results/0.7.0/summary.json`; this site includes a portable
mirror. The [upstream Caskade report](upstream_caskade.md) covers the later
compatibility fixes and their source hashes. Both records predate relocation of
the example Caustics and sampling helpers out of the runtime package. Workflow
configuration and example files describe what can run, not what has run.

(local-qualification)=

## 0.7.0 initial local qualification

Niagara 0.7.0 was tested locally on Windows with Python 3.11, Torch 2.12,
public pip Caskade 1.1.5 and Caustics 1.7.1 on two physical RTX 3090 GPUs.
Those runs produced the following results. This record predates the
[upstream-suite compatibility fixes](upstream_caskade.md). The full MALA runs
were not repeated for those fixes; their recorded source hashes still identify
the runtime they qualified.

| Check | Local result |
| --- | --- |
| Source test suite | 67 passed; no hardware skips |
| Independently installed wheel | 67 passed; no hardware skips |
| CUDA-hidden CPU suite | 42 passed; 25 hardware tests skipped |
| Notebook execution | 9 notebooks, 42 code cells and 18 image outputs; no execution errors or hardware skips |
| Scientific inference | All 21 parameters; one- and two-GPU MALA runs retained 32,000 draws each and passed every stated gate |

The full inference diagnostics were:

| Placement | Maximum R-hat | Minimum bulk ESS, rounded | Minimum tail ESS, rounded |
| --- | --- | --- | --- |
| One GPU | 1.003123 | 2037 | 1903 |
| Two GPUs | 1.003096 | 2055 | 1991 |

The 0.7 resource CLI completed a small smoke evaluation. The complete
performance plots still come from Niagara 0.6 and keep that version label.
CPU checks, logical device labels and rank semantics test different behavior
from physical two-GPU execution. See the current summary for exact
configurations and per-parameter gates.

## Upstream compatibility

The [complete published Caskade suite](upstream_caskade.md) checked Niagara
routing against unchanged upstream tests and notebooks at its recorded,
pre-relocation source hashes. Its report separates passing API tests, observed
split execution and native bypasses. It does not replace the scientific
inference qualification above or establish results for later source revisions.

## What must agree

Compare paths with the same published Caskade/Caustics model, parameter vector,
dtype, image size, PSF mode and loss. Test multiple parameter vectors to catch
results accidentally fixed to the capture example. Check every derivative
component, tensor device and parameter tie.

Test state and cleanup too. Active reassignment, exceptions, stream changes,
output independence, cache boundaries and native restoration can fail even
when the forward image is correct.

## Historical evidence

The 0.6 resource matrix compares native GPU Caskade, directly Inductor-compiled
GPU Caskade, Niagara on one GPU and source-only splitting. Its runtime and
memory records keep their original source hashes and version labels. The
earlier 0.5 long MALA runs check complete 21-parameter chains against their
recorded acceptance gates.

```{eval-rst}
:download:`Historical evidence layout <../../reports/README.md>`
```

Each measurement applies to its recorded source version. Smoke checks verify
execution. Assess inference convergence with the complete per-parameter
diagnostics and gates.

## Acceptance language

“Prepared” means the code or configuration exists. “Locally tested” means the
named checks ran. “Hardware-qualified” identifies the actual GPU topology and
paths tested; “Scientifically accepted” means the stated full inference gates
passed. Use the term that matches the evidence.
