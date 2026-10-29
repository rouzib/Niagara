# Testing

Run tests from the Niagara repository with its installed development environment:

```console
python -m pytest -q tests
python -m ruff check .
python -m ruff format --check .
python -m sphinx -W --keep-going -b html docs/source docs/build/html
```

Run the suite against the published Caskade distribution. Check for neighboring
checkouts that could override its import, and record `caskade.__file__` and the
version. The baseline manifest verifies its source bytes against the public
wheel.

For the complete published Caskade suite, run
`python scripts/validate_upstream_caskade.py --mode both`. The
[upstream-suite check](upstream_caskade.md) records which tests actually cross
GPU boundaries and which use native execution.

## Levels of evidence

| Level | What it establishes |
| --- | --- |
| Import/build/style checks | Package structure and basic distribution consistency |
| CPU numerical tests | Portable semantics and cleanup without CUDA |
| Real CUDA tests | Explicit device placement, streams and measured GPU paths |
| Installed-wheel tests | Delivered module bytes behave independently of source-path injection |
| Long scientific runs | The named inference workflow meets its complete acceptance gates |

Use physical GPUs for two-GPU tests; logical CPU labels check semantics only.
Guard CUDA tests with actual device visibility so CPU environments skip cleanly.
Assert output devices, parameter homes and cross-device behavior.

## Meaningful regression checks

The suite checks values and every relevant gradient, including detach behavior.
It also tests parameter ties, active contexts, ownership failures, metadata/model
guards, functional transforms, replicas, streams, independent outputs and
sampler transitions. Adapter tests cover existing calls, keyword flags, mode
specialization, cache limits and native restoration.

Run focused tests while implementing, then the complete suite appropriate to
the change. Investigate each failure before broadening or repeating the checks.

## Notebook checks

```console
python scripts/run_notebooks.py
```

The runner creates a private temporary kernel with the invoking project Python,
avoiding a globally registered `python3` kernel that might use another Torch
build. It saves outputs and an execution manifest with interpreter and version
details.

Sphinx renders saved notebook outputs without executing GPU workloads. Run the
notebooks with the project interpreter and record their environment. Report CPU
and physical two-GPU runs separately, and identify hardware skips as skips.

The runner discovers notebooks throughout `docs/source`, including the user
guide. To execute one chapter, select its path:

```console
python scripts/run_notebooks.py --notebook user_guide/backend.ipynb
```

The execution manifest counts actual image outputs as well as errors, hardware
skips and completed cells. Generate the shared two-GPU overview figures with
`python scripts/render_guide_assets.py`; their numerical record is stored with
the figures.


## Coverage qualification

```console
python scripts/validate_tests.py --mode gpu
python scripts/validate_tests.py --mode cpu
python scripts/validate_upstream_caskade.py --mode both --compile-checks eligible --coverage
```

The GPU profile requires two distinct CUDA devices and enforces at least 90%
combined statement/branch coverage. It records statement and branch percentages
separately, checks that every runtime module was measured, and verifies that the
runtime and test source hashes did not change during the run. The CPU profile
hides CUDA before importing Torch and reports hardware skips separately; it has
no GPU coverage gate.

JSON, JUnit and browsable HTML coverage reports are saved under
`results/0.7.0/upstream_complete/tests_gpu` and `tests_cpu`. The qualification
runner loads Torch and Caskade before starting coverage collection. On Windows
it also selects the Python tracer to avoid an import-time native GC failure.
These changes affect test instrumentation, not simulator execution or assertions.

The published-suite adapter reports compiled numerical probes separately from
dynamic calls. Eligible pure numerical entries must agree with routed execution
on original and changed inputs. Floating inputs also require complete gradient
agreement; integer inputs receive changed-value forward checks. Stateful caches,
random sampling and Python mutation remain dynamic, with eligibility reasons
recorded. A successful eager capture does not count as a compiled check.


The documentation includes the latest physical-GPU coverage browser:

<a href="../_static/coverage/index.html">Browse statement and branch coverage</a>.
The report measures Niagara's Python runtime; CUDA kernels and external Caskade
implementation lines are outside that denominator. Numerical, gradient and
device assertions provide the separate execution checks.
