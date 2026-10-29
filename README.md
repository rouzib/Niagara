# Niagara: multi-GPU execution for Caskade

Niagara runs [Caskade](https://github.com/ConnorStoneAstro/caskade) simulators
across explicitly chosen devices. It compiles their tensor operations for
repeated calls while keeping Caustics constructors, parameters and call sites
intact. You can put a source or lens on another GPU and still use ordinary
PyTorch gradients.

This repository contains the library, tests, examples, reproducible
benchmarks and Sphinx documentation. Public Python APIs provide type hints
and NumPy-style reference documentation.

## Installation

Use Python 3.11 or newer and install a CUDA-enabled **Torch 2.12** build suitable
for your driver before selecting GPU execution. From this source directory:

```console
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[full,dev,docs]"
```

| Need | Extra |
| --- | --- |
| Core placement/capture with pip Caskade | Base installation |
| Fused GPU kernels, including Windows | `compile` |
| Graphs and figures | `visualization` |
| All runtime/scientific features | `full` |
| Tests, packaging and notebooks | `dev` |
| Documentation site | `docs` |

The tested stack is Caskade 1.1.5, Caustics 1.7.1 and Torch 2.12.0+cu126.
The fusion extra pairs Torch 2.12 with Triton 3.7. CPU execution works without
CUDA, and replay uses TorchScript when Triton is absent. Install Graphviz's
`dot` program if you need graph rendering. Use this source tree or its wheel
to install Niagara: a different PyPI project uses the name `niagara`.

## Keep your Caustics calls

```python
import caustics
import niagara as ng

# Construct sim with your existing Caustics code, then opt in per simulator.
engine = ng.enable(sim, default="cuda:0", modules={"source": "cuda:1"})
image = sim(params)

q = params.detach().requires_grad_()
sim(q).square().sum().backward()

ng.disable(sim)
```

Omit `modules` to use one GPU. Niagara prepares each call form on first use;
pass `example=params` to prepare it at activation. Once all forms are ready,
`engine.session()` moves model-state checks to the boundaries of a fixed loop.
Use `compile=False` for dynamic Python that cannot be captured. Activation
changes the selected simulator instances; Caskade and Caustics remain installed.

Implicit tensor constructors inside a placed region follow its device. Explicit
`device` arguments, output buffers and tensor-derived constructors keep normal
PyTorch behavior. Use `factory_device=False` to retain ordinary constructor
defaults. Enable Niagara before entering a Caskade `ActiveContext`; manual
parameter fills then respect each parameter's device.

For direct control, use `DevicePlan`, `assign_devices`, `compile_model`,
`.transform()` and `ng.vmap`. Apply transforms to the pure graph before
compiling it. The `examples/` directory covers the opt-in adapter, pixelated
and multiplane lensing, and full 21-parameter inference.

## Documentation

```console
python -m sphinx -W --keep-going -b html docs/source docs/build/html
```

For the complete offline preview, serve the generated site locally:

```console
python -m http.server 8000 --bind 127.0.0.1 --directory docs/build/html
```

Open `http://127.0.0.1:8000/`. HTML text also opens directly from
`docs/build/html/index.html`, but browsers may block ES-module diagrams from
`file:` URLs. The site includes installation and first-simulation guides,
device ownership, compilation, derivatives, tasks/replicas, internals, executed
notebooks, API reference, testing and release procedures. Math and diagram
assets are bundled; this local preview needs no CDN or internet connection.
Read the Docs and documentation-test configuration are included.

Execute the notebook examples explicitly with the active project interpreter:

```console
python scripts/run_notebooks.py
```

The runner uses the active interpreter in a temporary kernel and records
GPU visibility and image outputs for each notebook. The guide displays those
saved results alongside the code.

## Validation and performance

```console
python scripts/validate_tests.py --mode gpu
python scripts/validate_tests.py --mode cpu
python scripts/validate_upstream_caskade.py --mode both --compile-checks eligible --coverage
python examples/opt_in_backend.py --device cuda:0 --source-device cuda:1
python benchmarks/runtime_memory.py --sizes 60 128 256 512 1024 --out results/resources
python benchmarks/plot_runtime_memory.py results/resources
```

Tests check values, gradients, explicit tensor devices, ownership/ties, active
contexts, guards, CUDA streams, compiler selection and installed-wheel behavior.
The GPU qualification requires two distinct devices and at least 90% combined
statement/branch coverage; its CPU counterpart reports hardware skips separately.
The published-suite adapter records selected compiled entry points, changed-input
and gradient comparisons, and actual device transfers. The direct references are
unmodified **pip Caskade on GPU**, both eager and directly Inductor-compiled.

The 0.6 rendering and memory results are in `docs/reports/0.6.0`. Guarded
one-GPU runtime was close to compiled Caskade. The tested split took longer
and used more forward memory, but reduced the backward tensor peak per device
at the largest size. These results depend on the workload, precision and
compiler as well as placement. Current validation records are in `results/`.

## Development

Library code lives in `src/niagara`; `tests`, `examples`, `benchmarks`,
`scripts` and `docs` have separate directories. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md), and build
distributions with `python -m build`. The core code is MIT licensed.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the third-party notices
and documentation asset licenses.

