# Installation

Niagara requires Python 3.11 or newer, PyTorch and Caskade. Local scientific
tests used public pip Caskade 1.1.5, Caustics 1.7.1 and Torch 2.12.0+cu126.
Other combinations allowed by the dependency ranges may still need testing.

## Install the standalone source

From the Niagara repository directory:

```console
python -m venv .venv
```

Activate the environment before installing. On PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[full,dev,docs]"
```

On Linux:

```bash
source .venv/bin/activate
python -m pip install -e '.[full,dev,docs]'
```

For GPU work, install a GPU driver and a CUDA-enabled Torch build that matches
your platform. Follow the [official Torch installation instructions](https://pytorch.org/get-started/locally/).
Check CUDA support in the environment before running a GPU example.

## Choose dependencies

| Need | Extra |
| --- | --- |
| Core placement and tensor-program APIs | base installation |
| Torch Inductor fusion on the qualified Torch/Triton pair | `compile` |
| Caustics workloads | `caustics` |
| Graphviz and scientific plotting Python packages | `visualization` |
| Inference examples and diagnostics | `mcmc` |
| All supported scientific options above | `full` |
| Tests and package checks | `dev` |
| This documentation | `docs` |

The fusion extra pairs Torch 2.12 with Triton 3.7. It selects `triton-windows`
on Windows and Triton on Linux. Without Triton, CUDA replay falls back to
TorchScript; CPU `auto` also uses TorchScript. Check this fallback with your
Torch version, since TorchScript is deprecated upstream.

Graphviz rendering also needs the external `dot` executable on `PATH`. Tensor
execution and matplotlib plots do not require that executable.

## Check the environment

```python
import caskade
import niagara
import torch

print(niagara.__version__)
print(caskade.__version__, caskade.__file__)
print(torch.__version__, torch.cuda.device_count())
```

Use explicit device indices such as `cuda:0`. Two-GPU examples need two
physical CUDA devices. Logical CPU devices can test semantics, but GPU memory
and timing require GPU execution. CPU-only tests skip guarded CUDA cases.

## Build the site locally

```console
python -m sphinx -W --keep-going -b html docs/source docs/build/html
```

For a local preview, serve the generated directory:

```console
python -m http.server 8000 --bind 127.0.0.1 --directory docs/build/html
```

Open `http://127.0.0.1:8000/`. The bundled math and Mermaid browser assets work
without a CDN or internet connection. Basic pages also open at
`docs/build/html/index.html`; use the local server for diagrams because browsers
may restrict ES modules loaded from `file:` URLs. Run notebooks separately
before building: Sphinx renders their saved outputs without executing them.

Install the supplied local source or wheel for this Niagara scientific add-on.
An unrelated public package can have the same name.
