# Contributing

Use Python 3.11+, a local virtual environment and `pip install -e ".[full,dev,docs]"`.
Install Caskade/Caustics as dependencies. Development uses Torch 2.12; testing
the GPU paths requires two physical CUDA devices.

Follow Caskade's Python style. Document public classes, functions, methods
and properties with NumPy-style Parameters, Returns and Raises sections,
adding examples where useful. Test observable behavior and cleanup after
failures. Avoid tests tied to implementation details.

Before submitting changes:

```console
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m sphinx -W --keep-going -b html docs/source docs/build/html
python -m build
```

Device code must validate homes and preserve parameter identities and ties.
Check gradients, stream ordering and allocation lifetimes together.
Use events and `record_stream` rather than global synchronization in ordinary
execution. CPU correctness and logical devices are not physical GPU evidence.

Run notebooks in the local environment and check every cell before saving
outputs. Documentation builds only render them. Each measurement must record
its version, dependency/import identity, precision, device plan, preparation
cost, timing method and validation errors. Preserve historical source hashes
and put new measurements in a fresh output directory.

CI runs CPU tests, strict documentation builds and distribution checks.
Run hardware tests separately; the optional self-hosted GPU workflow is manual.
Building an artifact does not publish or host it.

