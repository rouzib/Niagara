```{image} _static/niagara-logo.svg
:alt: Niagara
:width: 420px
:align: center
:class: niagara-logo
```

# Niagara documentation

Niagara runs Caskade scientific simulators across explicit devices. It uses
Caskade's `Module`, `Param` and parameter ties, with the same Caustics model
construction. Niagara adds region placement, tensor-program capture and
compilation for repeated execution across the available GPUs.

Keep calling an existing Caustics simulator with `sim(params)`, or use the
lower-level capture API to compose derivatives and batched inference. Enable
Niagara on the simulators that need it. Caskade and Caustics remain installed
dependencies with their own public APIs.

```{figure} _static/guide/classic_scene.png
:alt: Caustics lensed image components and a verified two-GPU comparison
:width: 100%

A classic Caustics lens simulation and its source/light components. The
[executed user guide](user_guide/index.md) shows the model, device placement,
cell outputs and numerical checks.
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Run your first simulator
:link: getting_started/first_simulation
:link-type: doc

Start on CPU, preserve native parameter handling, and enable prepared calls.
:::

:::{grid-item-card} Understand device regions
:link: internals/architecture
:link-type: doc

Follow the graph from ownership validation to local CUDA replay and transfers.
:::

:::{grid-item-card} Explore runnable notebooks
:link: notebooks/index
:link-type: doc

Check forward values and gradients before choosing a multi-GPU partition.
:::

::::

## Choose a path

- **New user:** [install Niagara](getting_started/installation.md), then run the
  [first simulator](getting_started/first_simulation.md).
- **Caustics user:** [enable the backend](user_guide/backend.ipynb) under existing
  calls, or [place regions explicitly](user_guide/placement.ipynb).
- **Inference developer:** read [capture and compilation](user_guide/compilation.ipynb)
  and [derivatives and batching](user_guide/derivatives.ipynb).
- **Contributor:** start with [architecture](internals/architecture.md), the
  [API reference](api/index.rst), and [validation](development/validation.md).

```{note}
Two GPUs do not automatically make a simulator faster or smaller. A partition
must create useful work or memory distribution that compensates for transfers.
The retained [performance measurements](user_guide/performance.md) include cases
where splitting is slower and forward replay retains more memory.
```

```{toctree}
:maxdepth: 4
:hidden:

getting_started/index
user_guide/index
internals/index
api/index
development/index
```
