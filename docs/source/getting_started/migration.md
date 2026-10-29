# Migration guide

Niagara 0.7 keeps the placement, capture and existing-call APIs. The package
uses a `src/niagara` layout with its own tests, examples, benchmarks and
documentation. See the current validation record for the software and hardware
configurations that have been tested.

## Existing Niagara callers

The explicit APIs remain the foundation:

```python
placement = ng.assign_devices(sim, ng.DevicePlan("cuda:0"))
compiled = ng.compile_model(sim, example_params)
image = compiled(example_params)
```

Continue using `CompiledModel.transform`, `ng.vmap`, `ReplicaPool`,
`AsyncGraphExecutor` and the graph/profile helpers for their documented tasks.
The instance adapter added in 0.6 lets existing `sim(params)` calls use Niagara
through `ng.enable`.

There are two different close contracts:

| Handle | What `close()` does |
| --- | --- |
| `Placement` from `assign_devices` | Restores classes; leaves tensors at their current homes |
| `Backend` from `enable` | Restores classes and co-locates simulator state at the plan's default home |

Disable or close the current handle before applying a new plan. Closing a
backend moves simulator state only; separately held inputs and outputs stay
on their existing devices.

## Earlier evidence

Versions 0.4 through 0.6 were tested against pip Caskade. Version 0.5 added
fused replay and device-region execution. Version 0.6 added the existing-call
adapter and isolated runtime/memory matrix. Each record keeps its original
version label.

Long converged MALA results apply to the releases that ran them. A later
transition smoke test checks execution only; repeat the full diagnostics to
assess convergence on that version. The
[validation guide](../development/validation.md) separates current checks from
historical results.

