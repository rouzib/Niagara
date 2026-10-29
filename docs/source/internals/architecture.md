# Runtime architecture

A Caskade region owns a device home: one module and the descendants that
inherit that home. Niagara places these regions as units. This ownership model
is separate from distributing individual scalar tensor operations or training
replicated `torch.nn.Module` models with DDP.

```{mermaid}
%%{init: {"flowchart": {"rankSpacing": 28, "nodeSpacing": 28}}}%%
flowchart TB
    A["Caskade model + parameters"] --> B["Validated device plan<br/>and instance-local routing"]
    B --> C["FX capture, static folding<br/>and prepared guards"]
    C --> D["Tensor compiler"]
    D --> E["CPU / ordinary autograd"]
    D --> F["Device-local CUDA replay"]
    E --> G["Public results"]
    F --> G
```

## Stateful boundary, pure numerical interior

Caskade's contexts fill dynamic parameters, preserve ties and evaluate
`@forward` methods. Placement routes calls through temporary module and parameter
subclasses on the selected instances. At each region boundary it moves inputs, filled parameter
arguments and returned tensors to their explicit homes.

Capture runs this machinery once to produce an FX tensor graph. Repeated calls
execute the graph directly. Guards check that its input metadata, execution
modes and known model state still match the captured program.

The high-level `Backend` intercepts selected existing calls and caches supported
specializations. Calls inside an active context use Niagara's boundary routing;
filled parameter state stays dynamic rather than entering a prepared program.
The lower-level `CompiledModel` lets callers compose pure graphs and apply
functional transforms directly.

## Modules and responsibilities

| Module | Responsibility |
| --- | --- |
| `placement` | Graph paths, ownership validation, device migration and forward-boundary routing |
| `backend` | Opt-in existing-call adapter, specialization cache and native restoration |
| `compiler` | Capture, constant folding, metadata/mode guards, transforms and compiler selection |
| `guards` | Prepared static-model checks |
| `replay` | Local fusion, fixed CUDA buffers, segmented device replay and transfer packing |
| `execution` | Independent replicas and explicit coarse tasks |
| `visualization` | Ownership/dependency views and real profiler export |

Caustics integration workloads and the MALA demonstration live under
`examples/niagara_examples`; they are not part of the installed runtime.

## Synchronization boundaries

PyTorch copies and explicit stream events order work during ordinary calls,
without global CUDA synchronization. Preparation warms the programs and
synchronizes its stream before capture. Timing experiments also synchronize
their measurement boundaries to obtain accurate samples; production calls
do not add those measurement barriers.

The runtime uses one process. Process-per-device isolation, multi-host
collectives, RPC and DDP are outside its supported scope, including when using
replicas or task workers.


## Tensor creation and active parameters

A device plan also governs implicit PyTorch tensor constructors inside a region.
For example, `torch.ones` and `torch.rand` create tensors on that region's home.
Explicit `device` arguments, output buffers and tensor-derived constructors such
as `ones_like` retain PyTorch's normal behavior. Set `factory_device=False` on the
plan to retain normal constructor defaults instead.

This policy is scoped to the executing call and thread. It leaves the global
Torch device unchanged and restores the caller's policy after errors and nested
calls. A copied context gets its own live factory scope when it executes; an
expired scope cannot suppress device routing. Capture records device choices
once, so cached compiled calls do not enter the Python factory policy.

Pointers evaluate where their declared dependencies live, then move the result
to the pointer's own home. Both operations preserve ordinary autograd. Manual
`fill_params` calls inside an already enabled active context route values to
their parameter homes immediately, including valid-range transforms. Enable
Niagara before entering the active context; changing placement while active
remains an error.

Modules stored only in metadata or closures are outside Caskade's owned DAG.
Enable such independent helpers separately or register them as owned regions.
The published-suite adapter enables independent helper graphs explicitly and
records those calls; the runtime does not infer ownership from opaque objects.
