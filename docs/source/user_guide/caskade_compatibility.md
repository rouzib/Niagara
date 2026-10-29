# Caskade guide compatibility

This page compares Niagara 0.7.0 with the published Caskade 1.1.5
[Advanced Guide](https://caskade.readthedocs.io/en/latest/notebooks/AdvancedGuide.html)
and [Hierarchical Models](https://caskade.readthedocs.io/en/latest/notebooks/HierarchicalModels.html).
The checks use PyTorch and cover the cases described below. They don't establish
compatibility for every Caskade model.

The earlier upstream notebook run used an adapter that enabled Niagara around
individual calls, then returned state to the CPU for the original assertions.
Some notebook operations ran outside an enabled model. That run predates the
fixes described here; see its [test record](../development/upstream_caskade.md).
The newer regression tests call Niagara's public API with persistent placement.
They cover specific paths, not every possible model or compiled program.

```{warning}
Niagara routes selected methods with recognized Python or NumPy randomness,
I/O and some Python mutations dynamically. Detection is incomplete: unknown
side effects and opaque external state can still be captured and reused. Use
`compile=False` when you need those effects to run on every call. Inspectable
metadata is guarded, but opaque object internals and arbitrary closures are not.
```

```{warning}
Standard Niagara HDF5 saves and appends support CUDA-resident values. On a
failed append, Niagara attempts to restore dataset lengths and markers; this
is best effort, not an atomic transaction. Custom writers and storage or
process failures can still leave changes. Check the file before retrying.
Custom persistence methods and descriptor-backed attributes have limits
described below.
```

## Reading the tables

- <span class="compat-status compat-works">Works</span> The stated case passed.
- <span class="compat-status compat-caution">Needs care</span> The feature has a condition or restriction.
- <span class="compat-status compat-fails">Does not work</span> The stated case failed or produced a stale result.
- <span class="compat-status compat-unverified">Not tested</span> There isn't enough evidence to draw a conclusion.

Each row describes one case and has one status. The description gives the
device placement and execution mode behind that status; related cases can have
different results.

## Advanced Guide

| Guide feature and case | Status | What we found |
| --- | --- | --- |
| Parameter values and overrides in routed calls | <span class="compat-status compat-works">Works</span> | Tested dynamic calls and overrides work. A parent can directly read a child parameter on another GPU; Niagara routes the value to the calling region while retaining the parameter's home. Dynamic and compiled-eager tests checked values and gradients. `.npvalue` copies to the host; ordinary CUDA tensors need a host copy for NumPy. |
| Static and dynamic parameter changes | <span class="compat-status compat-caution">Needs care</span> | Caskade's classifications remain available. Clear compiled programs after changing ordinary model state. Close placement before changing graph links. |
| Valid ranges and `ValidContext` with split placement | <span class="compat-status compat-works">Works</span> | Tested flat, grouped, batched, list and dict get/set paths match native Caskade. Flat values gather to the calling module's home; valid bounds follow the input device during transforms. |
| Scalar cyclic range transforms and `ValidContext` with split placement | <span class="compat-status compat-works">Works</span> | Direct utility calls after `assign_devices` were tested across two GPUs, CPU plus either GPU, and CPU plus both GPUs. For three scalar cyclic parameters and non-boundary inputs, `to_valid` and `from_valid` values and input gradients matched pip-installed CUDA Caskade; `ValidContext` wrapped values and preserved parameter and bound homes. |
| Other cyclic range cases with split placement | <span class="compat-status compat-unverified">Not tested</span> | The checks did not cover batched or grouped cyclic values, wrap-boundary gradients, compiled routed calls or gradients through `set_values`. |
| Temporary parameter overrides | <span class="compat-status compat-works">Works</span> | Tested routed calls respect Caskade's active context. Direct parameter utilities preserve the tested values and homes. |
| Parameter pointers and reparameterization | <span class="compat-status compat-caution">Needs care</span> | Dependencies need an unambiguous home. Cross-device results can retain gradients, but incompatible or invisible dependencies need an explicit plan. Recognized Python effects route dynamically; unknown effects still need care during capture. |
| Standard HDF5 save, append and load | <span class="compat-status compat-works">Works</span> | Tests passed with split GPU parameters, shared links and stored tensor attributes. Niagara converts standard CUDA values for HDF5; native GPU Caskade 1.1.5 append failed in the earlier check. |
| Custom HDF5 methods and descriptor-backed attributes | <span class="compat-status compat-caution">Needs care</span> | Custom persistence methods and descriptor-backed attributes have restrictions described below. |
| Inspectable metadata changes | <span class="compat-status compat-caution">Needs care</span> | Outside a fixed session, the high-level backend can recapture after a detected metadata-only change. Direct compiled programs reject the change; opaque internals remain unguarded. |
| Custom `Param` classes and serialization | <span class="compat-status compat-caution">Needs care</span> | Niagara temporarily changes parameter instance classes. Use `isinstance`, close placement before pickling and test custom serialization. |
| Flat parameter array packing | <span class="compat-status compat-works">Works</span> | Tested flat packing gathers values from split GPU homes to the calling module's home. |
| Slicing and reconstructing parameter arrays | <span class="compat-status compat-caution">Needs care</span> | Preserve group and batch structure when moving, slicing or packing values. |
| Dynamic hierarchical and nested `vmap` | <span class="compat-status compat-works">Works</span> | The tested examples passed on two GPUs. |
| Captured batched simulation with `ng.vmap` | <span class="compat-status compat-works">Works</span> | The split two-GPU simulator in the [example](#vmap-examples) matched sequential Caskade calls. The eager compiler was used. |
| Outer transform around an already compiled callable | <span class="compat-status compat-fails">Does not work</span> | This combination is rejected. See the [batched examples](#vmap-examples) for the supported alternative. |
| Other transform and device combinations | <span class="compat-status compat-unverified">Not tested</span> | The passing examples do not establish support for every transform and device combination. |
| Variable shapes, including `None` dimensions | <span class="compat-status compat-caution">Needs care</span> | Dynamic execution retains Caskade's shape rules. Compiled programs specialize input metadata, so new shapes or Python scalar values can need another specialization or recapture. |
| Grouped Gaussian with `vmap(grad)` on one GPU | <span class="compat-status compat-works">Works</span> | The tested grouped Gaussian matched native GPU Caskade. |
| Guide's direct `torch.vmap(torch.func.grad(...))` across GPUs | <span class="compat-status compat-fails">Does not work</span> | The split model failed with `Cannot access storage of TensorWrapper`. A direct PyTorch cross-GPU reproducer failed similarly. |
| Batched compiled gradients for the guide's combined model across GPUs | <span class="compat-status compat-works">Works</span> | In a separate five-lane check, the two parameter groups were packed into one tensor, a scalar brightness loss was captured, then transformed with `grad_and_value` and `ng.vmap`. Values and all four input gradients matched pip Caskade on CUDA. The grid and model state were held fixed in the closure. See the [grouped example](#grouped-vmap-example). |
| `NodeList`, `NodeTuple`, `NodeDict` and selections | <span class="compat-status compat-caution">Needs care</span> | Collections inside a model remain usable, but the placement root must be a `Module`. A separate collection sharing nodes with the model can prevent placement or close unless all owners are included or external ownership is removed. See the [ownership example](#shared-ownership-example). |
| Editing links while placement is active | <span class="compat-status compat-caution">Needs care</span> | Changes can prevent close. Close or disable first, then edit, validate and enable the new graph. Resolve shared-node placement by identity. |
| Dynamic pointer call caching and `@active_cache` | <span class="compat-status compat-works">Works</span> | Dynamic Caskade caching and the guide's outer `@active_cache` are supported. |
| Python effects in captured pointer calls | <span class="compat-status compat-caution">Needs care</span> | Recognized effects use dynamic routing; unrecognized effects can still run only during capture. |
| Live values in dynamic routing | <span class="compat-status compat-works">Works</span> | Dynamic routing and active contexts support the tested live values. |
| Guide's exact live-parameter example after the changes | <span class="compat-status compat-unverified">Not tested</span> | The exact example has not been rerun. Recognized printing routes the selected method dynamically; other scalar extraction may still fail capture. |

## Hierarchical Models

| Guide feature and case | Status | What we found |
| --- | --- | --- |
| Live-parameter contexts with dynamic routing | <span class="compat-status compat-works">Works</span> | The earlier dynamic check passed. |
| Guide's exact live-parameter example after the changes | <span class="compat-status compat-unverified">Not tested</span> | The exact example has not been rerun. Recognized printing routes the selected method dynamically; other scalar extraction may still fail capture. |
| Tested hierarchical batching and nested `vmap` | <span class="compat-status compat-works">Works</span> | The two-GPU dynamic hierarchy, outer vmap and varying batched parameters matched native GPU values and gradients. |
| Other nested functional derivatives | <span class="compat-status compat-unverified">Not tested</span> | The hierarchy check did not cover every nested functional derivative. |
| Published transdimensional Poisson population | <span class="compat-status compat-works">Works</span> | The published population class ran dynamically on two GPUs with one controlled Poisson draw per call. Images and x, y and packed-parameter gradients matched native GPU Caskade for draws of 3 and 7. No program was compiled; inspect `engine.dynamic_methods`. |
| Unknown effects in other population models | <span class="compat-status compat-caution">Needs care</span> | Detection of Python and NumPy effects is incomplete; use dynamic execution when effects must run on each call. |
| Independently owned helper model held in `meta` | <span class="compat-status compat-fails">Does not work</span> | `enable(root)` does not discover the helper for placement. Give it explicit placement, or make it part of a validated common graph. |
| Opaque metadata object internals | <span class="compat-status compat-caution">Needs care</span> | Inspectable metadata is guarded, but opaque object internals may not be. |

## Examples

(vmap-examples)=
### Batched calls with `ng.vmap`

Niagara has its own `ng.vmap` for a Niagara compiled program. It applies
PyTorch's vectorization to the captured graph before preparing a new batched
program. The existing simulator regions stay on their assigned devices; the
batch is not split between replicas.

This example needs two GPUs. It compares a split simulator with sequential
calls to pip-installed Caskade on `cuda:0`:

```python
import caskade as ck
import niagara as ng
import torch


class Source(ck.Module):
    def __init__(self, name):
        super().__init__(name)
        self.scale = ck.Param("scale", 0.7)

    @ck.forward
    def __call__(self, x, scale):
        return (scale * x).sin()


class Simulator(ck.Module):
    def __init__(self):
        super().__init__("sim")
        self.source = Source("source")

    @ck.forward
    def __call__(self, x):
        return self.source(x) + x.square()


batch = torch.tensor([[0.2, 0.4], [0.5, 0.7], [0.8, 1.0]], device="cuda:0")
reference = Simulator().to(device="cuda:0")
expected = torch.stack([reference(row) for row in batch])

sim = Simulator()
placement = ng.assign_devices(sim, ng.DevicePlan("cuda:0", {"source": "cuda:1"}))
base = ng.compile_model(sim, batch[0], backend="eager")
batched = ng.vmap(base, batch)
actual = batched(batch)

torch.testing.assert_close(actual, expected)
print(actual.device, actual.shape)
# cuda:0 torch.Size([3, 2])
```

This uses the eager compiler backend. It checks the result and its device; it
does not establish CUDA-graph replay or batch sharding.

An outer PyTorch transform on the already prepared `base` does not work:

```python
try:
    torch.vmap(base)(batch)
except ng.GuardError as error:
    print(f"{type(error).__name__}: {error}")
else:
    raise AssertionError("Expected a GuardError")
# GuardError: Apply transforms before compilation with .transform() or ng.vmap()
```

Use `ng.vmap(base, batch)(batch)` as above. PyTorch's `torch.vmap` can
also work with dynamically routed models in the cases listed in the tables;
the restriction above concerns a prepared Niagara callable. `ng.vmap`
is not a general replacement for every PyTorch functional transform.

The same API can batch a captured gradient for this simulator. Each row is
compared with an ordinary autograd call on the single-GPU Caskade reference:

```python
loss = base.transform(lambda f: lambda x: f(x).sum(), batch[0])
derivative = loss.transform(torch.func.grad, batch[0])
gradients = ng.vmap(derivative, batch)(batch)

expected_gradients = []
for row in batch:
    row = row.detach().clone().requires_grad_()
    expected_gradients.append(torch.autograd.grad(reference(row).sum(), row)[0])

torch.testing.assert_close(gradients, torch.stack(expected_gradients))
print(gradients)
# tensor([[1.0932, 1.4727],
#         [1.6576, 2.0176],
#         [2.1931, 2.5354]], device='cuda:0')
```

For the guide's grouped Gaussian model, another tested approach packed both
parameter groups into one tensor, sliced them inside a captured scalar
brightness loss, then used `transform(torch.func.grad_and_value, ...)`
followed by `ng.vmap`. On five inputs, its values and all four input
gradients matched native GPU Caskade. The test held its grid and model state
fixed in the closure; arbitrary closure state is not fully guarded. This does
not make the guide's direct dynamic cross-GPU `torch.vmap(grad)` call work.

```python
placement.close()
```

(grouped-vmap-example)=
The following shows that grouped-model alternative using the guide's existing
`model`, `x` and `y`, with its two parameter groups already configured and
on `cuda:0`. The model's first Gaussian is named `first`:

```python
group0, group1 = model.get_values()
group0_size = group0.shape[-1]
parameter_batch = torch.cat((group0, group1), dim=-1).detach().contiguous()

placement = ng.assign_devices(model, ng.DevicePlan("cuda:0", {"first": "cuda:1"}))


def brightness_sum(parameters):
    groups = (parameters[:group0_size], parameters[group0_size:])
    return model.brightness(x, y, groups).sum()


example = parameter_batch[0]
program = ng.compile_model(brightness_sum, example, backend="eager")
value_and_grad = program.transform(torch.func.grad_and_value, example, backend="eager")
gradients, values = ng.vmap(value_and_grad, parameter_batch, backend="eager")(parameter_batch)
print(gradients.shape, values.shape, gradients.device)
# torch.Size([5, 4]) torch.Size([5]) cuda:0
placement.close()
```

Keep the captured `model`, `x` and `y` fixed while using this program.
Because they are held in a closure, Niagara cannot fully guard changes to
their state. Capture a new program after changing them.

(shared-ownership-example)=
### Shared ownership during placement

A model outside the placement can share a child with the enabled model. In
this two-GPU example, the extra owner makes `engine.close()` reject the
placement. Detaching that owner allows the same engine to close:

```python
import caskade as ck
import niagara as ng


class Child(ck.Module):
    def __init__(self, name):
        super().__init__(name)
        self.scale = ck.Param("scale", 0.7)


class Simulator(ck.Module):
    def __init__(self):
        super().__init__("sim")
        self.child = Child("child")


model = Simulator()
engine = ng.enable(model, default="cuda:0", modules={"child": "cuda:1"},
                   compile=False)
outside = ck.Module("outside")
outside.shared = model.child

try:
    engine.close()
except ng.PlacementError as error:
    print(error)
else:
    raise AssertionError("Expected a PlacementError")
# child is shared with a model outside this plan; include the common root

outside.unlink("shared")
engine.close()
assert engine.closed
```

Alternatively, arrange the owners under a validated common root before
enabling placement. This example shows one rejected ownership change, not
every possible graph edit.

## Working with a split model

Use dynamic execution for models driven by Python control flow. If `child` is
a child module of the model, for example:

```python
import niagara as ng

engine = ng.enable(sim, default="cuda:0", modules={"child": "cuda:1"},
                   compile=False)
result = sim(inputs)
# Close before changing the graph or switching to a CPU persistence workflow.
engine.close()
```

The result normally stays on the root's device. Copy it to the host when you
need host I/O. Inside a routed call, a direct read of
`self.child.parameter.value` is copied to the caller's device if needed; the
stored parameter remains on its declared home. The tested scalar, vector and
pointer cases retained their gradients in dynamic and compiled-eager execution.

For split homes, `get_values("array")` gathers tensor values to the calling
module's home. The tested grouped and batched layouts match native Caskade.
List and dict values retain their parameter homes when accessed outside a
routed call. During `to_valid` and `from_valid`, bounds follow the input
tensor's device and are restored afterward. Flat, list and dict `set_values`
round-trips were checked under `ValidContext`; routed forward inputs and
`fill_params` have their own device handling.

For standard Caskade persistence methods, Niagara saves and appends CUDA
parameter values and directly stored CUDA tensor `saveattrs` using host copies.
The tests compare the resulting values, groups, batches and shared HDF5 links
with CPU Caskade. GPU tensor attributes are restored to their original devices
on load. CPU attribute loading retains Caskade's native behavior.

If directly stored CUDA tensor attributes coexist with descriptor-backed
`saveattrs`, custom attribute lookup or an overridden internal HDF5 writer,
Niagara rejects the save before writing; those callbacks could change snapshot
ordering. A CUDA attribute load through a custom setter or lookup is also
rejected. Descriptor-returned GPU tensors still use Caskade's behavior and may
be omitted with a warning. Public persistence overrides are left untouched and
must handle GPU serialization themselves. For these cases, close placement,
co-locate the needed values on CPU and verify the saved file.

On a standard append error, Niagara attempts to restore all recorded dataset
lengths and append markers. This is best effort: process or storage failures
and arbitrary custom writers can leave changes, so it is not an atomic
transaction. Pointer evaluation errors propagate instead of Caskade's silent
skip. Check the file and restore a known-good copy when needed.

An external selection that shares parameters is also an owner. Include the
owners under a common `Module` and resolve shared placement, or detach the
external selection first. `NodeList.clear()` worked for the tested list. Do
this before enabling or closing. If links have already changed and close
rejects the plan, restoring the original links may allow closure; it's better
to wait until after close before editing them.

## Compilation and Python behavior

Compiled entry points accept supported tensor pytrees and scalar metadata.
The tested NumPy array input was rejected, so convert arrays to tensors at
the boundary. If the model exposes a named method instead of `__call__`,
select it for compilation, for example with `methods="brightness"`.

The backend scans selected methods for common Python and NumPy random draws,
printing, I/O and some Python state mutations. When it recognizes an effect,
ordinary calls use dynamic routed execution instead of replay. Inspect
`engine.dynamic_methods` to see the selected methods and reasons. Explicit
`engine.prepare(...)` and direct `ng.compile_model(...)` reject recognized
effects with `CaptureError`; a fixed compiled session cannot run a dynamically
routed method.

The scan is incomplete. Unknown Python effects can still run at capture and be
reused or skipped on replay. Use `compile=False` when a method must execute
such effects on every call. In a two-GPU check of the published `BlobPopulation`
class, controlled draws of 3 and 7 produced populations of 3 and 7 on successive
calls, with one NumPy draw per call and identical Torch seeds in the native
comparison. Images and all x, y and packed-parameter gradients matched exactly.
No program was compiled; `engine.dynamic_methods["brightness"]` reported
`Python state mutation` in that check.

PyTorch emitted an `AccumulateGrad` stream-mismatch warning during the check.
The warning points to possible extra synchronization or CUDA graph capture
problems. This check used dynamic execution; it does not qualify compiled replay.

The state guard checks inspectable Caskade metadata, including supported
containers and tensors. Outside a fixed session, a detected metadata-only change
can make the high-level backend clear and recapture its programs. A direct
compiled program rejects the change, and a fixed session rejects mutation.
Other model changes still require an explicit clear. Guards cannot inspect
every opaque object, closure or global.

Recognized seeded PyTorch tensor random operations stay in the tensor program
instead of being constant-folded. A CPU TorchScript probe produced fresh
draws, but sequence equivalence across devices or backends was not established.
PyTorch's vmap randomness rules still apply. A Python or NumPy draw doesn't
automatically become a separate tensor draw for each lane.

Niagara routes a custom decorator outside `@ck.forward` when it preserves the
wrapped function with `functools.wraps`. The tested `@ck.active_cache` ordering
also works with such a decorator. Decorators that hide the wrapped function may
prevent Niagara from finding the boundary. Put `@ck.forward` outside those
decorators and check your model's behavior. An outer `torch.vmap` or
`torch.func.grad`
around an already prepared compiled method is rejected; apply supported
transforms before compilation. Cross-GPU functional transforms can still fail
in PyTorch itself. Ordinary `torch.autograd.grad` and backward worked for the
tested split numerical programs. Compare every relevant gradient for your model.

## What was checked

The earlier guide probes used pip-installed Caskade 1.1.5, Niagara 0.7.0 and
PyTorch 2.12.0+cu126 on two physical RTX 3090 GPUs. Their native comparisons ran
on CUDA, with CPU controls where relevant. They include the published hierarchy
and grouped-gradient classes. The compact
{download}`guide_compatibility.json <../_static/guide_compatibility.json>`
records that earlier audit and its source hashes; it predates the fixes above.

New regression tests cover capture, parameters and persistence against Caskade.
The device matrix tests two GPUs, CPU with either GPU, and CPU with both GPUs.
It compares values and gradients with native CUDA Caskade. Its mixed-device
compiled tests use the eager compiler backend; they do not establish CUDA-graph
replay support. The separate published
`BlobPopulation` comparison records both GPU identities, seeds, gradients and
source hashes in `results/0.7.0/compatibility_fixes/hierarchy/poisson_population.json`.
The unchanged Caskade 1.1.5 suite was also run through the test adapter in
three layouts: two GPUs, CPU plus one GPU, and CPU plus both GPUs. Each run
passed 952 tests and skipped two backend-specific cases. The adapter returns
outputs to the host for the original assertions, and not every case uses every
selected device. The two-GPU run qualified 15 CUDA-graph programs, including
12 gradient programs. The mixed-layout full suites used dynamic routing.

Across Niagara's source tests and the three upstream runs, combined statement
and branch coverage was 92.71%: 94.58% of statements and 88.60% of branches.
The {download}`compatibility test record <../_static/compatibility_fixes.json>`
contains the test counts, device connectivity and source hashes. A passing
example does not establish support for untested combinations.
