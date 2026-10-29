# Upstream Caskade tests

This historical check ran the complete Caskade 1.1.5 suite from its PyPI source
archive against the installed public package and the Niagara source recorded in
the linked validation record. It predates relocation of the example Caustics
and sampling helpers out of the Niagara runtime package. It collected 954 cases,
including four notebook tests; all 18 upstream inputs stayed unchanged. The
recorded results are not a qualification of later source hashes.
The adapter does not establish compatibility for every operation on a
persistently enabled model. See [Caskade guide compatibility](../user_guide/caskade_compatibility.md)
for direct API checks and observed limitations.

## Run the check

Use Niagara's development environment with two visible CUDA GPUs:

```console
python scripts/validate_upstream_caskade.py --mode both --coverage
```

The runner saves logs, JUnit results, source hashes, branch coverage and per-test
routing records under `results/0.7.0/upstream_complete`. Use a fresh `--output`
directory to retain an earlier run. Select `--mode native` or `--mode niagara`
to run one path, or `--default-device cuda:1` to reverse the GPU assignment.
Compiled qualification defaults to `--compile-checks eligible`; use `off` for a
routing-only run. `--compile-backend auto` records the backend Niagara selects.

The native control leaves Caskade's decorators and original tensor devices
unchanged. The Niagara adapter uses dynamic routing for the original calls,
then adds compiled checks for selected pure numerical entry points. Parameter
mutation, Python control flow and the original assertions remain part of the
upstream tests.

## Device placement

For an ordinary inactive call, the adapter assigns module regions to alternating
GPUs and gives parameters one explicit default home. Modules used as pointer
dependencies share that home. When several top-level modules share nodes, a
temporary common owner includes them in one placement plan; its links are
removed after the call.

Niagara receives the original arguments and runs the placed methods. The adapter
checks tensor output devices, then returns results and model state to the host
for the upstream assertions. These setup and host transfers are part of the test
adapter. This check does not measure runtime overhead or speedup.

A **split call** must observe methods on both physical GPUs and record a positive
transfer between them during that invocation. Passing tests with successful
split calls are counted separately from failed tests, expected error paths and
calls that use one GPU. Structural tests do not count toward GPU execution.

### Explicit active contexts

The adapter applies Niagara placement before entering an external
`ActiveContext`. It keeps parameter homes on CPU in these test scopes so that
upstream code can inspect and compare parameter values directly. Numerical
module methods still run on their assigned GPUs, with normal Niagara transfers
at each boundary. External results return to the host, and context exit restores
the original classes and parameter state.

This CPU parameter policy belongs to the test adapter. Normal Niagara usage
keeps the parameter homes declared in the device plan. Enable Niagara before
opening an active context; placement changes during activation remain invalid.
Manual `fill_params` calls on an enabled simulator route values to their declared
homes. Active calls use Niagara's dynamic routing because compiled state guards
require an inactive simulator.

### Helpers outside the graph

A helper stored only in metadata is outside Caskade's owned DAG. For such a call,
the adapter explicitly enables Niagara on the helper's independent inactive
graph, using the caller's GPU. Its result stays on that GPU for the surrounding
calculation. The helper's classes and host state are restored after the call.

This separate scope is recorded as `nested_owned_scope`; it is not native
execution. The package does not infer ownership from arbitrary metadata. In
application code, declare the helper in the owned graph or enable its independent
graph explicitly. Shared or active ownership must still pass placement checks.

## Tensor creation inside simulators

`DevicePlan.factory_device=True` is the default. Factories without a specified
device, such as `torch.tensor`, `torch.ones` and `torch.rand`, create tensors on
the current numerical region's device. Pointer calculations use their
dependencies' device. Explicit device arguments and output buffers retain their
requested placement.

The policy is local to the current call and thread; it does not change the
process-wide default device. Prepared compiled calls execute the captured tensor
program without re-entering the factory policy on each replay. Set
`factory_device=False` when the simulator already controls every factory device.

## Compiled qualification

The compiled checks cover selected pure core families: forward calls,
integration models, input packing, hierarchy, grouped parameters and valid-range
contexts. This is a defined qualification set, not a claim that every possible
simulator can be captured. Each skipped check records its reason. Random work,
active caches, attribute mutation, unavailable user source and notebook workflows
with plotting or other Python effects remain dynamic.

For each eligible test and entry point, the adapter captures a bound model method
with its state guards. It checks the compiled result at the original input and
at a changed input, then checks the input-metadata guard. Floating-point inputs
also receive a compiled scalar-loss gradient check against dynamic autograd at
both inputs, including every floating input component. Integer-only programs
receive changed-input value checks; their records state that ordinary autograd
derivatives do not apply. Later calls in the same test remain dynamic so that
changes to model state retain their original semantics.

A qualification failure fails the test. It does not silently select native
execution. The record lists the actual forward and gradient backends, captured
devices, value and gradient checks, and guard rejections. Tensor-work records
include input-dependent arithmetic by device and per-region descriptions.
View-only or constant-only regions are identified separately. These are FX
operator counts, not CUDA kernel counts or evidence of overlap; an empty local
CUDA graph warning is not hidden.

## Results and coverage

The recorded pre-relocation Niagara source suite passed 117 tests. With CUDA
hidden, 62 passed and 55 hardware cases skipped. Its branch-enabled source
coverage was 93.41% overall:
95.04% of statements and 89.21% of branches. The enforced overall coverage floor
is 90%; it does not require each file or each branch percentage to reach 90%.

The complete run collected the same 954 upstream cases in both modes:

| Run | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Published Caskade control | 952 | 0 | 2 |
| Niagara with compiled qualifications | 952 | 0 | 2 |
| Niagara source tests | 117 | 0 | 0 |
| Independently installed wheel | 117 | 0 | 0 |
| Niagara with CUDA hidden | 62 | 0 | 55 |

All four upstream notebooks pass without source or iteration changes. There are
zero native forward fallbacks, including the explicit active-context calls.
Thirty-six passing tests execute Niagara calls; 29 execute successful splits
across both GPUs. The run records 10,886 successful split calls and 18 closed
external contexts, with every class and inactive state restored.

Fifteen selected numerical programs passed CUDA graph compilation. All contain
input-dependent tensor computation; 14 perform arithmetic on both GPUs.
Thirty value checks compare original and changed inputs. Twelve floating-input
programs pass 24 gradient checks covering 242 components, and three integer
programs pass forward checks without claiming derivatives. All 15 changed-shape
calls are rejected by their input guards. Empty view-only local regions remain
identified in the per-region records; none of the complete qualified programs
is constant-only.

The completed pre-relocation upstream run, compiled qualifications and exact
source hashes are recorded below. The selected Caskade backend is Torch; its JAX and NumPy-specific
cases report their upstream skips. Initial `conftest.py` decorator aliases are
adapted so fixture-defined methods receive the same checks as directly defined
simulators.

```{eval-rst}
:download:`Upstream-suite validation record <../_static/upstream_caskade.json>`
```

The source archive is
[`caskade-1.1.5.tar.gz`](https://files.pythonhosted.org/packages/06/f4/c1912d689a7f43daf2f61d14831505bef63fb043d6a0d5df0c8e06b9cb46/caskade-1.1.5.tar.gz).
Its SHA256 is
`cf901badfca33a27a13c219e3a88640867b6ca9a1abfda7f428ad32986631d78`.
The runner checks this digest against PyPI's release metadata and verifies all
14 installed Caskade Python files against the published wheel manifest. That
wheel's SHA256 is
`89c54775efd1be83ba5f699583a7476e7c8a96c0f50e62a10b5cccb485f413be`.

Each run records the Niagara runtime, runner and adapter hashes, import paths,
versions, GPU names and physical UUIDs. It checks that all upstream inputs and
installed Caskade files remain unchanged. Per-test records distinguish routed
calls, compiled checks, explicit contexts, independent helpers and structural
checks.

The initial {ref}`0.7.0 qualification <local-qualification>` predates these
compatibility checks. Its MALA chains, saved guide outputs and performance
records retain their original source hashes. The full scientific MALA runs were
not repeated for these fixes.
