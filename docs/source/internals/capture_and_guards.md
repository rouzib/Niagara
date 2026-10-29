# Capture and guards

`make_fx` traces real tensor operations as the simulator runs through its
normal parameter and context machinery. Capture can record scope metadata.
It also checks grad, inference and autocast consistency, and checks the
versions of cloned inputs to detect mutation during capture.

## Constant folding

Preparation can evaluate pure subgraphs that are independent of dynamic inputs
and store their results as buffers. Static coordinates, for example, can be
built once. Values that an in-place tensor operation may mutate are not folded,
including values reached through views; otherwise replay could change a stored
buffer across calls. Other safe subgraphs may still be folded. Recognized
seeded PyTorch tensor random operators stay in the
program instead of being constant-folded. The high-level backend routes methods
with recognized Python or NumPy randomness dynamically. Detection is incomplete:
an unrecognized Python draw may still be frozen on replay. See the
[compatibility guide](../user_guide/caskade_compatibility.md). Random sequence
equivalence across devices and backends has not been established.

Stored constants can increase resident memory. Measure both runtime and memory
when evaluating folding and capture. Vary dynamic inputs in reference checks
to catch any input-dependent result that was accidentally stored as a constant.

## Device copies

Eager routing returns a module's result to its caller's home device so the
parent can combine it with its own tensors. A parent that passes the result
straight to another region can therefore create a detour such as CPU to
`cuda:0` to `cuda:1`.

Captured programs bypass consecutive copies when the intermediate preserves
dtype, shape and strides, does not request pinned memory, and no operation
between the copies may mutate the source. Copies that perform required dtype
or layout conversions remain. An intermediate copy also remains when another
computation uses it. The final copy still uses PyTorch's ordinary copy operation
and autograd behavior. The dependency graph shows the resulting route; use a
measured trace or synchronized timing to assess its performance.

## Input and execution-mode guards

A prepared input signature records the pytree structure and supported leaf
metadata: tensor shape, stride, dtype, device, requires-grad flag and layout.
Scalar flags are fixed metadata too. Each specialization also records the
grad-enabled, inference and autocast states.

A direct `CompiledModel` rejects changed metadata. Outside a fixed session, the
high-level backend can prepare another supported signature if the cache has
room and the model-state checks still pass.

## Static model guards

`StateGuard` builds its field checks once. Repeated calls check node, class and
field identity, plus the contents of supported mutable lists and dictionaries.
Shared tensors are checked once using their available version and metadata.
Immutable tuples are matched by identity: replacing a tuple invalidates the
guard even when its values compare equal.

Generated expressions contain only quoted field names and numeric indices.
Model values are stored separately, never interpolated into code. The checks
guard inspectable `node.meta` fields, supported containers and tensors. They
cannot inspect every opaque Python object's internals or arbitrary closure and
global state. Outside a fixed session, the high-level backend can clear and
recapture after a detected metadata-only change. A direct compiled program or
fixed session rejects that change. Other model changes still need an explicit
clear. Caskade's transient parameter values and memoization fields are treated
as runtime context rather than fixed configuration.

## Sessions and concurrency

A fixed session checks model state on entry and exit, while checking inputs
and modes on every call. The caller must keep the simulator unchanged inside
the block. Sessions provide no locking; concurrent host callers need separate
compiled objects and models because fixed CUDA buffers are single-caller.

Use `compiled.graph` only for controlled tensor composition: direct graph calls
bypass the wrapper guards. Keep external mutable state fixed when capturing a
Python closure, since capture cannot discover all of it.
