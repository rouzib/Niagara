# Supported scope and limits

Niagara executes Caskade simulators with PyTorch in one process. Its supported
scope covers the mechanisms below; other array backends and arbitrary Python
distribution are outside that scope.

## Supported mechanisms

- Explicit CPU/CUDA region placement with pre-execution graph validation.
- Caskade parameters, contexts and safe declared cross-device ties.
- Fixed tensor-program capture, guarded specializations and prepared sessions.
- Local/segmented CUDA replay, ordinary AD compiler paths and the documented
  functional derivative/batching APIs.
- Independent replicas, explicit coarse tasks, graph views and measured profiles.

## Restrictions that remain

- Prepared programs and Caskade's mutable context are single-caller. Separate
  models/programs are needed for concurrent stateful callers.
- Shapes, layouts, devices, dtypes, modes and static configuration specialize
  prepared work. Use explicit recapture after intentional changes.
- Arbitrary data-dependent Python, opaque mutable objects and invisible pointer
  dependencies are not exhaustively discoverable. Use native routed execution
  when fixed capture is inappropriate.
- No-tape CUDA replay does not support ordinary requires-grad inputs or
  trainable captured constants. Prepare explicit derivatives or use ordinary AD.
- Compose pure graphs before compilation through the documented APIs. General
  outer compiler/FX/vmap nesting is unsupported; detected invalid combinations
  raise errors, and other combinations have not been validated.
- The custom reverse-AD and convolution paths cover their documented cases.
  Other cross-GPU transforms depend on Torch and can remain unsupported.
- There is no automatic optimal placement, process-per-device/RPC runtime,
  multi-host qualification or DDP adapter in this release.
- A split can be slower and can retain more memory than a fused one-GPU model.
  Driver/context allocations are not included in PyTorch allocator counters.

For guide-by-guide examples and workarounds, including silent capture and
GPU persistence pitfalls, see [Caskade guide compatibility](../user_guide/caskade_compatibility.md).

Use the API contracts to determine supported behavior, and the validation
records to see which software, models and hardware have been tested.
