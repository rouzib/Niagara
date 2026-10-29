# Troubleshooting

## Device or placement errors

Use `cuda:0`, not an implicit `cuda`. Check physical device visibility with
`torch.cuda.device_count()`. A plan is rejected before migration when ownership
is ambiguous, a pointer requires incompatible dependency homes, or a node is
owned outside the supplied root. Include the common graph root and resolve
shared nodes by identity. Keep validation enabled so unsafe plans are rejected.

## A simulator configuration changed

Input metadata or execution-mode changes need a matching prepared program.
The high-level backend can prepare supported new signatures outside a session;
the lower-level `CompiledModel` requires explicit recapture. A detected
metadata-only change can trigger recapture through the high-level backend
outside a fixed session. Other model changes require an explicit cache clear.
Close or disable placement before editing the DAG or changing the device plan,
then revalidate and enable the new plan. Editing links while enabled can make
close fail. See the [Caskade compatibility guide](caskade_compatibility.md).

## Capture fails on Python control flow

Fixed capture cannot represent every data-dependent Python branch. Recognized
Python or NumPy randomness and other known effects route dynamically; inspect
`engine.dynamic_methods` for the methods and reasons. Explicit preparation of
those methods raises `CaptureError`, and fixed sessions cannot run them. The
scan does not find every side effect. Use `ng.enable(..., compile=False)` when
an unknown effect must run each time, or keep a pure tensor implementation.
Report capture failures as failures when comparing compiled execution.

## Gradients or batching fail

Apply `.transform()` and `ng.vmap` to a captured graph before final compilation.
Ordinary outer vmap of a prepared callable is rejected. Do not wrap enabled
dispatch inside another arbitrary FX/compiler capture. Default scalar reverse
AD has a specific lowering; other cross-GPU transform combinations rely on
native Torch support and can remain unsupported.

Check every parameter gradient against native autograd. Look at absolute error
and reference magnitude before interpreting a large pointwise relative error.

## No Triton or a Windows compiler error

The tested fusion extra pairs Torch 2.12 with Triton 3.7, using
`triton-windows` on Windows. Confirm both distributions and CUDA-enabled Torch.
CUDA replay can use TorchScript kernels without Triton; CPU auto uses that
fallback as well. An unsupported explicit compiler selection raises an error.
Inspect the prepared program to see the chosen backend.

## A native simulator fails after closing placement

The low-level `Placement.close()` restores classes without co-locating tensors.
Use a uniform plan for native co-location, or the high-level `Backend.close()`/
`ng.disable(sim)` contract to restore native execution at the default home.
Inputs held outside the model still need the matching device.

## Two GPUs are slower or use more memory

Inspect transfers and retained graph buffers before changing the model. Small
sequential regions may have no useful overlap. Host-staged copies on cards
without peer access can dominate. Compare against one compiled GPU and keep
preparation, latency, throughput and memory definitions separate.

## Graph rendering fails

Install the optional Python packages and make Graphviz's `dot` visible on
`PATH`. Graphviz supplies diagrams; tensor execution works independently of it.
The HTML documentation's built-in diagrams use bundled browser assets and do
not need `dot` at viewing time.
