# Device-region replay

CUDA replay runs a fixed tensor program using persistent input and output
buffers. Local Inductor fusion reduces elementwise launches before capture.
The TorchScript fallback uses the same fixed-buffer interface.

## One device

Preparation clones the example inputs, compiles and warms the local program
on a preparation stream, then captures a CUDA graph. Each call copies dynamic
inputs into fixed storage, replays the graph and returns copies of its outputs.
Later calls leave previously returned results unchanged.

Calls on the usual caller stream avoid redundant waits. If successive calls
use different streams, an event orders reuse of the fixed buffers. Input
storage records the relevant stream so the allocator preserves its lifetime.
The buffers remain single-caller even with this stream ordering.

## Multiple devices

```{mermaid}
flowchart LR
    P["Packed dynamic inputs"] --> A["GPU 0: lens and lens light"]
    A --> X["Pack compatible boundary tensors"]
    X --> B["GPU 1: source"]
    B --> C["GPU 0: image and PSF"]
    C --> O["Independent image"]
```

The split runtime groups FX operations by device and cross-device dependency
depth. Each region has a local replay program and holds the static buffers it
consumes, avoiding repeated copies of those buffers as dynamic inputs.

Cross-device copies run outside local CUDA capture. This supports the tested
GPU pair without peer access, where transfers can pass through host memory.
Compatible tensors from one source/dtype group are packed into fewer transfers
and unpacked into views on the destination.

Dependencies determine execution order. A sequential lens/source model stays
sequential after splitting. Extra graph pools, boundary buffers and output
copies can also increase resident memory. Measure the tradeoff with per-device
counters and a real trace.

## Gradients and fixed buffers

No-tape replay rejects ordinary requires-grad inputs and trainable captured
constants. To replay a captured functional derivative, make the gradient an
explicit output of the numerical program. That program can then replay
without building an autograd tape on each iteration.

The pure captured graph keeps differentiable transfers for ordinary autograd
paths. The segmented replay runner executes explicit tensor programs, with no
ordinary-autograd wrapper around its mutable replay buffers.
