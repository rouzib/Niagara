# Runtime and memory measurements

Niagara's performance depends on the scene, precision, partition, batch size,
compiler, driver and interconnect. Measure the configuration you will use.

## Compare the same work

Use unmodified published Caskade/Caustics on GPU as the scientific reference.
Compare native GPU execution with a directly Torch-compiled GPU control.
An eager CPU reference alone cannot isolate Niagara's benefit. Match dtype,
parameter values, image extent, PSF mode, scalar loss and derivative requirements.

Separate these costs:

1. Import, capture, compilation and preparation.
2. Synchronized end-to-end latency for a single call.
3. Repeated-call throughput with a defined batch of calls.
4. Ordinary forward plus loss plus backward.
5. Model/graph resident memory, execution peaks and preparation peaks.

Interleave timed methods after warmup to reduce order bias. Save the raw
samples and describe the synchronization boundary. Run the benchmark without
other GPU work started by the experiment.

## Memory definitions

The resource benchmark runs each method/size in a fresh subprocess. After
setup, it clears unused allocator blocks and resets peak counters. Sequential
calls keep one output, so the output history does not grow.

PyTorch's allocated tensor bytes differ from allocator-reserved pool bytes.
CUDA context, driver and non-PyTorch allocations are outside those counters.
For split work, a sum of per-device maxima is a conservative bound; the maxima
need not occur simultaneously. Report the per-device values as well as that sum.
See the official [CUDA memory definitions](https://docs.pytorch.org/docs/2.12/notes/cuda.html#memory-management).

## Retained 0.6 findings

The historical 0.6 matrix used two RTX 3090 cards without peer access, float64
LensSource images and the directly Inductor-compiled Caskade control. One-GPU
guarded latency was close to that compiled control. Splitting the source was
slower for this scene and retained more forward memory. At the largest
backward case it lowered the largest per-device tensor peak at a runtime cost.

These findings apply to the recorded release and machine. Current 0.7
validation is documented separately. The historical reports directory contains
the full curves, raw data and definitions:

```{eval-rst}
:download:`Historical evidence index <../../reports/README.md>`
```

```{figure} ../_static/results/0.6.0/runtime_forward_checked.svg
:alt: Historical Niagara 0.6 forward runtime against eager and Inductor-compiled GPU Caskade

**Historical Niagara 0.6.0:** normal guarded calls on the classic float64
Caustics scene, compared with public Caskade 1.1.5 on GPU. The single-call and
repeated-call panels have different measurement boundaries. Bands show the
interquartile range, not confidence intervals. The two-GPU source split is
slower for this scene; these curves are not new 0.7 measurements.
```

```{figure} ../_static/results/0.6.0/memory_forward.svg
:alt: Historical Niagara 0.6 forward tensor allocation and allocator reservation peaks

**Historical Niagara 0.6.0 forward rendering:** prepared replay retains more
memory than the directly Inductor-compiled Caskade control, and the split
retains still more. Tensor allocations and allocator reservations are shown
separately. Each point used an isolated worker; two-GPU totals sum per-device
maxima and are not simultaneous observed total peaks. Driver/context memory is
excluded. No forward-memory saving is claimed here.
```

```{figure} ../_static/results/0.6.0/memory_forward_backward.svg
:alt: Historical Niagara 0.6 forward and backward GPU allocator peaks

**Historical Niagara 0.6.0 ordinary forward/loss/backward:** at 1024 × 1024,
the split lowered the largest per-GPU tensor peak to about 170 MiB, compared
with 283 MiB for compiled one-GPU Caskade. Its summed tensor peaks were about
259 MiB, while runtime increased. This is a different workload from no-tape
forward replay. It establishes a version- and hardware-specific memory/runtime
tradeoff, not a new 0.7 performance or physical-board-memory claim.
```

## Reproduce the resource workflow

```console
python benchmarks/runtime_memory.py --sizes 60 128 256 512 1024 --tasks forward --out results/forward
python benchmarks/plot_runtime_memory.py results/forward
python benchmarks/runtime_memory.py --sizes 60 256 1024 --tasks forward_backward --out results/backward
python benchmarks/plot_runtime_memory.py results/backward --tasks forward_backward
```

The benchmark checks versions to identify the reference stack. Use fresh
output directories and save their environment/source hashes.

## Choose a strategy

Start with one fused GPU for a small sequential lensing scene. Split regions
when the memory distribution or independent work warrants it. Use replicas for
independent batches, and coarse tasks when the caller knows task dependencies.
Choose placement from measurements; replication and splitting do not search
for an optimal layout. Prepared graph buffers and protective output copies can
increase memory even when launches become cheaper.
