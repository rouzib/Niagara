# Notebooks

The [user-guide chapters](../user_guide/index.md) also contain executed cells
and simulator figures. These examples use small numerical targets and check
native agreement before prepared execution. The site displays saved outputs;
Sphinx does not start CUDA workloads while rendering documentation.

```{toctree}
:maxdepth: 1

01_cpu_first_simulator
02_two_gpu_caustics
```

Execute notebooks explicitly with the project's interpreter. The two-GPU
notebook checks hardware and reports a skip when it is unavailable. A skipped
run leaves physical-GPU execution unvalidated.

The [validation summary](../development/validation.md) identifies which notebook
and test executions completed for this release. For larger scientific targets,
use the Python examples and save their data and diagnostics. Small interactive
examples alone cannot establish convergence.
