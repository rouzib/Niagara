# How Niagara works

Niagara uses Caskade's graph ownership and parameter machinery to capture a
numerical program. Repeated calls then run that program without repeating the
stateful Python traversal. This section explains how the public APIs reach
each part of the runtime.

```{toctree}
:maxdepth: 2

architecture
capture_and_guards
device_replay
derivative_lowering
```
