# Your first simulator

This small model uses ordinary Caskade classes and decorators. Both parameters
are dynamic; the sampling grid is fixed model state.

```python
import caskade as ck
import niagara as ng
import torch


class Signal(ck.Module):
    def __init__(self):
        super().__init__("signal")
        self.amplitude = ck.Param("amplitude", None)
        self.phase = ck.Param("phase", None)
        self.grid = torch.linspace(-1, 1, 16, dtype=torch.float64)

    @ck.forward
    def __call__(self, amplitude, phase):
        return amplitude * torch.sin(self.grid + phase)


sim = Signal()
values = {"amplitude": 1.2, "phase": 0.3}
params = torch.tensor([values[p.name] for p in sim.dynamic_params], dtype=torch.float64)
reference = sim(params)
```

Read the packed order from the actual parameter objects. Constructor order
can differ from the model's parameter order.

## Enable prepared execution

```python
with ng.enable(sim, default="cpu", example=params) as engine:
    actual = sim(params)
    torch.testing.assert_close(actual, reference)

    q = params.detach().requires_grad_()
    gradient = torch.autograd.grad(sim(q).square().sum(), q)[0]
    assert torch.isfinite(gradient).all()

    with engine.session():
        actual = sim(params)
```

Keep using `sim(params)`. Supplying the example prepares that call immediately;
new supported metadata triggers preparation on the first matching call. The
requires-grad call has its own specialization. Leaving the context restores
native Caskade dispatch and brings all simulator state back to CPU.

## Move to CUDA

For this single region, set `default="cuda:0"` and put the packed inputs on
the same device. Keep dtype and shape stable across repeated calls. Measure
capture and compilation separately from production so their setup cost stays
visible.

A Caustics example is available directly from the source tree:

```console
python examples/opt_in_backend.py --device cuda:0 --source-device cuda:1 --size 60
```

The example uses a stock `LensSource` and checks its image and ordinary
gradients. Leaving the backend context restores native calls. Read
[device placement](../user_guide/placement.ipynb) before selecting a split.
