import caskade as ck
import pytest
import torch

import niagara as ng


class Leaf(ck.Module):
    def __init__(self, name):
        super().__init__(name)
        self.a = ck.Param("a", None)
        self.bias = ck.Param("bias", 0.1)

    @ck.forward
    def __call__(self, x, a, bias):
        return (x * a + bias).sin()


class Sim(ck.Module):
    def __init__(self):
        super().__init__("sim")
        self.left, self.right = Leaf("left"), Leaf("right")
        self._grid = torch.linspace(0.1, 1.0, 16, dtype=torch.float64)
        self.offset = 0.2

    @ck.forward
    def __call__(self):
        return self.right(self.left(self._grid)) + self.offset


GPU = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")


@pytest.mark.parametrize("home", ["cpu", pytest.param("cuda:0", marks=GPU)])
def test_direct_matches_caskade_and_restores(home):
    sim = Sim().to(dtype=torch.float64)
    p = torch.tensor([0.5, 0.7], dtype=torch.float64, requires_grad=True)
    reference = sim(p)
    grad = torch.autograd.grad(reference.sum(), p)[0]
    cls = type(sim)
    with ng.assign_devices(sim, ng.DevicePlan(home)):
        q = p.detach().to(home).requires_grad_()
        result = sim(q)
        torch.testing.assert_close(result.cpu(), reference)
        torch.testing.assert_close(torch.autograd.grad(result.sum(), q)[0].cpu(), grad)
        assert result.device == torch.device(home)
    assert type(sim) is cls
    assert not sim.active


@GPU
@pytest.mark.gpu
def test_split_all_gradients_and_devices():
    sim = Sim().to(dtype=torch.float64)
    p = torch.tensor([0.5, 0.7], dtype=torch.float64, requires_grad=True)
    reference = sim(p)
    grad = torch.autograd.grad(reference.square().sum(), p)[0]
    placement = ng.assign_devices(
        sim, ng.DevicePlan("cuda:0", {"right": "cuda:1"}, diagnostics=True)
    )
    q = p.detach().to("cuda:0").requires_grad_()
    result = sim(q)
    assert result.device == torch.device("cuda:0")
    assert sim.right.bias.value.device == torch.device("cuda:1")
    assert ("sim.right", "cuda:1") in placement.diagnostics()["observed_calls"]
    torch.testing.assert_close(result.cpu(), reference)
    torch.testing.assert_close(torch.autograd.grad(result.square().sum(), q)[0].cpu(), grad)
    assert placement.diagnostics()["bytes"]["cuda:0->cuda:1"] > 0


def test_plan_rejections_are_before_mutation():
    sim = Sim()
    cls = type(sim)
    for plan in (
        ng.DevicePlan("cuda"),
        ng.DevicePlan("cuda:999"),
        ng.DevicePlan("meta"),
        ng.DevicePlan("cpu", {"absent": "cpu"}),
    ):
        with pytest.raises(ng.PlacementError):
            ng.assign_devices(sim, plan)
        assert type(sim) is cls
    with ck.ActiveContext(sim), pytest.raises(ng.PlacementError):
        ng.assign_devices(sim)
    placement = ng.assign_devices(sim)
    with pytest.raises(ng.PlacementError):
        ng.assign_devices(sim)
    with ck.ActiveContext(sim), pytest.raises(ng.PlacementError):
        placement.close()
    placement.close()


@GPU
def test_shared_ownership_and_explicit_resolution():
    sim = Sim()
    del sim.right.bias
    sim.right.bias = sim.left.bias
    with pytest.raises(ng.PlacementError, match="Shared node"):
        ng.assign_devices(sim, ng.DevicePlan("cuda:0", {"right": "cuda:1"}))
    ng.assign_devices(
        sim, ng.DevicePlan("cuda:0", {"right": "cuda:1"}, parameters={sim.left.bias: "cuda:0"})
    )
    assert sim.right.bias is sim.left.bias
    assert sim(torch.tensor([0.5, 0.7], device="cuda:0")).device == torch.device("cuda:0")


@GPU
def test_cross_device_pointer_tie_preserves_gradient():
    sim = Sim().to(dtype=torch.float64)
    sim.right.a = sim.left.a
    assert sim.right.a.pointer
    x = torch.tensor([0.6], dtype=torch.float64, requires_grad=True)
    y = sim(x)
    g = torch.autograd.grad(y.sum(), x)[0]
    ng.assign_devices(sim, ng.DevicePlan("cuda:0", {"right": "cuda:1"}))
    z = x.detach().to("cuda:0").requires_grad_()
    out = sim(z)
    torch.testing.assert_close(out.cpu(), y)
    torch.testing.assert_close(torch.autograd.grad(out.sum(), z)[0].cpu(), g)


@GPU
def test_unsafe_pointer_dependencies_rejected():
    sim = Sim()
    sim.right.bias = lambda p: p["a"].value + p["bias"].value
    sim.right.bias.link((sim.left.a, sim.left.bias))
    with pytest.raises(ng.PlacementError, match="Pointer"):
        ng.assign_devices(
            sim, ng.DevicePlan("cuda:0", {"right": "cuda:1"}, parameters={sim.left.bias: "cuda:1"})
        )


def test_overrides_and_exception_cleanup():
    sim = Sim()
    ng.assign_devices(sim)
    p = torch.tensor([0.5, 0.7])
    with ck.ActiveContext(sim):
        sim.fill_params(p)
        out = sim.right(torch.ones(3), a=torch.tensor(0.9))
        torch.testing.assert_close(out, torch.full((3,), 1.0).sin())
        torch.testing.assert_close(sim.right.a.value, torch.tensor(0.7))
    with pytest.raises(ck.FillParamsError):
        sim(params=[])
    assert not sim.active


def test_explicit_tasks_dependencies_grad_mode():
    x = torch.tensor([2.0], requires_grad=True)
    with ng.AsyncGraphExecutor() as graph:
        a = graph.submit(lambda t: t.square(), x)
        b = graph.submit(lambda t: t * 3, a)
        y = b.result()
        torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0], torch.tensor([12.0]))
        assert len(graph.trace) == 2


@GPU
def test_gpu_explicit_tasks_and_cross_device_dependency():
    x = torch.tensor([2.0], device="cuda:0", requires_grad=True)
    with ng.AsyncGraphExecutor() as graph:
        a = graph.submit(lambda t: t.square(), x, device="cuda:0")
        b = graph.submit(lambda t: t * 3, a, device="cuda:1")
        y = b.result()
        assert y.device == torch.device("cuda:1")
        torch.testing.assert_close(
            torch.autograd.grad(y.sum(), x)[0], torch.tensor([12.0], device="cuda:0")
        )


@GPU
def test_async_inputs_from_nondefault_source_stream_and_cpu_result():
    source = torch.cuda.Stream(device=0)
    with torch.cuda.stream(source):
        x = torch.arange(128.0, device="cuda:0").square()
        with ng.AsyncGraphExecutor() as graph:
            future = graph.submit(lambda z: z * 3, x, device="cuda:1")
            result = future.result()
            actual = result.cpu()
    torch.testing.assert_close(actual, torch.arange(128.0).square() * 3)


@pytest.mark.skipif(
    not hasattr(ck, "active_cache"), reason="active_cache unavailable in stock Caskade 0.15"
)
@GPU
def test_active_cache_decorator_is_preserved_across_device_boundary():
    class Cached(Leaf):
        @ck.active_cache
        @ck.forward
        def __call__(self, x, a, bias):
            return (x * a + bias).sin()

    sim = Sim().to(dtype=torch.float64)
    del sim.right
    sim.right = Cached("cached")
    p = torch.tensor([0.5, 0.7], dtype=torch.float64)
    expected = (
        (sim._grid * p[0] + sim.left.bias.value).sin() * p[1] + sim.right.bias.value
    ).sin() + sim.offset
    ng.assign_devices(sim, ng.DevicePlan("cuda:0", {"right": "cuda:1"}))
    actual = sim(p.to("cuda:0"))
    torch.testing.assert_close(actual.cpu(), expected)
    assert not hasattr(sim.right, "_active_cache___call__")
