import pytest
import torch
from test_runtime import GPU, Sim

import niagara as ng


@pytest.mark.parametrize("backend", ["eager", "torchscript", "aot_eager"])
def test_capture_values_gradients_detach(backend):
    def f(x):
        return x.sin() + x.detach().square()

    x = torch.linspace(0.2, 0.8, 7, requires_grad=True)
    c = ng.compile_model(f, x, backend=backend)
    torch.testing.assert_close(c(x), f(x))
    torch.testing.assert_close(torch.autograd.grad(c(x).sum(), x)[0], x.cos())


def test_convolution_derivative_input_weight_bias_groups_stride():
    x = torch.linspace(0.1, 0.8, 4 * 6 * 6 + 4 * 2 * 3 * 3 + 4, dtype=torch.float64)

    def loss(params):
        image = params[:144].reshape(1, 4, 6, 6)
        weight = params[144:-4].reshape(4, 2, 3, 3)
        bias = params[-4:]
        return (
            torch.nn.functional.conv2d(image, weight, bias, stride=2, padding=1, groups=2)
            .sin()
            .square()
            .sum()
        )

    base = ng.compile_model(loss, x, backend="torchscript")
    derivative = base.transform(torch.func.grad_and_value, x)
    gradient, value = derivative(x)
    q = x.detach().requires_grad_()
    expected = loss(q)
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(gradient, torch.autograd.grad(expected, q)[0])


def test_functional_grad_vmap_jacobians_and_composition():
    x = torch.linspace(0.2, 0.8, 4, dtype=torch.float64)
    c = ng.compile_model(lambda t: t.sin(), x, backend="torchscript")
    loss = c.transform(lambda f: lambda z: f(z).square().sum(), x)
    derivative = loss.transform(torch.func.grad_and_value, x)
    batch = x.repeat(3, 1).contiguous()
    v = ng.vmap(derivative, batch, chunk_size=2)
    grad, value = v(batch)
    torch.testing.assert_close(grad, (2 * x.sin() * x.cos()).repeat(3, 1))
    torch.testing.assert_close(value, x.sin().square().sum().repeat(3))
    for transform in (torch.func.jacfwd, torch.func.jacrev):
        jac = c.transform(transform, x)
        torch.testing.assert_close(jac(x), torch.diag(x.cos()))
    with pytest.raises(ng.GuardError, match="before compilation"):
        torch.vmap(c)(batch)
    with pytest.raises(ng.GuardError):
        torch.vmap(lambda t: c(t).sum())(batch)


def test_input_mode_model_and_session_guards():
    sim = Sim()
    ng.assign_devices(sim)
    x = torch.tensor([0.5, 0.7])
    c = ng.compile_model(sim, x, backend="eager")
    for bad in (x.double(), x[:1], x.requires_grad_()):
        with pytest.raises(ng.GuardError):
            c(bad)
    x = x.detach()
    with torch.no_grad(), pytest.raises(ng.GuardError):
        c(x)
    with c.session():
        c(x)
    sim.offset = 0.3
    with pytest.raises(ng.GuardError, match="Simulator"):
        c(x)
    with pytest.raises(ng.GuardError):
        c.transform(torch.func.jacrev, x)


@pytest.mark.parametrize("change", ["parameter", "buffer", "placement", "shape"])
def test_state_mutation_guard(change):
    sim = Sim()
    placement = ng.assign_devices(sim)
    x = torch.tensor([0.5, 0.7])
    c = ng.compile_model(sim, x, backend="eager")
    if change == "parameter":
        sim.left.bias.value = 0.4
    elif change == "buffer":
        sim._grid.add_(1)
    elif change == "placement":
        placement.close()
    else:
        sim.left.a.shape = (1,)
    with pytest.raises(ng.GuardError):
        c(x)


@pytest.mark.parametrize("change", ["new_field", "list", "dict", "order", "requires_grad"])
def test_prepared_guard_checks_mutable_state(change):
    sim = Sim()
    sim.settings = {"a": [1, 2], "b": 3}
    ng.assign_devices(sim)
    x = torch.tensor([0.5, 0.7])
    c = ng.compile_model(sim, x, backend="eager")
    if change == "new_field":
        sim.new_setting = 5
    elif change == "list":
        sim.settings["a"][0] = 4
    elif change == "dict":
        sim.settings["b"] = 4
    elif change == "order":
        value = sim.settings.pop("a")
        sim.settings["a"] = value
    else:
        sim._grid.requires_grad_()
    with pytest.raises(ng.GuardError):
        c(x)


def test_compiled_calls_reject_active_context_including_sessions():
    import caskade as ck

    sim = Sim()
    ng.assign_devices(sim)
    x = torch.tensor([0.5, 0.7])
    c = ng.compile_model(sim, x, backend="eager")
    with c.session(), ck.ActiveContext(sim), pytest.raises(ng.GuardError, match="inactive"):
        c(x)


def test_capture_rejects_input_mutation_and_local_mode_changes():
    def mutate(x):
        return x.add_(1)

    x = torch.ones(3)
    with pytest.raises(ng.CaptureError, match="mutates"):
        ng.compile_model(mutate, x)
    torch.testing.assert_close(x, torch.ones(3))

    def mode_change(x):
        with torch.no_grad():
            return x.sin()

    with pytest.raises(ng.CaptureError, match="Local"):
        ng.compile_model(mode_change, x)


@GPU
@pytest.mark.parametrize("split", [False, True])
def test_gpu_compiler_and_batched_functional_grad(split):
    sim = Sim().to(dtype=torch.float64)
    ng.assign_devices(sim, ng.DevicePlan("cuda:0", {"right": "cuda:1"} if split else {}))
    x = torch.tensor([0.5, 0.7], device="cuda:0", dtype=torch.float64)
    c = ng.compile_model(sim, x, backend="torchscript", record_scopes=True)
    loss = c.transform(lambda f: lambda z: f(z).square().sum(), x)
    d = loss.transform(torch.func.grad_and_value, x)
    batch = x.repeat(4, 1)
    b = ng.vmap(d, batch)
    gradients, values = b(batch)
    z = x.detach().requires_grad_()
    value = sim(z).square().sum()
    ref = torch.autograd.grad(value, z)[0], value.detach()
    torch.testing.assert_close(gradients, ref[0].repeat(4, 1))
    torch.testing.assert_close(values, ref[1].repeat(4))
    assert gradients.device == values.device == torch.device("cuda:0")
    devices = {n["device"] for n in c.execution_plan()["nodes"]}
    assert "cuda:0" in devices
    if split:
        assert "cuda:1" in devices


@GPU
def test_cuda_replay_values_independent_outputs_and_functional_gradient():
    x = torch.tensor([0.3, 0.7], device="cuda:0")
    c = ng.compile_model(lambda z: z.sin().square().sum(), x, backend="cuda_graph")
    old = c(x)
    new = c(x + 0.2)
    torch.testing.assert_close(old, x.sin().square().sum())
    torch.testing.assert_close(new, (x + 0.2).sin().square().sum())
    d = c.transform(torch.func.grad_and_value, x, backend="cuda_graph")
    grad, value = d(x)
    torch.testing.assert_close(grad, 2 * x.sin() * x.cos())
    torch.testing.assert_close(value, old)


@GPU
def test_trainable_captured_constants_keep_autograd():
    x = torch.ones(3, device="cuda:0")
    scale = torch.tensor(0.4, device="cuda:0", requires_grad=True)
    c = ng.compile_model(lambda z: (scale * z).sin(), x, backend="auto")
    assert c.backend != "cuda_graph"
    torch.testing.assert_close(torch.autograd.grad(c(x).sum(), scale)[0], 3 * scale.cos())


@GPU
def test_replay_fallback_without_triton_and_scalar_pytree(monkeypatch):
    from niagara import replay

    find = replay.importlib.util.find_spec
    monkeypatch.setattr(
        replay.importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "triton" else find(name, *a, **k),
    )
    x = torch.ones(4, device="cuda:0")
    c = ng.compile_model(lambda args: args[0].sin() * args[1], (x, 2))
    torch.testing.assert_close(c((x, 2)), x.sin() * 2)
    with pytest.raises(ng.GuardError):
        c((x, 3))


@GPU
def test_segmented_replay_functional_gradients_batch_and_stream_switches():
    sim = Sim().to(dtype=torch.float64)
    ng.assign_devices(sim, ng.DevicePlan("cuda:0", {"right": "cuda:1"}))
    x = torch.tensor([0.5, 0.7], device="cuda:0", dtype=torch.float64)
    c = ng.compile_model(sim, x)
    assert set(c._run.region_devices.values()) == {"cuda:0", "cuda:1"}
    derivative = c.transform(lambda f: lambda z: f(z).square().sum(), x).transform(
        torch.func.grad_and_value, x
    )
    batch = torch.stack((x, x + 0.1, x + 0.2))
    b = ng.vmap(derivative, batch)
    gradients, values = b(batch)
    for i, row in enumerate(batch):
        q = row.detach().requires_grad_()
        reference = sim(q).square().sum()
        torch.testing.assert_close(gradients[i], torch.autograd.grad(reference, q)[0])
        torch.testing.assert_close(values[i], reference.detach())
    streams = [torch.cuda.Stream(device=0), torch.cuda.Stream(device=0)]
    outputs = []
    for i in range(8):
        with torch.cuda.stream(streams[i % 2]):
            data = x + i * 0.01
            outputs.append(c(data))
    for stream in streams:
        torch.cuda.current_stream(0).wait_stream(stream)
    for i, actual in enumerate(outputs):
        torch.testing.assert_close(actual, sim(x + i * 0.01))


@GPU
def test_replica_pool_preserves_order_and_all_gradients():
    x = torch.tensor(
        [[0.5, 0.7], [0.4, 0.8], [0.3, 0.9], [0.6, 0.4], [0.9, 0.1]],
        dtype=torch.float64,
        device="cuda:0",
        requires_grad=True,
    )
    with ng.ReplicaPool(lambda: Sim().to(dtype=torch.float64)) as pool:
        pool.prepare(x, backend="torchscript")
        out = pool(x)
        refsim = Sim().to(device="cuda:0", dtype=torch.float64)
        refsim._grid = refsim._grid.to("cuda:0")
        reference = torch.stack([refsim(row) for row in x])
        torch.testing.assert_close(out, reference)
        torch.testing.assert_close(
            torch.autograd.grad(out.sum(), x, retain_graph=True)[0],
            torch.autograd.grad(reference.sum(), x)[0],
        )
