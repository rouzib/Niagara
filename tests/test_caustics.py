import pytest
import torch
from test_runtime import GPU

import niagara as ng

pytest.importorskip("caustics")
from examples.niagara_examples.caustics import image_problem, multiplane_scene, pixelated_scene


@GPU
@pytest.mark.gpu
@pytest.mark.caustics
def test_split_noarg_image_matches_native_on_every_call():
    """Repeated no-argument calls must not mutate a captured image buffer."""
    p = image_problem(24, device="cuda:0")
    expected = p.model().detach().clone()
    plan = ng.DevicePlan(
        "cuda:0", {"lens": "cpu", "source": "cuda:1", "lens_light": "cuda:1"}
    )
    with ng.enable(p.model, plan) as engine:
        for _ in range(6):
            torch.testing.assert_close(p.model(), expected, atol=1e-10, rtol=1e-9)
        assert engine.programs


@pytest.mark.caustics
@pytest.mark.parametrize("home", ["cpu", pytest.param("cuda:0", marks=GPU)])
def test_full_image_all_21_gradients(home):
    p = image_problem(24, device=home)
    x = p.truth.clone().requires_grad_()
    reference = p.model(x)
    grad = torch.autograd.grad(reference.square().sum(), x)[0]
    assert len(p.names) == len(x) == 21
    ng.assign_devices(p.model, ng.DevicePlan(home))
    c = ng.compile_model(p.model, x, backend="torchscript")
    actual = c(x)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), x)[0], grad, atol=1e-10, rtol=1e-10
    )


@GPU
@pytest.mark.gpu
def test_split_image_all_gradients_and_ng_vmap():
    p = image_problem(32, device="cuda:0")
    x = p.truth.clone().requires_grad_()
    reference = p.model(x)
    grad = torch.autograd.grad(reference.square().sum(), x)[0]
    placement = ng.assign_devices(
        p.model, ng.DevicePlan("cuda:0", {"source": "cuda:1"}, diagnostics=True)
    )
    actual = p.model(x)
    assert actual.device == torch.device("cuda:0")
    assert p.model.source.Re.device == torch.device("cuda:1")
    assert ("sim.source", "cuda:1") in placement.diagnostics()["observed_calls"]
    assert placement.diagnostics()["bytes"]["cuda:0->cuda:1"] >= 2 * 38 * 38 * 8
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), x)[0], grad, atol=1e-10, rtol=1e-10
    )
    c = ng.compile_model(p.model, x.detach(), backend="torchscript")
    batch = x.detach().repeat(3, 1)
    out = ng.vmap(c, batch)(batch)
    torch.testing.assert_close(out, reference.detach().expand_as(out), atol=1e-12, rtol=1e-11)


@GPU
def test_pixelated_field_and_source_on_different_devices():
    model, x = pixelated_scene(32)
    x = x.to("cuda:0").requires_grad_()
    model.to(device="cuda:0")
    reference = model(x)
    grad = torch.autograd.grad(reference.sum(), x)[0]
    ng.assign_devices(model, ng.DevicePlan("cuda:0", {"source": "cuda:1"}))
    actual = model(x)
    assert model.lens.convergence_map.value.device == torch.device("cuda:0")
    assert model.source.Re.device == torch.device("cuda:1")
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], grad)


def test_prior_transform_roundtrip_and_non_truth_median():
    p = image_problem(16)
    torch.testing.assert_close(p.decode(p.encode(p.truth)), p.truth)
    assert not torch.allclose(p.decode(torch.zeros(21, dtype=torch.float64)), p.truth)


@GPU
def test_multiplane_static_order_ties_and_all_gradients():
    stock, x = multiplane_scene(12, fixed_order=False)
    ordered, y = multiplane_scene(12)
    x = x.requires_grad_()
    y = y.requires_grad_()
    expected = stock(x)
    reference = ordered(y)
    torch.testing.assert_close(reference, expected, atol=1e-10, rtol=1e-9)
    gref = torch.autograd.grad(reference.square().sum(), y)[0]
    torch.testing.assert_close(
        gref, torch.autograd.grad(expected.square().sum(), x)[0], atol=1e-8, rtol=1e-8
    )
    ng.assign_devices(
        ordered,
        ng.DevicePlan(
            "cuda:0",
            {ordered.cosmo: "cuda:0", "field": "cuda:1", "middle": "cuda:1", "perturber": "cuda:1"},
        ),
    )
    q = y.detach().to("cuda:0").requires_grad_()
    c = ng.compile_model(ordered, q, backend="torchscript", record_scopes=True)
    actual = c(q)
    assert actual.device == torch.device("cuda:0")
    assert ordered.perturber.x0.pointer
    assert ordered.middle.x0 in ordered.perturber.x0.children.values()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), q)[0].cpu(), gref, atol=1e-8, rtol=1e-8
    )
    assert any(n["scope"] for n in c.execution_plan()["nodes"])


@GPU
def test_split_scalar_posterior_captured_gradient_all_21_components():
    p = image_problem(16, device="cuda:0")
    ng.assign_devices(p.model, ng.DevicePlan("cuda:0", {"source": "cuda:1"}))
    z = torch.zeros_like(p.truth)
    reference_image = p.model(p.truth).detach()
    render = ng.compile_model(p.model, p.decode(z), backend="eager")

    def loss(f):
        return lambda z: (f(p.decode(z)) - reference_image).square().sum()

    posterior = render.transform(loss, z, backend="torchscript")
    derivative = posterior.transform(torch.func.grad_and_value, z, backend="torchscript")
    actual, value = derivative(z)
    q = z.detach().requires_grad_()
    expected = loss(p.model)(q)
    grad = torch.autograd.grad(expected, q)[0]
    torch.testing.assert_close(value, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual, grad, atol=1e-10, rtol=1e-10)


def test_direct_and_fft_psf_images_and_gradients():
    direct = image_problem(16)
    fft = image_problem(16, psf_mode="fft")
    x = direct.truth.requires_grad_()
    y = fft.truth.requires_grad_()
    a, b = direct.model(x), fft.model(y)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(
        torch.autograd.grad(a.square().sum(), x)[0],
        torch.autograd.grad(b.square().sum(), y)[0],
        rtol=1e-9,
        atol=1e-10,
    )
