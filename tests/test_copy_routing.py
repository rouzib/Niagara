"""Compiled copy shortcuts preserve snapshots, precision, layout and gradients."""

import pytest
import torch

import niagara as ng

GPU = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two physical CUDA devices")


def copies(program):
    return [n for n in program.graph.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten._to_copy.default]


@GPU
@pytest.mark.gpu
def test_cpu_result_goes_directly_to_its_gpu_consumer():
    def function(x):
        return x.to("cuda:0").to("cuda:1").square()

    x = torch.tensor([0.2, -0.7], dtype=torch.float64, requires_grad=True)
    compiled = ng.compile_model(function, x, backend="eager")
    assert len(copies(compiled)) == 1
    for _ in range(3):
        a, b = x.detach().clone().requires_grad_(), x.detach().clone().requires_grad_()
        actual, expected = compiled(a), function(b)
        assert actual.device == torch.device("cuda:1")
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), a)[0],
                                   torch.autograd.grad(expected.sum(), b)[0])


@GPU
@pytest.mark.gpu
def test_intermediate_device_computation_keeps_its_copy():
    def function(x):
        intermediate = x.to("cuda:0")
        return intermediate.square(), intermediate.to("cuda:1")

    x = torch.tensor([0.2, -0.7], dtype=torch.float64, requires_grad=True)
    compiled = ng.compile_model(function, x, backend="eager")
    assert len(copies(compiled)) == 2
    actual = compiled(x)
    expected = function(x)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(
        torch.autograd.grad(actual[0].sum() + actual[1].sum().to("cuda:0"), x)[0],
        2 * x.detach() + 1,
    )


@GPU
@pytest.mark.gpu
def test_mutation_between_copies_preserves_the_original_snapshot():
    def function(x):
        original = x * 2
        intermediate = original.to("cuda:0")
        original.add_(10)
        return intermediate.to("cuda:1")

    x = torch.tensor([0.2, -0.7], dtype=torch.float64, requires_grad=True)
    compiled = ng.compile_model(function, x, backend="eager")
    assert len(copies(compiled)) == 2
    result = compiled(x)
    torch.testing.assert_close(result.cpu(), 2 * x)
    torch.testing.assert_close(torch.autograd.grad(result.sum(), x)[0], torch.full_like(x, 2))


def test_intermediate_dtype_conversion_is_not_removed():
    def function(x):
        return x.to(torch.float32).to(torch.float64)

    x = torch.tensor([1 + 2 ** -30], dtype=torch.float64, requires_grad=True)
    compiled = ng.compile_model(function, x, backend="eager")
    assert len(copies(compiled)) == 2
    torch.testing.assert_close(compiled(x), function(x), rtol=0, atol=0)
    assert compiled(x).item() != x.item()
    torch.testing.assert_close(torch.autograd.grad(compiled(x).sum(), x)[0], torch.ones_like(x))


@GPU
@pytest.mark.gpu
def test_intermediate_contiguous_layout_is_preserved():
    def function(x):
        return x.to("cuda:0", memory_format=torch.contiguous_format).to("cuda:1")

    x = torch.arange(12, dtype=torch.float64).reshape(3, 4).t().requires_grad_()
    compiled = ng.compile_model(function, x, backend="eager")
    assert len(copies(compiled)) == 2
    actual, expected = compiled(x), function(x)
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], torch.ones_like(x))


@GPU
@pytest.mark.gpu
def test_direct_gpu_copy_respects_caller_streams_and_temporary_lifetimes():
    def function(x):
        return (x * 3).to("cpu").to("cuda:1")

    sample = torch.arange(32, device="cuda:0", dtype=torch.float64).requires_grad_()
    compiled = ng.compile_model(function, sample, backend="eager")
    assert len(copies(compiled)) == 1
    producer, consumer = torch.cuda.Stream(0), torch.cuda.Stream(1)
    producer.wait_stream(torch.cuda.current_stream(0))
    consumer.wait_stream(torch.cuda.current_stream(1))
    results, gradients = [], []
    with torch.cuda.stream(producer), torch.cuda.stream(consumer):
        for index in range(5):
            x = (sample.detach().clone() + index).requires_grad_()
            result = compiled(x)
            gradients.append(torch.autograd.grad(result.square().sum(), x)[0].clone())
            results.append(result.detach().clone())
            del x, result
            torch.empty_like(sample).fill_(-100)
    torch.cuda.current_stream(0).wait_stream(producer)
    torch.cuda.current_stream(1).wait_stream(consumer)
    for index, (result, gradient) in enumerate(zip(results, gradients, strict=True)):
        expected = torch.arange(32, dtype=torch.float64) + index
        torch.testing.assert_close(result.cpu(), expected * 3)
        torch.testing.assert_close(gradient.cpu(), expected * 18)
