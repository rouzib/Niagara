# Derivative preparation

The default scalar reverse-AD path captures ordinary autograd as an explicit
graph. For a scalar target $L(\theta)$, the graph returns $\nabla_\theta L$
and, with `grad_and_value`, the target value. Repeated calls execute this
numerical graph without accumulating an autograd history.

In the tested Torch release, a cross-device `torch.func` reverse transform can
fail through TensorWrapper storage access. Capturing scalar reverse AD avoids
that path. Other functional transforms still use native Torch behavior and
remain subject to its limitations.

## Convolution derivatives

Captured derivatives can contain convolution-backward operators. Niagara lowers
the tested non-transposed 2-D case, including input, weight and bias derivatives
with stride, padding, dilation and groups. Unsupported forms raise errors.
The lowering covers this case only; other ATen decompositions and arbitrary
higher-order modifier combinations need their own support and validation.

Tests compare the lowered derivatives to native autograd and check explicit
detach semantics. The Caustics inference workflow checks every dynamic
parameter and compares derivatives to finite differences.

## Composition order

1. Capture a simulator with fixed state and metadata.
2. Put the surrounding loss/prior into `.transform()`.
3. Apply the requested derivative and batching transforms to that pure graph.
4. Compile/replay the resulting numerical program.

Compose through these explicit entry points. An arbitrary outer transform on
an already prepared object can cross mutable parameter contexts or fixed
replay buffers, so general modifier nesting is unsupported.

## Scientific interpretation

Check derivatives separately from forward values. Cross-device tests must
verify which parameter receives each component; a summed scalar or total image
norm can hide a mismatch. Near a zero gradient, interpret relative error
alongside absolute error and reference magnitude. Assert device identities
to confirm that the test exercised the intended physical path.
