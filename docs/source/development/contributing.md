# Contributing

Install the editable project with the developer and documentation extras in
the local environment. Use Caskade's public `Module`, `Param`, context and
`@forward` semantics throughout Niagara.

Use explicit `torch.device` homes and small, readable functions. Write NumPy
docstrings that explain parameters, returns, exceptions and useful examples.
Import optional scientific and plotting dependencies lazily where possible,
and preserve the original exception context when a compiler fails.

## Development workflow

Put reusable examples in `examples`, benchmark workflows in `benchmarks` and
measured products in the results/report directories. For changes that affect
scientific or performance results, include the configuration, source hashes
and numerical checks.

Update the public API docstring, guide and relevant regression tests with each
behavior change. For device or gradient changes, compare returned values and
assert homes, ownership, cleanup and every relevant derivative.

## Review checklist

- Does the change preserve native parameter identities, ties and active contexts?
- Is ownership validated before migration or execution?
- Are stream/lifetime rules explicit without production-wide synchronization?
- Are metadata and static-state invalidations handled rather than ignored?
- Do scientific and performance claims identify their exact versions/hardware?
- Do source, installed wheel and documentation use the same API contract?

Avoid unrelated formatting or generated-data changes when making a focused
repair. Preserve raw records even when a later result supersedes them.
