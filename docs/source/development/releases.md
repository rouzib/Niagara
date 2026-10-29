# Building a release

A release contains a wheel, a source distribution and the validation records
for their source version.

## Local checks

1. Run the relevant test suite and CPU-only skip check.
2. Build the Sphinx site with warnings treated as errors.
3. Build wheel and source distribution with `python -m build`.
4. Install the wheel into an independent test target or environment.
5. Verify the import path and library bytes, then run the wheel's tests.
6. Check metadata and dependencies, and generate distribution checksums.

Runtime code lives in `src/niagara`. Keep tests, examples, documentation and
numerical products in their own directories, and exclude generated plots,
samples and virtual environments from the runtime wheel.

## Release records

Record the package version, baseline identity, source hashes, test results,
hardware configuration, scientific acceptance gates and relevant limits.
Version the archives and checksums. Describe observable behavior and API
changes in the changelog.

Document single-caller requirements, metadata specialization, supported
cross-GPU transforms and measured transfer/memory costs. Check API
compatibility separately from compiler availability.

Build, test and documentation workflows are configured in `.github/workflows`
and `.readthedocs.yaml`.
