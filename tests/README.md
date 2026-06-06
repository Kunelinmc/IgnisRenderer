# Test Layout

IgnisRenderer separates headless tests from real browser tests.

## Static Tests

`tests/static/` contains tests that run directly under Bun. These tests may use
mock canvas, WebGL, WebGPU, Worker, or DOM-like objects, but they must not depend
on a real browser page or device context.

Run all static tests:

```bash
bun run test:static
```

Run one static test:

```bash
bun tests/static/lighting/test_lighting.mjs
```

The static runner discovers every `test_*.mjs` file under `tests/static/`, so new
tests do not need to be added to a hard-coded manifest.

## Browser Tests

`tests/browser/` is reserved for tests that require a real browser runtime, such
as DOM integration, actual canvas presentation, WebGL contexts, or WebGPU device
access.

Run browser tests:

```bash
bun run test:browser
```

Browser tests are intentionally not included in `bun run test` because they may
require browser automation setup or host GPU capabilities.

## Benchmarks

`tests/benchmarks/` contains benchmark scripts and is not discovered by either
test runner.
