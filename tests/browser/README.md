# Browser Tests

Place only real browser-runtime tests in this directory. Tests that use fake
canvas, fake WebGL/WebGPU devices, fake Workers, or stubbed browser globals
belong under `tests/static/`.

Browser tests should use filenames that start with `test_` and end with `.mjs`
so `bun run test:browser` can discover them.
