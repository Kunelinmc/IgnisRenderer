# Browser Tests

Place only real browser-runtime tests in this directory. Tests that use fake
canvas, fake WebGL/WebGPU devices, fake Workers, or stubbed browser globals
belong under `tests/static/`.

## Playwright Fixtures

Playwright discovers `webgl_*.spec.ts` and `webgpu_*.spec.ts` under `playwright/`.
Each test uses `withBrowserFixture()` from `helpers/browserFixture.ts`, which
loads the shared `fixtures/browser.html` in the test's isolated page.

Browser fixture modules export `setup()`. They use `createBrowserTestSession()`
to return a serializable `result` and an asynchronous `dispose()` method.
Register cleanup immediately after acquiring a resource with the supplied
`defer()` callback. Cleanup runs in reverse order, including when setup fails.
All registered cleanup is attempted even if another cleanup fails.

Use `createTestCanvas(defer, width, height)` to create a canvas with explicit
bitmap and CSS dimensions and attach it before initializing a renderer. Each
renderer must receive a fresh backend instance. Raw WebGL contexts should
register context release, and raw WebGPU devices should register `destroy()`.

The setup callback passed to `withBrowserFixture()` runs in the browser and
must not capture variables from the Playwright process. Import the fixture
inside that callback. Relative fixture imports resolve from `browser.html`.
The result callback runs in Playwright and may perform assertions or take
screenshots while the session is still alive:

```ts
await withBrowserFixture(page, async () => {
	const { setup } = await import("../fixtures/webgl_auxiliary_raster.ts");
	return setup();
}, (result) => {
	expect(result.center[0]).toBeGreaterThan(0.8);
});
```

`withBrowserFixture()` waits for setup, reads the result through a `JSHandle`,
and disposes the browser session after the result callback finishes or throws.
It then releases the handle. Fixture modules must not publish results, ready
flags, or errors on `window`, or start tests as a module import side effect.
Case-specific backgrounds, browser mocks, and prototype changes must register
their own restoration callbacks. A fixture's `setup()` must resolve only after
the work required by its assertions or screenshots has finished.

Run `bun run typecheck:browser` to check browser fixtures and Playwright specs.
Static lifecycle regressions live under `tests/static/helpers/`.

## Standalone Probes

The Bun probe runner discovers filenames that start with `test_` and end with
`.mjs`. It does not create a browser or inject a DOM. Browser-dependent probes
such as Software Canvas HDR report a skip when those globals are unavailable.
