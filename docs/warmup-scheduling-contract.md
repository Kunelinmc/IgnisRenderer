# Warmup Scheduling Contract
## Scope
This document defines scheduling behavior for `Renderer.warmup(options)` and
backend warmup execution.

## Background
Warmup may compile shaders and create resources on the browser main thread.
Awaiting eager warmup during application bootstrap delays interaction until all
requested work completes.

## API/Contract
- `WarmupOptions.scheduling` must accept `"immediate"`, `"next-frame"`, or
  `"idle"`.
- `"immediate"` must preserve eager warmup behavior.
- `"next-frame"` must allow one animation frame before scene synchronization
  and backend warmup begin.
- `"idle"` must prefer `requestIdleCallback` and must fall back to animation
  frame or timer scheduling when idle callbacks are unavailable.
- `WarmupOptions.yieldIntervalMs` must define the approximate main-thread time
  budget between cooperative yield points.
- `yieldIntervalMs: 0` must disable cooperative timer or idle yields.
- `Renderer.warmup(options)` must remain awaitable and must resolve to
  `WarmupReport` after requested work completes.
- Applications that do not require warmup completion before interaction should
  start idle warmup without awaiting it during bootstrap.

## Usage
```ts
void renderer.warmup({ scheduling: "idle" }).then((report) => {
	if (report.failed > 0) {
		console.warn("Renderer warmup failures", report.errors);
	}
});
```

```ts
const report = await renderer.warmup({
	scheduling: "next-frame",
	yieldIntervalMs: 4,
});
```

```bash
bun tests/static/renderer/test_renderer_warmup_lightprobe.mjs
```

## Errors & Diagnostics
- Rejected warmup promises indicate setup errors outside backend compile error
  reporting and must be handled by fire-and-forget callers.
- Shader compilation failures must be returned through `WarmupReport.errors`.
- `WarmupOptions.onProgress` must continue to report completed backend phases.

## Compatibility / Breaking Changes
N/A. Existing `await renderer.warmup()` calls retain eager behavior.
