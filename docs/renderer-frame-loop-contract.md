# Renderer Frame Loop Contract

## Scope
This document defines automatic and manual frame scheduling through `Renderer`.

## Background
Applications need a convenient default render loop while retaining explicit
single-frame control for tools, tests, and externally scheduled runtimes.

## API/Contract
- `Renderer.renderLoop(): () => void`
	- Behavior contract: must schedule frames through `requestAnimationFrame`.
	- Behavior contract: must await each `renderFrame(nowMs)` call before
	  scheduling the next frame.
	- Behavior contract: must log frame failures through `Logger.error` and
	  continue scheduling later frames.
	- Output contract: must return an idempotent function that stops the loop and
	  cancels a pending animation frame request when possible.
	- Constraint: repeated calls while the loop is active must return the same
	  stop function and must not create another loop.
- `Renderer.renderFrame(nowMs): Promise<RenderFrameResult>`
	- Input contract: `nowMs` must use the animation-frame timestamp time base in
	  milliseconds.
	- Behavior contract: must render at most one frame and must not schedule a
	  later frame.
	- Constraint: concurrent calls must reject.
- `Renderer.renderScene(nowMs): Promise<RenderFrameResult>`
	- Compatibility contract: must remain a deprecated alias of
	  `renderFrame(nowMs)`.
	- Constraint: new application code must use `renderFrame(nowMs)` for manual
	  rendering or `renderLoop()` for automatic scheduling.
- `Renderer.destroy()`
	- Behavior contract: must stop the active render loop before waiting for an
	  in-progress frame and destroying the backend session.

## Usage
```ts
const stopRenderLoop = renderer.renderLoop();

// Stop automatic rendering when the application no longer needs it.
stopRenderLoop();

// Tools and externally scheduled runtimes may render one frame manually.
await renderer.renderFrame(performance.now());
```

## Errors & Diagnostics
- `Renderer render loop frame failed.`: logged when a frame rejects. The
  original error must be included in the diagnostic, and the loop must
  continue.
- `Renderer.renderFrame() cannot run concurrently.`: returned when another
  frame is still active.
- Errors from manually awaited `renderFrame()` calls must continue to reject to
  the caller without automatic logging by `Renderer`.

## Compatibility / Breaking Changes
`Renderer.renderScene()` is deprecated and retained only for compatibility.
Applications must use `Renderer.renderFrame()` for manual rendering or
`Renderer.renderLoop()` for automatic scheduling. Neither `renderFrame()` nor
the deprecated alias schedules subsequent frames based on backend
`frameScheduling`.
