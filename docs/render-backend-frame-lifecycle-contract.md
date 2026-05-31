# Render Backend Frame Lifecycle Contract
## Scope
This document defines the per-frame lifecycle contract for `IRenderBackend`
implementations used by `Renderer`.

## Background
Backends may allocate command encoders, frame targets, particle simulation
state, and temporal copy targets during `beginFrame`. A pass may throw before
`endFrame` runs. The renderer must therefore have a backend-agnostic abort path
that releases active frame state without submitting partial work.

## API/Contract
- `IRenderBackend.beginFrame(context)` must create backend frame state for
  `context`.
- `IRenderBackend.endFrame()` must finalize a successful frame.
- `IRenderBackend.abortFrame(error?)` may release active frame state after a
  failed `beginFrame`, `executePass`, `executeSharedPass`, or `endFrame`.
- `abortFrame` must be idempotent and must tolerate calls when no frame is
  active.
- `abortFrame` must not present to the canvas, submit new frame command buffers,
  copy motion history, or commit temporal history.
- `Renderer.renderScene` must call `abortFrame` when a backend frame has started
  and frame execution fails.
- `Renderer.renderScene` must rethrow the original frame error after abort
  cleanup.
- `WebGPUBackend` must defer resize, MSAA sample-count, and shader-runtime
  resource invalidation work while a frame is active.
- Deferred `WebGPUBackend` lifecycle work must run after `endFrame` or
  `abortFrame` clears active encoder state.
- Deferred `WebGPUBackend` resize work must use the latest queued width and
  height when multiple resize requests occur during one active frame.
- Deferred `WebGPUBackend` MSAA work must use the latest queued sample-count
  request when multiple MSAA changes occur during one active frame.
- Deferred `WebGPUBackend` shader-runtime work must invalidate WebGPU-owned
  caches and resources at the frame boundary. The public `ShaderRuntime` state
  may change immediately.
- Device loss, `destroy`, and full backend rollback must remain immediate
  recovery paths and must not wait for the deferred lifecycle queue.

## Usage
```ts
class Backend implements IRenderBackend {
	beginFrame(context: FrameContext): void {
		this.encoder = this.device.createCommandEncoder();
	}

	endFrame(): void {
		this.queue.submit([this.encoder.finish()]);
		this.encoder = null;
	}

	abortFrame(error?: unknown): void {
		this.encoder = null;
	}
}
```

```bash
bun tests/test_renderer_pipeline_registry.mjs
bun tests/test_webgpu_backend_cache_and_dependency.mjs
bun tests/test_webgpu_frame_executor_resilience.mjs
bun tests/test_webgpu_post_graph.mjs
```

## Errors & Diagnostics
- `renderer-postprocess-abort-failed`: triggered when post-process abort cleanup
  throws while preserving the original frame error.
- `renderer-backend-abort-failed`: triggered when backend abort cleanup throws
  while preserving the original frame error.
- Backend-specific `abortFrame` diagnostics should include the original `error`
  only for logging or telemetry.
- WebGPU deferred lifecycle flush failures should be reported as non-fatal
  diagnostics when preserving an earlier frame failure.

## Compatibility / Breaking Changes
`IRenderBackend.abortFrame` is additive and optional. Backends that implement it
must follow the abort constraints in this document.
