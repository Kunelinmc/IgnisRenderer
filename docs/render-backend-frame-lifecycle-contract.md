# Render Backend Frame Lifecycle Contract

## Scope
This document defines the per-frame lifecycle contract for `IRenderBackend` implementations managed by `FrameCoordinator`.

## Background
Rendering backends allocate command buffers and transient targets during frame execution. If a stage throws an error, the engine must abort the frame and release resources without submitting partial work to the GPU.

## API/Contract
- `IRenderBackend`
	- Must expose `id` for backend identification.
	- Must expose `attach(context)` for one-time binding to a renderer surface
	  and event sink.
	- Must expose frame lifecycle methods on the attached backend runtime.
	- Must throw if `attach(context)` is called more than once, including after
	  `destroy()`.
- `IRenderBackend.beginFrame(context: FrameContext)`
	- Behavior contract: must prepare command encoders, bind presentation attachments, and transition frame state.
	- Constraint: must throw if another frame is already active or if the backend is uninitialized.
- `IRenderBackend.executePass(pass: FramePass, context: FrameContext)`
	- Behavior contract: must execute the commands for the given `FramePass`.
	- Constraint: must throw if no frame is active.
- `IRenderBackend.skipPass(pass: FramePass)`
	- Behavior contract: called when a pass is disabled in the frame plan, allowing the backend to release/transition dependencies.
- `IRenderBackend.endFrame()`
	- Behavior contract: must finalize command encoders, submit command buffers, and present the frame.
	- Constraint: must throw if no frame is active.
- `IRenderBackend.abortFrame(error?: unknown)`
	- Behavior contract: must cancel/release active encoders and discard command buffers.
	- Constraint: must be idempotent and must not throw if no frame is active.
	- Constraint: must not present to the canvas, commit temporal history, or submit work.
- Deferred flushing:
	- Backends must defer resize, MSAA, and shader runtime compilation updates while a frame is active.
	- Deferred updates must be flushed immediately after `endFrame` or `abortFrame` clears the active frame state.

## Usage
```ts
// Inside FrameCoordinator execution loop
try {
	await backend.beginFrame(frameContext);
	for (const pass of framePlan.backendPasses) {
		if (pass.enabled) {
			await backend.executePass(pass, frameContext);
		} else {
			backend.skipPass?.(pass);
		}
	}
	await backend.endFrame();
} catch (error) {
	await backend.abortFrame(error);
	throw error;
}
```

## Errors & Diagnostics
- `beginFrame` called while a frame is active must throw an error.
- If aborting fails, the backend must catch the error, log a critical diagnostic, and rethrow the original frame error.

## Compatibility / Breaking Changes
- `IRenderBackend.createSession(context)` is removed.
- `IRenderBackend` instances are one-shot renderer runtimes. Applications must
  create a new backend instance for each `Renderer`.
- Backend profile, capability, extension, and frame lifecycle methods are read
  from the attached backend runtime.
- `IRenderBackend.executeSharedPass` is removed.
