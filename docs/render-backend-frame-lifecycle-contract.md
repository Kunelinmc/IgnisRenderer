# Render Backend Frame Lifecycle Contract

## Scope
This document defines the per-frame lifecycle contract for `IRenderBackendSession` implementations managed by `FrameCoordinator`.

## Background
Rendering backends allocate command buffers and transient targets during frame execution. If a stage throws an error, the engine must abort the frame and release resources without submitting partial work to the GPU.

## API/Contract
- `IRenderBackend`
	- Must expose `id` for provider identification.
	- Must expose `createSession(context)` for backend runtime creation.
	- Must not expose frame lifecycle methods or an implicit default session.
- `IRenderBackendSession.beginFrame(context: FrameContext)`
	- Behavior contract: must prepare command encoders, bind presentation attachments, and transition frame state.
	- Constraint: must throw if another frame is already active or if the session is uninitialized.
- `IRenderBackendSession.executePass(pass: FramePass, context: FrameContext)`
	- Behavior contract: must execute the commands for the given `FramePass`.
	- Constraint: must throw if no frame is active.
- `IRenderBackendSession.skipPass(pass: FramePass)`
	- Behavior contract: called when a pass is disabled in the frame plan, allowing the backend to release/transition dependencies.
- `IRenderBackendSession.endFrame()`
	- Behavior contract: must finalize command encoders, submit command buffers, and present the frame.
	- Constraint: must throw if no frame is active.
- `IRenderBackendSession.abortFrame(error?: unknown)`
	- Behavior contract: must cancel/release active encoders and discard command buffers.
	- Constraint: must be idempotent and must not throw if no frame is active.
	- Constraint: must not present to the canvas, commit temporal history, or submit work.
- Deferred flushing:
	- Backend sessions must defer resize, MSAA, and shader runtime compilation updates while a frame is active.
	- Deferred updates must be flushed immediately after `endFrame` or `abortFrame` clears the active frame state.

## Usage
```ts
// Inside FrameCoordinator execution loop
try {
	await session.beginFrame(frameContext);
	for (const pass of framePlan.backendPasses) {
		if (pass.enabled) {
			await session.executePass(pass, frameContext);
		} else {
			session.skipPass?.(pass);
		}
	}
	await session.endFrame();
} catch (error) {
	await session.abortFrame(error);
	throw error;
}
```

## Errors & Diagnostics
- `beginFrame` called while a frame is active must throw an error.
- If aborting fails, the session must catch the error, log a critical diagnostic, and rethrow the original frame error.

## Compatibility / Breaking Changes
- `IRenderBackend` direct frame lifecycle methods are removed. Frame orchestration is moved entirely to `IRenderBackendSession` and coordinated by `FrameCoordinator`.
- Backend providers no longer create an implicit session when profile,
  capability, extension, or frame methods are read.
- Backend providers may be identified with `IRenderBackend.id` without creating
  a session.
- `IRenderBackend.executeSharedPass` is removed.
