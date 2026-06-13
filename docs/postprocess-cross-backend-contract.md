# Post-Process Cross-Backend Contract

## Scope
This document defines the post-process runtime contract for `BackendPostProcessRuntime`, logical post-process passes, and session-owned execution.

## Background
To support decoupled backends, post-processing resources must be isolated per session. Instead of sharing single implementation instances globally, passes declare factories that allow each session to instantiate, cache, and manage its own implementations.

## API/Contract
- `PostProcessPass`
	- Must expose backend-specific implementations via a factory mapping.
- `PostProcessPassImplementationFactory`
	- Must be typed as: `(session: IRenderBackendSession) => PostProcessPassImplementation`.
- `BackendPostProcessRuntime`
	- Must be instantiated by each backend session.
	- Must cache instantiated pass implementations in a private registry.
	- At each frame boundary, must destroy and clear cached implementations for any passes that have been unregistered from the snapshot.
	- On session resize, must call `.invalidate()` on all cached implementations.
	- On session device loss or destruction, must call `.destroy()` on all cached implementations and clear the cache.

## Usage
```ts
import { PostProcessPass, IRenderBackendSession } from "../src";

class CustomPass extends PostProcessPass {
	constructor() {
		super({
			id: "custom-pass",
			implementations: {
				webgpu: (session: IRenderBackendSession) => new WebGPUPassImpl(session),
				webgl: (session: IRenderBackendSession) => new WebGLPassImpl(session),
			},
		});
	}
}
```

## Errors & Diagnostics
- Attempting to execute a pass that has no registered factory for the active backend must trigger a fallback pass execution warning.
- Exceptions thrown in custom pass implementations during execution must abort the post-process runtime frame.

## Compatibility / Breaking Changes
- `PostProcessPass` no longer accepts pre-instantiated implementation objects. It must accept factory functions.
- The global post-process registry bridge is removed. Caches are strictly scoped to the backend session.
