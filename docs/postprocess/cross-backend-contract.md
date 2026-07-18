# Post-Process Cross-Backend Contract

## Scope
This document defines the post-process runtime contract for `BackendPostProcessRuntime`, logical post-process passes, and backend-owned execution.

## Background
To support decoupled backends, post-processing resources must be isolated per backend runtime. Instead of sharing single implementation instances globally, passes declare factories that allow each backend to instantiate, cache, and manage its own implementations.

## API/Contract
- `PostProcessPass`
	- Must expose backend-specific implementations via a factory mapping.
- `PostProcessPassImplementationFactory`
	- Must be typed as: `(backend: IRenderBackend) => PostProcessPassImplementation`.
- `BackendPostProcessRuntime`
	- Must be instantiated by each backend.
	- Must cache instantiated pass implementations in a private registry.
	- At each frame boundary, must destroy and clear cached implementations for any passes that have been unregistered from the snapshot.
	- On backend resize, must call `.invalidate()` on all cached implementations.
	- On backend device loss or destruction, must call `.destroy()` on all cached implementations and clear the cache.
- `PostProcessPassImplementationMetadata.graph`
	- May declare backend-agnostic logical resource behavior for an implementation.
	- `color.access` must be one of `"none"`, `"read"`, or `"read-write"`;
	  `color.output` must be `"preserve"` or `"new-version"`.
	- Implementations that publish a GPU color target should declare
	  `color: { access: "read", output: "new-version" }`. Software
	  implementations that mutate the current buffer in place should declare
	  `color: { access: "read-write", output: "preserve" }`.
	- Built-in implementations are validated strictly for controlled output and
	  declared history updates. Custom implementations without `graph` metadata
	  remain compatibility-opaque; arbitrary raw GPU writes cannot be observed.
	- Required entries in `backendShared` must make the pass ineligible when the
	  active backend reports that the resource is unavailable. Entries marked
	  `optional: true` must not affect pass eligibility.
- `IPostProcessExecutor.isGraphResourceAvailable(resourceId)`
	- Must report readiness by backend-shared resource id, independently of the
	  ids of passes that consume the resource.
	- An omitted callback must treat declared backend-shared resources as
	  available for compatibility with executors that do not expose such state.
- `PostProcessGraphExecutionResult`
	- Must distinguish planned `outputColor` from `resolvedOutputColor` after
	  skipped-pass aliases are applied.
	- Must remain backend-internal and contain no native resource handles.
- `BackendPostProcessRuntime`
	- Must retain `lastAttempt` and `lastSuccessful` debug snapshots.
	- Must update `lastSuccessful` only after `commitFrame()` and must preserve it
	  when a later attempt aborts.

## Usage
```ts
import { PostProcessPass, IRenderBackend } from "ignisrenderer";

class CustomPass extends PostProcessPass {
	constructor() {
		super({
			id: "custom-pass",
			implementations: {
				webgpu: (backend: IRenderBackend) => new WebGPUPassImpl(backend),
				webgl: (backend: IRenderBackend) => new WebGLPassImpl(backend),
			},
		});
	}
}
```

## Errors & Diagnostics
- Attempting to execute a pass that has no registered factory for the active backend must trigger a fallback pass execution warning.
- Exceptions thrown in custom pass implementations during execution must abort the post-process runtime frame.
- A pass that uses a controlled color publication and then reports
  `{ ran: false }` must fail the active post-process frame. Raw encoder or
  context writes remain outside that guarantee.

## Compatibility / Breaking Changes
- `PostProcessPass` no longer accepts pre-instantiated implementation objects. It must accept factory functions.
- The global post-process registry bridge is removed. Caches are strictly scoped to the backend runtime.
