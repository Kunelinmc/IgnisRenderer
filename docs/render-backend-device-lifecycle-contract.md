# Render Backend Device Lifecycle Contract

## Scope
This document defines the device/context loss and restoration contract for `IRenderBackend`, the backend event sink, and the application-facing `Renderer` facade.

## Background
GPU-backed renderers can lose their device or context at runtime. WebGPU reports loss through `GPUDevice.lost`, and WebGL reports loss through `webglcontextlost` and `webglcontextrestored` events. IgnisRenderer exposes a backend-agnostic lifecycle contract through attached backend runtimes and events so that the renderer and applications can recover resource state without direct coupling.

## API/Contract
- `IRenderBackend.attach(context)`
	- Behavior contract: must bind the backend instance to one renderer-owned
	  surface and event sink.
	- Constraint: each backend instance may be attached only once.
	- Constraint: a second call must throw, including after `destroy()`.
- `IRenderBackend.id`
	- Output contract: must identify the backend implementation.
- `RenderBackendDeviceLostInfo`
	- Input contract: `reason` may contain a backend-specific loss reason.
	- Input contract: `message` may contain a diagnostic message.
- `RenderBackendEvent`
	- Must be a discriminated union representing backend state transitions.
	- `type: "device-lost"`: emitted when the backend observes device loss. Must carry `info` payload when diagnostics are available.
	- `type: "device-restored"`: emitted when the backend finishes context restoration.
	- `type: "render-invalidated"`: emitted when the backend invalidates visual state. Must carry a semantic `reason` of type `RenderDirtyReason`.
	- `type: "resource-lifecycle"`: emitted when backend-owned resources require renderer-side invalidation or destruction.
- `RenderBackendEventSink`
	- `emit(event: RenderBackendEvent): void`: method called by attached backends to dispatch events.
- `IRenderBackend.initialize()`
	- Behavior contract: must initialize the graphics context and acquire device resources.
	- Constraint: must throw when called before `attach(context)`, after the backend is destroyed, or when already initialized.
- `IRenderBackend.restore()`
	- Behavior contract: must rebuild the graphics context and device resources after loss.
	- Behavior contract: must trigger resource recovery and emit `device-restored` when complete.
- `IRenderBackend.destroy()`
	- Behavior contract: must release all device contexts, textures, buffers, and cached post-process implementations.
	- Constraint: must be idempotent.
- `Renderer.initialize()`
	- Behavior contract: must call `IRenderBackend.initialize()`.
	- Constraint: must throw if already initialized.
- `Renderer.restore()`
	- Behavior contract: must call `IRenderBackend.restore()`.
	- Behavior contract: resets the prepared-scene cache and marks the next frame dirty.
- `Renderer.destroy()`
	- Behavior contract: must wait for the active frame to finish, then call `IRenderBackend.destroy()`.
	- Constraint: must be idempotent.
- `RendererEvents.devicelost`
	- Output contract: emitted to the application after renderer-owned device-loss bookkeeping completes.
- `WebGPUBackend`
	- Must listen to `GPUDevice.lost`.
	- Must perform internal rollback, mark device as lost, and then emit `device-lost` event through `RenderBackendEventSink`.
- `WebGLBackend`
	- Must handle `webglcontextlost` by marking context as lost and emitting `device-lost`.
	- Must handle `webglcontextrestored` by restoring state and emitting `device-restored`.

## Usage
```ts
import { Renderer, WebGPUBackend } from "../src";

const backend = new WebGPUBackend();
console.info(`Using ${backend.id} backend`);
const renderer = new Renderer({
	canvas,
	backend,
	camera,
});

await renderer.initialize();

renderer.on("devicelost", ({ info }) => {
	console.warn(`Device lost: ${info?.message ?? "unknown"}`);
});

// Manual recovery
await renderer.restore();
```

## Errors & Diagnostics
- `Renderer.initialize() cannot be called multiple times.`: triggered when `initialize` is invoked on an already initialized renderer.
- `device-lost` event triggers warning logging with backend-supplied details.
- Context restoration failures must log `WebGL context restore failed` or throw appropriate errors.

## Compatibility / Breaking Changes
- `IRenderBackend.createSession(context)` and public backend session APIs are removed.
- Backend instances are one-shot renderer runtimes and must not be reused across renderers.
- `Renderer.onDeviceLost` and `Renderer.onBackendResourceEvent` are removed.
- `RendererBackendBridge` is removed.
- Backends must route all lifecycle notifications as events through `RenderBackendEventSink` instead of direct callbacks.
