# Render Backend Device Lifecycle Contract
## Scope
This document defines the public device/context loss and restoration contract for
`IRenderBackend` implementations and the `Renderer` facade.

## Background
GPU-backed renderers can lose their device or context at runtime. WebGPU reports
loss through `GPUDevice.lost`, and WebGL reports loss through
`webglcontextlost` and `webglcontextrestored` events. IgnisRenderer must expose
a backend-agnostic lifecycle contract so applications and tests can notify or
restore render resources without using private backend members.

## API/Contract
- `RenderBackendDeviceLostInfo`
	- Input contract: `reason` may contain a backend-specific loss reason.
	- Input contract: `message` may contain a diagnostic message.
- `IRenderBackend.onDeviceLost(info?: RenderBackendDeviceLostInfo)`
	- Behavior contract: implementations must mark device or context resources as
	  unavailable.
	- Behavior contract: implementations should release transient GPU resources
	  that are invalid after loss.
	- Behavior contract: implementations that own GPU-backed post-process
	  resources must notify the renderer before destroying the lost device.
	- Constraint: implementations must tolerate repeated calls.
- `IRenderBackend.restore(canvas?: HTMLCanvasElement)`
	- Behavior contract: implementations must rebuild device or context resources
	  for `canvas` or for the previously initialized canvas.
	- Constraint: implementations must throw a deterministic error when no canvas
	  is available.
- `Renderer.onDeviceLost(info?: RenderBackendDeviceLostInfo)`
	- Behavior contract: forwards `info` to `renderer.backend.onDeviceLost` when
	  the backend implements the method.
	- Behavior contract: resets renderer prepared-scene cache and marks the next
	  frame dirty.
	- Behavior contract: emits `RendererEvents.devicelost` after renderer-owned
	  device-loss bookkeeping completes.
- `Renderer.restore()`
	- Behavior contract: calls `renderer.backend.restore(renderer.canvas)` when
	  implemented.
	- Behavior contract: falls back to `renderer.backend.init(renderer.canvas)`
	  when the backend has no `restore` method.
	- Behavior contract: resizes the canvas and marks the next frame dirty.
- `Renderer.onBackendResourceEvent(event)`
	- Behavior contract: handles renderer-owned backend resource invalidation or
	  destruction for `event`.
	- Behavior contract: emits `RendererEvents.backendresourceevent` after
	  renderer-owned resource handling completes.
- `RendererEvents.devicelost`
	- Output contract: payload `backend` must identify the active backend type.
	- Output contract: payload `info` may contain the reported loss reason and
	  diagnostic message.
- `RendererEvents.backendresourceevent`
	- Output contract: payload must be the `RendererBackendResourceEvent` handled
	  by `Renderer.onBackendResourceEvent`.
- `RendererBackendBridge.onBackendResourceEvent(event)`
	- Input contract: `event.resource` must identify the backend resource family.
	- Input contract: `event.action` must be `"invalidate"` or `"destroy"`.
	- Behavior contract: for `resource: "postprocess"`, `Renderer` must
	  invalidate or destroy renderer-owned post-process state for `event.backend`.
	- Constraint: backends must emit `{ resource: "postprocess", action: "destroy" }`
	  before releasing a device or graphics context that owns post-process handles.
- `RendererBackendBridge.onDeviceLost(info?)`
	- Behavior contract: backends should notify the renderer after automatic
	  device/context loss is observed.
	- Constraint: backends must mark their own lost state before notifying the
	  renderer through this bridge callback.
- `WebGPUBackend` must route `GPUDevice.lost` through
  `WebGPUBackend.onDeviceLost`.
- `WebGPUBackend` must notify `RendererBackendBridge.onDeviceLost(info)` after
  automatic `GPUDevice.lost` handling.
- `WebGPUBackend` must call
  `RendererBackendBridge.onBackendResourceEvent({ resource: "postprocess", action: "destroy", backend: "webgpu" })`
  before rollback during automatic device-loss recovery, manual `restore()`,
  and `destroy()`.
- `WebGLBackend` must route `webglcontextlost` through
  `WebGLBackend.onDeviceLost` and `webglcontextrestored` through
  `WebGLBackend.restore`.
- `WebGLBackend` must notify `RendererBackendBridge.onDeviceLost(info)` after
  automatic `webglcontextlost` handling.

## Usage
```ts
import { Renderer, WebGPUBackend } from "../src";

const backend = new WebGPUBackend();
const renderer = new Renderer(backend, canvas, camera);
await renderer.init();

renderer.on("devicelost", ({ backend, info }) => {
	console.warn(`Renderer device lost on ${backend}: ${info?.message ?? "N/A"}`);
});
renderer.on("backendresourceevent", (event) => {
	console.info(`Backend resource ${event.action}: ${event.resource}`);
});

renderer.onDeviceLost({
	reason: "manual-recovery",
	message: "Application requested graphics device recovery.",
});

await renderer.restore();
renderer.requestRender();
```

```bash
bun tests/test_webgl_backend_stub.mjs
bun tests/test_webgpu_backend_cache_and_dependency.mjs
bun tests/test_renderer_dynamic_texture_updates.mjs
```

## Errors & Diagnostics
- `WebGPU backend cannot restore before a canvas has been initialized.`:
  triggered when `WebGPUBackend.restore()` is called without a provided or
  stored canvas.
- `WebGL backend cannot restore before a canvas has been initialized.`:
  triggered when `WebGLBackend.restore()` is called without a provided or stored
  canvas.
- WebGPU device loss should log the loss reason and message when available.
- WebGL context restoration failures should log `WebGL context restore failed`.

## Compatibility / Breaking Changes
`RendererBackendBridge.destroyPostProcessResources` is removed. Backends must
use `RendererBackendBridge.onBackendResourceEvent` for renderer-owned resource
cleanup. Backends may still omit `IRenderBackend.onDeviceLost` and
`IRenderBackend.restore`; `Renderer.restore` must fall back to `init` for such
backends.

`RendererEvents.devicelost`, `RendererEvents.backendresourceevent`, and
`RendererBackendBridge.onDeviceLost` are additive. Existing backends may omit the
bridge callback when they do not observe automatic device or context loss.
