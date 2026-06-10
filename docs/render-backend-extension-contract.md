# Render Backend Extension Contract
## Scope
This document defines the renderer backend extension registry used by
`IRenderBackend`, `Renderer`, and backend-owned optional integration APIs.

## Background
Renderer-facing optional capabilities must not add feature-specific public
properties to `IRenderBackend`. Backends expose optional integration APIs through
a stable extension registry only when the renderer must synchronously consume
that API outside normal backend pass execution.

Post-processing is not a backend extension. It is a normal backend pass named
`"postprocess"` and is gated by `BackendCapabilities.postProcess`.

## API/Contract
- `IRenderBackend.extensions` may expose a `RenderBackendExtensionRegistry`.
- `RenderBackendExtensionRegistry.getExtension(id)` must return the extension
  descriptor for `id`, or `null` when the backend does not expose it.
- `RenderBackendExtensionRegistry.listExtensions()` must return a stable
  snapshot of descriptors for the backend lifetime.
- `createRenderBackendExtensionRegistry(extensions)` must throw when duplicate
  extension ids are registered.
- `RenderBackendExtension.id` must identify one backend integration API.
- `RenderBackendExtension.insertionPoints` must describe where the extension is
  consumed. It must not add renderer frame stages by itself.
- `RenderBackendExtension.api` must contain the concrete backend integration API
  for that extension.
- `renderer.occlusion-culling` must expose an `OcclusionCullingBackendAdapter`.
- `renderer:prepared-scene:occlusion-visibility` must identify synchronous
  previous-frame visibility lookup during prepared-scene building.
- `backend:webgpu:frame-graph:after-depth` may describe WebGPU internal
  occlusion work after depth is available.
- Renderer must not execute `backend:webgpu:frame-graph:after-depth` as a
  renderer frame pass.
- `renderer.postprocess` must not be registered as a backend extension.
- Post-process support must be declared through
  `BackendCapabilities.postProcess` and backend handling of the `"postprocess"`
  pass.

## Usage
```ts
import {
	Renderer,
	WebGPUBackend,
	resolveOcclusionCullingBackendExtension,
} from "../src";

const backend = new WebGPUBackend();
const occlusion = resolveOcclusionCullingBackendExtension(backend)?.api;
console.assert(typeof occlusion?.getVisibilityProvider === "function");

const renderer = new Renderer(backend, canvas, camera);
await renderer.init();
```

```bash
bun tests/static/renderer/test_backend_extensions.mjs
bun tests/static/pipeline/test_frame_planner.mjs
```

## Errors & Diagnostics
- `createRenderBackendExtensionRegistry(extensions)` must throw an `Error` when
  two descriptors use the same `id`.
- Missing `renderer.occlusion-culling` must make prepared-scene occlusion
  visibility resolution return `null`; it must not hide draw packets.
- Missing post-process support must be represented by
  `BackendCapabilities.postProcess = false`; it must not emit a backend
  extension diagnostic.

## Compatibility / Breaking Changes
- `renderer.postprocess` is removed from `RenderBackendExtensionId`.
- `RENDERER_POST_PROCESS_EXTENSION_ID`,
  `RENDERER_POST_PROCESS_INSERTION_POINT`, and
  `resolvePostProcessBackendExtension(backend)` are removed.
- `IRenderBackend.postProcessAdapter` is removed with no extension replacement.
- `IRenderBackend.occlusionCullingAdapter` is removed. Code that needs
  occlusion visibility must use
  `resolveOcclusionCullingBackendExtension(backend)?.api`.
- Built-in Software, WebGL, and WebGPU post-process execution is backend-owned
  and must not be discovered through `IRenderBackend.extensions`.
