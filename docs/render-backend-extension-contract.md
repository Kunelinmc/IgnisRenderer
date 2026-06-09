# Render Backend Extension Contract
## Scope
This document defines the renderer backend extension registry used by
`IRenderBackend`, `Renderer`, and backend-owned optional integration APIs.

## Background
Renderer-facing optional capabilities must not add feature-specific properties to
`IRenderBackend`. Backends expose optional integration APIs through a stable
extension registry, and Renderer subsystems resolve the specific extension they
need by id.

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
- `RenderBackendExtension.api` must contain the concrete backend integration
  API for that extension.
- `renderer.postprocess` must expose a `PostProcessBackendAdapter`.
- `renderer.occlusion-culling` must expose an
  `OcclusionCullingBackendAdapter`.
- `renderer:postprocess` must identify the renderer-owned logical
  `postprocess` frame stage.
- `renderer:prepared-scene:occlusion-visibility` must identify synchronous
  previous-frame visibility lookup during prepared-scene building.
- `backend:webgpu:frame-graph:after-depth` may describe WebGPU internal
  occlusion work after depth is available. Renderer must not execute this
  insertion point as a renderer frame pass.

## Usage
```ts
import {
	Renderer,
	WebGPUBackend,
	resolvePostProcessBackendExtension,
} from "../src";

const backend = new WebGPUBackend();
const postProcess = resolvePostProcessBackendExtension(backend)?.api;
console.assert(postProcess?.backend === "webgpu");

const renderer = new Renderer(backend, canvas, camera);
await renderer.init();
```

```bash
bunx tsc --noEmit
bun tests/static/renderer/test_backend_extensions.mjs
```

## Errors & Diagnostics
- `createRenderBackendExtensionRegistry(extensions)` must throw an `Error` when
  two descriptors use the same `id`.
- `Renderer` must emit `"<backend>-postprocess-adapter-missing"` once when
  enabled post-process work exists but the `renderer.postprocess` extension is
  absent.
- Missing `renderer.occlusion-culling` must make prepared-scene occlusion
  visibility resolution return `null`; it must not hide draw packets.

## Compatibility / Breaking Changes
- `IRenderBackend.postProcessAdapter` is removed. Use
  `resolvePostProcessBackendExtension(backend)?.api`.
- `IRenderBackend.occlusionCullingAdapter` is removed. Use
  `resolveOcclusionCullingBackendExtension(backend)?.api`.
- Built-in Software, WebGL, and WebGPU backends must expose their optional
  integration APIs through `IRenderBackend.extensions`.
