# Render Backend Extension Contract

## Scope
This document defines the backend extension registry used by `IRenderBackendSession`, `Renderer`, and backend-owned optional integration APIs.

## Background
Renderer-facing optional capabilities must not add feature-specific properties to the main backend interface. Backends expose optional integration APIs through a stable, typed extension registry in the active backend session.

## API/Contract
- `IRenderBackend`
	- Must not expose `extensions` or an implicit default session.
- `IRenderBackendSession.extensions`
	- Must expose a `RenderBackendExtensionRegistry`.
- `RenderBackendExtensionKey<TApi>`
	- Must carry a unique string `id`.
- `RenderBackendExtensionRegistry.getBackendExtension(key)`
	- Behavior contract: must return the extension API for the specified `key`, or `null` if the backend does not implement it.
- `RenderBackendExtensionRegistry.requireBackendExtension(key)`
	- Behavior contract: must return the extension API or throw a deterministic error if the extension is unavailable.
- `OCCLUSION_CULLING_EXTENSION`
	- Must expose an `OcclusionCullingBackendAdapter` API.
- `PROBE_CAPTURE_EXTENSION`
	- Must expose a `ProbeWebGPUCaptureSource` API.
- `WEBGPU_COMPUTE_EXTENSION`
	- Must expose an `IWebGPUComputeFacade` API.
- Identity Persistence:
	- Extension API objects must maintain the same object identity for the lifetime of the backend session.
- Device Loss Behavior:
	- During a device-lost state, invoking operations on extension APIs must throw a clear, descriptive error.
	- After `restore()` completes, the existing extension API objects must resume normal operation.

## Usage
```ts
import { Renderer, WebGPUBackend, WEBGPU_COMPUTE_EXTENSION } from "../src";

const backend = new WebGPUBackend();
const renderer = new Renderer({ canvas, backend, camera });
await renderer.initialize();

const compute = renderer.getBackendExtension(WEBGPU_COMPUTE_EXTENSION);
if (compute) {
	const buffer = compute.createBuffer({ size: 1024, usage: BufferUsage.Storage });
}
```

## Errors & Diagnostics
- `Render backend extension "<id>" is unavailable.`: thrown when calling `requireBackendExtension` for an unsupported extension.
- Operations on extension APIs during device loss must throw an error with the message prefix `Device lost: `.

## Compatibility / Breaking Changes
- `IRenderBackend.extensions` is removed; extensions are now resolved through `IRenderBackendSession.extensions` via `renderer.getBackendExtension` or `renderer.requireBackendExtension`.
- Applications that manually create a session must query that session's
  extension registry, not the provider.
- Extensions must be queried via typed keys rather than raw string identifiers.
