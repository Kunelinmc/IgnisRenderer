# WebGPU OIT Contract
## Scope
This document defines the v1 contract for `Weighted Blended OIT` in the
WebGPU backend, including feature negotiation, runtime gating, pass ordering,
fallback behavior, and diagnostics.

## Background
`Weighted Blended OIT` allows transparent surfaces to be composited without
strict back-to-front sorting for supported content. In v1, OIT is implemented
for standard transparent meshes and alpha-blended particles, while
`transmission` remains on the legacy transparent path.

## API/Contract
- `BackendCapabilities.oit` must exist on all backends.
- `WebGPUBackend.capabilities.oit` must be `true`.
- `WebGLBackend.capabilities.oit` and `SoftwareBackend.capabilities.oit` must be
  `false`.
- `RendererFeatureRequest.enableOIT` must be accepted by feature resolution.
- `Renderer.features.enableOIT` defaults to `false`.
- `resolveFeatureState(...)` must auto-disable `enableOIT` when backend
  capability is `false` and must emit a feature warning.
- OIT must activate only when all runtime constraints are satisfied:
  - Backend is WebGPU.
  - MRT scene targets are available.
  - `sampleCount` is exactly `1`.
  - OIT runtime textures are available.
  - Native command encoder texture-copy access is available.
- When active, transparent packets must be partitioned:
  - `materialUsesTransmission(packet.material) === true`:
    route to legacy transmission path.
  - otherwise: route to OIT path.
- Particle routing must follow:
  - `ParticleBlendMode.Alpha` -> OIT particle pipeline (`fsMainOIT`).
  - `ParticleBlendMode.Additive` -> legacy additive pipeline (`fsMain`).
- OIT resolve must use a separate fullscreen pass and must not read/write the
  same texture simultaneously. The implementation must copy `sceneColorMain`
  into `oitSceneColorCopy`, then resolve back into `sceneColorMain`.

## Usage
```ts
import { Renderer } from "../src/renderers/Renderer";
import { WebGPUBackend } from "../src/renderers/WebGPUBackend";

const backend = new WebGPUBackend();
const renderer = new Renderer(backend, canvas, camera);
await renderer.init();

renderer.features.enableOIT = true;
await renderer.renderScene(performance.now());
```

```ts
// Feature negotiation contract:
// - If backend supports OIT, `enableOIT` remains true.
// - If backend does not support OIT, it is disabled and warning is emitted.
const resolved = resolveFeatureState(
	{ enableOIT: true },
	backend.capabilities,
	backend.type
);
```

## Errors & Diagnostics
- `webgpu-feature-oit`:
  emitted when `enableOIT=true` but backend capability `oit=false`.
- `webgpu-oit-disabled-mrt-unavailable`:
  emitted when OIT is requested but MRT targets are unavailable.
- `webgpu-oit-disabled-msaa`:
  emitted when OIT is requested with `sampleCount > 1`.
- `webgpu-oit-disabled-runtime`:
  emitted when runtime OIT resources or native copy capability are unavailable.
- `webgpu-oit-copy-scene-color-failed`:
  emitted when scene-color copy for OIT resolve fails.

All warnings should be emitted via `warn once` behavior.

## Compatibility / Breaking Changes
No breaking API changes are introduced. `enableOIT` is opt-in and defaults to
disabled. Unsupported backends and unsupported runtime conditions must fall
back to legacy transparent rendering.
