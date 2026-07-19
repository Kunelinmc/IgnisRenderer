# OIT Contract
## Scope
This document defines the v1 contract for `Weighted Blended OIT` in the
`WebGPUBackend` and `WebGLBackend`, including feature negotiation, runtime
gating, pass ordering, fallback behavior, and diagnostics.

## Background
`Weighted Blended OIT` allows transparent surfaces to be composited without
strict back-to-front sorting for supported content. In v1, OIT is implemented
for standard transparent meshes and alpha-blended particles, while
`transmission` remains on the legacy transparent path. `WebGLBackend` mirrors
the `WebGPUBackend` pass ordering, but resolves OIT through separate `accum`
and `reveal` draws because WebGL does not provide per-attachment blend state.
When WebGPU deferred lighting is active, OIT still runs after opaque deferred
lighting has resolved into `sceneColorMain`.

## API/Contract
- `BackendCapabilities.oit` must exist on all backends.
- `WebGPUBackend.profile.capabilities.oit` must be `true`.
- `WebGLBackend.profile.capabilities.oit` must be `true`.
- `SoftwareBackend.profile.capabilities.oit` must be `false`.
- `RendererFeatureRequest.enableOIT` must be accepted by feature resolution.
- `Renderer.features.enableOIT` defaults to `false`.
- `resolveFeatureState(...)` must auto-disable `enableOIT` when backend
  capability is `false` and must emit a feature warning.
- OIT must activate only when all runtime constraints are satisfied:
  - Backend is `WebGPU` or `WebGL`.
  - OIT runtime textures are available.
  - For `WebGPU`:
    - MRT scene targets must be available.
    - `sampleCount` must be exactly `1`.
    - Native command-encoder texture-copy access must be available.
  - For `WebGL`:
    - `EXT_color_buffer_float` must be available.
    - scene, post-process, and OIT framebuffers must be complete.
- When active, transparent packets must be partitioned:
  - `materialUsesTransmission(packet.material) === true`:
    route to legacy transmission path.
  - In `WebGLBackend`, `packet.material instanceof ShaderMaterial`:
    route to legacy transparent path.
  - otherwise:
    route to OIT path.
- Particle routing must follow:
  - `ParticleBlendMode.Alpha` -> OIT particle pipeline.
    - `WebGPUBackend` should use `fsMainOIT`.
    - `WebGLBackend` should use OIT pass-mode shading with separate `accum`
      and `reveal` draws.
  - `ParticleBlendMode.Additive` -> legacy additive pipeline.
- OIT resolve must use a separate fullscreen pass and must not read/write the
  same texture simultaneously.
  - `WebGPUBackend` must copy `sceneColorMain` into `oitSceneColorCopy` before
    any OIT accumulation draw, then resolve back into `sceneColorMain`.
  - If that in-frame copy fails while recording, WebGPU must use the legacy
    transparent path for the same frame. It must not silently discard OIT
    contributors after they have been classified.
  - `WebGLBackend` must copy `sceneColor` into `postColorTexture`, then resolve
    back into `sceneColor`.
- WebGPU deferred ordering contract:
  - OIT must not write G-buffer deferred material payload textures.
  - OIT must execute after `main-opaque` deferred lighting resolve.
  - `transmission` materials must remain on the legacy transparent path after
    OIT resolve or after alpha particles, matching existing pass ordering.

## Usage
```ts
import { Renderer } from "../src/rendering/Renderer";
import { WebGLBackend } from "../src/backends/webgl/WebGLBackend";

const backend = new WebGLBackend();
const renderer = new Renderer(backend, canvas, camera);
await renderer.initialize();

renderer.features.enableOIT = true;
await renderer.renderFrame(performance.now());
```

```ts
// Feature negotiation contract:
// - If backend supports OIT, `enableOIT` remains true.
// - If backend does not support OIT, it is disabled and warning is emitted.
const resolved = resolveFeatureState(
	{ enableOIT: true },
	renderer.backendProfile.capabilities,
	renderer.backendProfile.id
);
```

## Errors & Diagnostics
- `webgpu-oit-disabled-mrt-unavailable`:
  emitted when OIT is requested but MRT targets are unavailable.
- `webgpu-oit-disabled-msaa`:
  emitted when OIT is requested with `sampleCount > 1`.
- `webgpu-oit-disabled-runtime`:
  emitted when runtime OIT resources or native copy capability are unavailable.
- `webgpu-oit-copy-scene-color-failed`:
  emitted when scene-color copy for OIT resolve fails.
- `webgl-oit-disabled-runtime`:
  emitted when OIT is requested but WebGL float color-buffer OIT targets are
  unavailable.

All warnings should be emitted via `warn once` behavior.

## Compatibility / Breaking Changes
`enableOIT` remains opt-in and defaults to disabled. Capability inspection must
use `Renderer.backendProfile` or an explicit attached backend. Unattached
backends must not be used for runtime capability checks. Unsupported backends
and unsupported runtime conditions must fall back to legacy transparent
rendering.
