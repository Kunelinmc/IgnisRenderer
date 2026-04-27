# WebGL Backend V2 Contract

## Scope
This document defines the Phase 1 contract for `WebGLBackend` V2 in IgnisRenderer.
`WebGLBackend` must keep the public constructor and type name unchanged.

## Background
The previous WebGL path provided a V1-style pass execution and feature subset.
Phase 1 of V2 must align orchestration semantics with WebGPU while keeping
WebGL-specific implementation constraints.

## API/Contract
- `WebGLBackend` must report `capabilities.sh = true` and `capabilities.clusteredLighting = true`.
- `WebGLBackend` must keep `capabilities.ssgi = false`, `capabilities.ssr = false`, and `capabilities.volumetric = false` in Phase 1.
- `WebGLBackend` must expose `capabilities.fog = true` and support both postprocess fog and scene-mode fog.
- `WebGLBackend` must provide post-process pass registration methods:
  - `registerPostProcessPass(pass)`
  - `unregisterPostProcessPass(id)`
- The backend must validate pass dependency order per frame and must treat `skipPass` as an executed stage.
- SH lighting must use 16 coefficients and must be uploaded through texture-backed data for shader sampling.
- Clustered lighting must be CPU-built (`tile + z-slice`) and must provide runtime fallback to legacy forward lighting when requirements are not met.
- For non-perspective cameras, clustered lighting must be disabled for the frame and a warning key must be emitted.
- Forward-lighting uniform budgets must clamp to `4` directional lights, `16`
  point lights, and `8` spot lights. Extra lights must be ignored with the
  existing light-limit diagnostics.
- PBR materials with `transmissionFactor > 0` must be treated as transparent pass
  submissions even when `alphaMode` is `OPAQUE`.
- WebGL PBR shading must consume transmission through `uPBR.w` and must modulate
  composite alpha so transmissive surfaces do not render as fully opaque.

## Usage
```ts
import { Renderer, WebGLBackend } from "../src";

const backend = new WebGLBackend();
const renderer = new Renderer(backend, canvas, camera);
renderer.features.enableSH = true;
renderer.features.enableClusteredLighting = true;
renderer.features.clusteredLightingOptions = {
	tileSizePx: 64,
	zSlices: 24,
	maxLights: 256,
	maxLightsPerCluster: 64,
};

await renderer.init();
renderer.requestRender();
```

```bash
bun tests/test_webgl_backend_v2.mjs
```

## Errors & Diagnostics
- `webgl-clustered-perspective-only`: triggered when `enableClusteredLighting` is true on a non-perspective camera.
- `webgl-clustered-light-budget`: triggered when light count exceeds `clusteredLightingOptions.maxLights`.
- `webgl-clustered-texture-size-overflow`: triggered when clustered buffers cannot fit within texture capacity.
- `webgl-sh-ambient-texture-create-failed`: triggered when SH coefficient texture allocation fails.
- `webgl-sh-ambient-texture-upload-failed`: triggered when SH coefficient texture upload fails.

## Compatibility / Breaking Changes
- Public backend type name remains `WebGLBackend`.
- Capability semantics changed:
  - `sh` changed from disabled to enabled.
  - `clusteredLighting` changed from disabled to enabled.
- Forward-lighting point-light budget changed from `4` to `16` to match the
  WebGPU backend budget.
- Test entrypoint changed from `tests/test_webgl_backend_v1.mjs` to `tests/test_webgl_backend_v2.mjs`.
