# WebGPU SSGI Contract
## Scope
This document defines the contract for the built-in `ssgi` post-process pass in the WebGPU backend.

## Background
Screen-space global illumination provides a low-cost local indirect-light approximation by sampling visible scene color, albedo, normal, and depth buffers. The pass is intended for local color bounce and contact illumination, not full-scene physically complete global illumination.

## API/Contract
- `Renderer.features.enableSSGI` must default to `false`.
- `Renderer.features.enableSSGI = true` must enable the `ssgi` pass only when the active backend capability `ssgi` is `true`.
- `WebGPUBackend.capabilities.ssgi` must be `true`.
- `WebGLBackend.capabilities.ssgi` must be `false`.
- `SoftwareBackend.capabilities.ssgi` must be `false`.
- `Renderer.features.ssgiOptions.samples` must control the number of shader samples used by the pass.
- `ssgiOptions.samples` must default to `8`.
- `ssgiOptions.samples` must be clamped to `[1, 16]` before upload to the shader.
- `ssgiOptions.radius` must be clamped to `[1, 6]`.
- `ssgiOptions.intensity` must be clamped to `[0, +Infinity)`.
- `ssgiOptions.falloff` must be clamped to `[0.1, +Infinity)`.
- `ssgiOptions.depthPhi` must be clamped to `[0.01, +Infinity)`.
- `ssgiOptions.normalPhi` must be clamped to `[0.1, +Infinity)`.
- `ssgiOptions.albedoBoost` must be clamped to `[0, +Infinity)`.
- The `ssgi` pass must execute after `ssao` when `ssao` is enabled.
- The `taa` pass must execute after `ssgi` when both passes are enabled.
- The shader must preserve source alpha.

## Usage
```ts
import { Renderer, WebGPUBackend } from "ignis-renderer";

const renderer = new Renderer(new WebGPUBackend(), canvas, camera);
renderer.features.enableSSGI = true;
renderer.features.enableTAA = true;
renderer.features.ssgiOptions = {
	samples: 16,
	radius: 4,
	intensity: 0.45,
	falloff: 1.8,
	depthPhi: 1.4,
	normalPhi: 2.5,
	albedoBoost: 1.1,
};
```

```bash
bun tests/test_webgpu_postprocess_runtime_spatial.mjs
```

The command above must pass and verify that `ssgiOptions.samples` is uploaded to `WebGPUSSGIParams`.

## Errors & Diagnostics
- If `enableSSGI` is `true` on a backend where `capabilities.ssgi` is `false`, feature resolution must disable the flag and emit a warning with key pattern `"<backend>-feature-ssgi"`.
- Invalid numeric option values such as `NaN` and `Infinity` should fall back to defaults during pass execution.
- Values outside the supported range must be clamped before shader execution.

## Compatibility / Breaking Changes
The `samples` option is additive and backward compatible. Existing configurations without `ssgiOptions.samples` must continue to use `8` samples.
