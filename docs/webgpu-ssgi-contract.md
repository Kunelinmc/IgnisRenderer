# WebGPU SSGI Contract
## Scope
This document defines the contract for the built-in `ssgi` post-process pass in the WebGPU backend.

## Background
Screen-space global illumination provides a local indirect-light approximation by sampling visible scene color, albedo, normal, and depth buffers. The pass is intended for local color bounce and contact illumination, not full-scene physically complete global illumination.

## API/Contract
- `renderer.postProcess.enable("ssgi", options)` must enable the `ssgi` pass only when `backend.postProcessCapabilities.ssgi` is `true`.
- `renderer.postProcess.setOptions("ssgi", options)` must merge `SSGIOptions` without enabling the pass.
- `WebGPUBackend.postProcessCapabilities.ssgi` must be `true`.
- `WebGLBackend.postProcessCapabilities.ssgi` must be `false`.
- `SoftwareBackend.postProcessCapabilities.ssgi` must be `false`.
- `SSGIOptions.samples` must control the number of shader samples used by the pass.
- `SSGIOptions.samples` must default to `8` and must be clamped to `[1, 16]`.
- `SSGIOptions.radius` must be clamped to `[1, 6]`.
- `SSGIOptions.intensity` must be clamped to `[0, +Infinity)`.
- `SSGIOptions.falloff` must be clamped to `[0.1, +Infinity)`.
- `SSGIOptions.depthPhi` must be clamped to `[0.01, +Infinity)`.
- `SSGIOptions.normalPhi` must be clamped to `[0.1, +Infinity)`.
- `SSGIOptions.albedoBoost` must be clamped to `[0, +Infinity)`.
- The `ssgi` pass must execute after `ssao` when `ssao` is enabled.
- The `taa` pass must execute after `ssgi` when both passes are enabled.
- The shader must preserve source alpha.

## Usage
```ts
import { Renderer, WebGPUBackend } from "ignis-renderer";

const renderer = new Renderer(new WebGPUBackend(), canvas, camera);
renderer.postProcess.enable("ssgi", {
	samples: 16,
	radius: 4,
	intensity: 0.45,
	falloff: 1.8,
	depthPhi: 1.4,
	normalPhi: 2.5,
	albedoBoost: 1.1,
});
renderer.postProcess.enable("taa");
```

```bash
bun tests/test_webgpu_postprocess_runtime_spatial.mjs
```

## Errors & Diagnostics
- If `ssgi` is explicitly enabled on a backend where `backend.postProcessCapabilities.ssgi` is `false`, post-process resolution must disable the pass and emit warning key `"<backend>-postprocess-unsupported-ssgi"`.
- Invalid numeric option values such as `NaN` and `Infinity` should fall back to defaults during pass execution.
- Values outside the supported range must be clamped before shader execution.

## Compatibility / Breaking Changes
`Renderer.features.enableSSGI` and `Renderer.features.ssgiOptions` are removed. Code must use `renderer.postProcess.enable("ssgi", options)` or `renderer.postProcess.setOptions("ssgi", options)`.
