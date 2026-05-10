# Post-Process Color Filter Pass Contract
## Scope
This document defines the public contract for the built-in `color-filter` post-process pass across Software, WebGL, and WebGPU backends.

## Background
The renderer provides post-process effects through `renderer.postProcess`. The `color-filter` pass provides lightweight color grading after tone mapping without requiring a custom shader pipeline.

## API/Contract
- `renderer.postProcess.enable("color-filter", options)` must enable the pass and merge `ColorFilterOptions`.
- `renderer.postProcess.setOptions("color-filter", options)` must merge `ColorFilterOptions` without enabling the pass.
- `renderer.postProcess.disable("color-filter")` must disable the pass.
- `renderer.postProcess.reset("color-filter")` must clear explicit `color-filter` request state.
- `backend.postProcess.capabilities["color-filter"]` must report whether the active backend supports the pass.
- `SoftwareBackend.postProcess.capabilities["color-filter"]` must be `true`.
- `WebGLBackend.postProcess.capabilities["color-filter"]` must be `true`.
- `WebGPUBackend.postProcess.capabilities["color-filter"]` must be `true`.
- `ColorFilterOptions.brightness` must use range `[-1, 1]` and default to `0`.
- `ColorFilterOptions.saturation` must use range `[0, 2]` and default to `1`.
- `ColorFilterOptions.contrast` must use range `[0, 2]` and default to `1`.
- `ColorFilterOptions.temperature` must use range `[-1, 1]` and default to `0`.
- `ColorFilterOptions.tint` must use range `[-1, 1]` and default to `0`.
- The `color-filter` pass must execute after `tonemap`.
- The `color-filter` pass must execute before `fxaa`.

## Usage
```ts
import { Renderer, WebGPUBackend } from "ignis-renderer";

const renderer = new Renderer(new WebGPUBackend(), canvas, camera);
renderer.postProcess.enable("color-filter", {
	brightness: 0.05,
	saturation: 1.1,
	contrast: 1.05,
	temperature: 0.2,
	tint: -0.1,
});
```

```bash
bun tests/test_incremental_postfx_grading.mjs
```

## Errors & Diagnostics
- If `color-filter` is explicitly enabled on a backend where `backend.postProcess.capabilities["color-filter"]` is `false`, post-process resolution must disable the pass and emit warning key `"<backend>-postprocess-unsupported-color-filter"`.
- Invalid numeric option values such as `NaN` and `Infinity` should fall back to defaults during pass execution.
- Out-of-range values must be clamped to the contract ranges before shader or pixel evaluation.

## Compatibility / Breaking Changes
`Renderer.features.enableColorFilter` and `Renderer.features.colorFilterOptions` are removed. Code must use `renderer.postProcess.enable("color-filter", options)` or `renderer.postProcess.setOptions("color-filter", options)`.
