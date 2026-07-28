# Post-Process Color Filter Pass Contract
## Scope
This document defines the public contract for the engine-provided
`color-filter` post-process pass across Software, WebGL, and WebGPU backends.

## Background
The renderer provides post-process effects through `renderer.postProcess`. The `color-filter` pass provides lightweight color grading after tone mapping without requiring a custom shader pipeline.

## API/Contract
- `renderer.postProcess.registerPass(new ColorFilterPass({ enabled, options }))` must register the pass.
- `ColorFilterPass.enable(options)` must enable the pass and merge `ColorFilterOptions`.
- `ColorFilterPass.setOptions(options)` must merge `ColorFilterOptions` without enabling the pass.
- `ColorFilterPass.disable()` must disable the pass.
- `ColorFilterPass.resetOptions()` must restore the pass to its initial option state.
- `ColorFilterPass` must expose pass-owned Software, WebGL, and WebGPU implementations.
- Backends must not expose `postProcessCapabilities` for `color-filter`.
- `ColorFilterOptions.brightness` must use range `[-1, 1]` and default to `0`.
- `ColorFilterOptions.saturation` must use range `[0, 2]` and default to `1`.
- `ColorFilterOptions.contrast` must use range `[0, 2]` and default to `1`.
- `ColorFilterOptions.temperature` must use range `[-1, 1]` and default to `0`.
- `ColorFilterOptions.tint` must use range `[-1, 1]` and default to `0`.
- The `color-filter` pass must execute after `tonemap`.
- The `color-filter` pass must execute before `fxaa`.

## Usage
```ts
import { ColorFilterPass, Renderer, WebGPUBackend } from "ignisrenderer";

const renderer = new Renderer({
	backend: new WebGPUBackend(),
	canvas,
	camera,
});
const colorFilter = new ColorFilterPass({ enabled: true });
renderer.postProcess.registerPass(colorFilter);
colorFilter.setOptions({
	brightness: 0.05,
	saturation: 1.1,
	contrast: 1.05,
	temperature: 0.2,
	tint: -0.1,
});
```

```bash
bun tests/static/postprocess/test_incremental_postfx_grading.mjs
```

## Errors & Diagnostics
- If `color-filter` is explicitly enabled without an implementation for the
  active backend, post-process resolution must disable the pass.
- Invalid numeric option values such as `NaN` and `Infinity` should fall back to defaults during pass execution.
- Out-of-range values must be clamped to the contract ranges before shader or pixel evaluation.

## Compatibility / Breaking Changes
`Renderer.features.enableColorFilter`, `Renderer.features.colorFilterOptions`, and backend `postProcessCapabilities` are removed. Code must register `ColorFilterPass` and mutate the pass instance.
