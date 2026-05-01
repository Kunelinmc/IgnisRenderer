# Post-Process Color Filter Pass Contract
## Scope
This document defines the contract for the built-in `color-filter` post-process pass across Software, WebGL, and WebGPU backends.

## Background
The renderer already provides tone mapping, anti-aliasing, and cinematic post effects. A simple color-grading style filter is required for fast look adjustment without introducing custom shader pipelines.

## API/Contract
The renderer must expose the following feature controls:

- `Renderer.features.enableColorFilter: boolean`
  - `false` disables the pass.
  - `true` enables the pass when the backend capability `colorFilter` is available.
- `Renderer.features.colorFilterOptions: ColorFilterOptions`
  - `brightness`: range `[-1, 1]`, default `0`
  - `saturation`: range `[0, 2]`, default `1`
  - `contrast`: range `[0, 2]`, default `1`
  - `temperature`: range `[-1, 1]`, default `0`
  - `tint`: range `[-1, 1]`, default `0`

Execution order contract:

- The `color-filter` pass must execute after `tonemap`.
- The `color-filter` pass must execute before `fxaa`.
- The pass may be skipped automatically when `enableColorFilter` is `false`.

Capability contract:

- `SoftwareBackend.capabilities.colorFilter` must be `true`.
- `WebGLBackend.capabilities.colorFilter` must be `true`.
- `WebGPUBackend.capabilities.colorFilter` must be `true`.

## Usage
```ts
import { Renderer, WebGPUBackend } from "ignis-renderer";

const renderer = new Renderer(new WebGPUBackend(), canvas);
renderer.features.enableToneMapping = true;
renderer.features.enableColorFilter = true;
renderer.features.colorFilterOptions = {
	brightness: 0.05,
	saturation: 1.1,
	contrast: 1.05,
	temperature: 0.2,
	tint: -0.1,
};
```

Verifiable behavior:

- The frame output should change when one or more `colorFilterOptions` values differ from defaults.
- Setting all options to defaults with `enableColorFilter = true` should produce neutral output (no intentional color shift).

## Errors & Diagnostics
- If `enableColorFilter` is `true` on a backend where `capabilities.colorFilter` is `false`, feature resolution must disable the flag and emit a warning with key pattern `"<backend>-feature-color-filter"`.
- Invalid numeric option values (`NaN`, `Infinity`) should fall back to defaults during pass execution.
- Out-of-range values must be clamped to the contract ranges before shader/pixel evaluation.

## Compatibility / Breaking Changes
The API is additive and should be backward compatible. However, pass-chain assumptions in custom post-process integrations may require updates because a new built-in stage `color-filter` is inserted between `tonemap` and `fxaa`.
