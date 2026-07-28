# Screen-Space Global Illumination Pass Contract

## Scope

This document defines the contract and behavior for `ScreenSpaceGlobalIlluminationPass` (`ssgi`) across Software, WebGL, and WebGPU backends.

## Background

Screen-space global illumination provides local indirect-light approximation by sampling visible scene color, albedo, normal, and depth buffers. This pass is designed for local color bounce and contact illumination, rather than a full-scene physically complete global illumination solution.

## API/Contract

- `renderer.postProcess.registerPass(new ScreenSpaceGlobalIlluminationPass({ enabled, options }))` registers the `ssgi` pass.
- `ScreenSpaceGlobalIlluminationPass` must use pass id `"ssgi"`.
- The WebGPU implementation `describeExecution()` declaration must require
  `color`, `depth`, `normal`, and `albedo` logical G-buffer channels.
- **Backend Implementations**:
  - **WebGPU**: The primary backend implementing the v1 SSGI runtime.
  - **Software / WebGL**: Do not currently provide SSGI implementations. The
    planner must skip the pass and emit the missing-implementation diagnostic.
- `SSGIOptions` configuration parameters:
  - `samples`: Controls the number of shader samples. Defaults to `8`, clamped to `[1, 16]`.
  - `radius`: Clamped to `[1, 6]`.
  - `intensity`: Clamped to `[0, +Infinity)`.
  - `falloff`: Clamped to `[0.1, +Infinity)`.
  - `depthPhi`: Clamped to `[0.01, +Infinity)`.
  - `normalPhi`: Clamped to `[0.1, +Infinity)`.
  - `albedoBoost`: Clamped to `[0, +Infinity)`.
- Pass ordering constraints:
  - The `ssgi` pass must execute after `ssao` when `ssao` is enabled.
  - The `taa` pass must execute after `ssgi` when both passes are enabled.
- The SSGI shader must preserve the source alpha channel.
- The `ssgi` pass must execute through the logical pass implementation exported by `src/postprocess/passes/ScreenSpaceGlobalIlluminationPass.ts`.
- `WebGPUPostProcessRuntime` and WebGPU runtime delegates must not register or execute the `ssgi` kernel directly; execution is managed by the pass instance.

## Usage

```ts
import {
	Renderer,
	ScreenSpaceGlobalIlluminationPass,
	TemporalAntiAliasingPass,
	WebGPUBackend,
} from "ignisrenderer";

const renderer = new Renderer({
	backend: new WebGPUBackend(),
	canvas,
	camera,
});
renderer.postProcess.registerPass(new ScreenSpaceGlobalIlluminationPass({
	enabled: true,
	options: {
		samples: 16,
		radius: 4,
		intensity: 0.45,
		falloff: 1.8,
		depthPhi: 1.4,
		normalPhi: 2.5,
		albedoBoost: 1.1,
	},
}));
renderer.postProcess.registerPass(new TemporalAntiAliasingPass({ enabled: true }));
```

Run validation tests:
```bash
bun tests/static/postprocess/test_screen_space_global_illumination_pass.mjs
```

## Errors & Diagnostics

- If `ssgi` is enabled on Software or WebGL, the planner must emit
  `postprocess-implementation-missing-ssgi` and skip the pass.
- Invalid numeric option values such as `NaN` and `Infinity` fall back to defaults during execution.
- Out-of-range option values must be clamped before shader execution.

## Compatibility / Breaking Changes

`Renderer.features.enableSSGI`, `Renderer.features.ssgiOptions`, and backend `postProcessCapabilities` are removed. Applications must register `ScreenSpaceGlobalIlluminationPass` and configure the pass instance directly.
