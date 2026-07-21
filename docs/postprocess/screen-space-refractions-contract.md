# Screen-Space Refractions Pass Contract

## Scope

This document defines the contract for `ScreenSpaceRefractionsPass` (`ssrefraction`) and its cross-backend behavior across Software, WebGL, and WebGPU backends.

## Background

`ScreenSpaceRefractionsPass` resolves transmissive material background sampling as a logical post-process pass. Because transmissive materials are drawn after the opaque geometry pass, the backend must capture the opaque scene color and depth before rendering transmissive surfaces, and then perform ray-traced refraction using the captured data.

## API/Contract

- `ScreenSpaceRefractionsPass` must use pass id `"ssrefraction"`.
- `ScreenSpaceRefractionsPass` must be an engine-provided manually registered pass in the `temporal` placement at order `215`.
- The WebGPU implementation `describeExecution()` declaration must require
  `depth`, `motion`, `normal`, and `transmission` logical G-buffer channels.
- `LogicalGBufferSemantic` must include `transmission`.
- **Backend Implementations**:
  - **WebGPU**: The primary backend implementing the v1 refraction runtime.
  - **Software / WebGL**: Do not currently provide refraction implementations.
    The planner must skip the pass and emit the missing-implementation
    diagnostic.
- WebGPU must allocate `transmissionSceneColorCopy`, `transmissionLighting`, `gTransmissionSurface0`, `gTransmissionSurface1`, `gTransmissionSurface2`, and `transmissionDepth` only when `ssrefraction` is enabled and the frame contains at least one packet where `materialUsesTransmission(packet.material)` is true.
- WebGPU transmission surface texture packing:
  - `gTransmissionSurface0`: Normal (encoded) in `.xy`, linear depth in `.z`, and transmission weight in `.w`.
  - `gTransmissionSurface1`: `ior` in `.x`, `thickness` in `.y`, `roughness` in `.z`, and Fresnel average in `.w`.
  - `gTransmissionSurface2`: Background tint in `.rgb` and coverage in `.w`.
  - `transmissionLighting`: Forward surface lighting with environment or SH background transmission disabled.
- The transmission capture pass must use loaded `transmissionDepth` with depth writes enabled so the engine respects opaque occlusion and captures the nearest transmissive layer.
- The WebGPU post-process implementation must build a Hi-Z chain from opaque
  `gMotionDepth`, trace refracted rays, refine Hi-Z hits with opaque
  `gNormalRoughMetal` tangent-plane iterations, sample
  `transmissionSceneColorCopy`, and compose the result into the active scene
  color.
- `planeRefinementSteps` must control WebGPU tangent-plane refinement
  iterations, default to `3`, clamp to `0..8`, and use `0` to disable
  tangent-plane refinement while retaining the Hi-Z and binary refinement path.
- Offscreen hits, Hi-Z misses, and total internal reflection must sample `transmissionSceneColorCopy` at the original UV coordinate without offset.

## Usage

```ts
import { ScreenSpaceRefractionsPass } from "ignisrenderer";

renderer.postProcess.registerPass(
	new ScreenSpaceRefractionsPass({
		enabled: true,
		options: {
			maxSteps: 64,
			maxDistance: 50,
			thickness: 0.2,
			planeRefinementSteps: 3,
		},
	})
);
```

Run validation tests:
```bash
bun tests/static/postprocess/test_screen_space_refractions_pass.mjs
```

## Errors & Diagnostics

- `postprocess-requirement-missing-ssrefraction`: Triggered when a required
  logical `depth`, `motion`, `normal`, or `transmission` channel is unavailable
  for the frame.
- On Software or WebGL, the planner must emit
  `postprocess-implementation-missing-ssrefraction` and skip the pass.

## Compatibility / Breaking Changes

- `ScreenSpaceRefractionsPass` now requires the logical `normal` G-buffer channel. Backends or custom setups that do not supply the `normal` channel will have this pass disabled and log `postprocess-requirement-missing-ssrefraction`.
