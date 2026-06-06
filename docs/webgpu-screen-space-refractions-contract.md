# WebGPU Screen-Space Refractions Contract

## Scope

This document defines the v1 contract for `ScreenSpaceRefractionsPass` and its
WebGPU implementation.

## Background

`ScreenSpaceRefractionsPass` resolves transmissive material background sampling
as a logical post-process pass. WebGPU owns the required transmission surface
capture targets because transparent transmission materials do not participate in
the opaque deferred G-buffer.

## API/Contract

- `ScreenSpaceRefractionsPass` must use pass id `ssrefraction`.
- `ScreenSpaceRefractionsPass` must be a built-in pass in the `temporal`
  placement at order `215`.
- `ScreenSpaceRefractionsPass.getRequirements()` must require `depth`, `motion`,
  and `transmission` logical G-buffer channels.
- `LogicalGBufferSemantic` must include `transmission`.
- The WebGPU implementation must be the only v1 implementation. SoftwareBackend
  and WebGLBackend must emit the standard unsupported built-in warning when the
  pass is enabled.
- WebGPU must allocate `transmissionSceneColorCopy`, `transmissionLighting`,
  `gTransmissionSurface0`, `gTransmissionSurface1`, `gTransmissionSurface2`, and
  `transmissionDepth` only when `ssrefraction` is enabled and the frame contains
  at least one packet where `materialUsesTransmission(packet.material)` is true.
- `gTransmissionSurface0` must pack encoded normal in `.xy`, linear depth in
  `.z`, and transmission weight in `.w`.
- `gTransmissionSurface1` must pack `ior` in `.x`, `thickness` in `.y`,
  `roughness` in `.z`, and Fresnel average in `.w`.
- `gTransmissionSurface2` must pack background tint in `.rgb` and coverage in
  `.w`.
- `transmissionLighting` must contain forward surface lighting with environment
  or SH background transmission disabled.
- WebGPU must initialize `transmissionDepth` from the current opaque frame depth
  before drawing the transmission capture pass.
- The transmission capture pass must use loaded `transmissionDepth` with depth
  writes enabled so v1 respects opaque occlusion and captures the nearest
  transmissive layer.
- The WebGPU post-process implementation must build Hi-Z from opaque
  `gMotionDepth`, trace refracted rays, sample `transmissionSceneColorCopy`, and
  compose into the active scene color through `publishColorTarget`.
- Hi-Z miss, offscreen hits, and total internal reflection must sample
  `transmissionSceneColorCopy` at the original surface UV.

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
		},
	})
);
```

Run the focused validation with:

```bash
bun tests/static/postprocess/test_screen_space_refractions_pass.mjs
bun tests/static/webgpu/test_webgpu_frame_executor_resilience.mjs
```

## Errors & Diagnostics

- `software-postprocess-unsupported-ssrefraction`: triggered when the pass is
  enabled on SoftwareBackend.
- `webgl-postprocess-unsupported-ssrefraction`: triggered when the pass is
  enabled on WebGLBackend.
- `postprocess-requirement-missing-ssrefraction`: triggered when the logical
  `transmission` channel is unavailable for the frame.

## Compatibility / Breaking Changes

N/A
