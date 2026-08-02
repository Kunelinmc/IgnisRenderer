# Screen-Space Global Illumination

## Scope

This guide describes the WebGPU screen-space global illumination pass exposed
by `ScreenSpaceGlobalIlluminationPass`.

## Background

SSGI traces a small number of diffuse rays through the visible frame depth
hierarchy. It reuses temporal history and depth/normal-aware filtering to keep
the default cost suitable for real-time rendering.

SSGI is limited to geometry and radiance already visible on screen. It must not
be treated as a replacement for probes, baked lighting, or path tracing.

## API/Contract

- The built-in implementation supports WebGPU perspective cameras.
- The default trace runs at half resolution with one ray per pixel.
- The pass owns its temporal history and does not require TAA.
- `maxDistance`, `thickness`, and `normalBias` use world-space units.
- `downsample`, `raysPerPixel`, and `maxSteps` control the primary performance
  cost.
- Orthographic cameras must skip the pass and preserve the input color.

## Usage

```ts
import { ScreenSpaceGlobalIlluminationPass } from "ignisrenderer";

renderer.postProcess.registerPass(new ScreenSpaceGlobalIlluminationPass({
	enabled: true,
	options: {
		downsample: 2,
		raysPerPixel: 1,
		maxSteps: 24,
		maxDistance: 8,
		historyWeight: 0.9,
	},
}));
```

Applications should lower `maxSteps` before increasing `downsample` when SSGI
exceeds the desired GPU budget. Increasing `raysPerPixel` improves convergence
but multiplies the trace work directly.

## Errors & Diagnostics

- `webgpu-ssgi-orthographic-disabled` means the active camera is not supported.
- A missing `depth`, `normal`, `albedo`, `metallic`, or `motion` G-buffer
  semantic makes the pass ineligible.
- A missing `backend:frame-hiz` shared resource makes the pass ineligible.
- History is discarded after resize, camera reset, declaration changes, or
  option changes.

## Compatibility / Breaking Changes

The ray-marched implementation replaces the former screen-neighborhood gather.
The removed options and their replacements are documented in
`docs/migrations/ssgi-ray-marching-migration.md`.
