# SSGI Ray-Marching Migration

## Scope

This document describes migration from the screen-neighborhood SSGI kernel to
the WebGPU ray-marched SSGI implementation.

## Background

The previous pass sampled fixed pixel offsets and used depth/normal weights.
The replacement reconstructs surfaces, traces cosine-weighted hemisphere rays
through Hi-Z, accumulates temporal history, and applies separable bilateral
denoising.

## API/Contract

- Remove `samples`; use `raysPerPixel`.
- Remove pixel-space `radius`; use world-space `maxDistance`.
- Remove `falloff`; use `distanceFalloffExponent`.
- Remove `depthPhi`; use `denoiseDepthPhi`.
- Remove `normalPhi`; use `denoiseNormalPhi`.
- Remove `albedoBoost`; receiver albedo is applied during composition.
- Remove `createSSGIKernelParams()`; use the trace, denoise, and compose
  parameter packing helpers.

`ScreenSpaceGlobalIlluminationPass`, its `"ssgi"` id, and its registry usage do
not change.

## Usage

Before:

```ts
new ScreenSpaceGlobalIlluminationPass({
	enabled: true,
	options: {
		samples: 8,
		radius: 3,
		falloff: 1.5,
		depthPhi: 1.25,
		normalPhi: 2,
		albedoBoost: 1,
	},
});
```

After:

```ts
new ScreenSpaceGlobalIlluminationPass({
	enabled: true,
	options: {
		downsample: 2,
		raysPerPixel: 1,
		maxSteps: 24,
		maxDistance: 8,
		distanceFalloffExponent: 2,
		denoiseDepthPhi: 24,
		denoiseNormalPhi: 16,
	},
});
```

## Errors & Diagnostics

Type errors referencing removed option names indicate an incomplete migration.
Visual scale changes indicate that an old pixel-space `radius` was copied
directly into world-space `maxDistance`; tune the value for the scene scale.

## Compatibility / Breaking Changes

This migration is breaking. No legacy gather mode, deprecated aliases, or
automatic unit conversion is provided. WebGL and Software remain unsupported,
and orthographic WebGPU cameras skip SSGI.
