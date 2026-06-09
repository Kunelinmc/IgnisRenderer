# Particle Shadow Volume Contract
## Scope
This document defines the v1 contract for particle shadow casting through the
existing `ShadowMap` and `ShadowRenderSet` runtime.

## Background
Particle systems previously supported `receiveShadows` only. Billboard particle
shadow casting now uses light-space volume density that is associated with
active shadow slices, so `CSMShadowMap` and `SingleShadowMap` metadata can be
reused without coupling shadow strategy code to particle rendering.

## API/Contract
- `ParticleDefinition.castShadows` must control whether an alpha billboard
  definition contributes to particle shadow volume density.
- `ParticleDefinition.shadowDensity` must scale the injected billboard density.
- `ParticleDefinition.shadowSoftness` must control the spherical density falloff
  used by the v1 billboard particle kernel.
- `ParticleSystem.castShadows`, `ParticleSystem.shadowDensity`, and
  `ParticleSystem.shadowSoftness` must proxy the first definition for legacy
  callers.
- `ParticleDefinition.castShadows` must default to `true`.
- `ParticleDefinition.shadowDensity` must default to `1`.
- `ParticleDefinition.shadowSoftness` must default to `1`.
- `ParticleBlendMode.Additive` billboard particles must not cast particle
  volume shadows in v1, even when `castShadows` is `true`.
- Mesh particle definitions must use regular mesh shadow caster/transmitter
  draw packets on WebGPU instead of billboard particle shadow volume density.
- Particle shadow volume sampling must multiply the existing shadow visibility.
- Missing or inactive particle volume resources must return transmittance `1`.
- Backends may reduce particle shadow volume resolution when device limits are
  exceeded, but they must not fail the frame because of particle shadow volume
  allocation failure.
- Software and WebGPU must use a light-space `64x64x32` density grid per active
  directional shadow slice in v1.
- WebGL compatibility must use a packed texture approximation and must not imply
  that the WebGL backend supports the independent `volumetric` feature.
- WebGL must pack each `64x64x32` volume into 2D atlas tiles and sample that
  atlas from the scene shadow shader.

## Usage
```ts
import { ParticleSystem, ParticleBlendMode } from "../src/index";

const smoke = new ParticleSystem({
	definitions: [
		{
			lifetimeRange: [1, 3],
			speedRange: [0.5, 2],
			sizeRange: [1, 3],
			startColor: { r: 180, g: 180, b: 180, a: 0.7 },
			shape: {
				kind: "billboard",
				blendMode: ParticleBlendMode.Alpha,
			},
			castShadows: true,
			receiveShadows: true,
			shadowDensity: 0.75,
			shadowSoftness: 1.5,
		},
	],
});
```

Verification command:

```bash
bunx tsc --noEmit
```

## Errors & Diagnostics
- `webgl-particle-shadow-volume-texture-units` must be emitted when the WebGL
  fragment texture-unit budget cannot bind the packed volume atlas.
- `webgl-particle-shadow-volume-atlas-limit` must be emitted when the packed
  WebGL volume atlas exceeds `MAX_TEXTURE_SIZE`.
- `webgl-particle-shadow-volume-create-failed` must be emitted when WebGL cannot
  allocate the packed volume atlas texture.
- `webgpu-particle-gpu-upload-failed` may be emitted when WebGPU particle upload
  fails; the renderer must keep CPU particle batches available for shadow volume
  injection.

## Compatibility / Breaking Changes
- Existing particle systems will default to `castShadows: true`.
- Additive particle systems remain visually compatible because v1 excludes them
  from particle shadow volume injection.
- Legacy `ParticleSystemParams.castShadows`, `shadowDensity`, and
  `shadowSoftness` remain accepted as first-definition aliases.
