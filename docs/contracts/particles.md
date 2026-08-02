# Particles Contract

This document defines particle templates, mesh-particle rendering, simulation integration, and particle shadow behavior.

## Contract

### Particle templates

- `ParticleSystemParams.templates` must accept one to eight
  `ParticleTemplate` entries.
- `ParticleTemplate.weight` must control weighted random selection at spawn
  time. Non-finite weights must resolve to `1`; negative weights must resolve to
  `0`.
- At least one template weight must be positive.
- `ParticleTemplate.shape.kind: "billboard"` must render through the existing
  billboard particle path and must use `shape.texture`, `shape.atlas`, and
  `shape.blendMode`.
- `ParticleTemplate.shape.kind: "mesh"` must render the referenced
  `MeshAsset` primitives through their own primitive materials.
- WebGPU must emit mesh particle draw packets from
  `PARTICLE_MESH_TRANSIENT_BATCHES_KEY` and render them through the regular
  mesh material pipeline.
- WebGPU mesh particles must support opaque, transparent, OIT, and mesh shadow
  caster/transmitter passes according to each primitive material.
- WebGLBackend and SoftwareBackend must skip mesh particle templates and emit
  a once-only warning.
- WebGPU compute simulation may run only for compatible single-template
  additive billboard systems. Multi-template or mesh-template systems must
  use the CPU particle simulation path and may still render through WebGPU.
- Legacy `ParticleSystemParams.blendMode`, `texture`, `atlas`,
  `sizeOverLifetime`, `colorOverLifetime`, `receiveShadows`, `castShadows`,
  `shadowDensity`, and `shadowSoftness` must remain accepted as aliases for the
  first billboard template.
- Legacy `ParticleEmitterParams.lifetimeRange`, `speedRange`, `sizeRange`,
  `startColor`, `rotationRange`, and `angularVelocityRange` must remain accepted
  as aliases for the first template.

### Particle shadows

- `ParticleTemplate.castShadows` must control whether an alpha billboard
  template contributes to particle shadow volume density.
- `ParticleTemplate.shadowDensity` must scale the injected billboard density.
- `ParticleTemplate.shadowSoftness` must control the spherical density falloff
  used by the billboard particle kernel.
- `ParticleSystem.castShadows`, `ParticleSystem.shadowDensity`, and
  `ParticleSystem.shadowSoftness` must proxy the first template for legacy
  callers.
- `ParticleTemplate.castShadows` must default to `true`.
- `ParticleTemplate.shadowDensity` must default to `1`.
- `ParticleTemplate.shadowSoftness` must default to `1`.
- `ParticleBlendMode.Additive` billboard particles must not cast particle
  volume shadows, even when `castShadows` is `true`.
- Mesh particle templates must use regular mesh shadow caster/transmitter
  draw packets on WebGPU instead of billboard particle shadow volume density.
- Particle shadow volume sampling must multiply the existing shadow visibility.
- Missing or inactive particle volume resources must return transmittance `1`.
- Backends may reduce particle shadow volume resolution when device limits are
  exceeded, but they must not fail the frame because of particle shadow volume
  allocation failure.
- Software and WebGPU must use a light-space `64x64x32` density grid per active
  directional shadow slice.
- WebGL compatibility must use a packed texture approximation and must not imply
  that the WebGL backend supports the independent `volumetric` feature.
- WebGL must pack each `64x64x32` volume into 2D atlas tiles and sample that
  atlas from the scene shadow shader.

## Usage

### Particle templates

```ts
import {
	Material,
	MeshAsset,
	ParticleBlendMode,
	ParticleSystem,
} from "../src/index";

const shardMesh = MeshAsset.fromFaces([
	{
		material: new Material({ name: "Shard" }),
		vertices: [
			{ x: 0, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
			{ x: 1, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
			{ x: 0, y: 1, z: 0, normal: { x: 0, y: 0, z: 1 } },
		],
	},
]);

const impact = new ParticleSystem({
	emit: {
		rate: 0,
		bursts: [{ time: 0, count: 16 }],
		spread: 0.8,
	},
	templates: [
		{
			id: "spark",
			weight: 3,
			lifetimeRange: [0.2, 0.8],
			speedRange: [4, 8],
			sizeRange: [0.05, 0.2],
			startColor: { r: 255, g: 220, b: 120, a: 1 },
			shape: {
				kind: "billboard",
				blendMode: ParticleBlendMode.Additive,
			},
		},
		{
			id: "shard",
			weight: 1,
			lifetimeRange: [1, 2],
			speedRange: [1, 3],
			sizeRange: [0.2, 0.5],
			shape: {
				kind: "mesh",
				mesh: shardMesh,
			},
			castShadows: true,
		},
	],
});
```

Verification command:

```bash
bun tests/static/particles/test_default_particle_simulator.mjs
```

### Particle shadows

```ts
import { ParticleSystem, ParticleBlendMode } from "../src/index";

const smoke = new ParticleSystem({
	templates: [
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

## Diagnostics

### Particle templates

- `ParticleSystem supports at most 8 particle templates.` must be thrown when
  more than eight templates are provided.
- `ParticleSystem requires at least one positive template weight.` must be
  thrown when all sanitized template weights are zero.
- `webgl-particle-mesh-skipped` must be emitted once when WebGLBackend receives
  mesh particle batches.
- `software-particle-mesh-skipped` must be emitted once when SoftwareBackend
  receives mesh particle batches.

### Particle shadows

- `webgl-particle-shadow-volume-texture-units` must be emitted when the WebGL
  fragment texture-unit budget cannot bind the packed volume atlas.
- `webgl-particle-shadow-volume-atlas-limit` must be emitted when the packed
  WebGL volume atlas exceeds `MAX_TEXTURE_SIZE`.
- `webgl-particle-shadow-volume-create-failed` must be emitted when WebGL cannot
  allocate the packed volume atlas texture.
- `webgpu-particle-gpu-upload-failed` may be emitted when WebGPU particle upload
  fails; the renderer must keep CPU particle batches available for shadow volume
  injection.

## Compatibility

### Particle templates

- Legacy constructor fields remain accepted as first-template aliases.
- Legacy `emit` spawn-range fields remain accepted as first-template aliases.
- Code that reads `ParticleSystem.blendMode`, `texture`, `atlas`,
  `sizeOverLifetime`, `colorOverLifetime`, `receiveShadows`, `castShadows`,
  `shadowDensity`, or `shadowSoftness` continues to access the first template.
- Mesh particle templates are WebGPU-rendered only in this contract.

### Particle shadows

- Existing particle systems will default to `castShadows: true`.
- Additive particle systems remain visually compatible because they are
  excluded from particle shadow volume injection.
- Legacy `ParticleSystemParams.castShadows`, `shadowDensity`, and
  `shadowSoftness` remain accepted as first-template aliases.

## Verification

```bash
bun tests/static/particles/test_renderer_particle_stage.mjs
bunx tsc --noEmit
```

## Related Documents

- [Engine architecture](../architecture/engine.md)
- [Shadow contract](shadows.md)
- [Rendering contract](rendering.md)
