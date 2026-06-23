# Particle Template Contract
## Scope
This document defines the contract for independent particle templates and mesh
particle rendering in `ParticleSystem`.

## Background
`ParticleSystem` owns emitter timing, simulation limits, space mode, collisions,
sub-emitters, and LOD. `ParticleTemplate` owns the per-particle spawn ranges,
gradients, render shape, and shadow flags. This separation allows one emitter to
spawn multiple particle kinds without duplicating emitter state.

## API/Contract
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

## Usage
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
			startColor: { r: 255, g: 255, b: 255, a: 1 },
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

## Errors & Diagnostics
- `ParticleSystem supports at most 8 particle templates.` must be thrown when
  more than eight templates are provided.
- `ParticleSystem requires at least one positive template weight.` must be
  thrown when all sanitized template weights are zero.
- `webgl-particle-mesh-skipped` must be emitted once when WebGLBackend receives
  mesh particle batches.
- `software-particle-mesh-skipped` must be emitted once when SoftwareBackend
  receives mesh particle batches.

## Compatibility / Breaking Changes
- Legacy constructor fields remain accepted as first-template aliases.
- Legacy `emit` spawn-range fields remain accepted as first-template aliases.
- Code that reads `ParticleSystem.blendMode`, `texture`, `atlas`,
  `sizeOverLifetime`, `colorOverLifetime`, `receiveShadows`, `castShadows`,
  `shadowDensity`, or `shadowSoftness` continues to access the first template.
- Mesh particle templates are WebGPU-rendered only in this contract.
