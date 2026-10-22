# Particles Contract

This document defines particle templates, mesh-particle rendering, and simulation integration.

## Contract

### Particle templates

- `ParticleSystemParams.templates` must accept one to eight
  `ParticleTemplate` entries.
- `ParticleSystemParams.templates` must be the only constructor field for
  particle template configuration. When omitted or empty, the system must
  create one default billboard template.
- `ParticleEmitterParams` must contain only emission scheduling and spatial
  fields. Lifetime, speed, size, color, rotation, and render shape configuration
  must belong to `ParticleTemplate`.
- `ParticleTemplate.weight` must control weighted random selection at spawn
  time. Non-finite weights must resolve to `1`; negative weights must resolve to
  `0`.
- At least one template weight must be positive.
- `ParticleTemplate.shape.kind: "billboard"` must render through the existing
  billboard particle path and must use `shape.texture`, `shape.atlas`, and
  `shape.blendMode`.
- Billboard particle render items must apply `startColor` and
  `colorOverLifetime`. Mesh particle render items must not carry per-particle
  color or use particle color alpha for visibility; their appearance must come
  from each primitive's material.
- `ParticleTemplate.shape.kind: "mesh"` must render the referenced
  `MeshAsset` primitives through their own primitive materials.
- WebGPU must emit mesh particle draw packets from
	`PARTICLE_MESH_TRANSIENT_BATCHES_KEY` and render them through the regular
	mesh material pipeline.
- `src/particles/ParticleRenderBatch.ts` must own the backend-neutral billboard
  and mesh particle render-batch contracts. `src/pipeline/types.ts` must own
  only the transient keys used to transport those batches through a frame.
- WebGPU GPU draw batches must describe billboard instances only. Mesh-particle
  batches must use the backend-neutral contract and `prepareFramePackets`.
- `prepareFramePackets` must prepare current-view `DrawPacket` objects from
	emitted mesh-particle batches after particle simulation and before frame analysis.
- Mesh-particle packets must contribute to the opaque, transparent, reflective,
  and complete view packet collections. They must not contribute shadow caster
  or shadow transmitter submissions and must not receive shadows.
- Individual consumers must not rebuild mesh packets from
	`PARTICLE_MESH_TRANSIENT_BATCHES_KEY` or depend on a particle-specific packet
	accessor.
- The particle subsystem must expose backend-neutral render intent for visible
	mesh-particle templates before simulation emits the current frame's packets.
	The intent must identify transparent-material template work without comparing
  backend identifiers.
- Renderer frame planning must include that intent in `FramePassRequirements`
	when resolved render support enables mesh particles, so the
	`main-transparent` backend pass remains available for emitted mesh packets.
- Probe capture views must prepare their own packet set and resolve mesh-particle
	depth for the capture camera. Packet sets must not be reused across views or
	retained across frames. Planar reflection capture must exclude mesh-particle
	contributors because particle reflection is unsupported.
- WebGPU mesh particles must support opaque, transparent, and OIT passes
  according to each primitive material.
- WebGLBackend and SoftwareBackend must skip mesh particle templates and emit
  a once-only warning.
- WebGPU compute simulation may run only for compatible single-template
  additive billboard systems. Multi-template or mesh-template systems must
  use the CPU particle simulation path and may still render through WebGPU.

### Particle shadows

- Billboard and mesh particles must neither cast nor receive shadows.
- `ParticleTemplate`, `ParticleSystem`, and particle render batches must not
  expose shadow configuration fields.
- Particle render intent must not request shadow work. Shadow planning and
  backend shadow passes must consume only non-particle caster submissions.
- Particle shaders and simulation payloads must not carry shadow flags or
  sample shadow resources. Billboard render passes must not bind shadow
  textures, samplers, or uniforms.

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
		},
	],
});
```

Verification command:

```bash
bun tests/static/particles/test_default_particle_simulator.mjs
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

### Particle GPU upload

- `webgpu-particle-gpu-upload-failed` may be emitted when WebGPU particle upload
  fails; the renderer must keep CPU particle batches available for CPU
  rendering.

## Verification

```bash
bun tests/static/particles/test_renderer_particle_stage.mjs
bunx tsc --noEmit
```

## Related Documents

- [Engine architecture](../architecture/engine.md)
- [Shadow contract](shadows.md)
- [Rendering contract](rendering.md)
