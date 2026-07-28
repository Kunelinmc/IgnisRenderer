# LOD Mesh Instance Contract

## Scope
This document defines the runtime contract for `LODMeshInstance`, including
level configuration, distance-based level selection, and renderer-stage
integration requirements.

## Background
IgnisRenderer previously exposed `MeshInstance` and `CSGMeshInstance`, but it
did not provide a built-in mesh-level LOD container that could switch geometry
based on camera distance during the default render pipeline.

## API/Contract
The public surface must provide:

- `new LODMeshInstance(params)`
- `LODMeshInstance.levels: ReadonlyArray<LODMeshLevel>`
- `LODMeshInstance.activeLevelIndex: number`
- `LODMeshInstance.hysteresis: number`
- `LODMeshInstance.getLevels(): LODMeshLevel[]`
- `LODMeshInstance.setLevels(levels, activeLevelIndex?): this`
- `LODMeshInstance.setActiveLevelIndex(index): this`
- `LODMeshInstance.setHysteresis(value): this`
- `LODMeshInstance.resolveLevelIndex(distanceToCamera): number`
- `LODMeshInstance.updateLODByDistance(distanceToCamera, options?): boolean`
- `LODMeshInstance.updateLODForCamera(cameraWorldPosition, options?): boolean`

Behavioral requirements:

- `LODMeshInstance` construction must fail when `levels` is empty.
- Each level in `levels` must provide a valid `MeshAsset` in `level.mesh`.
- `level.distance` must be treated as a non-negative maximum distance bound for
that level, and levels must be resolved in ascending distance order.
- Distance resolution should use world-space Euclidean distance between
`cameraWorldPosition` and the instance world position.
- `hysteresis` must be non-negative and should reduce level flickering near
distance boundaries.
- When `LODMeshInstance` changes active mesh, implementation must replace
`meshInstance.mesh` and should invalidate scene transform/spatial state unless
`notifyScene` is explicitly disabled.
- Renderer default stage graph must execute `lod-resolve` after
`transform-update` and before `prepared-scene-build`.

## Usage
```ts
import { LODMeshInstance, MeshFactory, Material } from "../src";

const high = MeshFactory.createBox(
	{ x: 0, y: 0, z: 0 },
	2,
	2,
	2,
	new Material({ name: "High" })
).mesh;
const low = MeshFactory.createPlane(
	{ x: 0, y: 0, z: 0 },
	1,
	1,
	new Material({ name: "Low" })
).mesh;

const lodMesh = new LODMeshInstance({
	levels: [
		{ mesh: high, distance: 8 },
		{ mesh: low, distance: Number.POSITIVE_INFINITY },
	],
	hysteresis: 1,
});

// Manual update path (renderer handles this automatically in default stages)
lodMesh.updateLODForCamera({ x: 0, y: 0, z: 10 });
```

## Errors & Diagnostics
- `LODMeshInstance requires at least one LOD level`:
`levels` is empty or not an array.
- `LODMeshInstance level <index> must provide a MeshAsset`:
`levels[index].mesh` is not a `MeshAsset` instance.

## Compatibility / Breaking Changes
This change is additive. Existing `MeshInstance` and `CSGMeshInstance` flows
remain compatible.
