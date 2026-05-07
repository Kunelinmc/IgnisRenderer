# CSG v1 Contract

## Scope
This document defines the v1 Constructive Solid Geometry (CSG) runtime contract
for IgnisRenderer, including `CSG`/`CSGBuilder`, `CSGMeshInstance`, solver
selection, and geometry cache invalidation requirements.

## Background
IgnisRenderer previously had no built-in boolean mesh workflow. Runtime users
had to pre-bake CSG meshes externally and could not integrate CSG updates with
renderer stages, BVH, or physics collider refresh.

## API/Contract
The public surface must provide:

- `CSG.from(input): CSGBuilder`
- `builder.union(input): CSGBuilder`
- `builder.subtract(input): CSGBuilder`
- `builder.intersect(input): CSGBuilder`
- `builder.xor(input): CSGBuilder`
- `builder.toMeshAsset(options?): MeshAsset`
- `builder.solve(options?): CSGRebuildResult`
- `CSGMeshInstance.markCSGDirty(): void`
- `CSGMeshInstance.flushCSG(options?): CSGRebuildResult | Promise<CSGRebuildResult>`
- `CSGMeshInstance.setGraph(input): this`
- `CSGMeshInstance.setSolverPreference(value): this`
- `CSGMeshInstance.setExecutionMode(value): this`
- `CSGMeshInstance.setExecutor(value): this`
- `CSGExecutor.execute(graph, options): Promise<CSGRebuildResult>`

Behavioral requirements:

- CSG inputs must use triangle topology (`triangle-list`); non-triangle inputs
must fail with diagnostics.
- CSG inputs must be closed manifold meshes; non-closed inputs must fail with
diagnostics.
- v1 CSG must preserve only `position`, `normal`, and `uv0`.
- v1 CSG may drop `uv1`, `uv2`, `uv3`, vertex colors, skinning, and morph
data; when dropped, implementation should emit diagnostics.
- Output triangles must be capped by `maxOutputTriangles` (default `200000`).
- Solver preference `auto` must prefer registered wasm solver and must fall back
to builtin solver when wasm solve fails.
- `CSGMeshInstance` updates must mark scene transform dirty and must mark spatial
BVH bounds dirty for the instance.
- `IPrimitive.geometryVersion` must be incremented when CSG-generated geometry is
replaced, and render backends must re-upload geometry when version changes.
- `CSGMeshInstance.physicsSync="auto"` should call
`PhysicsSystem.rebuildColliders(target)` after successful CSG rebuild.

## Usage
```ts
import {
	CSG,
	buildCSGMeshAsset,
	CSGMeshInstance,
	MeshFactory,
	Material,
	Renderer,
} from "../src";

const left = MeshFactory.createBox(
	{ x: -0.2, y: 0, z: 0 },
	2,
	2,
	2,
	new Material({ name: "Left" })
);
const right = MeshFactory.createBox(
	{ x: 0.3, y: 0, z: 0 },
	2,
	2,
	2,
	new Material({ name: "Right" })
);

const csgMesh = new CSGMeshInstance({
	graph: CSG.from(left).subtract(right),
	physicsSync: "off",
});

const executor = {
	execute(graph, options) {
		return Promise.resolve(buildCSGMeshAsset(graph, options));
	},
};

csgMesh.setExecutionMode("worker").setExecutor(executor);

// Optional manual rebuild (sync by default)
const result = csgMesh.flushCSG({ maxOutputTriangles: 100000 });
if (result.ok) {
	// csgMesh.mesh is now replaced and geometryVersion is bumped.
}

// Builder one-shot bake
const bakedAsset = CSG.from(left).union(right).toMeshAsset();
```

## Errors & Diagnostics
- `csg-input-non-triangle-topology`:
Input primitive topology is not `triangle-list`.
- `csg-input-non-manifold`:
Input mesh is not closed manifold.
- `csg-input-invalid-indices`:
Primitive index count is not divisible by 3.
- `csg-input-index-out-of-range`:
Primitive index references a missing vertex.
- `csg-output-triangle-limit`:
Output exceeds `maxOutputTriangles`.
- `csg-solver-missing`:
`solverPreference="wasm"` but no wasm solver is registered.
- `csg-solver-auto-fallback`:
`solverPreference="auto"` wasm solve failed and builtin fallback was used.
- `csg-worker-fallback-sync`:
`executionMode="worker"` was requested but worker runtime is unavailable.

## Compatibility / Breaking Changes
This change is additive for runtime behavior and APIs, but it introduces
`IPrimitive.geometryVersion` as a required contract field in TypeScript source
construction paths.
