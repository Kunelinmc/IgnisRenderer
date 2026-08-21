# Geometry Contract

This document defines spatial indexing, level-of-detail mesh selection, and constructive solid geometry behavior.

## Contract

### Mesh bounds

- `MeshFactory.createPlane` must create an XZ-plane whose vertex normals and
  front-face winding both point toward positive Y.
- `MeshAsset.primitives` must be a frozen `ReadonlyArray<IPrimitive>` snapshot.
  Mutating the array passed to the constructor must not mutate the asset.
- A primitive must belong to at most one `MeshAsset`. Adding a duplicate
  primitive or a primitive owned by another asset must fail.
- `MeshAsset.setPrimitives`, `addPrimitive`, `replacePrimitive`, and
  `removePrimitive` must be the only structural primitive mutation paths.
- `MeshAsset.setPrimitiveGeometry` must replace primitive geometry, increment
  `IPrimitive.geometryVersion`, and invalidate only that primitive's cached
  local bounds.
- Callers that mutate an existing geometry buffer in place must call
  `MeshAsset.markPrimitiveGeometryDirty`. Direct writes to
  `IPrimitive.geometryVersion` are not supported.
- `MeshAsset.boundingBox` and `MeshAsset.boundingSphere` must be readonly,
  lazily refreshed derived values. `MeshAsset.boundsVersion` must change for
  every structural or geometry invalidation.
- Primitive and model bounding spheres must retain the exact existing
  AABB-center/max-vertex-distance definition. Implementations must not replace
  them with conservative primitive-sphere unions.
- Spatial indexes must use each `MeshInstance`'s own world-space AABB. Public
  `Node.getWorldBoundingBox()` must retain its subtree aggregate behavior.

### Spatial indexing

- `Scene.spatialIndexMode` must accept only `"bvh"` or `"hybrid"`.
- `Scene.setSpatialIndexMode(mode)` must reset the current spatial index and
  must rebuild lazily on the next frustum/ray query.
- `Scene.rebuildSpatialIndex(meshInstances)` must preserve incremental behavior:
  structural changes must call `upsert`/`remove`, and transform or dynamic-state
  changes must call `markDirty`.
- `SpatialIndex3D.queryFrustumInto(frustum, out, options?)` must clear `out`,
  append mesh-instance candidates whose world-space bounds overlap the frustum,
  and return the same `out` array.
- `SpatialIndex3D.queryBoundsInto(bounds, out, options?)` must clear `out`,
  append mesh-instance candidates whose world-space bounds overlap the input
  AABB, and return the same `out` array.
- `SpatialIndex3D.queryRayDetailedInto(origin, direction, out, options?)` must
  clear `out`, append ray candidates ordered by ascending `distance`, then by
  `entityId`, then by `meshInstance.id`, and return the same `out` array.
- Implementations may expose compatibility wrappers named `queryFrustum`,
  `queryBounds`, `queryRay`, and `queryRayDetailed`, but these wrappers are not
  part of the `SpatialIndex3D` interface contract.
- `BVH` must accept either a numeric leaf size or `BVHOptions`. `BVHOptions`
  may set `leafSize`, `buildStrategy`, `rebuildDirtyRatio`, and
  `rebuildSurfaceAreaInflation`.
- `BVHOptions.buildStrategy` must accept `"median"` or `"sah"`. The default
  must remain `"median"`.
- `HybridSpatialIndex` must classify dynamic instances by
  `isDynamicSpatialMeshInstance(meshInstance)` and route static instances to a
  `BVH`.
- `HybridSpatialIndexOptions.dynamicBackend` must accept `"auto"`, `"bvh"`, or
  `"octree"`. The default must be `"auto"`.
- `HybridSpatialIndexOptions.dynamicBVH` may configure the dynamic `BVH` when
  `dynamicBackend` is `"auto"` or `"bvh"`.
- `HybridSpatialIndexOptions.dynamicOctree` may configure the dynamic
  `LooseOctree` when `dynamicBackend` is `"auto"` or `"octree"`.
- `HybridSpatialIndex` with `dynamicBackend: "bvh"` must route dynamic
  instances to a `BVH`.
- `HybridSpatialIndex` with `dynamicBackend: "octree"` must route dynamic
  instances to a `LooseOctree`.
- `HybridSpatialIndex` with `dynamicBackend: "auto"` must select a dynamic
  backend during `rebuild(meshInstances?)`. It should use `BVH` for small
  dynamic sets or octree layouts with high parent/leaf-resident object pressure,
  and may use `LooseOctree` when dynamic objects subdivide cleanly.
- `HybridSpatialIndex` must maintain exclusive bucket membership during
  `upsert`, `remove`, and `markDirty`; query merging may rely on that exclusivity
  and must not require per-query identity de-duplication.
- `HybridSpatialIndex.markDirty(meshInstance)` must preserve incremental
  updates within the currently selected dynamic backend unless the instance
  migrates between static and dynamic buckets.
- `LooseOctree.markDirty(meshInstance)` should update the stored bounds in place
  when the updated AABB remains inside the current loose node.
- `BVH.markDirty(meshInstance)` should refit only the affected leaf and changed
  ancestor path when the dirty ratio remains below the rebuild threshold.
- `BVHOptions.rebuildSurfaceAreaInflation` must compare a tree-wide
  surface-area quality cost against the most recent full-rebuild baseline.
- Frustum traversal should append fully contained subtrees without testing
  descendant bounds. Bounds traversal should do the same when the query AABB
  fully contains a node AABB.
- `LooseOctree.queryRayDetailedInto` should tighten its traversal distance when
  finite `maxResults` already has enough hits.

### LOD mesh instances

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

### Constructive solid geometry

The public surface must provide:

- `CSG.from(input): CSGBuilder`
- `new CSGSolver(wasmSolvers?): CSGSolver`
- `builder.union(input): CSGBuilder`
- `builder.subtract(input): CSGBuilder`
- `builder.intersect(input): CSGBuilder`
- `builder.xor(input): CSGBuilder`
- `builder.toMeshAsset(options?): MeshAsset`
- `builder.solve(options?): CSGRebuildResult`
- `solver.buildMeshAsset(graph, options?): CSGRebuildResult`
- `solver.registerWasmSolver(adapter): void`
- `solver.unregisterWasmSolver(id): void`
- `solver.listWasmSolvers(): string[]`
- `solver.createEmptyResult(solverId?): CSGRebuildResult`
- `defaultCSGSolver: CSGSolver`
- `buildCSGMeshAsset(graph, options?): CSGRebuildResult`
- `registerWasmCSGSolver(adapter): void`
- `unregisterWasmCSGSolver(id): void`
- `listWasmCSGSolvers(): string[]`
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
- CSG must preserve `position`, `normal`, and `uv0`.
- CSG may drop `uv1`, `uv2`, `uv3`, vertex colors, skinning, and morph data;
  when dropped, the implementation should emit diagnostics.
- Output triangles must be capped by `maxOutputTriangles` (default `200000`).
- Solver preference `auto` must prefer registered wasm solver and must fall back
to builtin solver when wasm solve fails.
- Each `CSGSolver` instance must maintain an isolated wasm solver registry.
- Module-level helper functions must delegate to `defaultCSGSolver` for backward
compatible behavior.
- `CSGMeshInstance` updates must mark scene transform dirty and must mark spatial
BVH bounds dirty for the instance.
- `IPrimitive.geometryVersion` must be incremented when CSG-generated geometry is
replaced, and render backends must re-upload geometry when version changes.
- `CSGMeshInstance.physicsSync="auto"` should call
`PhysicsSystem.rebuildColliders(target)` after successful CSG rebuild.

## Usage

### Spatial indexing

```ts
import { Scene } from "../src/core/Scene";
import { HybridSpatialIndex } from "../src/spatial/HybridSpatialIndex";

const scene = new Scene();
scene.setSpatialIndexMode("hybrid");

const visible = scene.queryMeshInstancesInFrustum(camera, meshInstances);

const spatial = scene.rebuildSpatialIndex(meshInstances);
const out = [];
spatial.queryFrustumInto(camera.frustum, out);

const hybrid = new HybridSpatialIndex(meshInstances, {
	dynamicBackend: "auto",
	dynamicBVH: { buildStrategy: "median" },
	dynamicOctree: { looseness: 1.5 },
});
hybrid.queryRayDetailedInto(
	{ x: 0, y: 0, z: 5 },
	{ x: 0, y: 0, z: -1 },
	[],
	{ maxDistance: 100, maxResults: 1 },
);

const overlaps = scene
	.rebuildSpatialIndex(meshInstances)
	.queryBoundsInto(
		{
			min: { x: -1, y: -1, z: -1 },
			max: { x: 1, y: 1, z: 1 },
		},
		[]
	);
```

```bash
bun run bench:spatial-index --quick
bun run bench:spatial-index --out spatial-baseline.json
bun run bench:spatial-index --baseline spatial-baseline.json
```

### LOD mesh instances

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

### Constructive solid geometry

```ts
import {
	CSG,
	CSGSolver,
	buildCSGMeshAsset,
	defaultCSGSolver,
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

// Isolated solver instance with its own wasm registry
const isolatedSolver = new CSGSolver();
const graph = CSG.from(left).intersect(right).getGraph();
const isolatedResult = isolatedSolver.buildMeshAsset(graph);

// Legacy helper delegates to the shared default solver
defaultCSGSolver.registerWasmSolver({
	id: "example-wasm",
	solve(request) {
		return buildCSGMeshAsset(request.graph, {
			...request.options,
			solverPreference: "builtin",
		});
	},
});
defaultCSGSolver.unregisterWasmSolver("example-wasm");
```

## Diagnostics

### Spatial indexing

- `Scene.setSpatialIndexMode(mode)` must throw an error when `mode` is not
  `"bvh"` or `"hybrid"`.
- `queryRayDetailedInto` must throw when `direction` is zero-length.
- Diagnostics should compare `bvh` and `hybrid` result sets in tests to confirm
  behavioral parity before changing defaults.
- Spatial benchmark diagnostics may inspect class-private runtime fields to
  report dynamic backend selection, octree resident pressure, node visits, and
  object AABB tests. These diagnostics must not become part of `SpatialIndex3D`.
- `tests/benchmarks/bench_spatial_index.mjs` should remain a manual benchmark and
  must not be discovered by `bun run test`.

### LOD mesh instances

- `LODMeshInstance requires at least one LOD level`:
`levels` is empty or not an array.
- `LODMeshInstance level <index> must provide a MeshAsset`:
`levels[index].mesh` is not a `MeshAsset` instance.

### Constructive solid geometry

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

## Verification

```bash
bun run test:winding
bun tests/static/csg/test_csg_core.mjs
bunx tsc --noEmit
```

## Related Documents

- [Engine architecture](../architecture/engine.md)
- [Physics contract](physics.md)
- [Renderer contract](renderer.md)
