# Spatial Index Contract

## Scope
This document defines the runtime contract for `Scene` spatial indexing,
`SpatialIndex3D` query APIs, and the hybrid static/dynamic query path used by
camera frustum culling, physics mesh candidate generation, and fallback ray
picking.

## Background
The engine previously used a single `BVH` implementation for all mesh instances.
Scenes with high-frequency transform updates should avoid unnecessary static-tree
pressure, so a hybrid strategy is available. Dynamic instances may be sparse,
clustered, or heavily overlapping, so the dynamic bucket must choose an
acceleration structure that preserves broad-phase selectivity for the current
distribution. Query APIs use caller-owned output arrays to reduce per-frame
allocations in renderer, physics, and interaction hot paths.

## API/Contract
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
- `LooseOctree.queryRayDetailedInto` should tighten its traversal distance when
  finite `maxResults` already has enough hits.

## Usage
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

## Errors & Diagnostics
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

## Compatibility / Breaking Changes
This change is breaking for TypeScript consumers that type against
`SpatialIndex3D`:
- `queryFrustum`, `queryBounds`, `queryRay`, and `queryRayDetailed` are no longer
  interface methods;
- callers should use `queryFrustumInto`, `queryBoundsInto`, and
  `queryRayDetailedInto`;
- default `Scene.spatialIndexMode` remains `"bvh"`;
- class-level compatibility wrappers remain available on `BVH`, `LooseOctree`,
  and `HybridSpatialIndex` for direct callers.
