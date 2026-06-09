# Spatial Index Contract

## Scope
This document defines the runtime contract for `Scene` spatial indexing,
`SpatialIndex3D` query APIs, and the hybrid `BVH + LooseOctree` query path used
by camera frustum culling, physics mesh candidate generation, and fallback ray
picking.

## Background
The engine previously used a single `BVH` implementation for all mesh instances.
Scenes with high-frequency transform updates should avoid unnecessary static-tree
pressure, so a hybrid strategy is available. Query APIs use caller-owned output
arrays to reduce per-frame allocations in renderer, physics, and interaction hot
paths.

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
  `isDynamicSpatialMeshInstance(meshInstance)` and route:
  - static bucket: `BVH`
  - dynamic bucket: `LooseOctree`
- `HybridSpatialIndex` must maintain exclusive bucket membership during
  `upsert`, `remove`, and `markDirty`; query merging may rely on that exclusivity
  and must not require per-query identity de-duplication.
- `LooseOctree.markDirty(meshInstance)` should update the stored bounds in place
  when the updated AABB remains inside the current loose node.

## Usage
```ts
import { Scene } from "../src/core/Scene";

const scene = new Scene();
scene.setSpatialIndexMode("hybrid");

const visible = scene.queryMeshInstancesInFrustum(camera, meshInstances);

const spatial = scene.rebuildSpatialIndex(meshInstances);
const out = [];
spatial.queryFrustumInto(camera.frustum, out);

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
