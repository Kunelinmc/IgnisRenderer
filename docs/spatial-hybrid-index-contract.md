# Spatial Hybrid Index Contract

## Scope
This document defines the runtime contract for `Scene` spatial indexing mode
selection and the hybrid `BVH + LooseOctree` query path used by camera frustum
culling and fallback ray picking.

## Background
The engine previously used a single `BVH` implementation for all mesh instances.
Scenes with high-frequency transform updates should avoid unnecessary static-tree
pressure, so a hybrid strategy is introduced.

## API/Contract
- `Scene.spatialIndexMode` must accept only `"bvh"` or `"hybrid"`.
- `Scene.setSpatialIndexMode(mode)` must reset the current spatial index and
  must rebuild lazily on the next frustum/ray query.
- `Scene.rebuildSpatialIndex(meshInstances)` must preserve incremental behavior:
  structural changes must call `upsert`/`remove`, and transform or dynamic-state
  changes must call `markDirty`.
- `SpatialIndex3D.queryFrustum(frustum, options?)` must return mesh-instance
  candidates whose world-space bounds overlap the frustum.
- `SpatialIndex3D.queryRayDetailed(origin, direction, options?)` must return
  candidates ordered by ascending `distance`, then by `entityId`, then by
  `meshInstance.id`.
- `HybridSpatialIndex` must classify dynamic instances by
  `isDynamicSpatialMeshInstance(meshInstance)` and route:
  - static bucket: `BVH`
  - dynamic bucket: `LooseOctree`
- `HybridSpatialIndex` must de-duplicate merged results by instance identity.

## Usage
```ts
import { Scene } from "../src/core/Scene";

const scene = new Scene();
scene.setSpatialIndexMode("hybrid");

// Existing query APIs stay unchanged.
const visible = scene.queryMeshInstancesInFrustum(camera, meshInstances);
```

## Errors & Diagnostics
- `Scene.setSpatialIndexMode(mode)` must throw an error when `mode` is not
  `"bvh"` or `"hybrid"`.
- `queryRayDetailed` must throw when `direction` is zero-length.
- Diagnostics should compare `bvh` and `hybrid` result sets in tests to confirm
  behavioral parity before changing defaults.

## Compatibility / Breaking Changes
This change is backward compatible:
- default mode remains `"bvh"`;
- existing public query method signatures are unchanged;
- consumers may opt in to `"hybrid"` without changing call sites.
