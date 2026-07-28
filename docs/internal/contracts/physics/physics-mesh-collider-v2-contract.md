# Physics Mesh Collider V2 Contract

## Scope
This document defines the `MeshColliderDescriptorV2` contract, compatibility
rules for legacy mesh descriptors, and spatial-query integration requirements
for high-polygon mesh collisions.

## Background
Legacy mesh-collider paths rely on generic trimesh cooking behavior and may
cause avoidable CPU cost in scenes with high triangle counts. V2 introduces
explicit policy fields, local-space cook, and `Scene.spatial` candidate
generation so query workloads avoid global collider scans.

## API/Contract
Mesh collider descriptors must use `mode: "mesh"` and should provide explicit
runtime intent:

- `mode: "mesh"`
- `sourceNode?: Node`
- `meshPolicy?: "fixed" | "kinematic" | "dynamic"`
- `narrowphase?: "face-bvh" | "proxy"`
- `backendPreference?: "exact" | "approx"`

Normative requirements:

- Triangle cooking for V2 mesh colliders must use local-space geometry.
- Implementations must not require world-space cook for V2 descriptors.
- `meshPolicy` should default from body type when omitted.
- `narrowphase` should default to `"face-bvh"` when omitted.
- `backendPreference` should default by adapter policy:
  - Rapier family: `"exact"`
  - Ammo/Simple family: `"approx"`
- Legacy `mode: "trimesh-cook"` must be translated to V2 (`mode: "mesh"`) for
one compatibility version and must emit deprecation warning.
- Query filtering for mesh colliders should use `Scene.spatial` candidate sets
before adapter narrowphase.
- Query behavior must preserve filter/mask semantics from existing
`PhysicsQueryFilter`.
- Query pipelines must avoid false negatives. Conservative false positives may
be returned and then filtered in narrowphase.
- `rebuildColliders(target)` must recook mesh geometry only when geometry
version/key changes. Transform-only changes must not force full mesh recook.

## Usage
```ts
import { PhysicsSystem } from "../src/physics/PhysicsSystem";

physics.addCollider(body, {
	mode: "mesh",
	sourceNode: meshInstance,
	meshPolicy: "fixed",
	narrowphase: "face-bvh",
	backendPreference: "exact",
});
```

Compatibility example (`trimesh-cook` translation path):

```ts
physics.addCollider(body, {
	mode: "trimesh-cook",
	sourceNode: meshInstance,
});
// Translated internally to mode: "mesh" with deprecation warning.
```

## Errors & Diagnostics
- `mesh collider cook failed for node "<id>"`: triangle extraction failed for
`sourceNode`.
- Deprecation diagnostics should include that `mode: "trimesh-cook"` is
translated to `mode: "mesh"` and scheduled for major-version removal.
- If scene-spatial coverage is incomplete for physics mesh colliders, query
pipeline may fall back to adapter broadphase to avoid false negatives.

## Compatibility / Breaking Changes
This contract introduces moderate breaking changes with a compatibility window:

- New primary descriptor is `MeshColliderDescriptorV2` (`mode: "mesh"`).
- Legacy `mode: "trimesh-cook"` remains temporarily supported through automatic
translation and warning.
- Next major version may remove `trimesh-cook` support.
