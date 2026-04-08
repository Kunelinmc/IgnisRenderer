# Physics Query OBB Rotation Contract

## Scope
This document defines the `rotation` input contract added to
`PhysicsBoxCastQuery` and `PhysicsOverlapBoxQuery`.

## Background
Before this change, box-based physics queries only accepted `center` and
`halfExtents`, so callers could only express axis-aligned query boxes.

## API/Contract
The following query contracts are updated:

- `PhysicsBoxCastQuery`
- `PhysicsOverlapBoxQuery`

Input contract additions:

- `rotation` may be provided as `QuaternionTuple` in `[x, y, z, w]` order.
- `rotation` must be interpreted as query-box orientation in world space.
- If `rotation` is omitted, implementations must use identity
`[0, 0, 0, 1]`.
- Implementations must normalize finite quaternion input before evaluation.
- Non-finite quaternion components should be sanitized to identity-compatible
defaults.

Behavioral contract:

- `boxCast` must incorporate `rotation` when constructing box sweep volume.
- `overlapBox` must incorporate `rotation` when constructing overlap volume.
- Backends may use conservative broadphase approximation for rotated box tests
when exact narrow-phase is unavailable.

## Usage
```ts
import { PhysicsSystem } from "../src/physics/PhysicsSystem";

const physics = new PhysicsSystem();
physics.initSync();
physics.createWorld({ worldId: "main" });

const halfAngle = Math.PI / 8;
const rotation: [number, number, number, number] = [
	0,
	0,
	Math.sin(halfAngle),
	Math.cos(halfAngle),
];

const hit = physics.boxCast({
	worldId: "main",
	center: { x: 0, y: 0, z: 0 },
	halfExtents: { x: 1, y: 0.2, z: 0.2 },
	rotation,
	direction: { x: 0, y: 0, z: 1 },
	maxDistance: 20,
});
```

## Errors & Diagnostics
- `Physics query direction must be non-zero`:
`boxCast.direction` magnitude was zero or near-zero.
- `Physics query requires an active world, but no worlds are created`:
query was issued before `createWorld`.
- `Physics query.worldId is required when multiple worlds are active`:
query omitted `worldId` while multiple worlds were present.

## Compatibility / Breaking Changes
This change is additive and non-breaking:

- Existing query call sites without `rotation` remain valid.
- Runtime behavior for axis-aligned query boxes is unchanged.
