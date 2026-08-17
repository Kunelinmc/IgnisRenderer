# Physics Contract

This document defines physics runtime control, collision filtering, mesh colliders, spatial queries, and debug data ownership.

## Contract

### Runtime control

`PhysicsSystem` must expose the following public methods:

- `setLinearVelocity(target, velocity): void`
- `setAngularVelocity(target, velocity): void`
- `applyForce(target, force): void`
- `applyTorque(target, torque): void`
- `applyImpulse(target, impulse): void`
- `getLinearVelocity(target): IVector3 | null`
- `getAngularVelocity(target): IVector3 | null`
- `getBodyTransform(target): PhysicsTransform | null`
- `isSleeping(target): boolean | null`
- `getBodyStats(worldId): PhysicsBodyStats`
- `rebuildColliders(target): PhysicsColliderHandle[]`
- `removeCollider(collider): void`
- `setColliderSensor(collider, isSensor): void`
- `defineCollisionLayer(name, bitIndex): void`
- `setColliderCollisionFilter(collider, filter): void`
- `setColliderFriction(collider, friction): void`
- `setColliderRestitution(collider, restitution): void`
- `setBodyType(target, type): void`
- `setBodyMass(target, mass): void`
- `setBodyGravityScale(target, scale): void`
- `setBodyLinearDamping(target, value): void`
- `setBodyAngularDamping(target, value): void`
- `wakeUpBody(target): void`
- `destroyJoint(joint): void`
- `destroyCharacterController(controller): void`
- `bindSceneSpatial(scene): void`
- `bindSceneLifecycle(scene, options): () => void`

Contract rules:

- `target` must be one of:
`Node | PhysicsBodyHandle | string (bodyId) | PhysicsEntityId`.
- `target` must resolve to an existing body in an existing world, otherwise the
call must throw.
- All vector inputs (`velocity`, `force`, `torque`, `impulse`) must be passed in
world space as `IVector3`.
- Each dynamics-control call must forward to the active
`IPhysicsEngineAdapter`.
- Each dynamics-control call must mark the world dirty so the next
`step()` / `stepAsync()` is not skipped by sleeping-island optimization.
- Read-back methods must return `PhysicsSystem` cached state, not direct
adapter-native state, so worker adapters have the same synchronous behavior as
in-thread adapters.
- `getLinearVelocity()` and `getAngularVelocity()` must reflect descriptor
initial values, velocity setter calls, and the latest adapter step result.
- `getBodyTransform()` and `isSleeping()` must reflect the latest cached body
state after attach, animation-authority sync, or physics-authority step output.
- `getBodyStats(worldId)` must report cached `bodyCount`, `activeBodies`,
`sleepingBodies`, and `ccdBodies` for an existing world.
- `rebuildColliders(target)` must preserve stored descriptor semantics.
- `removeCollider(collider)` must remove only the selected collider and must not
detach the owning body.
- `destroyJoint(joint)` must remove only the selected joint and must not detach
the connected bodies.
- `destroyCharacterController(controller)` must remove only the selected
controller and must not detach the controlled body.
- Stale joint, controller, and collider handles passed to destruction/removal
methods should be treated as no-ops.
- `setColliderSensor(collider, isSensor)` must update the cached collider
descriptor and forward to `IPhysicsEngineAdapter.setColliderSensor()`.
- `defineCollisionLayer(name, bitIndex)` must register a unique layer name and a
unique bit index from `0` through `15`.
- Unspecified collider collision filters must resolve to group `default` and
`collidesWith: "all"`.
- `setColliderCollisionFilter(collider, filter)` must update the cached
descriptor and forward the encoded adapter collision filter.
- Unknown collision layer names, duplicate layer names, duplicate bit indexes,
and out-of-range bit indexes must throw.
- `setColliderFriction()` and `setColliderRestitution()` must update cached
collider material and forward through `IPhysicsEngineAdapter.setColliderMaterial()`.
- Ammo-backed adapters apply collider material updates to the owning rigid body;
for multi-collider bodies, the last material setter wins.
- Runtime body property setters must update the cached body descriptor, wake the
body, and mark the world dirty.
- `setBodyType()` must preserve existing body, collider, joint, and controller
handles and must update sleeping-island dynamic-body tracking.
- `setBodyType(target, "dynamic")` must throw when the body authority is
`"animation"`.
- `bindSceneLifecycle(scene, options)` must only auto-attach and auto-detach
`PhysicsBodyNode` instances.
- `bindSceneLifecycle()` must default to
`{ attachExisting: true, detachRemoved: true }` and return an unsubscribe
function.
- Reparenting a `PhysicsBodyNode` within the same `Scene` must not detach its
physics body.
- For mesh colliders (`mode: "mesh"`), `rebuildColliders(target)` must recook
only when geometry version/key changes.
- For mesh colliders, transform-only changes must update transform/spatial state
without forcing full triangle recook.
- `bindSceneSpatial(scene)` should bind `Scene.spatial` as the shared mesh
broadphase source for query candidate generation.
- Query filtering may fall back to adapter-native broadphase when bound scene
data is incomplete for tracked physics meshes.

### Oriented box queries

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
- Implementations should use `Matrix3.fromQuaternion` for shared row-major
quaternion-to-matrix conversion.
- Backends may use conservative broadphase approximation for rotated box tests
when exact narrow-phase is unavailable.

### Mesh colliders

Mesh collider descriptors must use `mode: "mesh"` and should provide explicit
runtime intent:

- `mode: "mesh"`
- `sourceNode?: Node`
- `meshPolicy?: "fixed" | "kinematic" | "dynamic"`
- `narrowphase?: "face-bvh" | "proxy"`
- `backendPreference?: "exact" | "approx"`

Normative requirements:

- Triangle cooking for mesh colliders must use local-space geometry.
- Implementations must not require world-space cook for mesh descriptors.
- `meshPolicy` should default from body type when omitted.
- `narrowphase` should default to `"face-bvh"` when omitted.
- `backendPreference` should default by adapter policy:
  - Rapier family: `"exact"`
  - Ammo/Simple family: `"approx"`
- Legacy `mode: "trimesh-cook"` must be translated to `mode: "mesh"` during
  the compatibility window and must emit a deprecation warning.
- Query filtering for mesh colliders should use `Scene.spatial` candidate sets
before adapter narrowphase.
- Query behavior must preserve filter/mask semantics from existing
`PhysicsQueryFilter`.
- Query pipelines must avoid false negatives. Conservative false positives may
be returned and then filtered in narrowphase.
- `rebuildColliders(target)` must recook mesh geometry only when geometry
version/key changes. Transform-only changes must not force full mesh recook.

### Debug visualization data

`PhysicsDebugGeometry` must contain:

- `lines: PhysicsDebugLine[]`
- `points: PhysicsDebugPoint[]`

`PhysicsDebugLine` must include:

- `kind: "collider" | "aabb" | "joint-axis" | "contact-point"`
- `worldId: PhysicsWorldId`
- `from: IVector3`
- `to: IVector3`

`PhysicsDebugLine` may include `bodyId`, `colliderId`, `jointId`, and `color`.

`PhysicsDebugPoint` must include:

- `kind: "contact-point"`
- `worldId: PhysicsWorldId`
- `position: IVector3`

`PhysicsDebugPoint` may include `bodyAId`, `bodyBId`, `normal`, and `color`.

The debug geometry contract must be renderer-agnostic. Renderers may consume the
geometry in a later overlay implementation, but physics modules must not submit
rendering work directly.

## Usage

### Runtime control

```ts
import { Node } from "../src/core/Node";
import { Scene } from "../src/core/Scene";
import { PhysicsSystem } from "../src/physics/PhysicsSystem";

const scene = new Scene();
const physics = new PhysicsSystem();
physics.initSync();
physics.createWorld({ worldId: "main", mode: "variable" });
physics.bindSceneSpatial(scene);

const node = new Node();
const body = physics.attachBody(node, {
	worldId: "main",
	body: { type: "dynamic" },
	authority: "physics",
});

physics.setLinearVelocity(body, { x: 2, y: 0, z: 0 });
physics.applyImpulse(body, { x: 0, y: 1, z: 0 });
physics.defineCollisionLayer("player", 1);
const collider = physics.addCollider(body, {
	mode: "explicit",
	shape: { kind: "sphere", radius: 1 },
	collision: { groups: ["player"], collidesWith: ["default"] },
});
physics.setColliderCollisionFilter(collider, {
	groups: ["player"],
	collidesWith: "all",
});
physics.setBodyGravityScale(body, 0.5);
const velocity = physics.getLinearVelocity(body);
physics.rebuildColliders(body);
physics.step(1 / 60);
const transform = physics.getBodyTransform(body);
```

### Oriented box queries

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

### Mesh colliders

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

### Debug visualization data

```ts
import type { PhysicsDebugGeometry } from "../src/physics/types";

const debugGeometry: PhysicsDebugGeometry = {
	lines: [
		{
			kind: "aabb",
			worldId: "main",
			bodyId: "physicsBody_1",
			from: { x: -1, y: 0, z: 0 },
			to: { x: 1, y: 0, z: 0 },
		},
	],
	points: [],
};
```

## Diagnostics

### Runtime control

- `Physics body "<id>" does not exist`: `target` resolved to a missing body
handle or body id.
- `Node "<id>" is not bound to any physics body`: `target` is a `Node` without
an attached body.
- `Entity "<id>" is not bound to a Node`: `target` is an unresolved ECS entity
id.
- `PhysicsSystem entity target requires setEntityNodeResolver()`: `target` is
`PhysicsEntityId` but no resolver is registered.
- `Physics collider "<id>" does not exist`: collider runtime control target is
missing.
- `Collider friction must be finite`: `setColliderFriction()` received `NaN` or
an infinite value.
- `Collider restitution must be finite`: `setColliderRestitution()` received
`NaN` or an infinite value.
- `Physics collision layer "<name>" is not defined`: a collider filter referenced
an unregistered layer.
- `Physics collision layer bitIndex must be between 0 and 15`: layer
registration used an invalid bit index.
- `Physics body mass must be a finite positive number`: `setBodyMass()` received
`NaN`, an infinite value, zero, or a negative value.

### Oriented box queries

- `Physics query direction must be non-zero`:
`boxCast.direction` magnitude was zero or near-zero.
- `Physics query requires an active world, but no worlds are created`:
query was issued before `createWorld`.
- `Physics query.worldId is required when multiple worlds are active`:
query omitted `worldId` while multiple worlds were present.

### Mesh colliders

- `mesh collider cook failed for node "<id>"`: triangle extraction failed for
`sourceNode`.
- Deprecation diagnostics should include that `mode: "trimesh-cook"` is
translated to `mode: "mesh"` and scheduled for major-version removal.
- If scene-spatial coverage is incomplete for physics mesh colliders, query
pipeline may fall back to adapter broadphase to avoid false negatives.

### Debug visualization data

Debug geometry producers should omit data they cannot compute. Missing
contact-manifold support must produce an empty `points` array instead of
throwing.

## Verification

```bash
bun tests/static/physics/test_physics_system_bindings.mjs
bun tests/static/physics/test_physics_adapter_contract.mjs
bunx tsc --noEmit
```

## Related Documents

- [Engine architecture](../architecture/engine.md)
- [Geometry contract](geometry.md)
- [Renderer contract](renderer.md)
