# PhysicsSystem Runtime Body Control API

## Scope
This document defines runtime body-control, named collision-filter, collider
rebuild, and opt-in `Scene` lifecycle contracts for `PhysicsSystem`.

## Background
`PhysicsSystem` controls simulation wakeup and step-skipping behavior at system
level. Direct adapter calls may bypass these invariants. Mesh-collider rebuild
behavior also requires explicit rules so transform updates do not trigger
high-cost geometry recook.

## API/Contract
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

## Usage
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

## Errors & Diagnostics
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

## Compatibility / Breaking Changes
This contract is breaking:

- `setCollisionMask()` has been removed from `PhysicsSystem` and replaced by
`setColliderCollisionFilter()`.
- Adapter and worker commands no longer expose `setCollisionMask`; they expose
`setColliderCollisionFilter` with an internal encoded filter.
- `mode: "mesh"` remains the primary mesh-collider entry.
