# PhysicsSystem Runtime Body Control API

## Scope
This document defines runtime body-control and collider-rebuild contracts for
`PhysicsSystem`, including `Scene` spatial-index binding for mesh query
candidate generation.

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
- `rebuildColliders(target): PhysicsColliderHandle[]`
- `bindSceneSpatial(scene): void`

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
- `rebuildColliders(target)` must preserve stored descriptor semantics.
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
physics.rebuildColliders(body);
physics.step(1 / 60);
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

## Compatibility / Breaking Changes
This contract is partially breaking at mesh-descriptor level and should be
adopted with migration guidance from `Mesh Collision V2`:

- `mode: "mesh"` is the primary mesh-collider entry.
- legacy `mode: "trimesh-cook"` may be translated for one compatibility version
with deprecation warning.
