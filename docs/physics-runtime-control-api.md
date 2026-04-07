# PhysicsSystem Runtime Body Control API

## Scope
This document defines the runtime rigid-body control methods exposed by
`PhysicsSystem` for velocity and external dynamics inputs.

## Background
Before this contract, body runtime controls existed at the adapter layer but were
not exposed through `PhysicsSystem`. Callers had to bypass `PhysicsSystem`, which
could desynchronize with system-level step skipping and wakeup behavior.

## API/Contract
`PhysicsSystem` now exposes the following public methods:

- `setLinearVelocity(target, velocity): void`
- `setAngularVelocity(target, velocity): void`
- `applyForce(target, force): void`
- `applyTorque(target, torque): void`
- `applyImpulse(target, impulse): void`

Contracts:

- `target` must be one of:
`Node | PhysicsBodyHandle | string (bodyId) | PhysicsEntityId`.
- `target` must resolve to an existing body in an existing world, otherwise the
call must throw.
- All vector inputs (`velocity`, `force`, `torque`, `impulse`) must be passed in
world space as `IVector3`.
- Each call must forward to the active `IPhysicsEngineAdapter`.
- Each call must mark the world as dirty for stepping, so the next
`PhysicsSystem.step()` or `PhysicsSystem.stepAsync()` must not be skipped by
sleeping-island optimization.

## Usage
```ts
import { Node } from "../src/core/Node";
import { PhysicsSystem } from "../src/physics/PhysicsSystem";

const physics = new PhysicsSystem();
physics.initSync();
physics.createWorld({ worldId: "main", mode: "variable" });

const node = new Node();
const body = physics.attachBody(node, {
	worldId: "main",
	body: { type: "dynamic" },
	authority: "physics",
});

physics.setLinearVelocity(body, { x: 2, y: 0, z: 0 });
physics.applyImpulse(body, { x: 0, y: 1, z: 0 });
physics.step(1 / 60);
```

## Errors & Diagnostics
- `Physics body "<id>" does not exist`:
`target` resolved to a missing body handle or body id.
- `Node "<id>" is not bound to any physics body`:
`target` is a `Node` without an attached body.
- `Entity "<id>" is not bound to a Node`:
`target` is an unresolved ECS entity id.
- `PhysicsSystem entity target requires setEntityNodeResolver()`:
`target` is `PhysicsEntityId` but no resolver is registered.

## Compatibility / Breaking Changes
This change is additive and non-breaking. Existing APIs and adapter contracts
remain unchanged.
