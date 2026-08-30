# ECS Projection Contract

This document defines the optional projection from a `Scene` into a
user-owned `experimentalECS.ECSWorld`.

## Construction and Ownership

- `new ECSWorld(scene)` must synchronously project the existing scene graph.
- `Scene`, `Renderer`, and `FrameCoordinator` must not create, retain, discover,
  or synchronize an `ECSWorld`.
- Entity identifiers are local to one world. Multiple worlds may project the
  same scene without sharing identifiers or component state.
- `ECSWorld.destroy()` must be idempotent, unsubscribe every scene listener,
  and prevent later scene changes from mutating the destroyed world.

## Incremental Projection

- Structure changes must create, remove, or reparent only the affected
  scene-backed entities. Unaffected entity identifiers must remain stable.
- Metadata and transform changes must update only nodes published by the scene
  change journal. An unchanged transform boundary must perform no component
  writes.
- `PathBinding` must be rebuilt only for initial projection, rename, attach, or
  reparent operations and their affected descendants.
- Existing component objects and matrix storage must be reused when values
  change.
- Query cache validity must depend on entity/component membership, not ordinary
  component value updates.

## Component Authority

- `Name`, `Visibility`, and `LocalTransform` are writable mirrored components.
  `setComponent()` must update the mapped `Node`, invalidate its scene when the
  value changes, and avoid recursive projection writes.
- `WorldTransform`, `Hierarchy`, `PathBinding`, `NodeRef`, and `NodeKind` are
  derived from the scene graph and must reject external mutation or removal on
  scene-backed entities.
- Scene-backed entities must not be destroyed through `ECSWorld.destroyEntity`;
  callers must detach the owning node through the scene graph.
- `SkeletonJoint` and entities created directly through `createEntity()` remain
  ECS-owned.

## Public Identity

- `Node.id` is the stable application and interaction identity.
- `Node` must not expose an intrinsic entity identifier.
- Applications resolve the optional mapping through
  `ecs.getEntityByNode(node)` and `ecs.getNodeByEntity(entity)`.

## Verification

```bash
bun tests/static/scene/test_scene_ecs_sync.mjs
bun run typecheck
```

## Related Documents

- [Engine architecture](../architecture/engine.md)
- [Rendering architecture](../architecture/rendering.md)
- [Migration guidance](../migrations/scene-ecs-decoupling.md)
