# Scene and ECS Decoupling

`Scene` no longer creates an ECS world or assigns entity identifiers to nodes.
Rendering, interaction, animation, physics, and prefab instantiation now use
scene nodes directly.

## Optional ECS Projection

Before:

```ts
const scene = new Scene();
const entity = node.entityId;
const world = scene.ecs;
```

After:

```ts
const scene = new Scene();
const world = new experimentalECS.ECSWorld(scene);
const entity = world.getEntityByNode(node);
```

The application owns `world` and must call `world.destroy()` when the
projection is no longer required.

## Node-First APIs

- Interaction selection and events use `Node` values.
- Animation bindings use `AnimationMixer.bindNode()`.
- Physics runtime targets use `Node`, body handles, or body ids.
- `NodePrefab.instantiate(scene)` returns the cloned root `Node`; applications
  with an ECS world may resolve its entity after instantiation.

## Related Documents

- [ECS projection contract](../contracts/ecs.md)
- [Engine architecture](../architecture/engine.md)
- [Interaction guide](../public/interaction.md)
