# Interaction

## Scope

This document defines the built-in interaction contract for selecting,
hovering, clicking, and transform-gizmo manipulation through independent
`Interactable` definitions registered on scene `Node` instances.

## Background

The interaction system is part of `src/interaction/`. Projects must use
`InteractionController` instead of the removed addon-level
`InteractionManager`. `Interactable` is an interaction-owned behavior contract;
it must not be stored as an ECS component and it must not require subclassing
`Node`.

## API/Contract

- `InteractionController` must be attached to a `Scene` and `Camera` before
  pointer input is processed. A `Renderer` must not be required.
- `Interactable` must be registered in `InteractionController.interactables`
  before a node can be hovered, clicked, selected, or selected by drag
  rectangle.
- `InteractableRegistry` must key interaction behavior by `Node` object
  identity. It must not mutate `Node` and must not add ECS components.
- `enabled: false` must disable all interaction for the node.
- `hoverable: false` must prevent hover callbacks and hover state.
- `selectable: false` must prevent click selection and drag selection.
- `priority` must break overlapping hit ties before distance and entity id.
- `selectionMode: "single"` must keep at most one selected entity.
- `selectionMode: "multiple"` must allow drag selection and shift-click
  selection changes.
- Object callbacks may be stored in `Interactable` as runtime functions:
  `onHoverEnter`, `onHoverLeave`, `onSelect`, `onDeselect`, and `onClick`.
- `InteractionController.updatePointer()` must return an `InteractionState`
  snapshot and must not request or submit rendering work.
- `InteractionController.getState()` must return the current selection, hover,
  transform-gizmo, and drag-rectangle state.

## Usage

```ts
import {
	InteractionController,
	Scene,
	Camera,
	type Interactable,
} from "ignisrenderer";

const controller = new InteractionController({
	selectionMode: "multiple",
});
controller.attach(scene, camera);

const interactable: Interactable = {
	priority: 10,
	onClick: ({ entityId }) => {
		console.log("clicked", entityId);
	},
};

controller.interactables.set(meshInstance, interactable);

canvas.addEventListener("pointerdown", (event) => {
	const interaction = controller.updatePointer({
		type: "down",
		button: event.button,
		screenX: event.clientX,
		screenY: event.clientY,
		shiftKey: event.shiftKey,
		viewportWidth: canvas.clientWidth,
		viewportHeight: canvas.clientHeight,
	});
	console.log(interaction.selectedEntityIds);
});
```

```bash
bun tests/static/scene/test_interaction_controller_selection.mjs
```

## Errors & Diagnostics

- If a visible mesh cannot be selected, verify that its entity has
  a registered `Interactable` in `InteractionController.interactables`.
- If callbacks do not run, verify that `enabled`, `hoverable`, and
  `selectable` are not set to `false` for the intended interaction.
- Verify that `InteractionController.attach()` was called and that the selected
  entity id appears in the `InteractionState.selectedEntityIds` returned by
  `updatePointer()` or `getState()`.
- If overlapping meshes select the wrong entity, inspect `priority`; higher
  values must win before distance.

## Compatibility / Breaking Changes

- `InteractionManager` was removed.
- Imports from `src/addons/InteractionManager` must be replaced with
  `src/interaction/InteractionController`.
- ECS `Interactable` components are removed. Code that used
  `scene.ecs.setComponent(entityId, "Interactable", value)` must register the
  target node through `controller.interactables.set(node, value)`.
- `InteractableComponent` remains a type alias for `Interactable` for source
  compatibility, but it is no longer part of `ECSComponentMap`.
- Meshes without a registered `Interactable` are not selected by default.
- Code that used `getSelection()` may continue reading the primary selected
  entity. Code that needs multi-selection must use `getSelectedEntities()`.
- `InteractionController.attach(renderer, scene, camera)` has been replaced by
  `InteractionController.attach(scene, camera)`.
- `INTERACTION_TRANSIENT_STATE_KEY` and the interaction pipeline transient
  types have been removed. Applications must consume `InteractionState`
  directly and decide when and how to render interaction visuals.
