# Interaction

## Scope

This document defines the built-in interaction contract for selecting,
hovering, clicking, and transform-gizmo manipulation through ECS
`Interactable` components.

## Background

The interaction system is part of `src/interaction/`. Projects must use
`InteractionController` instead of the removed addon-level
`InteractionManager`.

## API/Contract

- `InteractionController` must be attached to a `Renderer`, `Scene`, and
  `Camera` before pointer input is processed.
- `InteractableComponent` must be present on an entity before it can be
  hovered, clicked, selected, or selected by drag rectangle.
- `enabled: false` must disable all interaction for the entity.
- `hoverable: false` must prevent hover callbacks and hover state.
- `selectable: false` must prevent click selection and drag selection.
- `priority` must break overlapping hit ties before distance and entity id.
- `selectionMode: "single"` must keep at most one selected entity.
- `selectionMode: "multiple"` must allow drag selection and shift-click
  selection changes.
- Object callbacks may be stored in `InteractableComponent` as runtime
  functions: `onHoverEnter`, `onHoverLeave`, `onSelect`, `onDeselect`, and
  `onClick`.
- `InteractionController` must write `INTERACTION_TRANSIENT_STATE_KEY` during
  frame transient contribution so the built-in `interaction-outline` pass can
  render selected entities.

## Usage

```ts
import {
	InteractionController,
	Scene,
	Camera,
	Renderer,
	type InteractableComponent,
} from "ignisrenderer";

const controller = new InteractionController({
	selectionMode: "multiple",
});
controller.attach(renderer, scene, camera);

const interactable: InteractableComponent = {
	priority: 10,
	onClick: ({ entityId }) => {
		console.log("clicked", entityId);
	},
};

if (meshInstance.entityId !== null) {
	scene.ecs.setComponent(meshInstance.entityId, "Interactable", interactable);
}

canvas.addEventListener("pointerdown", (event) => {
	controller.updatePointer({
		type: "down",
		button: event.button,
		screenX: event.clientX,
		screenY: event.clientY,
		shiftKey: event.shiftKey,
		viewportWidth: canvas.clientWidth,
		viewportHeight: canvas.clientHeight,
	});
});
```

```bash
bun tests/static/scene/test_interaction_controller_selection.mjs
```

## Errors & Diagnostics

- If a visible mesh cannot be selected, verify that its entity has
  `Interactable`.
- If callbacks do not run, verify that `enabled`, `hoverable`, and
  `selectable` are not set to `false` for the intended interaction.
- If no outline is visible, verify that `InteractionController.attach()` was
  called and that the selected entity id appears in
  `INTERACTION_TRANSIENT_STATE_KEY.selectedEntityIds`.
- If overlapping meshes select the wrong entity, inspect `priority`; higher
  values must win before distance.

## Compatibility / Breaking Changes

- `InteractionManager` was removed.
- Imports from `src/addons/InteractionManager` must be replaced with
  `src/interaction/InteractionController`.
- Meshes without `Interactable` are no longer selected by default.
- Code that used `getSelection()` may continue reading the primary selected
  entity. Code that needs multi-selection must use `getSelectedEntities()`.
