# Interaction

Use the interaction APIs for picking, hover, selection, drag selection, and transform controls while retaining application ownership of event wiring and visuals.

## Overview

Use the interaction API to add hover, click, selection, drag selection, and
transform controls to scene nodes. The main entry point is
`InteractionController`, which connects browser input to a `Scene` and
`Camera`.

This API handles picking and interaction state. Your application remains in
control of event wiring, selection visuals, and when to render the next frame.

A basic interaction setup has three parts:

1. Create an `InteractionController` and attach it to the active `Scene` and
   `Camera`.
2. Register each interactive node with
   `controller.interactables.set(node, options)`.
3. Forward pointer or keyboard input through `updatePointer()`.

Only registered nodes participate in picking. Registration does not modify the
node, so the same scene objects can be used with or without the interaction
system. A `Renderer` is not required; you can also attach an optional
`PhysicsSystem` when physics-based picking is available.

## API

### Interaction workflow

#### Creating and attaching a controller

`InteractionController` accepts these commonly used options:

- `selectionMode`: `"single"` selects at most one node. `"multiple"` also
  enables shift-click toggling and drag-rectangle selection.
- `maxRayDistance`: limits how far picking rays can travel.
- `interactables`: supplies an existing `InteractableRegistry` when several
  systems need to share registrations.

Call `attach(scene, camera, physicsSystem?)` before forwarding input. Calling
`detach()` or `dispose()` clears hover, selection, drag, and gizmo state.

#### Registering nodes

An `Interactable` can include:

- `enabled`: set to `false` to temporarily disable all interaction.
- `hoverable`: set to `false` to ignore hover picking.
- `selectable`: set to `false` to ignore click and drag selection.
- `priority`: gives one overlapping node precedence over another. Higher values
  are considered first, followed by hit distance.
- `onHoverEnter`, `onHoverLeave`, `onSelect`, `onDeselect`, and `onClick`:
  callbacks scoped to that node.

Each callback receives the node, its entity id, the interaction phase, the
current selection, and the latest pointer state.

#### Forwarding input and reading state

`updatePointer()` accepts `"move"`, `"down"`, `"up"`, `"leave"`, `"cancel"`,
and `"key"` events. Pointer coordinates are viewport-local pixels; use the same
coordinate space for `screenX`, `screenY`, `viewportWidth`, and
`viewportHeight`.

Every call returns an `InteractionState` snapshot containing:

- `selectedEntityIds`
- `hoveredEntityId`
- the active `gizmo`, if any
- the active `dragRect`, if any

Use `getState()` to read the same information without processing input.
`getSelection()` returns the primary selected entity id, while
`getSelectedEntities()` returns the complete selection.

You can also subscribe to `hoverChanged`, `selectionChanged`, `click`,
`transformCommitted`, and `transformCancelled` with `controller.on()`.

For transform controls, forward keyboard events after selecting a node:
`G`, `R`, and `S` begin translate, rotate, and scale; `X`, `Y`, and `Z`
constrain an axis; holding Shift constrains the corresponding plane. `Enter`
or a left click commits the transform, while `Escape` or a right click cancels
it. `Q` toggles world/local space, and `.` toggles the transform pivot.

## Usage

### Interaction workflow

```ts
import {
	InteractionController,
	type Interactable,
} from "ignisrenderer";

const controller = new InteractionController({
	selectionMode: "multiple",
});
controller.attach(scene, camera);

const interactable: Interactable = {
	priority: 10,
	onClick: ({ node }) => {
		console.log("Clicked", node);
	},
};
controller.interactables.set(meshInstance, interactable);

type PointerInputType = "move" | "down" | "up" | "leave" | "cancel";

function forwardPointer(event: PointerEvent, type: PointerInputType): void {
	const bounds = canvas.getBoundingClientRect();
	const scaleX = canvas.width / bounds.width;
	const scaleY = canvas.height / bounds.height;

	const state = controller.updatePointer({
		type,
		button: event.button,
		screenX: (event.clientX - bounds.left) * scaleX,
		screenY: (event.clientY - bounds.top) * scaleY,
		shiftKey: event.shiftKey,
		ctrlKey: event.ctrlKey,
		metaKey: event.metaKey,
		altKey: event.altKey,
		viewportWidth: canvas.width,
		viewportHeight: canvas.height,
	});

	console.log(state.selectedEntityIds);
}

canvas.addEventListener("pointermove", (event) => {
	forwardPointer(event, "move");
});
canvas.addEventListener("pointerdown", (event) => {
	forwardPointer(event, "down");
});
canvas.addEventListener("pointerup", (event) => {
	forwardPointer(event, "up");
});

window.addEventListener("keydown", (event) => {
	controller.updatePointer({
		type: "key",
		key: event.key,
		shiftKey: event.shiftKey,
	});
});
```

If your application renders only on demand, request a new frame after an
interaction changes a visual state. `InteractionController` does not submit
rendering work itself.

## Troubleshooting

### Interaction workflow

- Nothing can be selected: confirm that `attach()` uses the active scene and
  camera, and that the node is registered in `controller.interactables`.
- The pointer selects the wrong location: convert browser coordinates to
  canvas-local coordinates and account for CSS-to-canvas scaling.
- Hover works but selection does not: check whether `selectable` is `false`.
  If neither works, also check `enabled` and `hoverable`.
- An overlapping node wins unexpectedly: compare `priority` values. Nodes with
  equal priority are ordered by hit distance, then entity id.
- Drag selection does not add nodes: use `selectionMode: "multiple"` and make
  sure the drag rectangle spans at least a few viewport pixels.
- State changes but the display does not: request or render another frame in
  your application.

## Compatibility

### Interaction workflow

Current code should use `InteractionController` and register nodes through
`controller.interactables`. The earlier `InteractionManager` API and ECS-based
`Interactable` components are no longer supported.

When migrating older code:

- Replace `attach(renderer, scene, camera)` with `attach(scene, camera)`.
- Replace ECS component registration with
  `controller.interactables.set(node, interactable)`.
- Keep using `getSelection()` for the primary selection, or switch to
  `getSelectedEntities()` for multi-selection.
- Read `InteractionState` directly to draw outlines, drag rectangles, or gizmo
  UI.

`InteractableComponent` remains as a deprecated alias for `Interactable`, but
new code should use `Interactable`.

## Related Documents

- [Renderer](renderer.md)
- [Geometry contract](../contracts/geometry.md)
