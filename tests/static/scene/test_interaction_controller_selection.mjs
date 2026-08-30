import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { MeshFactory } from "../../../src/meshes/MeshFactory.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { InteractionController } from "../../../src/interaction/InteractionController.ts";

function createScene() {
	const scene = new Scene();
	const camera = new Camera();
	scene.add(camera);
	return { scene, camera };
}

function pointer(type, x, y, extra = {}) {
	return {
		type,
		screenX: x,
		screenY: y,
		viewportWidth: 200,
		viewportHeight: 200,
		...extra,
	};
}

function sync(scene, camera) {
	scene.updateWorldMatrices();
	camera.updateMatrices();
}

function runDefaultAndCallbackTests() {
	const { scene, camera } = createScene();
	const box = MeshFactory.createBox(
		{ x: 0, y: 0, z: -5 },
		2,
		2,
		2,
		new PBRMaterial()
	);
	scene.add(box);
	sync(scene, camera);

	const controller = new InteractionController();
	controller.attach(scene, camera, null);

	controller.updatePointer(pointer("down", 99.5, 99.5, { button: 0 }));
	controller.updatePointer(pointer("up", 99.5, 99.5, { button: 0 }));
	assert.equal(controller.getSelection(), null);

	controller.interactables.set(box, {
		selectable: false,
	});
	controller.updatePointer(pointer("down", 99.5, 99.5, { button: 0 }));
	controller.updatePointer(pointer("up", 99.5, 99.5, { button: 0 }));
	assert.equal(controller.getSelection(), null);

	controller.interactables.set(box, {
		enabled: false,
	});
	controller.updatePointer(pointer("down", 99.5, 99.5, { button: 0 }));
	controller.updatePointer(pointer("up", 99.5, 99.5, { button: 0 }));
	assert.equal(controller.getSelection(), null);

	const callbacks = [];
	controller.interactables.set(box, {
		onHoverEnter: (context) => callbacks.push(context.phase),
		onHoverLeave: (context) => callbacks.push(context.phase),
		onSelect: (context) => callbacks.push(context.phase),
		onDeselect: (context) => callbacks.push(context.phase),
		onClick: (context) => callbacks.push(context.phase),
	});

	controller.updatePointer(pointer("move", 99.5, 99.5));
	const interactionState = controller.updatePointer(
		pointer("down", 99.5, 99.5, { button: 0 })
	);
	assert.equal(controller.getSelection(), box);
	assert.deepEqual(interactionState.selectedNodes, [box]);
	assert.equal(interactionState.hoveredNode, box);
	assert.deepEqual(controller.getState(), interactionState);

	controller.updatePointer(pointer("move", 0, 0));
	controller.updatePointer(pointer("down", 0, 0, { button: 0 }));
	controller.updatePointer(pointer("up", 0, 0, { button: 0 }));
	assert.equal(controller.getSelection(), null);
	assert.deepEqual(callbacks, [
		"hover-enter",
		"select",
		"click",
		"hover-leave",
		"deselect",
	]);

	controller.detach();
	assert.deepEqual(controller.getState(), {
		selectedNodes: [],
		hoveredNode: null,
		gizmo: null,
		dragRect: null,
	});
}

function runPriorityTest() {
	const { scene, camera } = createScene();
	const low = MeshFactory.createBox(
		{ x: 0, y: 0, z: -5 },
		2,
		2,
		2,
		new PBRMaterial()
	);
	const high = MeshFactory.createBox(
		{ x: 0, y: 0, z: -5 },
		2,
		2,
		2,
		new PBRMaterial()
	);
	scene.add(low);
	scene.add(high);
	sync(scene, camera);

	const controller = new InteractionController();
	controller.interactables.set(low, { priority: 0 });
	controller.interactables.set(high, { priority: 10 });
	controller.attach(scene, camera, null);
	controller.updatePointer(pointer("down", 99.5, 99.5, { button: 0 }));
	assert.equal(controller.getSelection(), high);
}

function runMultipleSelectionTest() {
	const { scene, camera } = createScene();
	const left = MeshFactory.createBox(
		{ x: -1.5, y: 0, z: -5 },
		1,
		1,
		1,
		new PBRMaterial()
	);
	const right = MeshFactory.createBox(
		{ x: 1.5, y: 0, z: -5 },
		1,
		1,
		1,
		new PBRMaterial()
	);
	scene.add(left);
	scene.add(right);
	sync(scene, camera);

	const controller = new InteractionController({ selectionMode: "multiple" });
	controller.interactables.set(left, {});
	controller.interactables.set(right, {});
	controller.attach(scene, camera, null);
	controller.updatePointer(pointer("down", 0, 0, { button: 0 }));
	controller.updatePointer(pointer("move", 199, 199));
	controller.updatePointer(pointer("up", 199, 199, { button: 0 }));

	const selected = controller.getSelectedNodes().slice().sort((a, b) =>
		a.id.localeCompare(b.id)
	);
	assert.deepEqual(selected, [left, right].sort((a, b) => a.id.localeCompare(b.id)));
	assert.equal(controller.getSelection(), selected[0]);
}

function runPhysicsSceneBoundaryTest() {
	const { scene, camera } = createScene();
	const otherScene = createScene().scene;
	const oldBox = MeshFactory.createBox(
		{ x: 0, y: 0, z: -5 },
		2,
		2,
		2,
		new PBRMaterial(),
	);
	otherScene.add(oldBox);
	const physics = {
		raycastAll: () => [{ bodyId: "old-scene-body", distance: 1 }],
		resolveHitNode: () => oldBox,
	};

	const controller = new InteractionController();
	controller.interactables.set(oldBox, {});
	controller.attach(scene, camera, physics);
	controller.updatePointer(pointer("down", 100, 100, { button: 0 }));

	assert.equal(
		controller.getSelection(),
		null,
		"A physics hit from another scene must not be selectable",
	);
}

runDefaultAndCallbackTests();
runPriorityTest();
runMultipleSelectionTest();
runPhysicsSceneBoundaryTest();

console.log("InteractionController selection tests passed");
