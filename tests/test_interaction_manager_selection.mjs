import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Scene } from "../src/core/Scene.ts";
import { MeshFactory } from "../src/meshes/MeshFactory.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import {
	INTERACTION_TRANSIENT_STATE_KEY,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	ANIMATION_SIM_DELTA_TIME_MS_KEY,
} from "../src/pipeline/types.ts";
import { InteractionManager } from "../src/addons/InteractionManager.ts";

function run() {
	const scene = new Scene();
	const camera = new Camera();
	scene.add(camera);
	const box = MeshFactory.createBox(
		{ x: 0, y: 0, z: -5 },
		2,
		2,
		2,
		new PBRMaterial()
	);
	scene.add(box);
	scene.updateWorldMatrices();
	camera.updateMatrices();

	const contributors = new Set();
	const fakeRenderer = {
		scene,
		camera,
		requestRender: () => {},
		registerFrameTransientContributor: (contributor) => contributors.add(contributor),
		unregisterFrameTransientContributor: (contributor) => contributors.delete(contributor),
	};

	const manager = new InteractionManager();
	manager.attach(fakeRenderer, scene, camera, null);

	manager.updatePointer({
		type: "move",
		screenX: 99.5,
		screenY: 99.5,
		viewportWidth: 200,
		viewportHeight: 200,
	});
	manager.updatePointer({
		type: "down",
		button: 0,
		screenX: 99.5,
		screenY: 99.5,
		viewportWidth: 200,
		viewportHeight: 200,
	});

	assert.equal(manager.getSelection(), box.entityId);

	const transient = new Map();
	transient.set(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 0);
	transient.set(ANIMATION_SIM_DELTA_TIME_MS_KEY, 0);
	for (const contributor of contributors) {
		contributor({
			now: 0,
			deltaTime: 0,
			scene,
			camera,
			transient,
		});
	}
	const interactionState = transient.get(INTERACTION_TRANSIENT_STATE_KEY);
	assert.ok(interactionState);
	assert.equal(interactionState.selectedEntityIds[0], box.entityId);
	assert.equal(interactionState.outline.shape, "circle");

	manager.setOutlineStyle({ shape: "square" });
	transient.clear();
	transient.set(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 0);
	transient.set(ANIMATION_SIM_DELTA_TIME_MS_KEY, 0);
	for (const contributor of contributors) {
		contributor({
			now: 0,
			deltaTime: 0,
			scene,
			camera,
			transient,
		});
	}
	const updatedInteractionState = transient.get(INTERACTION_TRANSIENT_STATE_KEY);
	assert.ok(updatedInteractionState);
	assert.equal(updatedInteractionState.outline.shape, "square");

	manager.detach();
	assert.equal(contributors.size, 0);

	console.log("InteractionManager selection tests passed");
}

run();
