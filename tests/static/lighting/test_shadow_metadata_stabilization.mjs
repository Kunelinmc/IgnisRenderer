import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { ShadowPlanner } from "../../../src/lights/shadows/ShadowPlanner.ts";

function plan(plannerState, scene, light, radius, enableShadows = true) {
	const camera = { near: 0.1, far: 100, fov: 60, aspectRatio: 1, position: { x: 0, y: 4, z: 16 }, up: { x: 0, y: 1, z: 0 },
		getWorldPosition(target = {}) { return Object.assign(target, this.position); },
		getWorldDirection(local, target = {}) { return Object.assign(target, local.y === 1 ? this.up : { x: 0, y: 0, z: -1 }); } };
	return ShadowPlanner.plan({ manager: scene.shadows, lights: [light], camera, cameraPosition: camera.position,
		sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius }, casterIntent: { meshPackets: [], hasTransparentCasters: false, hasParticleCasters: false, estimatedParticleCapacity: 0 },
		backendKey: "software", enableShadows }, plannerState);
}

const scene = new Scene();
const light = scene.add(new DirectionalLight());
const definition = scene.shadows.createSingle({ size: 1024 });
scene.shadows.bind(light, definition);
const plannerState = ShadowPlanner.createState();
const first = plan(plannerState, scene, light, 100);
const second = plan(plannerState, scene, light, 10);
assert.equal(first.lights[0].slices.length, 1);
assert.equal(second.lights[0].slices.length, 1);
definition.update({ size: 512 });
const reset = plan(plannerState, scene, light, 10);
assert.equal(reset.lights[0].effectiveResolution, 512);

const csm = scene.shadows.createCascaded({
	size: 1024,
	cascadeCounts: { directional: 1 },
});
scene.shadows.bind(light, csm);
const csmPlannerState = ShadowPlanner.createState();
const largeBounds = plan(csmPlannerState, scene, light, 100);
const shrunkBounds = plan(csmPlannerState, scene, light, 10);
const disabled = plan(csmPlannerState, scene, light, 10, false);
const afterDisable = plan(csmPlannerState, scene, light, 10);
assert.notDeepEqual(
	shrunkBounds.lights[0].slices[0].projection.elements,
	afterDisable.lights[0].slices[0].projection.elements,
);
assert.notDeepEqual(
	largeBounds.lights[0].slices[0].projection.elements,
	shrunkBounds.lights[0].slices[0].projection.elements,
);
assert.equal(disabled.lights.length, 0);
console.log("shadow projection history tests passed");
