import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { ShadowPlanner } from "../../../src/lights/shadows/ShadowPlanner.ts";

function createCamera() {
	return {
		near: 0.1, far: 100, fov: 60, aspectRatio: 16 / 9,
		position: { x: 0, y: 4, z: 16 }, up: { x: 0, y: 1, z: 0 },
		getWorldPosition(target = { x: 0, y: 0, z: 0 }) { return Object.assign(target, this.position); },
		getWorldDirection(local, target = { x: 0, y: 0, z: 0 }) {
			return Object.assign(target, local.y === 1 ? this.up : { x: 0, y: 0, z: -1 });
		},
	};
}

function testDirectionalCsmPreparedSlices() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight());
	scene.shadows.bind(sun, scene.shadows.createCascaded({
		size: 1024, cascadeCounts: { directional: 4 }, blendRatio: 0.2,
	}));
	const camera = createCamera();
	const plan = ShadowPlanner.plan({
		manager: scene.shadows, lights: [sun], camera, cameraPosition: camera.position,
		sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 80 },
		casterIntent: { meshPackets: [], hasTransparentCasters: false, hasParticleCasters: false, estimatedParticleCapacity: 0 },
		enableShadows: true,
		backendKey: "webgpu",
	}, ShadowPlanner.createState());
	assert.equal(plan.lights[0].effectiveTechnique, "cascaded");
	assert.equal(plan.lights[0].slices.length, 4);
	assert.ok(plan.lights[0].slices.every((slice) => slice.viewProjection));
}

testDirectionalCsmPreparedSlices();
console.log("CSM planner tests passed");
