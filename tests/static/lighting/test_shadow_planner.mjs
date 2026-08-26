import assert from "node:assert/strict";

import { Scene } from "../../../src/core/Scene.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { PointLight } from "../../../src/lights/PointLight.ts";
import { SpotLight } from "../../../src/lights/SpotLight.ts";
import { ShadowPlanner } from "../../../src/lights/shadows/ShadowPlanner.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function createCamera() {
	const position = { x: 0, y: 4, z: 16 };
	return {
		near: 0.1,
		far: 100,
		fov: 60,
		aspectRatio: 16 / 9,
		position,
		up: { x: 0, y: 1, z: 0 },
		getWorldPosition(target = { x: 0, y: 0, z: 0 }) {
			Object.assign(target, position);
			return target;
		},
		getWorldDirection(localDirection, target = { x: 0, y: 0, z: 0 }) {
			const source = localDirection.y === 1 ?
				{ x: 0, y: 1, z: 0 }
			: { x: 0, y: 0, z: -1 };
			Object.assign(target, source);
			return target;
		},
	};
}

function createIntent(hasCasters = true) {
	return {
		meshPackets: hasCasters ? [createTestDrawPacket({
			worldBounds: { center: { x: 0, y: 0, z: 0 }, radius: 10 },
		})] : [],
		hasTransparentCasters: false,
		hasParticleCasters: false,
		estimatedParticleCapacity: 0,
	};
}

function planScene(scene, options = {}) {
	const plannerState = options.plannerState ?? ShadowPlanner.createState();
	const camera = createCamera();
	return ShadowPlanner.plan({
		manager: scene.shadows,
		lights: scene.ecs.findLights(),
		backendKey: options.backendKey ?? "webgpu",
		camera,
		sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 80 },
		casterIntent: options.intent ?? createIntent(),
		enableShadows: true,
		hasTransmissionCasters: false,
		needsAtlasFallback: options.needsAtlasFallback ?? false,
	}, plannerState);
}

function testCapabilityAndFilterFallbacksAreExplicit() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight({ intensity: 2 }));
	const point = scene.add(new PointLight({ intensity: 10 }));
	scene.shadows.bind(sun, scene.shadows.createVariance({ priority: 1 }));
	scene.shadows.bind(point, scene.shadows.createSingle({ priority: 100 }));

	const plan = planScene(scene);
	assert.deepEqual(plan.lights.map((light) => light.lightId), [sun.id]);
	assert.equal(plan.lights[0].filterMode, "pcf");
	assert.ok(plan.diagnostics.some((item) => item.code === "filter-fallback"));
	assert.ok(plan.diagnostics.some((item) =>
		item.code === "unsupported-light-type" && item.lightId === point.id
	));
}

function testBudgetDegradesCascadeBeforeResolution() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight({ intensity: 1 }));
	scene.shadows.bind(sun, scene.shadows.createCascaded({
		size: 4096,
		cascadeCounts: { directional: 4 },
	}));

	const plan = planScene(scene);
	assert.equal(plan.lights.length, 1);
	assert.equal(plan.lights[0].slices.length, 3);
	assert.equal(plan.lights[0].slices[0].resolution, 2048);
	assert.ok(plan.diagnostics.some((item) => item.code === "budget-degraded"));
}

function testPagedJobsAreExplicitAndCasterGated() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight());
	scene.shadows.bind(sun, scene.shadows.createPaged());

	const pagedOnly = planScene(scene);
	assert.deepEqual(pagedOnly.jobs.map((job) => job.technique), ["paged"]);
	assert.equal(pagedOnly.hasPagedWork, true);

	const withFallback = planScene(scene, { needsAtlasFallback: true });
	assert.deepEqual(
		withFallback.jobs.map((job) => job.technique),
		["paged", "atlas-fallback"]
	);

	const noCasters = planScene(scene, { intent: createIntent(false) });
	assert.equal(noCasters.hasRasterWork, false);
	assert.equal(noCasters.jobs.length, 0);
	assert.equal(noCasters.lights.length, 1);
}

function testPagedStorageIsDirectionalOnly() {
	const scene = new Scene();
	const spot = scene.add(new SpotLight());
	scene.shadows.bind(spot, scene.shadows.createPaged());

	const plan = planScene(scene);
	assert.equal(plan.lights[0].storage, "atlas");
	assert.ok(plan.diagnostics.some((item) => item.code === "storage-fallback"));
}

function testSingleCascadeCsmFallbackIsReported() {
	const scene = new Scene();
	const spot = scene.add(new SpotLight());
	scene.shadows.bind(spot, scene.shadows.createCascaded({
		cascadeCounts: { spot: 1 },
	}));

	const plan = planScene(scene);
	assert.equal(plan.lights[0].effectiveTechnique, "single");
	assert.equal(plan.lights[0].fallbackReason, "projection-fallback");
	assert.ok(plan.diagnostics.some((item) => item.code === "projection-fallback"));
}

function testFixedBackendPoliciesRemainDistinct() {
	const pointScene = new Scene();
	const point = pointScene.add(new PointLight());
	pointScene.shadows.bind(point, pointScene.shadows.createCascaded({
		cascadeCounts: { point: 2 },
	}));

	const softwarePlan = planScene(pointScene, { backendKey: "software" });
	assert.equal(softwarePlan.lights.length, 1);
	assert.equal(softwarePlan.lights[0].effectiveTechnique, "cascaded");
	assert.equal(softwarePlan.lights[0].slices.length, 12);

	const webglPlan = planScene(pointScene, { backendKey: "webgl" });
	assert.equal(webglPlan.lights.length, 0);
	assert.ok(webglPlan.diagnostics.some((item) =>
		item.code === "unsupported-light-type"
	));

	const pagedScene = new Scene();
	const sun = pagedScene.add(new DirectionalLight());
	pagedScene.shadows.bind(sun, pagedScene.shadows.createPaged());
	const webglPagedPlan = planScene(pagedScene, { backendKey: "webgl" });
	assert.equal(webglPagedPlan.lights[0].storage, "atlas");
	assert.ok(webglPagedPlan.diagnostics.some((item) =>
		item.code === "storage-fallback"
	));
}

function testCameraWorldPositionDrivesLightSelection() {
	const scene = new Scene();
	const spots = Array.from({ length: 9 }, (_, index) => {
		const spot = scene.add(new SpotLight({
			position: { x: index * 10, y: 0, z: 0 },
		}));
		scene.shadows.bind(spot, scene.shadows.createSingle());
		return spot;
	});
	const camera = createCamera();
	camera.position = { x: 0, y: 0, z: 0 };
	camera.getWorldPosition = (target = { x: 0, y: 0, z: 0 }) => {
		Object.assign(target, { x: 80, y: 0, z: 0 });
		return target;
	};
	const plan = ShadowPlanner.plan({
		manager: scene.shadows,
		lights: spots,
		backendKey: "webgpu",
		camera,
		sceneBounds: { center: { x: 40, y: 0, z: 0 }, radius: 80 },
		casterIntent: createIntent(false),
		enableShadows: true,
		hasTransmissionCasters: false,
		needsAtlasFallback: false,
	}, ShadowPlanner.createState());
	const selectedIds = new Set(plan.lights.map((light) => light.lightId));
	assert.equal(selectedIds.has(spots[0].id), false);
	assert.equal(selectedIds.has(spots[8].id), true);
}

function testPlanStaysImmutableAcrossPlanningFrames() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight());
	scene.shadows.bind(sun, scene.shadows.createSingle());
	const plannerState = ShadowPlanner.createState();
	const plan = planScene(scene, { plannerState });
	assert.deepEqual(plan.diagnostics, []);
	const planHash = (value) => JSON.stringify({
		revision: value.revision,
		lights: value.lights.map((light) => ({
			lightId: light.lightId,
			definition: light.definition,
			filterMode: light.filterMode,
			storage: light.storage,
			slices: light.slices.map((slice) => ({
				...slice,
				view: slice.view.elements,
				projection: slice.projection.elements,
				viewProjection: slice.viewProjection.elements,
			})),
		})),
		jobs: value.jobs,
		diagnostics: value.diagnostics,
	});
	const before = planHash(plan);
	const nextPlan = planScene(scene, { plannerState });
	assert.equal(planHash(plan), before);
	assert.notEqual(nextPlan, plan);
	assert.equal(nextPlan.revision, plan.revision + 1);
	assert.equal(planScene(scene).revision, 1);
	assert.ok(Object.isFrozen(plan));
	assert.ok(Object.isFrozen(plan.jobs));
	assert.ok(Object.isFrozen(plan.lights[0].slices[0].view));
	assert.ok(Object.isFrozen(plan.lights[0].slices[0].view.elements));
}

function run() {
	testCapabilityAndFilterFallbacksAreExplicit();
	testBudgetDegradesCascadeBeforeResolution();
	testPagedJobsAreExplicitAndCasterGated();
	testPagedStorageIsDirectionalOnly();
	testSingleCascadeCsmFallbackIsReported();
	testFixedBackendPoliciesRemainDistinct();
	testCameraWorldPositionDrivesLightSelection();
	testPlanStaysImmutableAcrossPlanningFrames();
	console.log("Shadow planner tests passed");
}

run();
