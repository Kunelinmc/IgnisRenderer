import assert from "node:assert/strict";

import { Scene } from "../../../src/core/Scene.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { PointLight } from "../../../src/lights/PointLight.ts";
import { SpotLight } from "../../../src/lights/SpotLight.ts";
import { ShadowPlanner } from "../../../src/pipeline/shadows/ShadowPlanner.ts";
import { resolveLegacyShadowMaps } from "../../../src/pipeline/shadows/LegacyShadowPlanAdapter.ts";

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

function createCapabilities({ budget = 48, paged = true } = {}) {
	return {
		backendKey: "planner-test",
		supportsFilterModes: ["pcf"],
		lightTypes: {
			directional: {
				projections: ["single", "cascaded"],
				storage: paged ? ["atlas", "paged"] : ["atlas"],
				maxLights: 4,
				maxCascadedLights: 1,
			},
			point: {
				projections: [],
				storage: [],
				maxLights: 0,
				maxCascadedLights: 0,
			},
			spot: {
				projections: ["single"],
				storage: ["atlas"],
				maxLights: 8,
				maxCascadedLights: 0,
			},
		},
		supportsTransmission: true,
		supportsDirectionalCSM: true,
		supportsSpotCSM: false,
		supportsPointCSM: false,
		maxDynamicShadowCost: budget,
		supportsPagedShadowRendering: paged,
		maxPagedShadowPages: 2048,
		pagedShadowPageSizeRange: [64, 256],
	};
}

function createIntent(hasCasters = true) {
	return {
		meshPackets: hasCasters ? [{
			worldBounds: { center: { x: 0, y: 0, z: 0 }, radius: 10 },
		}] : [],
		hasTransparentCasters: false,
		hasParticleCasters: false,
		estimatedParticleCapacity: 0,
	};
}

function planScene(scene, options = {}) {
	const planner = options.planner ?? new ShadowPlanner();
	const camera = createCamera();
	return planner.plan({
		manager: scene.shadows,
		lights: scene.ecs.findLights(),
		capabilities: options.capabilities ?? createCapabilities(),
		camera,
		cameraPosition: camera.position,
		sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 80 },
		casterIntent: options.intent ?? createIntent(),
		enableShadows: true,
		hasTransmissionCasters: false,
		needsAtlasFallback: options.needsAtlasFallback ?? false,
	});
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
		size: 1024,
		cascadeCounts: { directional: 4 },
	}));

	const plan = planScene(scene, {
		capabilities: createCapabilities({ budget: 1.1 }),
	});
	assert.equal(plan.lights.length, 1);
	assert.equal(plan.lights[0].slices.length, 1);
	assert.equal(plan.lights[0].slices[0].resolution, 512);
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
	assert.equal(resolveLegacyShadowMaps(noCasters).size, 0);
}

function testPagedStorageIsDirectionalOnly() {
	const scene = new Scene();
	const spot = scene.add(new PointLight());
	const pagedCapabilities = createCapabilities();
	pagedCapabilities.lightTypes.point = {
		projections: ["single"],
		storage: ["paged"],
		maxLights: 1,
		maxCascadedLights: 0,
	};
	scene.shadows.bind(spot, scene.shadows.createPaged());

	const plan = planScene(scene, { capabilities: pagedCapabilities });
	assert.equal(plan.lights[0].storage, "atlas");
	assert.ok(plan.diagnostics.some((item) => item.code === "storage-fallback"));
}

function testSingleCascadeCsmFallbackIsReported() {
	const scene = new Scene();
	const spot = scene.add(new SpotLight());
	const capabilities = createCapabilities();
	capabilities.lightTypes.spot = {
		projections: ["single"],
		storage: ["atlas"],
		maxLights: 1,
		maxCascadedLights: 0,
	};
	scene.shadows.bind(spot, scene.shadows.createCascaded({
		cascadeCounts: { spot: 1 },
	}));

	const plan = planScene(scene, { capabilities });
	assert.equal(plan.lights[0].effectiveTechnique, "single");
	assert.equal(plan.lights[0].fallbackReason, "projection-fallback");
	assert.ok(plan.diagnostics.some((item) => item.code === "projection-fallback"));
}

function testPlanStaysImmutableWhenLegacyPlacementChanges() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight());
	scene.shadows.bind(sun, scene.shadows.createSingle());
	const plan = planScene(scene);
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
	const legacy = resolveLegacyShadowMaps(plan);
	legacy.get(sun).slices[0].atlasRect = {
		offsetX: 0,
		offsetY: 0,
		size: 512,
		tileSize: 512,
		localTileX: 0,
		localTileY: 0,
		localTileSpan: 1,
	};
	assert.equal(planHash(plan), before);
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
	testPlanStaysImmutableWhenLegacyPlacementChanges();
	console.log("Shadow planner tests passed");
}

run();
