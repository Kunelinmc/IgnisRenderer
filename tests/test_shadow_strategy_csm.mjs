import assert from "node:assert/strict";
import { Scene } from "../src/core/Scene.ts";
import { AreaLight } from "../src/lights/AreaLight.ts";
import { DirectionalLight } from "../src/lights/DirectionalLight.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { PointLight } from "../src/lights/PointLight.ts";
import { SpotLight } from "../src/lights/SpotLight.ts";
import {
	createShadowRenderSet,
	normalizeShadowConfig,
} from "../src/lights/shadows/ShadowMapping.ts";
import { updateShadowMapMetadata } from "../src/pipeline/ShadowMetadata.ts";
import { selectCSMDirectionalLights } from "../src/pipeline/ShadowStrategyRegistry.ts";
import { collectWebGPULighting } from "../src/renderers/webgpu/lights.ts";
import { collectWebGLLights } from "../src/renderers/webgl/WebGLLightCollector.ts";

function createSceneBounds(radius = 80) {
	return {
		center: { x: 0, y: 0, z: 0 },
		radius,
	};
}

function createCamera(overrides = {}) {
	const position = overrides.position ?? { x: 0, y: 4, z: 16 };
	const up = overrides.up ?? { x: 0, y: 1, z: 0 };
	const forward = overrides.forward ?? { x: 0, y: 0, z: -1 };
	return {
		near: overrides.near ?? 0.1,
		far: overrides.far ?? 100,
		fov: overrides.fov ?? 60,
		aspectRatio: overrides.aspectRatio ?? 16 / 9,
		position,
		up,
		getWorldPosition(target = { x: 0, y: 0, z: 0 }) {
			target.x = position.x;
			target.y = position.y;
			target.z = position.z;
			return target;
		},
		getWorldDirection(localDirection, target = { x: 0, y: 0, z: 0 }) {
			const useUpDirection =
				Math.abs(localDirection.x - up.x) <= 1e-6 &&
				Math.abs(localDirection.y - up.y) <= 1e-6 &&
				Math.abs(localDirection.z - up.z) <= 1e-6;
			const source = useUpDirection ? up : forward;
			target.x = source.x;
			target.y = source.y;
			target.z = source.z;
			return target;
		},
	};
}

function createDirectionalCSMLight({
	priority = 0,
	intensity = 1,
	lambda = 0.65,
	maxDistance,
	blendRatio = 0.1,
	cascadeCount = 4,
	stabilize = true,
	direction = { x: 0, y: -1, z: -0.3 },
} = {}) {
	const scene = new Scene();
	const light = new DirectionalLight({
		intensity,
		direction,
	});
	scene.add(light);
	const shadowMap = scene.shadows.createCSM({
		size: 1024,
		priority,
		lambda,
		maxDistance,
		blendRatio,
		stabilize,
		cascadeCounts: {
			directional: cascadeCount,
		},
	});
	scene.shadows.bind(light, shadowMap);
	return light;
}

function createOrthographicCamera(overrides = {}) {
	const camera = createCamera(overrides);
	const bounds = overrides.bounds ?? {
		left: -12,
		right: 12,
		bottom: -6,
		top: 6,
	};
	return {
		...camera,
		type: "orthographic",
		fov: overrides.fov ?? 60,
		size: overrides.size ?? 12,
		getBounds() {
			return bounds;
		},
	};
}

function createSpotCSMLight({
	priority = 0,
	intensity = 1,
	range = 80,
} = {}) {
	const scene = new Scene();
	const light = new SpotLight({
		intensity,
		range,
		direction: { x: 0, y: -1, z: 0 },
		outerAngle: Math.PI / 3,
	});
	scene.add(light);
	const shadowMap = scene.shadows.createCSM({
		size: 1024,
		priority,
		lambda: 0.65,
		blendRatio: 0.1,
		stabilize: true,
		cascadeCounts: {
			spot: 4,
		},
	});
	scene.shadows.bind(light, shadowMap);
	return light;
}

function createRenderSetForBoundLight(light) {
	const shadowConfig = light.scene?.shadows.getLegacyShadowConfig(light);
	assert.ok(shadowConfig, `Expected bound shadow config for light ${light.id}`);
	return createShadowRenderSet(shadowConfig);
}

function testCSMSplitsMonotonicAndCovered() {
	const camera = createCamera({ near: 0.2, far: 120 });
	const light = createDirectionalCSMLight({
		lambda: 0.65,
		maxDistance: 80,
		cascadeCount: 4,
	});
	const renderSet = createRenderSetForBoundLight(light);

	updateShadowMapMetadata(renderSet, light, createSceneBounds(120), {
		camera,
	});

	assert.equal(renderSet.effectiveStrategyType, "csm");
	assert.equal(renderSet.slices.length, 4);
	assert.ok(Math.abs(renderSet.slices[0].splitNear - 0.2) < 1e-6);
	assert.ok(Math.abs(renderSet.slices[3].splitFar - 80) < 1e-6);
	for (let index = 0; index < renderSet.slices.length; index++) {
		const slice = renderSet.slices[index];
		assert.ok(slice.splitNear < slice.splitFar);
		if (index > 0) {
			assert.ok(
				renderSet.slices[index - 1].splitFar <= slice.splitNear + 1e-6
			);
		}
	}
}

function testLambdaBoundarySplits() {
	const camera = createCamera({ near: 0.1, far: 100 });
	const bounds = createSceneBounds(100);

	const uniformLight = createDirectionalCSMLight({
		lambda: 0,
		maxDistance: 100,
		cascadeCount: 4,
	});
	const uniformSet = createRenderSetForBoundLight(uniformLight);
	updateShadowMapMetadata(uniformSet, uniformLight, bounds, { camera });

	const logLight = createDirectionalCSMLight({
		lambda: 1,
		maxDistance: 100,
		cascadeCount: 4,
	});
	const logSet = createRenderSetForBoundLight(logLight);
	updateShadowMapMetadata(logSet, logLight, bounds, { camera });

	const near = 0.1;
	const far = 100;
	const t = 1 / 4;
	const expectedUniform = near + (far - near) * t;
	const expectedLog = near * Math.pow(far / near, t);
	assert.ok(Math.abs(uniformSet.slices[0].splitFar - expectedUniform) < 1e-5);
	assert.ok(Math.abs(logSet.slices[0].splitFar - expectedLog) < 1e-5);
	assert.ok(logSet.slices[0].splitFar < uniformSet.slices[0].splitFar);
}

function testDirectionalCSMSingleCascadeCoversMaxDistance() {
	const camera = createCamera({ near: 0.2, far: 120 });
	const light = createDirectionalCSMLight({
		lambda: 0.65,
		maxDistance: 80,
		cascadeCount: 1,
	});
	const renderSet = createRenderSetForBoundLight(light);

	updateShadowMapMetadata(renderSet, light, createSceneBounds(120), {
		camera,
	});

	assert.equal(renderSet.effectiveStrategyType, "csm");
	assert.equal(renderSet.slices.length, 1);
	assert.ok(Math.abs(renderSet.slices[0].splitNear - 0.2) < 1e-6);
	assert.ok(Math.abs(renderSet.slices[0].splitFar - 80) < 1e-6);
}

function readCascadeOrthoSpan(slice) {
	const projection = slice.shadowMap.projectionMatrix;
	assert.ok(projection);
	const m00 = projection.elements[0][0];
	const m11 = projection.elements[1][1];
	const width = Math.abs(2 / m00);
	const height = Math.abs(2 / m11);
	return { width, height };
}

function testCSMStabilizedExtentIsCameraRotationInvariant() {
	const light = createDirectionalCSMLight({
		lambda: 0.65,
		maxDistance: 80,
		cascadeCount: 4,
	});
	const forwardA = { x: 0, y: 0, z: -1 };
	const invSqrt2 = 1 / Math.sqrt(2);
	const forwardB = { x: invSqrt2, y: 0, z: -invSqrt2 };
	const cameraA = createCamera({ near: 0.1, far: 100, forward: forwardA });
	const cameraB = createCamera({ near: 0.1, far: 100, forward: forwardB });
	const bounds = createSceneBounds(120);
	const renderSetA = createRenderSetForBoundLight(light);
	const renderSetB = createRenderSetForBoundLight(light);

	updateShadowMapMetadata(renderSetA, light, bounds, { camera: cameraA });
	updateShadowMapMetadata(renderSetB, light, bounds, { camera: cameraB });

	for (let index = 0; index < renderSetA.slices.length; index++) {
		const spanA = readCascadeOrthoSpan(renderSetA.slices[index]);
		const spanB = readCascadeOrthoSpan(renderSetB.slices[index]);
		assert.ok(
			Math.abs(spanA.width - spanB.width) < 1e-6,
			`Cascade ${index} width should be stable across camera rotation`
		);
		assert.ok(
			Math.abs(spanA.height - spanB.height) < 1e-6,
			`Cascade ${index} height should be stable across camera rotation`
		);
	}
}

function testOrthographicCSMUsesOrthoBoundsInsteadOfFov() {
	const light = createDirectionalCSMLight({
		lambda: 0.65,
		maxDistance: 80,
		cascadeCount: 4,
	});
	const cameraA = createOrthographicCamera({
		near: 0.1,
		far: 100,
		fov: 30,
	});
	const cameraB = createOrthographicCamera({
		near: 0.1,
		far: 100,
		fov: 120,
	});
	const bounds = createSceneBounds(120);
	const renderSetA = createRenderSetForBoundLight(light);
	const renderSetB = createRenderSetForBoundLight(light);

	updateShadowMapMetadata(renderSetA, light, bounds, { camera: cameraA });
	updateShadowMapMetadata(renderSetB, light, bounds, { camera: cameraB });

	for (let index = 0; index < renderSetA.slices.length; index++) {
		const spanA = readCascadeOrthoSpan(renderSetA.slices[index]);
		const spanB = readCascadeOrthoSpan(renderSetB.slices[index]);
		assert.ok(Math.abs(spanA.width - spanB.width) < 1e-6);
		assert.ok(Math.abs(spanA.height - spanB.height) < 1e-6);
	}
}

function testDirectionalCSMDepthIncludesCasterBounds() {
	const camera = createCamera({
		near: 0.1,
		far: 20,
		aspectRatio: 1,
	});
	const light = createDirectionalCSMLight({
		cascadeCount: 1,
		maxDistance: 20,
		stabilize: false,
		direction: { x: 0, y: -1, z: 0 },
	});
	const renderSet = createRenderSetForBoundLight(light);
	const casterBounds = {
		center: { x: 0, y: 80, z: -8 },
		radius: 2,
	};

	updateShadowMapMetadata(renderSet, light, casterBounds, { camera });

	const viewProjection = renderSet.slices[0].shadowMap.viewProjectionMatrix;
	assert.ok(viewProjection);
	const clip = Matrix4.transformPoint(viewProjection, casterBounds.center);
	const ndcZ = clip.z / clip.w;
	assert.ok(
		ndcZ >= -1 && ndcZ <= 1,
		`Caster bounds should be inside CSM depth range, got ndcZ=${ndcZ}`
	);
}

function distanceToNearestInteger(value) {
	return Math.abs(value - Math.round(value));
}

function testCSMStabilizedViewTranslationSnapsToTexelGrid() {
	const light = createDirectionalCSMLight({
		lambda: 0.65,
		maxDistance: 80,
		cascadeCount: 4,
		stabilize: true,
	});
	const bounds = createSceneBounds(120);
	const baseCameraPosition = { x: 0, y: 4, z: 16 };
	const cameraA = createCamera({
		near: 0.1,
		far: 100,
		position: baseCameraPosition,
	});
	const renderSetA = createRenderSetForBoundLight(light);
	updateShadowMapMetadata(renderSetA, light, bounds, { camera: cameraA });

	const cascadeA = renderSetA.slices[0];
	const spanA = readCascadeOrthoSpan(cascadeA);
	const texelSize = spanA.width / cascadeA.shadowMap.size;
	const moveDistance = texelSize * 0.37;
	const cameraB = createCamera({
		near: 0.1,
		far: 100,
		position: {
			x: baseCameraPosition.x + moveDistance,
			y: baseCameraPosition.y,
			z: baseCameraPosition.z + moveDistance * 0.5,
		},
	});
	const renderSetB = createRenderSetForBoundLight(light);
	updateShadowMapMetadata(renderSetB, light, bounds, { camera: cameraB });

	const viewA = renderSetA.slices[0].shadowMap.viewMatrix;
	const viewB = renderSetB.slices[0].shadowMap.viewMatrix;
	assert.ok(viewA && viewB);

	const deltaXInTexelUnits = (viewB.elements[0][3] - viewA.elements[0][3]) / texelSize;
	const deltaYInTexelUnits = (viewB.elements[1][3] - viewA.elements[1][3]) / texelSize;
	assert.ok(
		distanceToNearestInteger(deltaXInTexelUnits) < 1e-4,
		`Stabilized CSM x-translation should move in texel steps, got ${deltaXInTexelUnits}`
	);
	assert.ok(
		distanceToNearestInteger(deltaYInTexelUnits) < 1e-4,
		`Stabilized CSM y-translation should move in texel steps, got ${deltaYInTexelUnits}`
	);
}

function testCSMUnstabilizedViewTranslationRemainsContinuous() {
	const light = createDirectionalCSMLight({
		lambda: 0.65,
		maxDistance: 80,
		cascadeCount: 4,
		stabilize: false,
	});
	const bounds = createSceneBounds(120);
	const baseCameraPosition = { x: 0, y: 4, z: 16 };
	const cameraA = createCamera({
		near: 0.1,
		far: 100,
		position: baseCameraPosition,
	});
	const renderSetA = createRenderSetForBoundLight(light);
	updateShadowMapMetadata(renderSetA, light, bounds, { camera: cameraA });

	const cascadeA = renderSetA.slices[0];
	const spanA = readCascadeOrthoSpan(cascadeA);
	const texelSize = spanA.width / cascadeA.shadowMap.size;
	const moveDistance = texelSize * 0.37;
	const cameraB = createCamera({
		near: 0.1,
		far: 100,
		position: {
			x: baseCameraPosition.x + moveDistance,
			y: baseCameraPosition.y,
			z: baseCameraPosition.z + moveDistance * 0.5,
		},
	});
	const renderSetB = createRenderSetForBoundLight(light);
	updateShadowMapMetadata(renderSetB, light, bounds, { camera: cameraB });

	const viewA = renderSetA.slices[0].shadowMap.viewMatrix;
	const viewB = renderSetB.slices[0].shadowMap.viewMatrix;
	assert.ok(viewA && viewB);

	const deltaXInTexelUnits = (viewB.elements[0][3] - viewA.elements[0][3]) / texelSize;
	const deltaYInTexelUnits = (viewB.elements[1][3] - viewA.elements[1][3]) / texelSize;
	const nearestX = distanceToNearestInteger(deltaXInTexelUnits);
	const nearestY = distanceToNearestInteger(deltaYInTexelUnits);
	assert.ok(
		nearestX > 1e-3 || nearestY > 1e-3,
		"Unstabilized CSM should allow continuous sub-texel shadow movement"
	);
}

function testBlendRatioNormalization() {
	const clampedLow = normalizeShadowConfig({
		strategy: "csm",
		blendRatio: -2,
	});
	const clampedHigh = normalizeShadowConfig({
		strategy: "csm",
		blendRatio: 2,
	});
	assert.equal(clampedLow.strategy, "csm");
	assert.equal(clampedHigh.strategy, "csm");
	assert.equal(clampedLow.blendRatio, 0);
	assert.equal(clampedHigh.blendRatio, 1);
}

function testBackendFallbackToSingleMap() {
	const camera = createCamera({ near: 0.1, far: 90 });
	const light = createDirectionalCSMLight({
		maxDistance: 90,
		cascadeCount: 4,
	});
	const renderSet = createRenderSetForBoundLight(light);
	const warnings = [];

	updateShadowMapMetadata(renderSet, light, createSceneBounds(100), {
		camera,
		backendCapabilities: {
			backendKey: "webgl",
			supportsSingleMap: true,
			supportsDirectionalCSM: false,
			maxCsmDirectionalLights: 0,
		},
		onWarning: (key, message) => warnings.push({ key, message }),
	});

	assert.equal(renderSet.requestedStrategyType, "csm");
	assert.equal(renderSet.effectiveStrategyType, "single-map");
	assert.equal(renderSet.slices.length, 1);
	assert.equal(renderSet.resolvedConfig.strategy, "single-map");
	assert.equal(warnings.length, 1);
	assert.ok(warnings[0].key.includes(`webgl-${light.id}`));
}

function testUnsupportedPositionalCSMFallbackToSingleMap() {
	const spotLight = createSpotCSMLight();
	const spotRenderSet = createRenderSetForBoundLight(spotLight);
	updateShadowMapMetadata(spotRenderSet, spotLight, createSceneBounds(60), {
		backendCapabilities: {
			backendKey: "webgpu",
			supportsSingleMap: true,
			supportsDirectionalCSM: true,
			supportsSpotCSM: false,
			supportsPointCSM: false,
			maxCsmDirectionalLights: 1,
		},
	});
	assert.equal(spotRenderSet.requestedStrategyType, "csm");
	assert.equal(spotRenderSet.effectiveStrategyType, "single-map");
	assert.equal(spotRenderSet.slices.length, 1);

	const scene = new Scene();
	const pointLight = new PointLight({ range: 80 });
	scene.add(pointLight);
	scene.shadows.bind(
		pointLight,
		scene.shadows.createCSM({
			size: 1024,
			cascadeCounts: {
				point: 2,
			},
		})
	);
	const pointRenderSet = createRenderSetForBoundLight(pointLight);
	updateShadowMapMetadata(pointRenderSet, pointLight, createSceneBounds(60), {
		backendCapabilities: {
			backendKey: "webgpu",
			supportsSingleMap: true,
			supportsDirectionalCSM: true,
			supportsSpotCSM: false,
			supportsPointCSM: false,
			maxCsmDirectionalLights: 1,
		},
	});
	assert.equal(pointRenderSet.requestedStrategyType, "csm");
	assert.equal(pointRenderSet.effectiveStrategyType, "single-map");
	assert.equal(pointRenderSet.slices.length, 1);
}

function testCSMSelectionPriority() {
	const lightA = createDirectionalCSMLight({ priority: 1, intensity: 10 });
	const lightB = createDirectionalCSMLight({ priority: 3, intensity: 1 });
	const lightC = createDirectionalCSMLight({ priority: 3, intensity: 5 });
	const selected = selectCSMDirectionalLights([lightA, lightB, lightC], 2);
	assert.equal(selected.size, 2);
	assert.ok(selected.has(lightB));
	assert.ok(selected.has(lightC));
	assert.ok(!selected.has(lightA));
}

function testSpotLightCSMUsesCascadeSlices() {
	const light = createSpotCSMLight();
	const renderSet = createRenderSetForBoundLight(light);
	const emptyDirectionalBudget = new Set();

	updateShadowMapMetadata(renderSet, light, createSceneBounds(60), {
		backendCapabilities: {
			backendKey: "webgpu",
			supportsSingleMap: true,
			supportsDirectionalCSM: true,
			supportsSpotCSM: true,
			maxCsmDirectionalLights: 0,
		},
		allowCSMDirectionalLights: emptyDirectionalBudget,
	});

	assert.equal(renderSet.requestedStrategyType, "csm");
	assert.equal(renderSet.effectiveStrategyType, "csm");
	assert.equal(renderSet.slices.length, 4);
	for (let index = 0; index < renderSet.slices.length; index++) {
		assert.ok(renderSet.slices[index].shadowMap.viewProjectionMatrix);
	}

	const shadowMaps = new Map([[light, renderSet]]);
	const webgpuLighting = collectWebGPULighting(
		[light],
		true,
		false,
		true,
		shadowMaps,
		false
	);
	assert.equal(webgpuLighting.spotShadows[0]?.strategyType, "csm");
	assert.equal(webgpuLighting.spotShadows[0]?.cascadeCount, 4);

	const webglLighting = collectWebGLLights(
		[light],
		true,
		true,
		shadowMaps,
		false,
		null,
		false
	);
	assert.equal(webglLighting.spotShadows[0]?.strategyType, "csm");
	assert.equal(webglLighting.spotShadows[0]?.cascadeCount, 4);
}

function testPointLightSingleMapUsesSceneShadowBinding() {
	const scene = new Scene();
	const light = new PointLight({ range: 60 });
	scene.add(light);
	scene.shadows.bind(light, scene.shadows.createSingle({ size: 1024 }));
	const renderSet = createRenderSetForBoundLight(light);
	updateShadowMapMetadata(renderSet, light, createSceneBounds(50));

	assert.equal(renderSet.effectiveStrategyType, "single-map");
	assert.ok(renderSet.slices[0].shadowMap.viewProjectionMatrix);
	assert.ok(renderSet.slices[0].splitFar > renderSet.slices[0].splitNear);
}

function testAreaLightSingleMapUsesSceneShadowBinding() {
	const scene = new Scene();
	const light = new AreaLight({ range: 80 });
	scene.add(light);
	scene.shadows.bind(light, scene.shadows.createSingle({ size: 1024 }));
	const renderSet = createRenderSetForBoundLight(light);
	updateShadowMapMetadata(renderSet, light, createSceneBounds(70));

	assert.equal(renderSet.effectiveStrategyType, "single-map");
	assert.ok(renderSet.slices[0].shadowMap.viewProjectionMatrix);
	assert.ok(renderSet.slices[0].splitFar > renderSet.slices[0].splitNear);
}

function run() {
	testCSMSplitsMonotonicAndCovered();
	testLambdaBoundarySplits();
	testDirectionalCSMSingleCascadeCoversMaxDistance();
	testCSMStabilizedExtentIsCameraRotationInvariant();
	testOrthographicCSMUsesOrthoBoundsInsteadOfFov();
	testDirectionalCSMDepthIncludesCasterBounds();
	testCSMStabilizedViewTranslationSnapsToTexelGrid();
	testCSMUnstabilizedViewTranslationRemainsContinuous();
	testBlendRatioNormalization();
	testBackendFallbackToSingleMap();
	testUnsupportedPositionalCSMFallbackToSingleMap();
	testCSMSelectionPriority();
	testSpotLightCSMUsesCascadeSlices();
	testPointLightSingleMapUsesSceneShadowBinding();
	testAreaLightSingleMapUsesSceneShadowBinding();
	console.log("Shadow strategy CSM tests passed");
}

run();
