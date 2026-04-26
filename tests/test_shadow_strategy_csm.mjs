import assert from "node:assert/strict";
import { AreaLight } from "../src/lights/AreaLight.ts";
import { DirectionalLight } from "../src/lights/DirectionalLight.ts";
import { PointLight } from "../src/lights/PointLight.ts";
import { SpotLight } from "../src/lights/SpotLight.ts";
import {
	createShadowRenderSet,
	normalizeShadowConfig,
} from "../src/lights/ShadowMapping.ts";
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
} = {}) {
	const light = new DirectionalLight({
		intensity,
		direction: { x: 0, y: -1, z: -0.3 },
	});
	light.castShadow = true;
	light.shadow = {
		strategy: "csm",
		size: 1024,
		priority,
		lambda,
		maxDistance,
		blendRatio,
		cascadeCount,
		stabilize: true,
	};
	return light;
}

function createSpotCSMLight({
	priority = 0,
	intensity = 1,
	range = 80,
} = {}) {
	const light = new SpotLight({
		intensity,
		range,
		direction: { x: 0, y: -1, z: 0 },
		outerAngle: Math.PI / 3,
	});
	light.castShadow = true;
	light.shadow = {
		strategy: "csm",
		size: 1024,
		priority,
		lambda: 0.65,
		blendRatio: 0.1,
		cascadeCount: 4,
		stabilize: true,
	};
	return light;
}

function testCSMSplitsMonotonicAndCovered() {
	const camera = createCamera({ near: 0.2, far: 120 });
	const light = createDirectionalCSMLight({
		lambda: 0.65,
		maxDistance: 80,
		cascadeCount: 4,
	});
	const renderSet = createShadowRenderSet(light.shadow);

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
	const uniformSet = createShadowRenderSet(uniformLight.shadow);
	updateShadowMapMetadata(uniformSet, uniformLight, bounds, { camera });

	const logLight = createDirectionalCSMLight({
		lambda: 1,
		maxDistance: 100,
		cascadeCount: 4,
	});
	const logSet = createShadowRenderSet(logLight.shadow);
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
	const renderSetA = createShadowRenderSet(light.shadow);
	const renderSetB = createShadowRenderSet(light.shadow);

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
	const renderSet = createShadowRenderSet(light.shadow);
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

function testSpotLightCSMUsesSingleSliceEquivalentPath() {
	const light = createSpotCSMLight();
	const renderSet = createShadowRenderSet(light.shadow);
	const emptyDirectionalBudget = new Set();

	updateShadowMapMetadata(renderSet, light, createSceneBounds(60), {
		backendCapabilities: {
			backendKey: "webgpu",
			supportsSingleMap: true,
			supportsDirectionalCSM: true,
			maxCsmDirectionalLights: 0,
		},
		allowCSMDirectionalLights: emptyDirectionalBudget,
	});

	assert.equal(renderSet.requestedStrategyType, "csm");
	assert.equal(renderSet.effectiveStrategyType, "csm");
	assert.ok(renderSet.slices[0].shadowMap.viewProjectionMatrix);
	for (let index = 1; index < renderSet.slices.length; index++) {
		assert.equal(renderSet.slices[index].shadowMap.viewProjectionMatrix, null);
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
	assert.equal(webgpuLighting.spotShadows[0]?.cascadeCount, 1);

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
	assert.equal(webglLighting.spotShadows[0]?.cascadeCount, 1);
}

function testPointLightSingleMapUsesCastShadowProperty() {
	const light = new PointLight({
		range: 60,
		castShadow: true,
	});
	assert.ok(light.shadow, "PointLight should provide a default shadow config.");

	const renderSet = createShadowRenderSet(light.shadow);
	updateShadowMapMetadata(renderSet, light, createSceneBounds(50));

	assert.equal(renderSet.effectiveStrategyType, "single-map");
	assert.ok(renderSet.slices[0].shadowMap.viewProjectionMatrix);
	assert.ok(renderSet.slices[0].splitFar > renderSet.slices[0].splitNear);
}

function testAreaLightSingleMapUsesCastShadowProperty() {
	const light = new AreaLight({
		range: 80,
		castShadow: true,
	});
	assert.ok(light.shadow, "AreaLight should provide a default shadow config.");

	const renderSet = createShadowRenderSet(light.shadow);
	updateShadowMapMetadata(renderSet, light, createSceneBounds(70));

	assert.equal(renderSet.effectiveStrategyType, "single-map");
	assert.ok(renderSet.slices[0].shadowMap.viewProjectionMatrix);
	assert.ok(renderSet.slices[0].splitFar > renderSet.slices[0].splitNear);
}

function run() {
	testCSMSplitsMonotonicAndCovered();
	testLambdaBoundarySplits();
	testCSMStabilizedExtentIsCameraRotationInvariant();
	testBlendRatioNormalization();
	testBackendFallbackToSingleMap();
	testCSMSelectionPriority();
	testSpotLightCSMUsesSingleSliceEquivalentPath();
	testPointLightSingleMapUsesCastShadowProperty();
	testAreaLightSingleMapUsesCastShadowProperty();
	console.log("Shadow strategy CSM tests passed");
}

run();
