import assert from "node:assert/strict";
import {
	ShaderSource
} from "../../../src/shaders/ShaderSource.ts";
import {
	collectWebGPUEnvironment,
	collectWebGPULightingCatalog,
	collectWebGPULighting,
	createWebGPUClusteredLightingData,
	createWebGPUFrameFeatureRegistry,
	createWebGPULightingState,
	WEBGPU_VOLUMETRIC_LIGHTING_DATA
} from "../../../src/backends/webgpu/index.ts";
import {
	LightProbe
} from "../../../src/lights/LightProbe.ts";
import {
	AreaLight
} from "../../../src/lights/AreaLight.ts";
import {
	DirectionalLight
} from "../../../src/lights/DirectionalLight.ts";
import {
	PointLight
} from "../../../src/lights/PointLight.ts";
import {
	SpotLight
} from "../../../src/lights/SpotLight.ts";
import {
	ReflectionProbe
} from "../../../src/lights/ReflectionProbe.ts";
import {
	Matrix4
} from "../../../src/maths/Matrix4.ts";
import {
	SH
} from "../../../src/maths/SH.ts";
import {
	Logger
} from "../../../src/foundation/Logger.ts";
import {
	Node
} from "../../../src/core/Node.ts";
import {
	ShadowMap,
	createShadowRenderSet
} from "../../../src/lights/shadows/ShadowMapping.ts";
import {
	MAX_AREA_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS
} from "../../../src/backends/constants.ts";
import {
	WEBGPU_CLUSTERED_LIGHT_TYPE_AREA,
	WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT
} from "../../../src/backends/webgpu/constants.ts";

import {
	createEnvironmentSnapshot,
	createTinyCubeTexture,
	createTinyTexture
} from "../../helpers/webgpu-bridge.mjs";
const previousGPUShaderStage = globalThis.GPUShaderStage;
globalThis.GPUShaderStage = {
	...(previousGPUShaderStage ?? {}),
	VERTEX: previousGPUShaderStage?.VERTEX ?? 1,
	FRAGMENT: previousGPUShaderStage?.FRAGMENT ?? 2,
	COMPUTE: previousGPUShaderStage?.COMPUTE ?? 4,
};
ShaderSource.resetConfiguration();
Logger.reset();

function testEnvironmentCollection() {
	const environment = createTinyTexture(1);
	const probeMap = createTinyTexture(3);
	const sh = SH.empty();
	sh[0] = { r: 10, g: 10, b: 10 };
	const probeA = new ReflectionProbe({
		shape: "box",
		prefilteredMap: probeMap,
	});
	const probeB = new ReflectionProbe({
		shape: "sphere",
		prefilteredMap: createTinyTexture(3),
	});

	const prioritized = collectWebGPUEnvironment(
		{
			environment: createEnvironmentSnapshot(environment),
			lights: [probeA, probeB],
		},
		true,
		sh
	);
	assert.equal(prioritized.environmentTexture, environment);
	assert.ok(prioritized.envSpecularTexture);
	assert.notEqual(prioritized.envSpecularTexture, environment);
	assert.equal(prioritized.envSpecularFallbackTexture, null);
	assert.equal(prioritized.envSpecularMaxMipLevel, 2);
	assert.equal(prioritized.envSpecularFallbackMaxMipLevel, 0);
	assert.equal(prioritized.reflectionProbeCount, 2);
	assert.equal(prioritized.reflectionProbes.length, 2);
	assert.equal(prioritized.hasSHAmbient, true);
	assert.ok(prioritized.brdfLUTTexture);

	const fallback = collectWebGPUEnvironment(
		{
			environment: createEnvironmentSnapshot(null, null),
			lights: [probeA, probeB],
		},
		true,
		sh
	);
	assert.equal(fallback.environmentTexture, null);
	assert.ok(fallback.envSpecularTexture);
	assert.equal(fallback.envSpecularFallbackTexture, null);
	assert.equal(fallback.reflectionProbeCount, 2);

	const failedEnvironment = createTinyTexture(1);
	failedEnvironment.markAsLoadErrorFallback();
	const fallbackFromFailedEnvironment = collectWebGPUEnvironment(
		{
			environment: createEnvironmentSnapshot(failedEnvironment),
			lights: [probeA],
		},
		true,
		sh
	);
	assert.equal(fallbackFromFailedEnvironment.environmentTexture, null);
	assert.ok(fallbackFromFailedEnvironment.envSpecularTexture);
	assert.equal(fallbackFromFailedEnvironment.reflectionProbeCount, 1);
	assert.ok(
		fallbackFromFailedEnvironment.warnings.some(
			(warning) =>
				warning.key === "webgpu-environment-background-load-error-fallback"
		)
	);

	const failedOnlyEnvironment = collectWebGPUEnvironment(
		{
			environment: createEnvironmentSnapshot(failedEnvironment, null),
			lights: [],
		},
		true,
		sh
	);
	assert.equal(failedOnlyEnvironment.environmentTexture, null);
	assert.equal(failedOnlyEnvironment.envSpecularTexture, null);
	assert.equal(failedOnlyEnvironment.reflectionProbeCount, 0);
	assert.ok(failedOnlyEnvironment.brdfLUTTexture);

	const disabledFallback = collectWebGPUEnvironment(
		{
			environment: createEnvironmentSnapshot(environment, environment),
			lights: [],
		},
		true,
		sh
	);
	assert.equal(disabledFallback.environmentTexture, environment);
	assert.equal(disabledFallback.envSpecularTexture, environment);
	assert.equal(disabledFallback.envSpecularFallbackTexture, null);
	assert.equal(disabledFallback.reflectionProbeCount, 0);
	assert.ok(disabledFallback.brdfLUTTexture);
	assert.equal(disabledFallback.envSpecularMaxMipLevel, 0);
}

function testEnvironmentCollectionWithCubeTextures() {
	const environment = createTinyCubeTexture(2, 0.5);
	const probeMap = createTinyCubeTexture(3, 0.75);
	const probe = new ReflectionProbe({
		shape: "sphere",
		prefilteredMap: probeMap,
	});

	const state = collectWebGPUEnvironment(
		{
			environment: createEnvironmentSnapshot(environment),
			lights: [probe],
		},
		false,
		null
	);
	assert.ok(state.environmentTexture);
	assert.ok(state.envSpecularTexture);
	assert.notEqual(state.environmentTexture, environment);
	assert.notEqual(state.envSpecularTexture, probeMap);
	assert.equal(state.environmentTexture.width, 4);
	assert.equal(state.environmentTexture.height, 2);
	assert.equal(state.envSpecularTexture.width, 4);
	assert.equal(state.envSpecularTexture.height, 2);
	assert.equal(state.reflectionProbeCount, 1);
	assert.equal(state.envSpecularMaxMipLevel, 2);
}

function testEnvironmentCollectionUsesParentedProbeCaptureOrigin() {
	const probeMap = createTinyCubeTexture(3, 0.75);
	const model = new Node();
	model.position.set(4, 0, 0);
	const probe = new ReflectionProbe({
		shape: "box",
		prefilteredMap: probeMap,
	});
	model.addChild(probe);
	probe.position.set(2, 0, 0);
	model.updateWorldMatrix();

	const state = collectWebGPUEnvironment(
		{
			environment: null,
			lights: [probe],
		},
		false,
		null
	);
	assert.equal(state.reflectionProbeCount, 1);
	assert.deepEqual(state.reflectionProbes[0].captureWorldPosition, [4, 0, 0]);
}

function testLightProbeDCAmbientFallbackWhenSHDisabled() {
	const sh = SH.empty();
	sh[0] = { r: 120, g: 60, b: 30 };
	const probe = new LightProbe({ sh });
	const withoutSH = collectWebGPULighting([probe], true, false);
	assert.ok(withoutSH.ambientColor[0] > 0);
	assert.ok(withoutSH.ambientColor[1] > 0);

	const withSH = collectWebGPULighting([probe], true, true);
	assert.equal(withSH.ambientColor[0], 0);
	assert.equal(withSH.ambientColor[1], 0);
	assert.equal(withSH.ambientColor[2], 0);
}

function testEnvironmentSynthesizesSHAmbientFromLightProbeWhenMissingFrameSH() {
	const sh = SH.empty();
	sh[0] = { r: 80, g: 40, b: 20 };
	sh[5] = { r: 3, g: 2, b: 1 };
	const probe = new LightProbe({ sh });
	const state = collectWebGPUEnvironment(
		{
			environment: null,
			lights: [probe],
		},
		true,
		null
	);
	assert.equal(state.enableSH, true);
	assert.equal(state.hasSHAmbient, true);
	assert.ok(state.shAmbientCoeffs);
	assert.ok(Math.abs(state.shAmbientCoeffs[0].r - 80) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[0].g - 40) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[0].b - 20) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[5].r - 3) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[5].g - 2) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[5].b - 1) < 1e-6);
}

function testEnvironmentCollectsLocalizedLightProbesWithoutPollutingGlobalSH() {
	const globalSH = SH.empty();
	globalSH[5] = { r: 4, g: 2, b: 1 };
	const globalProbe = new LightProbe({ sh: globalSH });

	const localASh = SH.empty();
	localASh[5] = { r: 90, g: 45, b: 22.5 };
	const localA = new LightProbe({
		sh: localASh,
		shape: "sphere",
		radius: 2,
		priority: 5,
	});
	localA.position.set(0, 0, 0);
	localA.updateWorldMatrix();
	localA.markRuntimeDirty();

	const localBSh = SH.empty();
	localBSh[5] = { r: 60, g: 30, b: 15 };
	const localB = new LightProbe({
		sh: localBSh,
		shape: "box",
		halfExtents: { x: 2, y: 2, z: 2 },
		priority: 5,
	});
	localB.position.set(0.5, 0, 0);
	localB.updateWorldMatrix();
	localB.markRuntimeDirty();

	const state = collectWebGPUEnvironment(
		{
			environment: null,
			lights: [globalProbe, localA, localB],
			camera: {
				getWorldPosition() {
					return { x: 0, y: 0, z: 0 };
				},
			},
		},
		true,
		null
	);
	assert.equal(state.localLightProbeCount, 2);
	assert.equal(state.localLightProbes.length, 2);
	assert.equal(state.hasSHAmbient, true);
	assert.ok(state.shAmbientCoeffs);
	assert.ok(Math.abs(state.shAmbientCoeffs[5].r - 4) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[5].g - 2) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[5].b - 1) < 1e-6);
	assert.equal(state.localLightProbes[0].priority, 5);
}

function testWebGPUShadowBiasAvoidsSlopeOffset() {
	const light = new DirectionalLight();
	const shadowMap = new ShadowMap(1024, {
		shadowBias: 0.008,
		shadowSlopeBias: 0.03,
		shadowTexelBias: 1,
		shadowMaxBias: 0.05,
	});
	shadowMap.viewProjectionMatrix = Matrix4.identity();
	const state = collectWebGPULighting(
		[light],
		true,
		false,
		true,
		new Map([[light, shadowMap]])
	);
	const shadow = state.directionalShadows[0];
	assert.ok(shadow.enabled);
	assert.ok(Math.abs(shadow.depthBias - (0.008 + 1 / 1024)) < 1e-6);
	assert.ok(Math.abs(shadow.slopeBias - 0.03) < 1e-6);
	assert.equal(shadow.pcssEnabled, false);
	assert.equal(shadow.pcssRadius, 0);
	assert.equal(shadow.shadowSamples, 16);
	assert.equal(shadow.shadowSearchSamples, 16);
}

function testWebGPUShadowPCSSParams() {
	const light = new DirectionalLight();
	const shadowMap = new ShadowMap(1024, {
		shadowBias: 0.008,
		shadowSlopeBias: 0.03,
		shadowTexelBias: 1,
		shadowMaxBias: 0.05,
		shadowPCF: 1.5,
		shadowRadius: 6,
		shadowSamples: 20,
		shadowSearchSamples: 14,
	});
	shadowMap.viewProjectionMatrix = Matrix4.identity();
	const state = collectWebGPULighting(
		[light],
		true,
		false,
		true,
		new Map([[light, shadowMap]])
	);
	const shadow = state.directionalShadows[0];
	assert.equal(shadow.pcfRadius, 1.5);
	assert.equal(shadow.pcssEnabled, true);
	assert.equal(shadow.pcssRadius, 6);
	assert.equal(shadow.shadowSamples, 20);
	assert.equal(shadow.shadowSearchSamples, 14);
}

function testWebGPUPointLightLimit() {
	const withinLimit = Array.from(
		{ length: MAX_POINT_LIGHTS },
		() => new PointLight()
	);
	const withinState = collectWebGPULighting(withinLimit, true, false);
	assert.equal(withinState.pointLights.length, MAX_POINT_LIGHTS);
	assert.ok(
		!withinState.warnings.some(
			(warning) => warning.key === "webgpu-point-limit"
		)
	);

	const overLimit = Array.from(
		{ length: MAX_POINT_LIGHTS + 2 },
		() => new PointLight()
	);
	const overState = collectWebGPULighting(overLimit, true, false);
	assert.equal(overState.pointLights.length, MAX_POINT_LIGHTS);
	assert.ok(
		overState.warnings.some((warning) => warning.key === "webgpu-point-limit")
	);
}

function testWebGPUAreaLightCollection() {
	const light = new AreaLight({
		color: { r: 255, g: 128, b: 0 },
		intensity: 2,
		position: { x: 1, y: 2, z: 3 },
		width: 20,
		height: 10,
		range: 50,
	});
	const state = collectWebGPULighting([light], true, false);
	assert.equal(state.areaLights.length, 1);
	assert.equal(
		state.warnings.some((warning) => warning.key === "webgpu-light-rectArea"),
		false
	);

	const area = state.areaLights[0];
	assert.deepEqual(area.position, [1, 2, 3]);
	assert.deepEqual(area.right, [1, 0, 0]);
	assert.deepEqual(area.up, [0, 0, 1]);
	assert.deepEqual(area.normal, [0, 1, 0]);
	assert.equal(area.width, 20);
	assert.equal(area.height, 10);
	assert.equal(area.range, 50);
	assert.equal(area.areaScale, 200);
	assert.equal(area.color[0], 2);
	assert.equal(area.color[2], 0);

	const overLimit = Array.from(
		{ length: MAX_AREA_LIGHTS + 1 },
		() => new AreaLight()
	);
	const overState = collectWebGPULighting(overLimit, true, false);
	assert.equal(overState.areaLights.length, MAX_AREA_LIGHTS);
	assert.ok(
		overState.warnings.some((warning) => warning.key === "webgpu-area-limit")
	);

	const clusteredOverState = collectWebGPULighting(
		overLimit,
		true,
		false,
		false,
		undefined,
		true
	);
	const clusteredOverCatalog = collectWebGPULightingCatalog(
		overLimit,
		true,
		false
	);
	const clusteredOverData = createWebGPUClusteredLightingData(
		clusteredOverCatalog,
		MAX_AREA_LIGHTS + 1
	);
	assert.equal(clusteredOverState.areaLights.length, MAX_AREA_LIGHTS);
	assert.equal(clusteredOverData.lights.length, MAX_AREA_LIGHTS + 1);
	assert.equal(
		clusteredOverData.lights[MAX_AREA_LIGHTS].type,
		WEBGPU_CLUSTERED_LIGHT_TYPE_AREA
	);
	assert.equal(
		clusteredOverState.warnings.some(
			(warning) => warning.key === "webgpu-area-limit"
		),
		false
	);
}

function testWebGPUClusteredSpotShadowBudgetFallback() {
	const lights = Array.from(
		{ length: MAX_SPOT_LIGHTS + 1 },
		() => new SpotLight()
	);
	const shadowMaps = new Map();
	for (const light of lights) {
		const renderSet = createShadowRenderSet({ strategy: "single-map" });
		renderSet.slices[0].shadowMap.viewProjectionMatrix = Matrix4.identity();
		shadowMaps.set(light, renderSet);
	}

	const state = collectWebGPULighting(
		lights,
		true,
		false,
		true,
		shadowMaps,
		true
	);
	const catalog = collectWebGPULightingCatalog(
		lights,
		true,
		false,
		true,
		shadowMaps
	);
	const clusteredData = createWebGPUClusteredLightingData(
		catalog,
		MAX_SPOT_LIGHTS + 1
	);
	const clusteredSpots = clusteredData.lights.filter(
		(light) => light.type === WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT
	);
	assert.equal(state.spotLights.length, MAX_SPOT_LIGHTS);
	assert.equal(state.spotShadows.length, MAX_SPOT_LIGHTS);
	assert.equal(clusteredSpots.length, MAX_SPOT_LIGHTS + 1);
	assert.equal(clusteredSpots[MAX_SPOT_LIGHTS - 1].castsShadow, true);
	assert.equal(clusteredSpots[MAX_SPOT_LIGHTS].castsShadow, false);
	assert.equal(clusteredSpots[MAX_SPOT_LIGHTS].shadowIndex, MAX_SPOT_LIGHTS);
	assert.ok(
		clusteredData.warnings.some(
			(warning) => warning.key === "webgpu-clustered-spot-shadow-budget"
		)
	);
}

function testWebGPUFrameFeatureRegistryGatesVolumetricLighting() {
	const light = new PointLight({ range: 8, intensity: 2 });
	const catalog = collectWebGPULightingCatalog([light], true, false);
	const lightingState = createWebGPULightingState(catalog, false);
	const registry = createWebGPUFrameFeatureRegistry();
	const baseContext = {
		frameContext: {},
		scene: {
			camera: {
				type: "perspective",
				near: 0.1,
				far: 100,
			},
		},
		featureState: {
			enableLighting: true,
			enableClusteredLighting: false,
			clusteredLightingOptions: {},
			postProcess: {
				isEnabled: () => false,
			},
		},
		lightingCatalog: catalog,
		lightingState,
		renderWidth: 64,
		renderHeight: 64,
	};
	const disabledStore = registry.prepareFrame(baseContext);
	assert.equal(disabledStore.has(WEBGPU_VOLUMETRIC_LIGHTING_DATA), false);

	const enabledStore = registry.prepareFrame({
		...baseContext,
		featureState: {
			...baseContext.featureState,
			postProcess: {
				isEnabled: (id) => id === "volumetric",
			},
		},
	});
	const volumetric = enabledStore.get(WEBGPU_VOLUMETRIC_LIGHTING_DATA);
	assert.ok(volumetric);
	assert.equal(volumetric.lights.length, 1);
	assert.equal(volumetric.lights[0].type, 1);

	const secondLight = new PointLight({ range: 16, intensity: 4 });
	const clusteredCatalog = collectWebGPULightingCatalog(
		[light, secondLight],
		true,
		false
	);
	const clusteredLightingState = createWebGPULightingState(
		clusteredCatalog,
		true
	);
	const clusteredStore = registry.prepareFrame({
		...baseContext,
		featureState: {
			...baseContext.featureState,
			enableClusteredLighting: true,
			clusteredLightingOptions: { maxLights: 1 },
			postProcess: {
				isEnabled: (id) => id === "volumetric",
			},
		},
		lightingCatalog: clusteredCatalog,
		lightingState: clusteredLightingState,
	});
	const clusteredVolumetric = clusteredStore.get(WEBGPU_VOLUMETRIC_LIGHTING_DATA);
	assert.ok(clusteredVolumetric);
	assert.equal(clusteredVolumetric.lights.length, 1);
}

async function run() {
	try {
		await testEnvironmentCollection();
		await testEnvironmentCollectionWithCubeTextures();
		await testEnvironmentCollectionUsesParentedProbeCaptureOrigin();
		await testLightProbeDCAmbientFallbackWhenSHDisabled();
		await testEnvironmentSynthesizesSHAmbientFromLightProbeWhenMissingFrameSH();
		await testEnvironmentCollectsLocalizedLightProbesWithoutPollutingGlobalSH();
		await testWebGPUShadowBiasAvoidsSlopeOffset();
		await testWebGPUShadowPCSSParams();
		await testWebGPUPointLightLimit();
		await testWebGPUAreaLightCollection();
		await testWebGPUClusteredSpotShadowBudgetFallback();
		await testWebGPUFrameFeatureRegistryGatesVolumetricLighting();
		console.log("WebGPU bridge lighting/environment tests passed");
	} finally {
		ShaderSource.resetConfiguration();
		Logger.reset();
		if (previousGPUShaderStage === undefined) {
			delete globalThis.GPUShaderStage;
		} else {
			globalThis.GPUShaderStage = previousGPUShaderStage;
		}
	}
}
await run();
