import assert from "node:assert/strict";
import { AmbientLight } from "../../../src/lights/AmbientLight.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { LightProbe } from "../../../src/lights/LightProbe.ts";
import { PointLight } from "../../../src/lights/PointLight.ts";
import { ReflectionProbe } from "../../../src/lights/ReflectionProbe.ts";
import { SpotLight } from "../../../src/lights/SpotLight.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { SH } from "../../../src/maths/SH.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { Node } from "../../../src/core/Node.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { collectWebGLLights } from "../../../src/backends/webgl/WebGLLightCollector.ts";
import { planWebGLScenePrograms } from "../../../src/backends/webgl/WebGLSceneProgramPlanner.ts";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../../../src/backends/constants.ts";
import { createTinyCubeTexture, runWebGLBackendFile } from "../../helpers/webgl-backend.mjs";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function testLightCollectorLimitsAndWarnings() {
	const warnings = [];
	const warn = (key, message) => warnings.push({ key, message });
	const lights = [new AmbientLight()];

	for (let i = 0; i < MAX_DIRECTIONAL_LIGHTS + 2; i++) {
		lights.push(new DirectionalLight({ intensity: 1 + i * 0.1 }));
	}
	for (let i = 0; i < MAX_POINT_LIGHTS + 2; i++) {
		lights.push(new PointLight({ range: 100 + i }));
	}
	for (let i = 0; i < MAX_SPOT_LIGHTS + 2; i++) {
		lights.push(new SpotLight({ range: 100 + i }));
	}

	const state = collectWebGLLights(lights, {
		enableLighting: true,
		warn,
	});
	assert.equal(state.directionalLights.length, MAX_DIRECTIONAL_LIGHTS);
	assert.equal(state.pointLights.length, MAX_POINT_LIGHTS);
	assert.equal(state.spotLights.length, MAX_SPOT_LIGHTS);
	assert.ok(warnings.some((warning) => warning.key === "webgl-directional-light-limit"));
	assert.ok(warnings.some((warning) => warning.key === "webgl-point-light-limit"));
	assert.ok(warnings.some((warning) => warning.key === "webgl-spot-light-limit"));
}

function testLightProbeAmbientAndReflectionProbeSpecularCollection() {
	const warnings = [];
	const warn = (key, message) => warnings.push({ key, message });
	const sh = SH.empty();
	sh[0] = { r: 120, g: 60, b: 30 };
	const probeMap = new Texture({
		data: new Float32Array(4 * 2 * 4),
		width: 4,
		height: 2,
		colorSpace: "HDR",
	});
	probeMap.mipmaps = [new Float32Array(4 * 2 * 4), new Float32Array(2 * 1 * 4)];
	const lightProbe = new LightProbe({ sh });
	const reflectionProbe = new ReflectionProbe({
		prefilteredMap: probeMap,
		shape: "box",
	});

	const withoutSH = collectWebGLLights([lightProbe, reflectionProbe], {
		enableLighting: true,
		warn,
	});
	assert.ok(withoutSH.ambientColor[0] > 0);
	assert.ok(withoutSH.ambientColor[1] > 0);
	assert.ok(withoutSH.ambientColor[2] > 0);
	assert.ok(withoutSH.envSpecularMap);
	assert.equal(withoutSH.reflectionProbeCount, 1);
	assert.equal(withoutSH.reflectionProbes.length, 1);
	assert.equal(
		warnings.some((warning) => warning.key === "webgl-light-unsupported-lightProbe"),
		false,
	);

	const withSH = collectWebGLLights([lightProbe, reflectionProbe], {
		enableLighting: true,
		warn,
		enableSH: true,
	});
	assert.equal(withSH.ambientColor[0], 0);
	assert.equal(withSH.ambientColor[1], 0);
	assert.equal(withSH.ambientColor[2], 0);
}

function testLightCollectorCollectsLocalizedLightProbes() {
	const warnings = [];
	const warn = (key, message) => warnings.push({ key, message });
	const globalProbe = new LightProbe({ sh: SH.empty() });

	const localProbe = new LightProbe({
		sh: SH.empty(),
		shape: "box",
		halfExtents: { x: 2, y: 2, z: 2 },
		priority: 7,
	});
	localProbe.position.set(0, 0, 0);
	localProbe.updateWorldMatrix();
	localProbe.markRuntimeDirty();

	const state = collectWebGLLights([globalProbe, localProbe], {
		enableLighting: true,
		warn,
		enableSH: true,
		cameraWorldPosition: { x: 0, y: 0, z: 0 },
	});
	assert.equal(state.localLightProbeCount, 1);
	assert.equal(state.localLightProbes.length, 1);
	assert.equal(state.localLightProbes[0].priority, 7);
	assert.equal(state.localLightProbes[0].shape, 1);
	assert.equal(state.ambientColor[0], 0);
	assert.equal(warnings.length, 0);
}

function testLightCollectorSupportsCubeTextureEnvironmentMaps() {
	const warn = () => {};
	const cubeProbeMap = createTinyCubeTexture(3, 0.75);
	const cubeEnvironment = createTinyCubeTexture(2, 0.5);
	const reflectionProbe = new ReflectionProbe({
		prefilteredMap: cubeProbeMap,
		shape: "sphere",
	});

	const probeState = collectWebGLLights([reflectionProbe], {
		enableLighting: true,
		warn,
		environmentTexture: cubeEnvironment,
	});
	assert.ok(probeState.envSpecularMap);
	assert.notEqual(probeState.envSpecularMap, cubeProbeMap);
	assert.equal(probeState.envSpecularFallbackMap, null);
	assert.equal(probeState.envSpecularMap.width, 4);
	assert.equal(probeState.envSpecularMap.height, 2);
	assert.equal(probeState.reflectionProbeCount, 1);

	const environmentState = collectWebGLLights([], {
		enableLighting: true,
		warn,
		environmentTexture: cubeEnvironment,
	});
	assert.ok(environmentState.envSpecularMap);
	assert.notEqual(environmentState.envSpecularMap, cubeEnvironment);
	assert.equal(environmentState.envSpecularFallbackMap, null);
	assert.equal(environmentState.envSpecularMap.width, 4);
	assert.equal(environmentState.envSpecularMap.height, 2);
	assert.equal(environmentState.reflectionProbeCount, 0);
}

function testLightCollectorUsesParentedProbeCaptureOrigin() {
	const warn = () => {};
	const probeMap = createTinyCubeTexture(3, 0.75);
	const model = new Node();
	model.position.set(5, 0, 0);
	const probe = new ReflectionProbe({
		prefilteredMap: probeMap,
		shape: "box",
	});
	model.addChild(probe);
	probe.position.set(2, 0, 0);
	model.updateWorldMatrix();

	const state = collectWebGLLights([probe], {
		enableLighting: true,
		warn,
	});
	assert.equal(state.reflectionProbeCount, 1);
	assert.deepEqual(state.reflectionProbes[0].captureWorldPosition, [5, 0, 0]);
}

function testLightCollectorUsesProbeCaptureOriginWhenParentedToSceneRoot() {
	const warn = () => {};
	const probeMap = createTinyCubeTexture(3, 0.75);
	const scene = new Scene();
	const probe = new ReflectionProbe({
		prefilteredMap: probeMap,
		shape: "box",
	});
	scene.add(probe);
	probe.position.set(5, 0, 0);
	scene.updateWorldMatrices();

	const state = collectWebGLLights([probe], {
		enableLighting: true,
		warn,
	});
	assert.equal(state.reflectionProbeCount, 1);
	assert.deepEqual(state.reflectionProbes[0].captureWorldPosition, [5, 0, 0]);
}

function testLightCollectorDoesNotExposeEnvironmentSpecularFallbackMap() {
	const warnings = [];
	const warn = (key, message) => warnings.push({ key, message });
	const environment = createTinyCubeTexture(2, 0.5);
	const state = collectWebGLLights([], {
		enableLighting: true,
		warn,
		environmentTexture: environment,
	});
	assert.ok(state.envSpecularMap);
	assert.equal(state.envSpecularFallbackMap, null);
	assert.equal(state.reflectionProbeCount, 0);
	assert.equal(state.reflectionProbes.length, 0);
	assert.equal(
		warnings.some((warning) => warning.key.startsWith("webgl-environment-")),
		false,
	);
}

function testLightCollectorShadowBias() {
	const light = new DirectionalLight();
	const shadowPlan = createShadowPlan(light, { size: 1024 });
	const state = collectWebGLLights([light], {
		enableLighting: true,
		warn: () => {},
		enableShadows: true,
		shadowPlan,
	});
	const shadow = state.directionalShadows[0];
	assert.ok(shadow.enabled);
	assert.ok(Math.abs(shadow.depthBias - (0.008 + 1 / 1024)) < 1e-6);
	assert.ok(Math.abs(shadow.slopeBias - 0.03) < 1e-6);
	assert.equal(shadow.shadowMapSize, 1024);
	assert.equal(shadow.pcssEnabled, false);
	assert.equal(shadow.pcssRadius, 0);
	assert.equal(shadow.shadowSamples, 16);
	assert.equal(shadow.shadowSearchSamples, 16);
}

function testLightCollectorPCSSShadowParams() {
	const light = new DirectionalLight();
	const shadowPlan = createShadowPlan(light, {
		size: 1024,
		sampling: { pcfRadius: 1.25, radius: 5, samples: 24, searchSamples: 12 },
	});
	const state = collectWebGLLights([light], {
		enableLighting: true,
		warn: () => {},
		enableShadows: true,
		shadowPlan,
	});
	const shadow = state.directionalShadows[0];
	assert.equal(shadow.pcfRadius, 1.25);
	assert.equal(shadow.pcssEnabled, true);
	assert.equal(shadow.pcssRadius, 5);
	assert.equal(shadow.shadowSamples, 24);
	assert.equal(shadow.shadowSearchSamples, 12);
}

function testLightCollectorDirectionalCSMShadowData() {
	const light = new DirectionalLight();
	const shadowPlan = createShadowPlan(light, {
		size: 512,
		sliceSize: 256,
		cascades: 4,
		blendRatio: 0.2,
	});

	const state = collectWebGLLights([light], {
		enableLighting: true,
		warn: () => {},
		enableShadows: true,
		shadowPlan,
	});
	const shadow = state.directionalShadows[0];
	assert.equal(shadow.enabled, true);
	assert.equal(shadow.strategyType, "csm");
	assert.equal(shadow.cascadeCount, 4);
	assert.equal(shadow.cascadeBlendRatio, 0.2);
	assert.equal(shadow.shadowMapBaseSize, 512);
	assert.equal(shadow.shadowMapSize, 256);
	assert.ok(shadow.cascadeViewProjectionMatrices[3]);
	assert.deepEqual(shadow.cascadeSplits[0], [0, 10, 0, 0]);
	assert.deepEqual(shadow.cascadeSplits[1], [10, 20, 1, 0]);
	assert.deepEqual(shadow.cascadeSplits[2], [20, 30, 0, 1]);
	assert.deepEqual(shadow.cascadeSplits[3], [30, 40, 1, 1]);
}

function createShadowPlan(light, options = {}) {
	const size = options.size ?? 1024;
	const sliceSize = options.sliceSize ?? size;
	const cascades = options.cascades ?? 1;
	const slices = Array.from({ length: cascades }, (_, index) => ({
		index, resolution: sliceSize, view: Matrix4.identity(), projection: Matrix4.identity(), viewProjection: Matrix4.identity(), lightDirection: { x: 0, y: -1, z: 0 }, splitNear: index * 10, splitFar: (index + 1) * 10,
	}));
	return { revision: 1, jobs: [], diagnostics: [], hasRasterWork: false, hasTransmissionWork: false, hasPagedWork: false, lights: [{ light, lightId: light.id, definition: { bias: { constant: 0.008, slope: 0.03, texel: 1, max: 0.05, normal: 1, normalMin: 0.05 }, sampling: { filterMode: "pcf", pcfRadius: 1, radius: 0, strength: 1, samples: 16, searchSamples: 16 }, projection: { blendRatio: options.blendRatio ?? 0 } }, requestedTechnique: cascades > 1 ? "cascaded" : "single", effectiveTechnique: cascades > 1 ? "cascaded" : "single", requestedCascadeCount: cascades, effectiveCascadeCount: cascades, requestedResolution: size, effectiveResolution: size, sampling: options.sampling ?? { filterMode: "pcf", pcfRadius: 1, radius: 0, strength: 1, samples: 16, searchSamples: 16 }, filterMode: "pcf", storage: "atlas", priority: 0, cost: 1, score: 1, slices }] };
}

function testSceneProgramPlannerEnumeratesRuntimeTransmittanceAlternatives() {
	const light = new DirectionalLight();
	const material = new PBRMaterial();
	const context = {
		features: {
			enableLighting: true,
			enableShadows: true,
			enableSH: false,
			enableClusteredLighting: false,
			enableOIT: false,
		},
		shadowPlan: createShadowPlan(light),
		viewCamera: {
			getWorldPosition() {
				return { x: 0, y: 0, z: 0 };
			},
		},
		scene: {
			lights: [light],
			environment: { lightingEnabled: false, iblTexture: null },
		},
	};
	const plan = planWebGLScenePrograms(context, [material], ["single", "mrt"]);
	const keys = [...plan.sceneVariants.keys()];
	assert.ok(keys.some((key) => key.includes("shdt:0")));
	assert.ok(keys.some((key) => key.includes("shdt:1")));
	assert.ok(keys.some((key) => key.includes("out:single")));
	assert.ok(keys.some((key) =>
		key.includes("gbuf:1") && key.includes("out:mrt")
	));
}

function testSceneProgramPlannerReadsMaterialFromDrawSubmission() {
	const material = new PBRMaterial();
	material.map = { id: "base-map" };
	const context = {
		features: {
			enableLighting: false,
			enableShadows: false,
			enableSH: false,
			enableClusteredLighting: false,
			enableOIT: false,
		},
		viewCamera: {
			getWorldPosition() {
				return { x: 0, y: 0, z: 0 };
			},
		},
		scene: {
			lights: [],
			environment: { lightingEnabled: false, iblTexture: null },
		},
	};
	const packet = createTestDrawPacket({ material });
	const plan = planWebGLScenePrograms(context, [packet], ["mrt"]);
	const keys = [...plan.sceneVariants.keys()];

	assert.ok(keys.some((key) =>
		key.includes("out:mrt") && key.includes("base:1")
	));
	assert.equal(keys.some((key) => key.includes("base:0")), false);
}

await runWebGLBackendFile(
	[
		testLightCollectorLimitsAndWarnings,
		testLightProbeAmbientAndReflectionProbeSpecularCollection,
		testLightCollectorCollectsLocalizedLightProbes,
		testLightCollectorSupportsCubeTextureEnvironmentMaps,
		testLightCollectorUsesParentedProbeCaptureOrigin,
		testLightCollectorUsesProbeCaptureOriginWhenParentedToSceneRoot,
		testLightCollectorDoesNotExposeEnvironmentSpecularFallbackMap,
		testLightCollectorShadowBias,
		testLightCollectorPCSSShadowParams,
		testLightCollectorDirectionalCSMShadowData,
		testSceneProgramPlannerEnumeratesRuntimeTransmittanceAlternatives,
		testSceneProgramPlannerReadsMaterialFromDrawSubmission,
	],
	"WebGL light collection tests",
);
