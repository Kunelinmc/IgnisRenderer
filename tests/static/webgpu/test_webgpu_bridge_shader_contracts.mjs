import assert from "node:assert/strict";
import {
	readFileSync
} from "node:fs";
import {
	ShaderSource
} from "../../../src/shaders/ShaderSource.ts";
import {
	ShaderBackendCompileStage,
	ShaderRuntime
} from "../../../src/shaders/runtime/index.ts";
import { WEBGPU_TEST_PROFILE } from "../shaders/shaderDirectiveTestProfiles.mjs";
import {
	createWebGPUMaterialUniformData,
	materialSupportsWebGPUDeferredLighting,
	packMatrix4ForWGSL,
	remapClipSpaceDepth
} from "../../../src/backends/webgpu/index.ts";
import {
	WebGPUBackend
} from "../../../src/backends/webgpu/WebGPUBackend.ts";
import {
	createWebGPUPipelineLayouts
} from "../../../src/backends/webgpu/WebGPUPipelineLayouts.ts";
import {
	WEBGPU_FLAT_MATERIAL_UNIFORM_LAYOUT,
	WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT,
	WEBGPU_OBJECT_UNIFORM_LAYOUT,
	WEBGPU_PBR_MATERIAL_UNIFORM_LAYOUT,
	WEBGPU_PHONG_MATERIAL_UNIFORM_LAYOUT,
} from "../../../src/backends/webgpu/bufferLayouts.ts";
import {
	createWebGPURequiredLimits
} from "../../../src/backends/webgpu/WebGPUDeviceCapabilities.ts";
import {
	resolveFeatureState
} from "../../../src/pipeline/FeatureResolver.ts";
import {
	Matrix4
} from "../../../src/maths/Matrix4.ts";
import {
	PBRMaterial,
	PBRMaterialFeature,
	PBRMaterialTextureFeature
} from "../../../src/materials/PBRMaterial.ts";
import {
	Material,
	ShadingModel
} from "../../../src/materials/Material.ts";
import {
	PhongMaterial
} from "../../../src/materials/PhongMaterial.ts";
import {
	ShaderMaterial
} from "../../../src/materials/ShaderMaterial.ts";
import {
	Texture
} from "../../../src/core/Texture.ts";
import {
	Logger
} from "../../../src/foundation/Logger.ts";
import {
	UnlitMaterial
} from "../../../src/materials/UnlitMaterial.ts";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_AREA_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS
} from "../../../src/backends/constants.ts";
import {
	WEBGPU_DEFERRED_BASE_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_TARGET_COUNT,
	WEBGPU_DEFERRED_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT,
	WEBGPU_GBUFFER_READ_TEXTURE_COUNT,
	WEBGPU_MODEL_BINDING_FLAT_MATERIAL,
	WEBGPU_MODEL_BINDING_MATERIAL_COMMON,
	WEBGPU_MODEL_BINDING_PBR_MATERIAL,
	WEBGPU_MODEL_BINDING_PHONG_MATERIAL,
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_SCENE_FRAME_FRAGMENT_TEXTURE_COUNT,
	WEBGPU_PLANAR_REFLECTION_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_DECAL_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLER_COUNT,
	WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_TEXTURE_SLOT,
	WEBGPU_TEXTURE_SLOT_COUNT,
	GBufferSlot
} from "../../../src/backends/webgpu/constants.ts";
import {
	createResolvedPostProcess
} from "../../helpers/postprocess.mjs";

const previousGPUShaderStage = globalThis.GPUShaderStage;
globalThis.GPUShaderStage = {
	...(previousGPUShaderStage ?? {}),
	VERTEX: previousGPUShaderStage?.VERTEX ?? 1,
	FRAGMENT: previousGPUShaderStage?.FRAGMENT ?? 2,
	COMPUTE: previousGPUShaderStage?.COMPUTE ?? 4,
};

function readWGSLStructFieldNames(source, structName) {
	const match = source.match(new RegExp(`struct\\s+${structName}\\s*\\{([\\s\\S]*?)\\}`));
	assert.ok(match, `missing WGSL struct ${structName}`);
	return [...match[1].matchAll(/^\s*([A-Za-z_]\w*)\s*:/gm)].map((entry) => entry[1]);
}

function readLayoutFieldNames(layout) {
	assert.equal(layout.rootSchema.kind, "struct");
	return layout.rootSchema.fields.map((field) => field.name);
}
ShaderSource.resetConfiguration();
Logger.reset();

function testBackendDoesNotExposeDeviceServiceForwarders() {
	for (const method of [
		"device",
		"queue",
		"getShaderDirectiveCacheTag",
		"isOcclusionCullingEnabled",
		"onDeviceLost",
		"getFrameSceneTargetMode",
		"captureProbeFace",
		"getCurrentColorView",
		"getCurrentDepthView",
		"getTimestampDurationsMs",
		"createPassTimestampWrites",
		"createBuffer",
		"createTexture",
		"createSampler",
		"createShaderModule",
		"createPipeline",
		"createComputePipeline",
		"createBindingGroup",
		"createTextureView",
		"createCommandEncoder",
		"writeBuffer",
		"writeTexture",
		"copyTextureToTexture",
		"submit",
		"getTextureForSlot",
		"registerExternalTexture",
		"unregisterExternalTexture",
		"postProcessRuntime",
	]) {
		assert.equal(method in WebGPUBackend.prototype, false, method);
	}
}

function testMatrixPackingAndDepthRemap() {
	const matrix = new Matrix4([
		[1, 2, 3, 4],
		[5, 6, 7, 8],
		[9, 10, 11, 12],
		[13, 14, 15, 16],
	]);

	assert.deepEqual(
		Array.from(packMatrix4ForWGSL(matrix)),
		[1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
	);
	assert.equal(remapClipSpaceDepth(-1, 1), 0);
	assert.equal(remapClipSpaceDepth(1, 1), 1);
}

function testTransformComposition() {
	const transform = {
		position: { x: 3, y: -2, z: 5 },
		rotation: { x: Math.PI / 4, y: Math.PI / 6, z: -Math.PI / 3 },
		scale: { x: 2, y: 3, z: 4 },
	};
	const expected = Matrix4.multiply(
		Matrix4.fromTranslation([
			transform.position.x,
			transform.position.y,
			transform.position.z,
		]),
		Matrix4.multiply(
			Matrix4.rotationFromEuler(
				transform.rotation.x,
				transform.rotation.y,
				transform.rotation.z
			),
			Matrix4.fromScale([
				transform.scale.x,
				transform.scale.y,
				transform.scale.z,
			])
		)
	);

	assert.deepEqual(
		Matrix4.fromTransform(transform).elements,
		expected.elements
	);
}

function testMaterialAdaptation() {
	const pbr = new PBRMaterial({
		albedo: { r: 128, g: 64, b: 32 },
		roughness: 0.25,
		metalness: 0.75,
		reflectance: 0.6,
	});
	const pbrData = createWebGPUMaterialUniformData(pbr);
	assert.equal(pbrData.shadingFamily, "pbr");
	assert.ok(Math.abs(pbrData.common.baseColorFactor[0] - 128 / 255) < 1e-6);
	assert.ok(Math.abs(pbrData.lighting.surfaceParams0[0] - 0.25) < 1e-6);
	assert.ok(Math.abs(pbrData.lighting.surfaceParams0[1] - 0.75) < 1e-6);
	assert.ok(Math.abs(pbrData.lighting.surfaceParams0[2] - 0.6) < 1e-6);
	assert.equal(pbrData.common.textureSlots.length, WEBGPU_TEXTURE_SLOT_COUNT);
	assert.deepEqual(pbrData.lighting.pbrMasks, [0, 0, 0, 0]);
	pbr.albedoMapUV = 2;
	pbr.normalMapUV = 3;
	const pbrUVData = createWebGPUMaterialUniformData(pbr);
	assert.equal(pbrUVData.common.textureSlots[0].transformB[1], 2);
	assert.equal(pbrUVData.common.textureSlots[2].transformB[1], 3);
	pbr.anisotropyStrength = 0.75;
	pbr.anisotropyRotation = Math.PI / 2;
	pbr.anisotropyMap = new Texture({
		data: new Uint8ClampedArray([255, 128, 128, 255]),
		width: 1,
		height: 1,
		colorSpace: "Linear",
	});
	pbr.anisotropyMapUV = 2;
	const pbrAnisotropyData = createWebGPUMaterialUniformData(pbr);
	assert.ok(Math.abs(pbrAnisotropyData.lighting.anisotropyParams[0] - 0.75) < 1e-6);
	assert.ok(Math.abs(pbrAnisotropyData.lighting.anisotropyParams[1]) < 1e-6);
	assert.ok(Math.abs(pbrAnisotropyData.lighting.anisotropyParams[2] - 1) < 1e-6);
	assert.equal(
		pbrAnisotropyData.common.textureSlots[WEBGPU_TEXTURE_SLOT.ANISOTROPY].transformB[1],
		2
	);
	assert.equal(
		pbrAnisotropyData.common.textureSlots[WEBGPU_TEXTURE_SLOT.ANISOTROPY].transformB[3],
		0
	);
	assert.equal(
		pbrAnisotropyData.common.textureSlots[WEBGPU_TEXTURE_SLOT.ANISOTROPY].map,
		pbr.anisotropyMap
	);
	assert.equal(
		pbrAnisotropyData.lighting.pbrMasks[0],
		PBRMaterialFeature.ANISOTROPY
	);
	assert.equal(
		pbrAnisotropyData.lighting.pbrMasks[1],
		PBRMaterialTextureFeature.ANISOTROPY_MAP
	);
	assert.equal(pbrAnisotropyData.pipelineKey, pbrData.pipelineKey);
	assert.equal(materialSupportsWebGPUDeferredLighting(pbr), true);

	const genericPBR = new Material({
		shading: ShadingModel.PBR,
		map: new Texture({
			data: new Uint8ClampedArray([255, 255, 255, 255]),
			width: 1,
			height: 1,
		}),
	});
	const genericPBRData = createWebGPUMaterialUniformData(genericPBR);
	assert.deepEqual(genericPBRData.lighting.pbrMasks, [1, 1, 0, 0]);

	const transmissivePBR = new PBRMaterial({ transmissionFactor: 1 });
	assert.equal(materialSupportsWebGPUDeferredLighting(transmissivePBR), false);

	const phong = new PhongMaterial({
		diffuse: { r: 128, g: 128, b: 128 },
		specular: { r: 255, g: 128, b: 64 },
		shininess: 24,
	});
	const phongData = createWebGPUMaterialUniformData(phong);
	assert.equal(phongData.shadingFamily, "phong");
	assert.ok(
		phongData.common.baseColorFactor[0] > 0.2 && phongData.common.baseColorFactor[0] < 0.22
	);
	assert.equal(phongData.lighting.ambientShininess[3], 24);
	assert.ok(phongData.lighting.specular[0] > 0.9);
	for (const shininess of [0, 1, 32, 128]) {
		const data = createWebGPUMaterialUniformData(
			new PhongMaterial({ shininess })
		);
		assert.equal(data.lighting.ambientShininess[3], shininess);
	}

	const unlit = new UnlitMaterial({
		diffuse: { r: 255, g: 32, b: 16 },
	});
	const unlitData = createWebGPUMaterialUniformData(unlit);
	assert.equal(unlitData.shadingFamily, "unlit");
	assert.equal(unlitData.lighting, null);

	const shader = new ShaderMaterial({
		uniformBindings: [
			{ name: "time", type: "f32", value: 2 },
			{ name: "mode", type: "i32", value: 7 },
			{ name: "flags", type: "u32", value: 3 },
			{ name: "tint", type: "vec4f", value: [1, 0.5, 0.25, 1] },
		],
	});
	const shaderData = createWebGPUMaterialUniformData(shader);
	assert.notEqual(shaderData.shaderUniforms.data, null);
	assert.ok(shaderData.shaderUniforms.byteLength >= 32);
	assert.ok(shaderData.shaderUniforms.cacheKey.includes("time:f32"));
	assert.equal(shaderData.shaderUniforms.valueRevision, shader.uniformValueRevision);
	const shaderRevision = shader.shaderRevision;
	shader.setUniform("time", 3);
	const shaderDataUpdated = createWebGPUMaterialUniformData(shader);
	assert.equal(shaderDataUpdated.shaderUniforms.valueRevision, shader.uniformValueRevision);
	assert.equal(shader.shaderRevision, shaderRevision);
}

function testFeatureGate() {
	const featureState = resolveFeatureState(
		{
			enableLighting: true,
			enableSH: true,
			enableReflection: true,
			enableEnvironment: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			environment: false,
			clusteredLighting: false,
			oit: false,
		},
		"webgpu"
	);

	assert.equal(featureState.enableLighting, true);
	assert.equal(featureState.enableSH, false);
	assert.equal("enableShadows" in featureState, false);
	assert.equal(featureState.enableReflection, false);
	assert.equal(featureState.enableEnvironment, false);
	assert.ok(featureState.warnings.length >= 3);

	const postProcess = createResolvedPostProcess(
		{
			ssgi: { enabled: true },
			ssr: { enabled: true },
			bloom: { enabled: true },
			tonemap: { enabled: true },
			gamma: { enabled: true },
		},
		"software"
	);
	assert.equal(postProcess.isEnabled("gamma"), true);
	assert.equal(postProcess.isEnabled("tonemap"), true);
	assert.equal(postProcess.isEnabled("ssgi"), false);
	assert.equal(postProcess.isEnabled("ssr"), false);
	assert.equal(postProcess.isEnabled("bloom"), false);
	assert.equal(postProcess.getOptions("gamma") !== null, true);
	assert.deepEqual(postProcess.getWarnings(), []);
}

async function testSceneShaderCoverage() {
	const WEBGPU_SCENE_SHADER = (await ShaderSource.load("webgpu.scene")).source.code;
	for (const [name, layout] of [
		["ObjectUniforms", WEBGPU_OBJECT_UNIFORM_LAYOUT],
		["MaterialCommonUniforms", WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT],
		["PBRMaterialUniforms", WEBGPU_PBR_MATERIAL_UNIFORM_LAYOUT],
		["PhongMaterialUniforms", WEBGPU_PHONG_MATERIAL_UNIFORM_LAYOUT],
		["FlatMaterialUniforms", WEBGPU_FLAT_MATERIAL_UNIFORM_LAYOUT],
	]) {
		assert.deepEqual(
			readWGSLStructFieldNames(WEBGPU_SCENE_SHADER, name),
			readLayoutFieldNames(layout),
			`${name} CPU and WGSL member order must match`,
		);
	}
	const WEBGPU_DEFERRED_LIGHTING_SHADER = (await ShaderSource.load(
		"webgpu.deferredLighting"
	)).source.code;
	const WEBGPU_ENVIRONMENT_SHADER = (await ShaderSource.load(
		"webgpu.environment"
	)).source.code;
	const WEBGPU_SSAO_SHADER = (await ShaderSource.load(
		"webgpu.postprocess.ssao"
	)).source.code;
	const WEBGPU_SSGI_SHADER = (await ShaderSource.load(
		"webgpu.postprocess.ssgi"
	)).source.code;
	const WEBGPU_SSR_SHADER = (await ShaderSource.load(
		"webgpu.postprocess.ssr"
	)).source.code;

	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"let pointCount = u32(frame.lightCounts.y + 0.5);"
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"frameLights.pointLights[i].positionRange.xyz - input.worldPosition"
		)
	);
	assert.ok(WEBGPU_SCENE_SHADER.includes("let areaCount = areaLightCount();"));
	assert.ok(
		/evaluateAreaLight\(\s*frameLights\.areaLights\[i\],\s*input\.worldPosition,\s*sampleIndex\s*\)/.test(
			WEBGPU_SCENE_SHADER
		)
	);
	assert.ok(WEBGPU_SCENE_SHADER.includes("CLUSTER_LIGHT_TYPE_AREA"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("clusteredRecordToAreaLight"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("if (!isClusteredLightingEnabled())"));
	assert.ok(
		/evaluateAreaLight\(\s*areaRecord,\s*input\.worldPosition,\s*sampleIndex\s*\)/.test(
			WEBGPU_SCENE_SHADER
		)
	);
	assert.ok(WEBGPU_SCENE_SHADER.includes("AREA_LIGHT_SAMPLE_COUNT"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("sampleDirectionalShadowVisibility"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("sampler_comparison"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("textureSampleCompareLevel("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("textureLoad(shadowTransmittanceAtlas"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("texture_depth_2d"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("MAX_SHADOW_SEARCH_SAMPLES: i32 = 12"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("MAX_SHADOW_FILTER_SAMPLES: i32 = 7"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("depthProjectionParams"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("linearizeShadowDepth"));
	assert.equal(WEBGPU_SCENE_SHADER.includes("MAX_PCSS_SEARCH_SAMPLES"), false);
	assert.equal(WEBGPU_SCENE_SHADER.includes("vogelDiskSample"), false);
	assert.ok(!WEBGPU_SCENE_SHADER.includes("decodePackedShadowDepth"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("let shadowNormal = normal;"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("let pbrShadowNormal = pbrNormal;"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("shadowData.paramsC.x"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("calculateIrradianceFromSH"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("localLightProbeCounts"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("selectTopTwoLocalLightProbes"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("sampleBlendedLocalLightProbeIrradiance"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("sampleBlendedLocalLightProbeRadiance"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("sampleEnvironmentSpecular"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("resolveSpecularEnergyCompensation"));
	assert.ok(
		(WEBGPU_SCENE_SHADER.match(/evaluateOpaquePBRLight\(/g)?.length ?? 0) >= 6
	);
	assert.equal(
		(WEBGPU_SCENE_SHADER.match(/fn evaluateOpaquePBRLight\(/g)?.length ?? 0),
		1
	);
	assert.ok(WEBGPU_DEFERRED_LIGHTING_SHADER.includes("DeferredPBRContext"));
	assert.ok(
		WEBGPU_DEFERRED_LIGHTING_SHADER.includes("evaluateOpaquePBRLight(")
	);
	assert.ok(
		/evaluateAreaLight\(\s*frameLights\.areaLights\[i\],\s*surface\.worldPosition,\s*sampleIndex\s*\)/.test(
			WEBGPU_DEFERRED_LIGHTING_SHADER
		)
	);
	assert.ok(WEBGPU_DEFERRED_LIGHTING_SHADER.includes("CLUSTER_LIGHT_TYPE_AREA"));
	assert.ok(WEBGPU_DEFERRED_LIGHTING_SHADER.includes("clusteredRecordToAreaLight"));
	assert.ok(
		/evaluateAreaLight\(\s*areaRecord,\s*surface\.worldPosition,\s*sampleIndex\s*\)/.test(
			WEBGPU_DEFERRED_LIGHTING_SHADER
		)
	);
	assert.ok(
		WEBGPU_DEFERRED_LIGHTING_SHADER.includes("pbr.energyCompensation")
	);
	assert.ok(WEBGPU_SCENE_SHADER.includes("fn encodeOctahedralNormal("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("return encodeOctahedralNormal(vn);"));
	assert.ok(
		WEBGPU_DEFERRED_LIGHTING_SHADER.includes(
			"let vn = decodeOctahedralNormal(encoded);"
		)
	);
	assert.ok(!WEBGPU_DEFERRED_LIGHTING_SHADER.includes("sqrt(z2)"));
	for (const postProcessShader of [
		WEBGPU_SSAO_SHADER,
		WEBGPU_SSGI_SHADER,
		WEBGPU_SSR_SHADER,
	]) {
		assert.ok(postProcessShader.includes("fn octahedralWrap("));
		assert.ok(!postProcessShader.includes("sqrt(z2)"));
	}
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(2)"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(4) var<uniform> fog"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(7) var brdfLUTTexture"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(9) var shadowComparisonSampler"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(2) @binding(0)"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("if (isClusteredLightingEnabled())"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("decodeClusteredLightRef"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@location(4) gMotionDepth"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("frame.prevViewProjection"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("object.prevModelMatrix"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(1) @binding(29) var iridescenceTexture"));
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"@group(1) @binding(31) var iridescenceThicknessTexture"
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"@group(1) @binding(34) var<uniform> animationParams"
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"@group(1) @binding(35) var<storage, read> jointMatrices"
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"@group(1) @binding(33) var anisotropyTexture"
		)
	);
	assert.ok(!WEBGPU_SCENE_SHADER.includes("anisotropyTextureTransformA"));
	assert.ok(!WEBGPU_SCENE_SHADER.includes("anisotropyTextureTransformB"));
	assert.ok(!WEBGPU_SCENE_SHADER.includes("var iridescenceSampler"));
	assert.ok(!WEBGPU_SCENE_SHADER.includes("var iridescenceThicknessSampler"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("applySkinning("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("struct SceneFragmentOITOutput"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("struct GBufferFragmentOutput"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("fn fsMainGBufferPBR("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("gMaterialExt0Out"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("gMaterialExt3Out"));
	assert.ok(WEBGPU_DEFERRED_LIGHTING_SHADER.includes("gMaterialExt3In"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("fn evaluateOpaquePhongLight("));
	assert.ok(WEBGPU_DEFERRED_LIGHTING_SHADER.includes("fn evaluateOpaquePhongLight("));
	assert.ok(
		WEBGPU_SCENE_SHADER.includes("vec4<f32>(phongSpecular, shininess)")
	);
	assert.ok(
		WEBGPU_DEFERRED_LIGHTING_SHADER.includes(
			"select(clamp(specular.a, 0.0, 1.0), max(specular.a, 0.0), isPhong)"
		)
	);
	assert.ok(WEBGPU_DEFERRED_LIGHTING_SHADER.includes("anisotropyTangent"));
	assert.ok(
		WEBGPU_DEFERRED_LIGHTING_SHADER.includes(
			"specular = resolveAnisotropicSpecular("
		)
	);
	assert.ok(
		WEBGPU_DEFERRED_LIGHTING_SHADER.includes(
			"resolveAnisotropicReflectionDirection(\n\t\t\t\tsurface.normal"
		)
	);
	assert.ok(WEBGPU_SCENE_SHADER.includes("fn resolveOITWeight("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("fn buildSceneOITOutput("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("fn fsMainOITPBR("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("fn fsMainTransmissionCapturePBR("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@builtin(vertex_index)"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@builtin(front_facing) frontFacing"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("doubleSided && !frontFacing"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("dot(pbrNormal, geometryNormal) < 0.0"));
	assert.ok(!WEBGPU_SCENE_SHADER.includes("doubleSided && dot(normal, viewDir)"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@location(8) weights1"));
	assert.ok(WEBGPU_ENVIRONMENT_SHADER.includes("@group(0) @binding(1)"));
	assert.ok(WEBGPU_ENVIRONMENT_SHADER.includes("prevViewProjection"));
	assert.ok(WEBGPU_ENVIRONMENT_SHADER.includes("taaJitterCurrentPrev"));
	assert.ok(WEBGPU_ENVIRONMENT_SHADER.includes("atan2(direction.x, direction.z)"));
	assert.ok(WEBGPU_ENVIRONMENT_SHADER.includes("frame.environmentOptionsB.z < 0.5"));
	assert.ok(
		WEBGPU_ENVIRONMENT_SHADER.includes(
			"return vec4<f32>(max(skyColor, vec3<f32>(0.0)), 1.0);"
		)
	);
}

async function testGBufferSlotShaderABI() {
	const sceneShader = (await ShaderSource.load("webgpu.scene")).source.code;
	const slots = [
		["gAlbedoAlpha", GBufferSlot.AlbedoAlpha],
		["gNormalRoughMetal", GBufferSlot.NormalRoughMetal],
		["gEmissiveOcclusion", GBufferSlot.EmissiveOcclusion],
		["gMotionDepth", GBufferSlot.MotionDepth],
		["gSpecular", GBufferSlot.Specular],
		["gCoatSheen", GBufferSlot.CoatSheen],
		["gSheenReflectance", GBufferSlot.SheenReflectance],
	];
	assert.equal(WEBGPU_DEFERRED_COLOR_TARGET_COUNT, slots.length);
	for (const [name, slot] of slots) {
		assert.ok(
			sceneShader.includes(`@location(${slot}) ${name}`),
			`GBufferSlot.${name.slice(1)} must match the WGSL output location`
		);
	}
}

function testScenePipelineLimitConstantsMatchLayout() {
	const device = {
		bindGroupLayouts: [],
		pipelineLayouts: [],
		createBindGroupLayout(desc) {
			const layout = { desc };
			this.bindGroupLayouts.push(layout);
			return layout;
		},
		createPipelineLayout(desc) {
			const layout = { desc };
			this.pipelineLayouts.push(layout);
			return layout;
		},
	};
	const layouts = createWebGPUPipelineLayouts(device);
	const modelBindings = (family) => new Set(
		layouts.modelBindGroupLayouts[family].desc.entries.map((entry) => entry.binding)
	);
	assert.ok(modelBindings("pbr").has(WEBGPU_MODEL_BINDING_MATERIAL_COMMON));
	assert.ok(modelBindings("pbr").has(WEBGPU_MODEL_BINDING_PBR_MATERIAL));
	assert.equal(modelBindings("pbr").has(WEBGPU_MODEL_BINDING_PHONG_MATERIAL), false);
	assert.ok(modelBindings("phong").has(WEBGPU_MODEL_BINDING_PHONG_MATERIAL));
	assert.ok(modelBindings("flat").has(WEBGPU_MODEL_BINDING_FLAT_MATERIAL));
	assert.equal(modelBindings("unlit").has(WEBGPU_MODEL_BINDING_PBR_MATERIAL), false);
	assert.equal(modelBindings("unlit").has(WEBGPU_MODEL_BINDING_PHONG_MATERIAL), false);
	assert.equal(modelBindings("unlit").has(WEBGPU_MODEL_BINDING_FLAT_MATERIAL), false);
	const getFragmentEntries = (pipelineLayout) =>
		pipelineLayout.desc.bindGroupLayouts.flatMap((layout) =>
			layout.desc.entries.filter(
				(entry) => (entry.visibility & GPUShaderStage.FRAGMENT) !== 0
			)
		);
	const sceneFragmentEntries = getFragmentEntries(layouts.scenePipelineLayouts.pbr);
	const deferredFragmentEntries = getFragmentEntries(
		layouts.deferredLightingPipelineLayout
	);
	const planarReflectionFragmentEntries = getFragmentEntries(
		layouts.planarReflectionPipelineLayouts.pbr
	);
	const sceneSampledTextureCount = sceneFragmentEntries.filter(
		(entry) => !!entry.texture
	).length;
	const deferredSampledTextureCount = deferredFragmentEntries.filter(
		(entry) => !!entry.texture
	).length;
	const planarReflectionSampledTextureCount =
		planarReflectionFragmentEntries.filter((entry) => !!entry.texture).length;
	const samplerCount = sceneFragmentEntries.filter(
		(entry) => !!entry.sampler
	).length;
	const planarReflectionSamplerCount = planarReflectionFragmentEntries.filter(
		(entry) => !!entry.sampler
	).length;

	assert.equal(
		sceneSampledTextureCount,
		WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT
	);
	assert.equal(
		deferredSampledTextureCount,
		WEBGPU_DEFERRED_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT
	);
	assert.equal(
		deferredSampledTextureCount,
		WEBGPU_SCENE_FRAME_FRAGMENT_TEXTURE_COUNT +
			WEBGPU_GBUFFER_READ_TEXTURE_COUNT
	);
	assert.equal(
		planarReflectionSampledTextureCount,
		WEBGPU_PLANAR_REFLECTION_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT
	);
	const decalFragmentEntries = getFragmentEntries(
		layouts.decalPipelineLayout
	);
	const decalSampledTextureCount = decalFragmentEntries.filter(
		(entry) => !!entry.texture
	).length;
	assert.equal(
		decalSampledTextureCount,
		WEBGPU_DECAL_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT
	);
	assert.equal(
		WEBGPU_DECAL_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
		26,
		"the compact decal fragment layout must bind 26 sampled textures"
	);
	assert.equal(
		WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
		decalSampledTextureCount,
		"device negotiation must request the decal sampled-texture limit"
	);
	assert.equal(
		layouts.deferredLightingPipelineLayout.desc.bindGroupLayouts[1],
		layouts.deferredUnusedBindGroupLayout
	);
	assert.equal(layouts.scenePipelineLayouts.pbr.desc.bindGroupLayouts.length, 3);
	assert.equal(
		layouts.sceneGBufferPipelineLayouts.pbr.desc.bindGroupLayouts.length,
		4
	);
	assert.equal(
		layouts.sceneGBufferPipelineLayouts.pbr.desc.bindGroupLayouts[3],
		layouts.gbufferWriteBindGroupLayout
	);
	assert.equal(
		layouts.gbufferWriteBindGroupLayout.desc.entries.length,
		WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT
	);
	assert.equal(layouts.decalPipelineLayout.desc.bindGroupLayouts.length, 4);
	assert.equal(
		layouts.decalPipelineLayout.desc.bindGroupLayouts[3],
		layouts.decalOutputBindGroupLayout
	);
	assert.equal(layouts.decalBatchPipelineLayout.desc.bindGroupLayouts.length, 4);
	assert.equal(
		layouts.decalBatchPipelineLayout.desc.bindGroupLayouts[3],
		layouts.decalBatchBindGroupLayout
	);
	assert.equal(
		layouts.decalOutputBindGroupLayout.desc.entries.length,
		WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT
	);
	assert.equal(
		layouts.decalOutputBindGroupLayout.desc.entries.filter(
			(entry) => (entry.visibility & GPUShaderStage.FRAGMENT) !== 0
		).length,
		WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT
	);
	assert.equal(layouts.decalBatchBindGroupLayout.desc.entries.length, 13);
	assert.equal(
		layouts.decalBatchBindGroupLayout.desc.entries.filter(
			(entry) => (entry.visibility & GPUShaderStage.FRAGMENT) !== 0
		).length,
		0
	);
	assert.equal(
		layouts.decalBatchBindGroupLayout.desc.entries.filter(
			(entry) =>
				(entry.visibility & GPUShaderStage.COMPUTE) !== 0 &&
				!!entry.storageTexture
		).length,
		WEBGPU_GBUFFER_READ_TEXTURE_COUNT
	);
	assert.equal(layouts.deferredUnusedBindGroupLayout.desc.entries.length, 0);
	assert.equal(samplerCount, WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLER_COUNT);
	assert.equal(
		planarReflectionSamplerCount,
		WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLER_COUNT
	);
	assert.ok(samplerCount <= 16);
	assert.ok(planarReflectionSamplerCount <= 16);
	assert.equal(WEBGPU_DEFERRED_BASE_COLOR_BYTES_PER_SAMPLE, 32);
	assert.equal(WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE, 56);
	const requiredLimits = createWebGPURequiredLimits({
		maxSampledTexturesPerShaderStage: 32,
		maxSamplersPerShaderStage: 16,
		maxStorageBuffersPerShaderStage: 8,
		maxStorageTexturesPerShaderStage: 4,
		maxColorAttachments: 8,
		maxColorAttachmentBytesPerSample: 128,
		maxTextureDimension2D: 16384,
	});
	assert.equal(requiredLimits.maxColorAttachmentBytesPerSample, 56);
	assert.equal(
		WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE +
			WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT * 8,
		72
	);
}

function testDecalBatchLayoutHonorsStorageTextureLimit() {
	const device = {
		limits: {
			maxStorageTexturesPerShaderStage: 4,
			maxStorageBuffersPerShaderStage: 8,
		},
		bindGroupLayouts: [],
		pipelineLayouts: [],
		createBindGroupLayout(desc) {
			for (const visibility of [
				GPUShaderStage.FRAGMENT,
				GPUShaderStage.COMPUTE,
			]) {
				const storageTextureCount = desc.entries.filter(
					(entry) =>
						(entry.visibility & visibility) !== 0 &&
						!!entry.storageTexture
				).length;
				assert.ok(storageTextureCount <= 4);
			}
			const layout = { desc };
			this.bindGroupLayouts.push(layout);
			return layout;
		},
		createPipelineLayout(desc) {
			const layout = { desc };
			this.pipelineLayouts.push(layout);
			return layout;
		},
	};
	const layouts = createWebGPUPipelineLayouts(device);

	assert.equal(layouts.decalBatchBindGroupLayout.desc.entries.length, 0);
	assert.equal(
		layouts.decalPipelineLayout.desc.bindGroupLayouts[3],
		layouts.decalOutputBindGroupLayout
	);
	assert.equal(
		layouts.decalBatchPipelineLayout.desc.bindGroupLayouts[3],
		layouts.decalBatchBindGroupLayout
	);
	assert.equal(
		layouts.decalOutputBindGroupLayout.desc.entries.length,
		WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT
	);
}

function testShadowDepthLayoutMatchesTransmittanceShaderBinding() {
	const shadowDepthShader = readFileSync(
		"src/shaders/webgpu/shadow/depth.wgsl",
		"utf8"
	);
	const shadowPassSource = readFileSync(
		"src/backends/webgpu/WebGPUShadowCasterRenderer.ts",
		"utf8"
	);
	const depthLayoutBindingPattern = new RegExp([
		"label: \"WebGPUShadowDepthBindGroupLayout\"",
		"binding: 2,",
		"visibility: GPUShaderStage\\.VERTEX,",
		"buffer: \\{ type: \"read-only-storage\" \\}",
	].join("[\\s\\S]*"));
	const depthBindGroupEntryPattern = new RegExp([
		"label: `WebGPUShadowDepthMvpBindGroup_",
		"binding: 2,",
		"resource: \\{ buffer: transmittanceBuffer \\}",
	].join("[\\s\\S]*"));

	assert.ok(
		shadowDepthShader.includes(
			"@group(0) @binding(2) var<storage, read> shadowTransmittance"
		)
	);
	assert.match(shadowPassSource, depthLayoutBindingPattern);
	assert.match(shadowPassSource, depthBindGroupEntryPattern);
	assert.ok(shadowDepthShader.includes("let jointOffset = instanceData.jointBaseOffset;"));
	assert.ok(
		shadowDepthShader.includes(
			"let morphWeightOffset = instanceData.morphWeightBaseOffset;"
		)
	);
	assert.equal(shadowDepthShader.includes("localInstanceIndex *"), false);
}

async function testParticleShaderDepthConsistency() {
	const WEBGPU_PARTICLE_SHADER = (await ShaderSource.load("webgpu.particle")).source.code;

	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes(
			"let currJitter = frame.taaJitterCurrentPrev.xy * clipPosition.w;"
		)
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes(
			"clipPosition.z = clipPosition.z * 0.5 + clipPosition.w * 0.5;"
		)
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes("let currentDepth = ndc.z * 0.5 + 0.5;")
	);
	assert.ok(WEBGPU_PARTICLE_SHADER.includes("struct ParticleUVTransform"));
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes("@group(1) @binding(2) var<uniform>")
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes("@group(0) @binding(4) var<uniform> fog")
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes(
			"input.uv.x * particleUVTransform.transformA.x"
		)
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes(
			"rotatedUV + particleUVTransform.transformA.zw"
		)
	);
	assert.ok(WEBGPU_PARTICLE_SHADER.includes("struct ParticleOITOutput"));
	assert.ok(WEBGPU_PARTICLE_SHADER.includes("particleShadowSampleCounts"));
	assert.ok(WEBGPU_PARTICLE_SHADER.includes("particleLinearizeShadowDepth"));
	assert.ok(WEBGPU_PARTICLE_SHADER.includes("fn fsMainOIT("));
	assert.ok(WEBGPU_PARTICLE_SHADER.includes("fn resolveParticleOITWeight("));
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes(
			"for (var index = 0u; index < directionalCount; index = index + 1u)"
		)
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes(
			"@group(0) @binding(6) var shadowTransmittanceAtlas"
		)
	);
	assert.equal(WEBGPU_PARTICLE_SHADER.includes("directionalShadows[0]"), false);
}

async function testWebGPUShaderConstantTokenInjection() {
	const rawSceneShader = (await ShaderSource.load("webgpu.scene")).source.code;
	const rawEnvironmentShader = (await ShaderSource.load("webgpu.environment")).source.code;
	const rawParticleShader = (await ShaderSource.load("webgpu.particle")).source.code;
	const rawSSRShader = (await ShaderSource.load("webgpu.postprocess.ssr")).source.code;
	const rawClusteredCullShader =
		(await ShaderSource.load("webgpu.clusteredLightingCull")).source.code;
	const rawPlanarReflectionCompositeShader =
		(
			await ShaderSource.load(
				"webgpu.utility.planarReflectionComposite"
			)
		).source.code;
	assert.deepEqual(
		readWGSLStructFieldNames(rawPlanarReflectionCompositeShader, "ObjectUniforms"),
		readLayoutFieldNames(WEBGPU_OBJECT_UNIFORM_LAYOUT),
	);
	assert.deepEqual(
		readWGSLStructFieldNames(rawPlanarReflectionCompositeShader, "MaterialCommonUniforms"),
		readLayoutFieldNames(WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT),
	);
	assert.ok(rawSceneShader.includes("__WEBGPU_MAX_DIRECTIONAL_LIGHTS__"));

	const compileStage = new ShaderBackendCompileStage({
		runtime: new ShaderRuntime({ mode: "strict" }),
		profile: WEBGPU_TEST_PROFILE,
		mode: "strict",
	});
	const compileShader = async (code, label, sourceKind) => {
		const result = await compileStage.compileAsync({
			code,
			language: "wgsl",
			stage: "unknown",
			label,
			sourceKind,
		});
		return result.code;
	};
	const WEBGPU_SCENE_SHADER = await compileShader(
		rawSceneShader,
		"test-webgpu-scene-shader",
		"builtin-scene"
	);
	const WEBGPU_ENVIRONMENT_SHADER = await compileShader(
		rawEnvironmentShader,
		"test-webgpu-environment-shader",
		"builtin-environment"
	);
	const WEBGPU_PARTICLE_SHADER = await compileShader(
		rawParticleShader,
		"test-webgpu-particle-shader",
		"particle"
	);
	const WEBGPU_SSR_SHADER = await compileShader(
		rawSSRShader,
		"test-webgpu-ssr-shader",
		"postprocess"
	);
	const WEBGPU_CLUSTERED_CULL_SHADER = await compileShader(
		rawClusteredCullShader,
		"test-webgpu-clustered-cull-shader",
		"clustered"
	);
	const WEBGPU_PLANAR_REFLECTION_COMPOSITE_SHADER = await compileShader(
		rawPlanarReflectionCompositeShader,
		"test-webgpu-planar-reflection-composite-shader",
		"builtin-scene"
	);

	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`directionalLights: array<DirectionalLightData, ${MAX_DIRECTIONAL_LIGHTS}>`
		)
	);
	assert.ok(WEBGPU_SCENE_SHADER.includes("const PI: f32 = 3.14159265359;"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("const INV_PI: f32 = 0.31830988618;"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("compensateCascadeDepthRange"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("length(shadowClipDepthRow) * 0.5"));
	assert.ok(WEBGPU_PARTICLE_SHADER.includes("cascadeDepthBiasScale"));
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`pointLights: array<PointLightData, ${MAX_POINT_LIGHTS}>`
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`spotLights: array<SpotLightData, ${MAX_SPOT_LIGHTS}>`
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`areaLights: array<AreaLightData, ${MAX_AREA_LIGHTS}>`
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`shAmbientCoeffs: array<vec4<f32>, ${WEBGPU_SH_COEFFICIENT_COUNT}>`
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`localLightProbeWorldToProbeRow0: array<vec4<f32>, ${MAX_LOCAL_LIGHT_PROBES}>`
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`reflectionProbes: array<ReflectionProbeData, ${MAX_REFLECTION_PROBES}>`
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`textureTransformA: array<vec4<f32>, ${WEBGPU_TEXTURE_SLOT_COUNT}>`
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`textureTransformB: array<vec4<f32>, ${WEBGPU_TEXTURE_SLOT_COUNT}>`
		)
	);
	assert.ok(WEBGPU_SCENE_SHADER.includes("pbrMasks: vec4<u32>"));
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"if (hasPBRTexture(material, PBR_TEXTURE_METALLIC_ROUGHNESS_MAP))"
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"hasPBRFeature(material, PBR_FEATURE_TRANSMISSION)"
		)
	);
	assert.ok(
		WEBGPU_ENVIRONMENT_SHADER.includes("struct FrameCameraUniforms")
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes("struct FrameCameraUniforms")
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes("struct FrameShadowUniforms")
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes("@group(0) @binding(11) var<uniform> frameShadows")
	);
	assert.ok(
		WEBGPU_SSR_SHADER.includes("struct FrameCameraUniforms")
	);
	assert.ok(
		WEBGPU_CLUSTERED_CULL_SHADER.includes("struct FrameCameraUniforms")
	);
	assert.ok(
		WEBGPU_SSR_SHADER.includes(
			"@group(0) @binding(6) var composePlanarReflectionMask"
		)
	);
	assert.ok(
		WEBGPU_PLANAR_REFLECTION_COMPOSITE_SHADER.includes(
			`textureTransformA: array<vec4<f32>, ${WEBGPU_TEXTURE_SLOT_COUNT}>`
		)
	);
	assert.ok(
		WEBGPU_PLANAR_REFLECTION_COMPOSITE_SHADER.includes(
			"@group(2) @binding(0) var reflectionTexture"
		)
	);
	assert.ok(!WEBGPU_SCENE_SHADER.includes("__WEBGPU_"));
	assert.ok(!WEBGPU_ENVIRONMENT_SHADER.includes("__WEBGPU_"));
	assert.ok(!WEBGPU_PARTICLE_SHADER.includes("__WEBGPU_"));
	assert.ok(!WEBGPU_SSR_SHADER.includes("__WEBGPU_"));
	assert.ok(!WEBGPU_CLUSTERED_CULL_SHADER.includes("__WEBGPU_"));
	assert.ok(!WEBGPU_PLANAR_REFLECTION_COMPOSITE_SHADER.includes("__WEBGPU_"));
}

async function run() {
	try {
		await testBackendDoesNotExposeDeviceServiceForwarders();
		await testMatrixPackingAndDepthRemap();
		await testTransformComposition();
		await testMaterialAdaptation();
		await testFeatureGate();
		await testSceneShaderCoverage();
		await testGBufferSlotShaderABI();
		await testScenePipelineLimitConstantsMatchLayout();
		await testDecalBatchLayoutHonorsStorageTextureLimit();
		await testShadowDepthLayoutMatchesTransmittanceShaderBinding();
		await testParticleShaderDepthConsistency();
		await testWebGPUShaderConstantTokenInjection();
		console.log("WebGPU bridge shader contracts tests passed");
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
