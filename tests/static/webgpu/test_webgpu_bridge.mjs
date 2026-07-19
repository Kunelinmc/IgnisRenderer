import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { WebGPUFrameServiceOwner as WebGPURenderResources } from "../../../src/backends/webgpu/WebGPUFrameServiceOwner.ts";
import { WebGPUFrameOrchestrator as WebGPUFrameExecutor } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameOrchestrator.ts";
import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";
import {
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../../../src/shaders/runtime/index.ts";
import {
	collectWebGPUEnvironment,
	collectWebGPULightingCatalog,
	collectWebGPULighting,
	createWebGPUClusteredLightingData,
	createWebGPUFrameFeatureRegistry,
	createWebGPULightingState,
	createWebGPUMaterialUniformData,
	materialSupportsWebGPUDeferredLighting,
	packMatrix4ForWGSL,
	remapClipSpaceDepth,
	WEBGPU_VOLUMETRIC_LIGHTING_DATA,
	WEBGPU_FRAME_CAMERA_UNIFORM_FLOATS,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_FLOATS,
	WEBGPU_FRAME_LIGHT_UNIFORM_FLOATS,
	WEBGPU_FRAME_SHADOW_UNIFORM_FLOATS,
} from "../../../src/backends/webgpu/index.ts";
import { WebGPUReflectionProbeCapturePass } from "../../../src/backends/webgpu/WebGPUReflectionProbeCapturePass.ts";
import { createWebGPUPipelineLayouts } from "../../../src/backends/webgpu/WebGPUPipelineLayouts.ts";
import { resolveFeatureState } from "../../../src/pipeline/FeatureResolver.ts";
import { BufferUsage, TextureFormat } from "../../../src/backends/types.ts";
import { LightProbe } from "../../../src/lights/LightProbe.ts";
import { AreaLight } from "../../../src/lights/AreaLight.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { PointLight } from "../../../src/lights/PointLight.ts";
import { SpotLight } from "../../../src/lights/SpotLight.ts";
import { ReflectionProbe } from "../../../src/lights/ReflectionProbe.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { computeHaltonJitterNDC } from "../../../src/maths/Misc.ts";
import { SH } from "../../../src/maths/SH.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { PhongMaterial } from "../../../src/materials/PhongMaterial.ts";
import { AlphaMode } from "../../../src/materials/Material.ts";
import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { CubeTexture } from "../../../src/core/CubeTexture.ts";
import { float16BitsToFloat32 } from "../../../src/foundation/Float16.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { Node } from "../../../src/core/Node.ts";
import { UnlitMaterial } from "../../../src/materials/UnlitMaterial.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { ShadowMap, createShadowRenderSet } from "../../../src/lights/shadows/ShadowMapping.ts";
import { PARTICLE_TRANSIENT_BATCHES_KEY } from "../../../src/pipeline/types.ts";
import { ParticleBlendMode } from "../../../src/particles/types.ts";
import {
	WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT,
	WEBGPU_PARTICLE_VERTEX_LAYOUTS,
} from "../../../src/backends/webgpu/bufferLayouts.ts";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_AREA_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
} from "../../../src/backends/constants.ts";
import {
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_TARGET_COUNT,
	WEBGPU_DEFERRED_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT,
	WEBGPU_GBUFFER_READ_TEXTURE_COUNT,
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_SHADOW_ATLAS_COLUMNS,
	WEBGPU_SCENE_FRAME_FRAGMENT_TEXTURE_COUNT,
	WEBGPU_PLANAR_REFLECTION_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_DECAL_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLER_COUNT,
	WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
	WEBGPU_MODEL_BINDING_JOINT_MATRICES,
	WEBGPU_MODEL_BINDING_MORPH_NORMAL,
	WEBGPU_MODEL_BINDING_MORPH_POSITION,
	WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
	WEBGPU_MODEL_BINDING_ANISOTROPY_TEXTURE,
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	WEBGPU_CLUSTERED_LIGHT_TYPE_AREA,
	WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT,
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT,
	GBufferSlot,
} from "../../../src/backends/webgpu/constants.ts";
import { WebGPUGeometryRegistry } from "../../../src/backends/webgpu/WebGPUGeometryRegistry.ts";
import { WebGPUTextureRegistry } from "../../../src/backends/webgpu/WebGPUTextureRegistry.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

globalThis.GPUShaderStage ??= {
	VERTEX: 1,
	FRAGMENT: 2,
	COMPUTE: 4,
};

function nearlyEqual(actual, expected, epsilon = 1e-6) {
	assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected}`);
}

function createMainFrameOptions(options = {}) {
	return {
		scopeKey: "main",
		sceneTargetMode: "mrt",
		...options,
	};
}

import {
	FakeCommandEncoder as FakeRenderEncoder,
	FakeWebGPUBackend as FakeBackend,
} from "../../helpers/fakes.mjs";

function createModel(materials) {
	const mesh = MeshAsset.fromFaces(
		materials.map((material, index) => ({
			material,
			vertices: [
				{
					x: index,
					y: 0,
					z: 0,
					u: 0,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: index + 1,
					y: 0,
					z: 0,
					u: 1,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: index,
					y: 1,
					z: 0,
					u: 0,
					v: 1,
					normal: { x: 0, y: 0, z: 1 },
				},
			],
		}))
	);
	return new MeshInstance({ mesh });
}

function createPacket(model) {
	model.updateWorldMatrix(model.parent?.worldMatrix);
	const primitive = model.mesh.primitives[0];
	const worldMatrix = model.worldMatrix;
	return {
		id: `${model.id}:${primitive.id}`,
		meshInstance: model,
		mesh: model.mesh,
		primitive,
		material: primitive.material,
		geometry: primitive.geometry,
		worldMatrix,
		normalMatrix: Matrix4.normalMatrix(worldMatrix),
		worldBounds: primitive.boundingSphere,
		sortDepth: 1,
		pipelineKey: "test",
		passFlags: 0,
	};
}

function createFrame(packet) {
	const cameraPosition = { x: 0, y: 0, z: 5 };
	return {
		sceneBounds: packet.mesh.boundingSphere,
		lights: [],
		camera: {
			viewProjectionMatrix: Matrix4.identity(),
			viewMatrix: Matrix4.identity(),
			position: cameraPosition,
			getWorldPosition() {
				return this.position;
			},
			fov: 60,
			aspectRatio: 1,
			type: "perspective",
		},
		environment: createEnvironmentSnapshot(null, null),
		meshInstances: [packet.meshInstance],
		shadowMaps: new Map(),
		opaquePackets: [packet],
		transparentPackets: [],
		shadowCasterPackets: [],
		shadowTransmitterPackets: [],
		reflectivePackets: [],
		decalPackets: [],
	};
}

function createFrameContext(frame, features, options = {}) {
	const width = options.width ?? 1;
	const height = options.height ?? 1;
	return {
		viewCamera: frame.camera,
		attachments: { width, height },
		features,
		postProcess: options.postProcess ?? createResolvedPostProcess(),
		shadowMaps: frame.shadowMaps,
		scene: options.scene ?? frame,
		shCoeffs: options.shCoeffs ?? SH.empty(),
		shAmbientCoeffs: options.shAmbientCoeffs ?? SH.empty(),
		worldMatrix: options.worldMatrix ?? Matrix4.identity(),
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width, height }],
			dirtyTileSize: Math.max(width, height),
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			postProcessStartPass: null,
			reasonMask: 0,
			temporalHistoryReset: options.temporalHistoryReset === true,
		},
		transient: options.transient ?? new Map(),
	};
}

function createFrameContextWithFeatures(
	frame,
	featureRequest,
	capabilities,
	options = {}
) {
	return createFrameContext(
		frame,
		resolveFeatureState(featureRequest, capabilities, "webgpu"),
		options
	);
}

function createPreparedFrameResources(options = {}) {
	const scopeKey = options.scopeKey ?? "test";
	const sceneTargetMode = options.sceneTargetMode ?? "single";
	return {
		scopeKey,
		sceneTargetMode,
		frameBinding: { bindGroup: { label: `${scopeKey}:frame` } },
		decalFrameBinding: { bindGroup: { label: `${scopeKey}:decal-frame` } },
		environmentBinding: null,
		clusteredSceneBinding: null,
		lightingState: {},
		featureData: { get: () => undefined },
		featureState: {},
		environmentState: {},
		jointMatrixMap: new Map(),
		morphWeightMap: new Map(),
	};
}

function createFrameScopeAdapter(resources) {
	return {
		prepare: (context, options) => resources.prepareFrame(context, options),
		updateParticleShadowVolumes() {},
		destroy() {},
	};
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
	assert.ok(Math.abs(pbrData.baseColorFactor[0] - 128 / 255) < 1e-6);
	assert.ok(Math.abs(pbrData.surfaceParams0[0] - 0.25) < 1e-6);
	assert.ok(Math.abs(pbrData.surfaceParams0[1] - 0.75) < 1e-6);
	assert.ok(Math.abs(pbrData.surfaceParams0[2] - 0.6) < 1e-6);
	assert.equal(pbrData.textureSlots.length, WEBGPU_TEXTURE_SLOT_COUNT);
	pbr.albedoMapUV = 2;
	pbr.normalMapUV = 3;
	const pbrUVData = createWebGPUMaterialUniformData(pbr);
	assert.equal(pbrUVData.textureSlots[0].transformB[1], 2);
	assert.equal(pbrUVData.textureSlots[2].transformB[1], 3);
	pbr.anisotropyStrength = 0.75;
	pbr.anisotropyRotation = Math.PI / 2;
	pbr.anisotropyMap = new Texture(
		new Uint8ClampedArray([255, 128, 128, 255]),
		1,
		1,
		"Linear"
	);
	pbr.anisotropyMapUV = 2;
	const pbrAnisotropyData = createWebGPUMaterialUniformData(pbr);
	assert.ok(Math.abs(pbrAnisotropyData.anisotropyParams[0] - 0.75) < 1e-6);
	assert.ok(Math.abs(pbrAnisotropyData.anisotropyParams[1]) < 1e-6);
	assert.ok(Math.abs(pbrAnisotropyData.anisotropyParams[2] - 1) < 1e-6);
	assert.equal(pbrAnisotropyData.anisotropyTexture.transformB[1], 2);
	assert.equal(pbrAnisotropyData.anisotropyTexture.transformB[3], 1);
	assert.equal(materialSupportsWebGPUDeferredLighting(pbr), true);

	const transmissivePBR = new PBRMaterial({ transmissionFactor: 1 });
	assert.equal(materialSupportsWebGPUDeferredLighting(transmissivePBR), false);

	const phong = new PhongMaterial({
		diffuse: { r: 128, g: 128, b: 128 },
		specular: { r: 255, g: 128, b: 64 },
		shininess: 24,
	});
	const phongData = createWebGPUMaterialUniformData(phong);
	assert.ok(
		phongData.baseColorFactor[0] > 0.2 && phongData.baseColorFactor[0] < 0.22
	);
	assert.equal(phongData.materialFlags[0], 0);
	assert.equal(phongData.phongAmbientShininess[3], 24);
	assert.ok(phongData.phongSpecularShading[0] > 0.9);

	const unlit = new UnlitMaterial({
		diffuse: { r: 255, g: 32, b: 16 },
	});
	const unlitData = createWebGPUMaterialUniformData(unlit);
	assert.equal(unlitData.materialFlags[0], 2);

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
			enableShadows: true,
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
	assert.equal(featureState.enableShadows, true);
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
	const WEBGPU_SCENE_SHADER = await ShaderSource.load("webgpu.scene.raw");
	const WEBGPU_DEFERRED_LIGHTING_SHADER = await ShaderSource.load(
		"webgpu.deferredLighting.raw"
	);
	const WEBGPU_ENVIRONMENT_SHADER = await ShaderSource.load(
		"webgpu.environment.raw"
	);
	const WEBGPU_SSAO_SHADER = await ShaderSource.load(
		"webgpu.postprocess.ssao.raw"
	);
	const WEBGPU_SSGI_SHADER = await ShaderSource.load(
		"webgpu.postprocess.ssgi.raw"
	);
	const WEBGPU_SSR_SHADER = await ShaderSource.load(
		"webgpu.postprocess.ssr.raw"
	);

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
		(WEBGPU_SCENE_SHADER.match(
			/specular = specular \* energyCompensation;/g
		)?.length ?? 0) >= 5
	);
	assert.ok(WEBGPU_DEFERRED_LIGHTING_SHADER.includes("DeferredPBRContext"));
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
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(4) var envSpecularFallbackTexture"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(5) var envSpecularFallbackSampler"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(6) var<uniform> fog"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(9) var brdfLUTTexture"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(13) var shadowComparisonSampler"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(2) @binding(0)"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("if (isClusteredLightingEnabled())"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("decodeClusteredLightRef"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@location(4) gMotionDepth"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("frame.prevViewProjection"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("model.prevModelMatrix"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(1) @binding(29) var iridescenceTexture"));
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"@group(1) @binding(31) var iridescenceThicknessTexture"
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"@group(1) @binding(30) var<uniform> animationParams"
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"@group(1) @binding(32) var<storage, read> jointMatrices"
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"@group(1) @binding(37) var anisotropyTexture"
		)
	);
	assert.ok(!WEBGPU_SCENE_SHADER.includes("var iridescenceSampler"));
	assert.ok(!WEBGPU_SCENE_SHADER.includes("var iridescenceThicknessSampler"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("applySkinning("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("struct SceneFragmentOITOutput"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("struct GBufferFragmentOutput"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@fragment\nfn fsMainGBuffer("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("gMaterialExt0Out"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("gMaterialExt3Out"));
	assert.ok(WEBGPU_DEFERRED_LIGHTING_SHADER.includes("gMaterialExt3In"));
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
	assert.ok(WEBGPU_SCENE_SHADER.includes("@fragment\nfn fsMainOIT("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@fragment\nfn fsMainTransmissionCapture("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@builtin(vertex_index)"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@location(8) weights1"));
	assert.ok(WEBGPU_ENVIRONMENT_SHADER.includes("@group(0) @binding(1)"));
	assert.ok(WEBGPU_ENVIRONMENT_SHADER.includes("prevViewProjection"));
	assert.ok(WEBGPU_ENVIRONMENT_SHADER.includes("taaJitterCurrentPrev"));
	assert.ok(WEBGPU_ENVIRONMENT_SHADER.includes("atan2(direction.x, direction.z)"));
	assert.ok(WEBGPU_ENVIRONMENT_SHADER.includes("frame.environmentOptionsB.z < 0.5"));
	assert.ok(WEBGPU_ENVIRONMENT_SHADER.includes("frame.options.w > 0.5"));
	assert.ok(
		WEBGPU_ENVIRONMENT_SHADER.includes(
			"linearToSrgb(max(skyColor, vec3<f32>(0.0)))"
		)
	);
	assert.ok(
		WEBGPU_ENVIRONMENT_SHADER.includes(
			"return vec4<f32>(max(skyColor, vec3<f32>(0.0)), 1.0);"
		)
	);
}

async function testGBufferSlotShaderABI() {
	const sceneShader = await ShaderSource.load("webgpu.scene.raw");
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
	const getFragmentEntries = (pipelineLayout) =>
		pipelineLayout.desc.bindGroupLayouts.flatMap((layout) =>
			layout.desc.entries.filter(
				(entry) => (entry.visibility & GPUShaderStage.FRAGMENT) !== 0
			)
		);
	const sceneFragmentEntries = getFragmentEntries(layouts.scenePipelineLayout);
	const deferredFragmentEntries = getFragmentEntries(
		layouts.deferredLightingPipelineLayout
	);
	const planarReflectionFragmentEntries = getFragmentEntries(
		layouts.planarReflectionPipelineLayout
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
		WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
		decalSampledTextureCount
	);
	assert.equal(
		layouts.deferredLightingPipelineLayout.desc.bindGroupLayouts[1],
		layouts.deferredUnusedBindGroupLayout
	);
	assert.equal(layouts.scenePipelineLayout.desc.bindGroupLayouts.length, 3);
	assert.equal(
		layouts.sceneGBufferPipelineLayout.desc.bindGroupLayouts.length,
		4
	);
	assert.equal(
		layouts.sceneGBufferPipelineLayout.desc.bindGroupLayouts[3],
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
	assert.equal(layouts.decalBatchBindGroupLayout.desc.entries.length, 15);
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
	assert.equal(WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE, 56);
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
		"src/backends/webgpu/WebGPUShadowPass.ts",
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
}

async function testParticleShaderDepthConsistency() {
	const WEBGPU_PARTICLE_SHADER = await ShaderSource.load("webgpu.particle.raw");

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
		WEBGPU_PARTICLE_SHADER.includes("@group(0) @binding(6) var<uniform> fog")
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
	assert.ok(WEBGPU_PARTICLE_SHADER.includes("fn fsMainOIT("));
	assert.ok(WEBGPU_PARTICLE_SHADER.includes("fn resolveParticleOITWeight("));
}

async function testWebGPUShaderConstantTokenInjection() {
	const rawSceneShader = await ShaderSource.load("webgpu.scene.raw");
	const rawEnvironmentShader = await ShaderSource.load("webgpu.environment.raw");
	const rawParticleShader = await ShaderSource.load("webgpu.particle.raw");
	const rawSSRShader = await ShaderSource.load("webgpu.postprocess.ssr.raw");
	const rawClusteredCullShader =
		(await ShaderSource.load("webgpu.clusteredLightingCull.composite")).code;
	const rawPlanarReflectionCompositeShader =
		(
			await ShaderSource.load(
				"webgpu.utility.planarReflectionComposite.composite"
			)
		).code;
	assert.ok(rawSceneShader.includes("__WEBGPU_MAX_DIRECTIONAL_LIGHTS__"));

	const compileStage = new ShaderBackendCompileStage({
		backend: "webgpu",
		runtime: new ShaderRuntime({ mode: "strict" }),
		profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
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
		WEBGPU_PARTICLE_SHADER.includes("@group(0) @binding(15) var<uniform> frameShadows")
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

function createTinyTexture(mips = 1) {
	const texture = new Texture(new Float32Array([1, 1, 1, 1]), 1, 1, "HDR");
	texture.mipmaps = Array.from(
		{ length: mips },
		() => new Float32Array([1, 1, 1, 1])
	);
	return texture;
}

function createTinyCubeTexture(mips = 1, value = 1) {
	const createFace = () => new Float32Array([value, value, value, 1]);
	const faceMipmaps = [];
	for (let level = 1; level < mips; level++) {
		faceMipmaps.push(
			Array.from({ length: 6 }, () => createFace())
		);
	}
	return new CubeTexture({
		faces: Array.from({ length: 6 }, () => createFace()),
		faceMipmaps,
		size: 1,
		colorSpace: "HDR",
	});
}

function createEnvironmentSnapshot(
	backgroundTexture = null,
	iblTexture = null
) {
	return {
		backgroundEnabled: true,
		lightingEnabled: true,
		backgroundTexture,
		iblTexture,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};
}

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

function testRenderResourcesResolveComputeFacadeFromBackend() {
	const backend = new FakeBackend();
const resources = new WebGPURenderResources(backend);

	assert.equal(backend.getComputeFacadeCalls, 1);
	assert.equal(
		typeof resources._computeFacade.createComputePipeline,
		"function"
	);

	resources.destroy();
}

function testRenderResourcesLeaveShaderRuntimeSubscriptionToBackend() {
	const backend = new FakeBackend();
	let listenerCount = 0;
	backend.shaderRuntime = {
		revision: 1,
		getMode: () => "strict",
		onDidChange: () => {
			listenerCount++;
			return () => {
				listenerCount--;
			};
		},
	};
	const resources = new WebGPURenderResources(backend);

	assert.equal(listenerCount, 0);

	resources.destroy();
	assert.equal(listenerCount, 0);
}

function testFrameExecutorConsumesComputeFacadeFromHost() {
	const backend = new FakeBackend();
	const resourcesStub = {
		sceneFrameLayout: null,
		createFrameScope() {
			return {
				prepare() { throw new Error("not used by this test"); },
				updateParticleShadowVolumes() {},
				destroy() {},
			};
		},
	};
	const executor = new WebGPUFrameExecutor(backend, resourcesStub);

	assert.equal(backend.getComputeFacadeCalls, 0);
	assert.equal(typeof executor.getDebugState, "function");
}

async function testRenderResourcesUseCopyDstForUploads() {
	const backend = new FakeBackend();
	const model = createModel([
		new PBRMaterial({
			albedo: { r: 255, g: 255, b: 255 },
		}),
	]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const draw = await resources.getDrawResources(packet, frameResources);

	assert.ok(draw);
	const firstDraw = draw[0];
	assert.ok(firstDraw);
	assert.equal(firstDraw.frameBinding.desc.entries.length, 17);
	assert.ok(
		firstDraw.frameBinding.desc.entries.some((entry) => entry.binding === 7)
	);
	assert.ok(
		firstDraw.frameBinding.desc.entries.some((entry) => entry.binding === 10)
	);
	for (const binding of [0, 14, 15, 16]) {
		assert.ok(
			firstDraw.frameBinding.desc.entries.some(
				(entry) => entry.binding === binding
			)
		);
	}
	assert.equal(
		firstDraw.modelBinding.desc.entries.length,
		1 +
			WEBGPU_TEXTURE_SLOT_COUNT +
			WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT +
			7
	);
	assert.ok(
		firstDraw.modelBinding.desc.entries.some(
			(entry) => entry.binding === WEBGPU_MODEL_BINDING_SHADER_UNIFORMS
		)
	);
	assert.ok(
		firstDraw.modelBinding.desc.entries.some(
			(entry) => entry.binding === WEBGPU_MODEL_BINDING_ANISOTROPY_TEXTURE
		)
	);
	assert.equal(
		firstDraw.pipeline.desc.layout,
		backend.device.pipelineLayouts[0]
	);
	assert.equal(
		firstDraw.pipeline.desc.layout.desc.bindGroupLayouts.length,
		3
	);
	assert.equal(firstDraw.pipeline.desc.fragment.targets.length, 5);
	assert.deepEqual(
		firstDraw.pipeline.desc.fragment.targets.map((target) => target.format),
		["rgba16float", "rgba8unorm", "rgba16float", "rgba16float", "rgba16float"]
	);
	const modelBindingIndices = firstDraw.modelBinding.desc.entries.map(
		(entry) => entry.binding
	);
	assert.ok(modelBindingIndices.includes(29));
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_ANIMATION_PARAMS)
	);
	assert.ok(modelBindingIndices.includes(31));
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_JOINT_MATRICES)
	);
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_MORPH_WEIGHTS)
	);
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_MORPH_POSITION)
	);
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_MORPH_NORMAL)
	);
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_ANISOTROPY_TEXTURE)
	);
	assert.equal(modelBindingIndices.includes(38), false);
	const sceneVertexAttributes =
		firstDraw.pipeline.desc.vertex.buffers[0].attributes;
	assert.ok(
		sceneVertexAttributes.some((attribute) => attribute.shaderLocation === 8)
	);
	assert.ok(
		backend.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Vertex) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	);
	assert.ok(
		backend.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Index) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	);
	assert.ok(
		backend.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Uniform) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	);
	for (const [label, size] of [
		["WebGPUFrameCameraUniforms", WEBGPU_FRAME_CAMERA_UNIFORM_FLOATS * 4],
		["WebGPUFrameLightUniforms", WEBGPU_FRAME_LIGHT_UNIFORM_FLOATS * 4],
		["WebGPUFrameShadowUniforms", WEBGPU_FRAME_SHADOW_UNIFORM_FLOATS * 4],
		["WebGPUFrameEnvironmentUniforms", WEBGPU_FRAME_ENVIRONMENT_UNIFORM_FLOATS * 4],
	]) {
		assert.ok(
			backend.bufferDescs.some(
				(desc) => desc.label === label && desc.size === size
			)
		);
	}
}

async function testWebGPUBlendMaterialsUseTransparentPipelineState() {
	const backend = new FakeBackend();
	const material = new PBRMaterial({
		albedo: { r: 255, g: 255, b: 255 },
		opacity: 0.6,
	});
	material.alphaMode = AlphaMode.Blend;
	const model = createModel([material]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.opaquePackets = [];
	frame.transparentPackets = [packet];
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const draw = await resources.getDrawResources(packet, frameResources);
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(pipelineDesc.fragment.targets.length, 5);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.srcFactor,
		"src-alpha"
	);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.dstFactor,
		"one-minus-src-alpha"
	);
	assert.equal(pipelineDesc.fragment.targets[1].writeMask, 0);
	assert.equal(pipelineDesc.fragment.targets[2].writeMask, 0);
	assert.equal(pipelineDesc.fragment.targets[3].writeMask, 0);
	assert.equal(
		pipelineDesc.fragment.targets[4].blend?.alpha?.dstFactor,
		"one-minus-src-alpha"
	);
}

async function testWebGPUTransmissionMaterialsUseTransparentPipelineState() {
	const backend = new FakeBackend();
	const material = new PBRMaterial({
		albedo: { r: 255, g: 255, b: 255 },
		roughness: 0.05,
		metalness: 0,
		transmissionFactor: 1,
		ior: 1.52,
	});
	const model = createModel([material]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.opaquePackets = [];
	frame.transparentPackets = [packet];
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const draw = await resources.getDrawResources(packet, frameResources);
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(pipelineDesc.fragment.targets.length, 5);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.srcFactor,
		"src-alpha"
	);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.dstFactor,
		"one-minus-src-alpha"
	);
	assert.equal(
		pipelineDesc.fragment.targets[4].blend?.color?.srcFactor,
		"src-alpha"
	);
	assert.equal(
		pipelineDesc.fragment.targets[4].blend?.color?.dstFactor,
		"one-minus-src-alpha"
	);
}

async function testWebGPUEarlyZPrepassOpaquePipelineHasDepthOnlyState() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const draw = await resources.getDrawResources(packet, frameResources, {
		drawMode: "early-z-prepass",
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.layout.desc.bindGroupLayouts.length, 3);
	assert.equal(typeof pipelineDesc.fragment, "undefined");
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, true);
	assert.equal(pipelineDesc.depthStencil.depthCompare, "less");
}

async function testWebGPUEarlyZPrepassMaskPipelineUsesMaskDepthFragment() {
	const backend = new FakeBackend();
	const material = new PBRMaterial();
	material.alphaMode = AlphaMode.Mask;
	const model = createModel([material]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const draw = await resources.getDrawResources(packet, frameResources, {
		drawMode: "early-z-prepass",
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.layout.desc.bindGroupLayouts.length, 3);
	assert.equal(pipelineDesc.fragment.entryPoint, "fsMainDepthMask");
	assert.equal(pipelineDesc.fragment.targets.length, 0);
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, true);
	assert.equal(pipelineDesc.depthStencil.depthCompare, "less");
}

async function testWebGPUEarlyZColorPipelineUsesReadOnlyDepthState() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const draw = await resources.getDrawResources(packet, frameResources, {
		drawMode: "early-z-color",
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(pipelineDesc.depthStencil.depthCompare, "less-equal");
}

async function testWebGPUEarlyZShaderMaterialDepthContract() {
	const backend = new FakeBackend();
	const shaderMaterial = new ShaderMaterial({
		name: "EarlyZShaderMask",
		alphaMode: AlphaMode.Mask,
		vertexEntryPoint: "customVs",
		depthFragmentEntryPoint: "customDepth",
		depthFragmentCode: /* wgsl */ `
@fragment
fn customDepth() {
}
`,
		chunks: [
			{
				language: "wgsl",
				stage: "vertex",
				code: /* wgsl */ `
@vertex
fn customVs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(position, 1.0);
}
`,
			},
		],
	});
	const supportedModel = createModel([shaderMaterial]);
	const supportedPacket = createPacket(supportedModel);
	const frame = createFrame(supportedPacket);
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const supportedDraw = await resources.getDrawResources(
		supportedPacket,
		frameResources,
		{
			drawMode: "early-z-prepass",
		}
	);
	assert.ok(supportedDraw && supportedDraw.length > 0);
	assert.equal(
		supportedDraw[0].pipeline.desc.fragment.entryPoint,
		"customDepth"
	);

	const missingContractMaterial = new ShaderMaterial({
		name: "EarlyZShaderMaskMissingDepth",
		alphaMode: AlphaMode.Mask,
		vertexEntryPoint: "customVs",
		chunks: [
			{
				language: "wgsl",
				stage: "vertex",
				code: /* wgsl */ `
@vertex
fn customVs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(position, 1.0);
}
`,
			},
		],
	});
	const unsupportedModel = createModel([missingContractMaterial]);
	const unsupportedPacket = createPacket(unsupportedModel);
	const unsupportedDraw = await resources.getDrawResources(
		unsupportedPacket,
		frameResources,
		{
			drawMode: "early-z-prepass",
		}
	);
	assert.equal(unsupportedDraw, null);
}

async function testWebGPUShaderMaterialDepthWriteFalseSkipsDepthPrepass() {
	const backend = new FakeBackend();
	const shaderMaterial = new ShaderMaterial({
		name: "DepthReadShader",
		depthWrite: false,
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFs",
		chunks: [
			{
				language: "wgsl",
				stage: "vertex",
				code: /* wgsl */ `
@vertex
fn customVs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(position, 1.0);
}
`,
			},
			{
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: /* wgsl */ `
@fragment
fn customFs() -> @location(0) vec4<f32> {
	return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`,
			},
		],
	});
	const model = createModel([shaderMaterial]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const prepassDraw = await resources.getDrawResources(packet, frameResources, {
		sceneTargetMode: "single",
		drawMode: "early-z-prepass",
	});
	assert.equal(prepassDraw, null);

	const draw = await resources.getDrawResources(packet, frameResources, {
		sceneTargetMode: "single",
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(pipelineDesc.depthStencil.depthCompare, "less");

	const earlyZColorDraw = await resources.getDrawResources(
		packet,
		frameResources,
		{
			sceneTargetMode: "single",
			drawMode: "early-z-color",
		}
	);
	assert.ok(earlyZColorDraw && earlyZColorDraw.length > 0);
	const earlyZColorPipelineDesc = earlyZColorDraw[0].pipeline.desc;
	assert.equal(earlyZColorPipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(earlyZColorPipelineDesc.depthStencil.depthCompare, "less");
}

async function testWebGPUShaderMaterialCustomUniformBufferBinding() {
	const backend = new FakeBackend();
	const shaderMaterial = new ShaderMaterial({
		name: "CustomUniformShader",
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFs",
		uniformBindings: [
			{ name: "time", type: "f32", value: 1 },
		],
		chunks: [
			{
				language: "wgsl",
				stage: "vertex",
				code: /* wgsl */ `
@vertex
fn customVs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(position, 1.0);
}
`,
			},
			{
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: /* wgsl */ `
@fragment
fn customFs() -> @location(0) vec4<f32> {
	return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`,
			},
		],
	});
	const model = createModel([shaderMaterial]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const firstDraw = await resources.getDrawResources(packet, frameResources);
	assert.ok(firstDraw && firstDraw.length > 0);
	const firstUniformEntry = firstDraw[0].modelBinding.desc.entries.find(
		(entry) => entry.binding === WEBGPU_MODEL_BINDING_SHADER_UNIFORMS
	);
	assert.ok(firstUniformEntry);
	assert.ok(
		String(firstUniformEntry.resource.label).startsWith(
			"ShaderMaterialUniform_"
		)
	);
	assert.deepEqual(firstUniformEntry.resource.lastWrite.slice(0, 4), [
		0,
		0,
		128,
		63,
	]);

	const firstResource = firstUniformEntry.resource;
	shaderMaterial.setUniform("time", 2);
	const secondDraw = await resources.getDrawResources(packet, frameResources);
	const secondUniformEntry = secondDraw[0].modelBinding.desc.entries.find(
		(entry) => entry.binding === WEBGPU_MODEL_BINDING_SHADER_UNIFORMS
	);
	assert.strictEqual(secondUniformEntry.resource, firstResource);
	assert.deepEqual(secondUniformEntry.resource.lastWrite.slice(0, 4), [
		0,
		0,
		0,
		64,
	]);

	shaderMaterial.setUniformBinding({
		name: "transform",
		type: "mat4x4f",
		value: Matrix4.identity(),
	});
	const thirdDraw = await resources.getDrawResources(packet, frameResources);
	const thirdUniformEntry = thirdDraw[0].modelBinding.desc.entries.find(
		(entry) => entry.binding === WEBGPU_MODEL_BINDING_SHADER_UNIFORMS
	);
	assert.notStrictEqual(thirdUniformEntry.resource, firstResource);
	assert.ok(thirdUniformEntry.resource.size > firstResource.size);
}

async function testWebGPUOITTransparentPipelineUsesDualTargets() {
	const backend = new FakeBackend();
	const material = new PBRMaterial({
		albedo: { r: 255, g: 255, b: 255 },
		opacity: 0.6,
	});
	material.alphaMode = AlphaMode.Blend;
	const model = createModel([material]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.opaquePackets = [];
	frame.transparentPackets = [packet];
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
				enableOIT: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				oit: true,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const draw = await resources.getDrawResources(packet, frameResources, {
		transparentPipelineMode: "oit",
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(pipelineDesc.fragment.entryPoint, "fsMainOIT");
	assert.equal(pipelineDesc.fragment.targets.length, 2);
	assert.equal(pipelineDesc.fragment.targets[0].format, "rgba16float");
	assert.equal(pipelineDesc.fragment.targets[1].format, "r8unorm");
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.srcFactor,
		"one"
	);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.dstFactor,
		"one"
	);
	assert.equal(
		pipelineDesc.fragment.targets[1].blend?.color?.srcFactor,
		"zero"
	);
	assert.equal(
		pipelineDesc.fragment.targets[1].blend?.color?.dstFactor,
		"one-minus-src"
	);
}

async function testWebGPUOITTransmissionMaterialsStayLegacyPipeline() {
	const backend = new FakeBackend();
	const material = new PBRMaterial({
		albedo: { r: 255, g: 255, b: 255 },
		roughness: 0.05,
		metalness: 0,
		transmissionFactor: 1,
		ior: 1.52,
	});
	const model = createModel([material]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.opaquePackets = [];
	frame.transparentPackets = [packet];
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
				enableOIT: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				oit: true,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const draw = await resources.getDrawResources(packet, frameResources, {
		transparentPipelineMode: "transmission",
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.fragment.entryPoint, "fsMain");
	assert.equal(pipelineDesc.fragment.targets.length, 5);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.srcFactor,
		"src-alpha"
	);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.dstFactor,
		"one-minus-src-alpha"
	);
}

async function testWebGPUOITParticlePipelinesSplitAlphaAndAdditive() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend);
	await resources.init();

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableShadows: true,
			enableOIT: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			environment: false,
			oit: true,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	resources.beginFrameResourceLifecycle();
	const frameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		createMainFrameOptions()
	);

	const texture = new Texture(
		new Uint8Array([255, 255, 255, 255]),
		1,
		1,
		"sRGB"
	);
	const context = {
		camera: frame.camera,
		attachments: { width: 16, height: 16 },
		features,
		postProcess: createResolvedPostProcess(),
		shadowMaps: frame.shadowMaps,
		scene: { ...frame, particleSystems: [] },
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		transient: new Map([
			[
				PARTICLE_TRANSIENT_BATCHES_KEY,
				[
					{
						systemId: "particleSystem-oit-alpha",
						blendMode: ParticleBlendMode.Alpha,
						texture,
						receiveShadows: true,
						particles: [
							{
								position: { x: 0, y: 0, z: 0 },
								size: 1,
								color: { r: 255, g: 255, b: 255, a: 1 },
								rotation: 0,
								depth: 1,
								uvRect: { u0: 0, v0: 0, u1: 1, v1: 1 },
							},
						],
					},
					{
						systemId: "particleSystem-oit-add",
						blendMode: ParticleBlendMode.Additive,
						texture,
						receiveShadows: false,
						particles: [
							{
								position: { x: 0, y: 0, z: 0 },
								size: 1,
								color: { r: 255, g: 255, b: 255, a: 1 },
								rotation: 0,
								depth: 1,
								uvRect: { u0: 0, v0: 0, u1: 1, v1: 1 },
							},
						],
					},
				],
			],
		]),
	};
	const encoder = new FakeRenderEncoder();
	const renderTarget = { width: 16, height: 16, destroy() {} };
	const alphaCount = await resources.renderParticles(
		encoder,
		context,
		{
			label: "WebGPUParticlesOIT_Test",
			colorAttachments: [
				{
					view: renderTarget,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: renderTarget,
					loadOp: "load",
					storeOp: "store",
				},
			],
			depth: renderTarget,
		},
		frameResources,
		"single",
		{
			includeBlendModes: [ParticleBlendMode.Alpha],
			pipelineMode: "oit",
		}
	);
	assert.equal(alphaCount, 1);

	const additiveCount = await resources.renderParticles(
		encoder,
		context,
		{
			label: "WebGPUParticlesAdd_Test",
			colorAttachments: [
				{
					view: renderTarget,
					loadOp: "load",
					storeOp: "store",
				},
			],
			depth: renderTarget,
		},
		frameResources,
		"single",
		{
			includeBlendModes: [ParticleBlendMode.Additive],
			pipelineMode: "legacy",
		}
	);
	assert.equal(additiveCount, 1);

	const oitPipeline = backend.pipelines.find(
		(pipeline) => pipeline.label === "WebGPUParticlePipeline_oit-alpha_single"
	);
	assert.ok(oitPipeline);
	assert.equal(oitPipeline.desc.fragment.entryPoint, "fsMainOIT");
	assert.equal(oitPipeline.desc.fragment.targets.length, 2);
	assert.equal(oitPipeline.desc.fragment.targets[1].format, "r8unorm");

	const additivePipeline = backend.pipelines.find(
		(pipeline) => pipeline.label === "WebGPUParticlePipeline_additive_single"
	);
	assert.ok(additivePipeline);
	assert.equal(additivePipeline.desc.fragment.entryPoint, "fsMain");
	assert.equal(additivePipeline.desc.fragment.targets.length, 1);
	assert.equal(
		additivePipeline.desc.fragment.targets[0].blend?.color?.dstFactor,
		"one"
	);
}

async function testWebGPUEnvironmentCombinationsRegression() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const baseScene = createFrame(packet);
	const resources = new WebGPURenderResources(backend);
	await resources.init();

	const caps = {
		sh: true,
		shadows: true,
		reflection: false,
		environment: true,
		ssao: false,
		taa: false,
		ssr: false,
		volumetric: false,
		fog: false,
		motionBlur: false,
		dof: false,
		bloom: false,
		clusteredLighting: true,
	};

	const shAmbient = SH.empty();
	shAmbient[0] = { r: 12, g: 12, b: 12 };
	const probeMap = createTinyTexture(2);
	const probe = new ReflectionProbe({
		shape: "box",
		prefilteredMap: probeMap,
	});

	const cases = [
		{
			environment: createEnvironmentSnapshot(
				createTinyTexture(1),
				createTinyTexture(1)
			),
			lights: [probe],
			enableSH: true,
			expectEnvironment: true,
		},
		{
			environment: createEnvironmentSnapshot(null, null),
			lights: [probe],
			enableSH: true,
			expectEnvironment: false,
		},
		{
			environment: createEnvironmentSnapshot(null, null),
			lights: [],
			enableSH: false,
			expectEnvironment: false,
		},
	];

	for (const scenario of cases) {
		const scene = {
			...baseScene,
			environment: scenario.environment,
			lights: scenario.lights,
		};
		const features = resolveFeatureState(
			{
				enableLighting: true,
				enableGamma: true,
				enableSH: scenario.enableSH,
				enableShadows: true,
				enableEnvironment: true,
			},
			caps,
			"webgpu"
		);
		const frameResources = resources.prepareFrame(
			{
				viewCamera: scene.camera,
				attachments: { width: 16, height: 16 },
				features,
				postProcess: createResolvedPostProcess(),
				shadowMaps: scene.shadowMaps,
				scene,
				shCoeffs: SH.empty(),
				shAmbientCoeffs: scenario.enableSH ? shAmbient : SH.empty(),
				worldMatrix: Matrix4.identity(),
				transient: new Map(),
			},
			createMainFrameOptions()
		);

		const environmentResources =
			await resources.getEnvironmentResources(frameResources);
		assert.equal(!!environmentResources, scenario.expectEnvironment);
		const draw = await resources.getDrawResources(packet, frameResources);
		assert.ok(draw);
	}
}

async function testScopedSceneTargetModesUseDistinctBindings() {
	const backend = new FakeBackend();
	backend.canvasFormat = "bgra8unorm";
	backend.canvasDepthFormat = "depth24plus";
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.environment = createEnvironmentSnapshot(
		createTinyTexture(1),
		createTinyTexture(1)
	);
	const resources = new WebGPURenderResources(backend);
	await resources.init();

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableShadows: true,
			enableEnvironment: true,
			enableClusteredLighting: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			environment: true,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);

	const mainFrameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		{
			scopeKey: "main",
			sceneTargetMode: "single",
		}
	);
	const mainFrameBinding = mainFrameResources.frameBinding;
	const captureFrameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		{
			scopeKey: "test-capture",
			sceneTargetMode: "mrt",
			temporalStateMode: "disabled",
		}
	);
	assert.notEqual(captureFrameResources.frameBinding, mainFrameBinding);

	const captureDrawResources = await resources.getDrawResources(
		packet,
		captureFrameResources
	);
	assert.ok(captureDrawResources);
	assert.equal(captureDrawResources[0].pipeline.label.endsWith("_mrt"), true);
	assert.equal(mainFrameResources.frameBinding, mainFrameBinding);

	const environmentResources =
		await resources.getEnvironmentResources(mainFrameResources);
	assert.ok(environmentResources);
	assert.equal(
		environmentResources.pipeline.label,
		"WebGPUEnvironmentPipeline_single"
	);
	assert.equal(
		environmentResources.pipeline.desc.depthStencil.format,
		backend.canvasDepthFormat
	);

	const drawResources = await resources.getDrawResources(
		packet,
		mainFrameResources,
		{ sceneTargetMode: "single" }
	);
	assert.ok(drawResources);
	assert.equal(drawResources[0].pipeline.label.endsWith("_single"), true);
	assert.equal(
		drawResources[0].pipeline.desc.depthStencil.format,
		backend.canvasDepthFormat
	);
}

async function testSampleCountOverrideUsesSingleSampleCapturePipelines() {
	const backend = new FakeBackend();
	backend.canvasFormat = "bgra8unorm";
	backend.canvasDepthFormat = "depth24plus";
	const msaa = {
		sampleCount: 4,
		resolveSupportedSampleCount: (requested) => Math.max(1, Math.floor(requested)),
		fallbackToSingleSample: () => false,
	};
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.environment = createEnvironmentSnapshot(
		createTinyTexture(1),
		createTinyTexture(1)
	);
	const resources = new WebGPURenderResources(backend, msaa);
	await resources.init();

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableShadows: false,
			enableEnvironment: true,
			enableClusteredLighting: true,
		},
		{
			sh: false,
			shadows: false,
			reflection: false,
			environment: true,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	const frameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		{
			scopeKey: "main",
			sceneTargetMode: "mrt",
		}
	);

	const mainDrawResources = await resources.getDrawResources(
		packet,
		frameResources
	);
	assert.ok(mainDrawResources);
	assert.equal(mainDrawResources[0].pipeline.desc.sampleCount, 4);

	const captureDrawResources = await resources.getDrawResources(
		packet,
		frameResources,
		{
			sceneTargetMode: "mrt",
			drawMode: "reflection-capture",
			sampleCountOverride: 1,
		}
	);
	assert.ok(captureDrawResources);
	assert.equal(captureDrawResources[0].pipeline.desc.sampleCount, 1);

	const mainEnvironment =
		await resources.getEnvironmentResources(frameResources, "mrt");
	assert.ok(mainEnvironment);
	assert.equal(mainEnvironment.pipeline.desc.sampleCount, 4);

	const captureEnvironment =
		await resources.getEnvironmentResources(frameResources, "mrt", {
			sampleCountOverride: 1,
		});
	assert.ok(captureEnvironment);
	assert.equal(captureEnvironment.pipeline.desc.sampleCount, 1);
}

async function testReflectionProbeCaptureUsesCanvasAttachmentFormats() {
	const backend = new FakeBackend();
	backend.canvasFormat = "bgra8unorm";
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const preparedScene = {
		...createFrame(packet),
		particleSystems: [],
		hasActiveAnimations: false,
		spatialIndex: null,
	};
	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableClusteredLighting: true,
			enableEnvironment: false,
			enableShadows: false,
			enableReflection: false,
			enableOIT: false,
			enableSSAO: false,
			enableSSGI: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableFog: false,
			enableMotionBlur: false,
			enableDOF: false,
			enableBloom: false,
		},
		{
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			oit: false,
			ssao: false,
			ssgi: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	const frameContext = {
		camera: preparedScene.camera,
		attachments: { width: 1, height: 1 },
		features,
		postProcess: createResolvedPostProcess(),
		shadowMaps: preparedScene.shadowMaps,
		scene: preparedScene,
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width: 1, height: 1 }],
			dirtyTileSize: 1,
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			reasonMask: 0,
			temporalHistoryReset: true,
		},
		transient: new Map(),
	};
	const resources = {
		createFrameScope() { return createFrameScopeAdapter(resources); },
		prepareFrame(_context, options = {}) {
			return createPreparedFrameResources(options);
		},
		releaseScope() {},
		async buildClusteredLighting() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources() {
			return null;
		},
		async renderParticles() {
			return 0;
		},
		buildParticleMeshDrawPackets() {
			return [];
		},
	};
	const probe = new ReflectionProbe({
		includeMeshes: false,
		includeEnvironment: false,
		includeTransparent: false,
		includeParticles: false,
		includeShadows: false,
	});
	const capturePass = new WebGPUReflectionProbeCapturePass(backend, resources);
	const probeCache = probe.getRuntimeCache();
	const result = await capturePass.captureFace({
		frameContext,
		targetId: probe.id,
		targetKind: "reflection",
		captureWorldPosition: probeCache.captureWorldPosition,
		captureFar: probe.captureFar,
		faceIndex: 0,
		faceSize: 1,
		includeEnvironment: false,
		includeMeshes: false,
		includeTransparent: false,
		includeParticles: false,
		includeShadows: false,
	});

	assert.ok(result);
	assert.equal(result.length, 4);
	assert.equal(backend.createTextureCalls.length >= 2, true);
	assert.equal(backend.createTextureCalls[0].format, TextureFormat.RGBA16Float);
	assert.equal(
		backend.createTextureCalls.some(
			(call) => call.format === TextureFormat.Depth32Float
		),
		true
	);
}

async function testReflectionProbeCaptureUsesParentWorldPositionAsOrigin() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const preparedScene = {
		...createFrame(packet),
		particleSystems: [],
		hasActiveAnimations: false,
		spatialIndex: null,
	};
	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableClusteredLighting: true,
			enableEnvironment: false,
			enableShadows: false,
			enableReflection: false,
			enableOIT: false,
			enableSSAO: false,
			enableSSGI: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableFog: false,
			enableMotionBlur: false,
			enableDOF: false,
			enableBloom: false,
		},
		{
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			oit: false,
			ssao: false,
			ssgi: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	const frameContext = {
		camera: preparedScene.camera,
		attachments: { width: 1, height: 1 },
		features,
		postProcess: createResolvedPostProcess(),
		shadowMaps: preparedScene.shadowMaps,
		scene: preparedScene,
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width: 1, height: 1 }],
			dirtyTileSize: 1,
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			reasonMask: 0,
			temporalHistoryReset: true,
		},
		transient: new Map(),
	};
	const preparedCameraPositions = [];
	const resources = {
		createFrameScope() { return createFrameScopeAdapter(resources); },
		prepareFrame(context, options = {}) {
			preparedCameraPositions.push(
				context.viewCamera.getWorldPosition({ x: 0, y: 0, z: 0 })
			);
			return createPreparedFrameResources(options);
		},
		releaseScope() {},
		async buildClusteredLighting() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources() {
			return null;
		},
		async renderParticles() {
			return 0;
		},
		buildParticleMeshDrawPackets() {
			return [];
		},
	};
	const modelRoot = new Node();
	modelRoot.position.set(3, 0, 0);
	const probe = new ReflectionProbe({
		includeMeshes: false,
		includeEnvironment: false,
		includeTransparent: false,
		includeParticles: false,
		includeShadows: false,
	});
	modelRoot.addChild(probe);
	probe.position.set(2, 0, 0);
	modelRoot.updateWorldMatrix();
	const capturePass = new WebGPUReflectionProbeCapturePass(backend, resources);
	const probeCache = probe.getRuntimeCache();

	await capturePass.captureFace({
		frameContext,
		targetId: probe.id,
		targetKind: "reflection",
		captureWorldPosition: probeCache.captureWorldPosition,
		captureFar: probe.captureFar,
		faceIndex: 0,
		faceSize: 1,
		includeEnvironment: false,
		includeMeshes: false,
		includeTransparent: false,
		includeParticles: false,
		includeShadows: false,
	});

	assert.ok(preparedCameraPositions.length >= 1);
	assert.deepEqual(preparedCameraPositions[0], { x: 3, y: 0, z: 0 });
}

async function testParticleUVLayoutAndUniformBinding() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend);

	await resources.init();
	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableShadows: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			environment: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);

	const frameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		createMainFrameOptions()
	);

	const texture = new Texture(
		new Uint8Array([255, 255, 255, 255]),
		1,
		1,
		"sRGB"
	);
	texture.repeat = { x: 2, y: 3 };
	texture.offset = { x: 0.25, y: -0.5 };
	texture.rotation = Math.PI / 4;

	const context = {
		camera: frame.camera,
		attachments: { width: 16, height: 16 },
		features,
		postProcess: createResolvedPostProcess(),
		shadowMaps: frame.shadowMaps,
		scene: {
			...frame,
			particleSystems: [],
		},
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		transient: new Map([
			[
				PARTICLE_TRANSIENT_BATCHES_KEY,
				[
					{
						systemId: "particleSystem-test",
						blendMode: ParticleBlendMode.Alpha,
						texture,
						receiveShadows: true,
						particles: [
							{
								position: { x: 0, y: 0, z: 0 },
								size: 1,
								color: { r: 255, g: 255, b: 255, a: 1 },
								rotation: 0,
								depth: 1,
								uvRect: { u0: 0, v0: 0, u1: 1, v1: 1 },
							},
						],
					},
				],
			],
		]),
	};

	const encoder = new FakeRenderEncoder();
	const renderTarget = { width: 16, height: 16, destroy() {} };
	await resources.renderParticles(
		encoder,
		context,
		{
			label: "WebGPUParticles_TestUV",
			colorAttachments: [
				{
					view: renderTarget,
					loadOp: "load",
					storeOp: "store",
				},
			],
			depth: renderTarget,
		},
		frameResources,
		"single"
	);

	const particleLayout = backend.device.bindGroupLayouts.find(
		(layout) => layout.desc.label === "WebGPUParticleBindGroupLayout"
	);
	assert.ok(particleLayout);
	assert.equal(particleLayout.desc.entries.length, 3);

	const particlePipeline = backend.pipelines.find((pipeline) =>
		String(pipeline.label).startsWith("WebGPUParticlePipeline_")
	);
	assert.ok(particlePipeline);
	assert.deepEqual(
		particlePipeline.desc.vertex.buffers,
		WEBGPU_PARTICLE_VERTEX_LAYOUTS
	);

	const particleBinding = backend.bindingGroups.find(
		(binding) => binding.label === "ParticleBinding_particleSystem-test"
	);
	assert.ok(particleBinding);
	assert.equal(particleBinding.desc.entries.length, 3);

	const uvBuffer = particleBinding.desc.entries[2].resource;
	assert.ok(uvBuffer?.lastWrite);
	const uvTransform = Array.from(uvBuffer.lastWrite);
	assert.equal(uvTransform.length, 8);
	assert.ok(Math.abs(uvTransform[0] - 2) < 1e-6);
	assert.ok(Math.abs(uvTransform[1] - 3) < 1e-6);
	assert.ok(Math.abs(uvTransform[2] - 0.25) < 1e-6);
	assert.ok(Math.abs(uvTransform[3] + 0.5) < 1e-6);
	assert.ok(Math.abs(uvTransform[4] - Math.cos(Math.PI / 4)) < 1e-6);
	assert.ok(Math.abs(uvTransform[5] - Math.sin(Math.PI / 4)) < 1e-6);
}

async function testFrameBindingReplacementDestroysOldBinding() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend);
	await resources.init();

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableEnvironment: true,
		},
		{
			sh: false,
			shadows: false,
			reflection: false,
			environment: true,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);

	const firstFrameResources = resources.prepareFrame(
		createFrameContext(
			{
				...frame,
				environment: createEnvironmentSnapshot(
					createTinyTexture(1),
					createTinyTexture(1)
				),
			},
			features
		),
		createMainFrameOptions()
	);
	const firstEnvironment =
		await resources.getEnvironmentResources(firstFrameResources);
	assert.ok(firstEnvironment);
	const firstBinding = firstEnvironment.frameBinding;
	assert.equal(firstBinding.destroyed, false);

	const secondFrameResources = resources.prepareFrame(
		createFrameContext(
			{
				...frame,
				environment: createEnvironmentSnapshot(
					createTinyTexture(1),
					createTinyTexture(1)
				),
			},
			features
		),
		createMainFrameOptions()
	);
	assert.equal(firstBinding.destroyed, true);

	const secondEnvironment =
		await resources.getEnvironmentResources(secondFrameResources);
	assert.ok(secondEnvironment);
	assert.notEqual(secondEnvironment.frameBinding, firstBinding);
}

function createWebGPUFrameContextForTemporalTest(
	frame,
	features,
	postProcess,
	width,
	height,
	temporalHistoryReset = false
) {
	return {
		viewCamera: frame.camera,
		attachments: { width, height },
		features,
		postProcess,
		shadowMaps: frame.shadowMaps,
		scene: frame,
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width, height }],
			dirtyTileSize: Math.max(width, height),
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			postProcessStartPass: null,
			reasonMask: 0,
			temporalHistoryReset,
		},
		transient: new Map(),
	};
}

function readLatestFrameCameraUniformField(backend, field) {
	const frameWrites = backend.writeCalls.filter(
		(call) =>
			call[0] === "writeBuffer" &&
			call[1]?.label === "WebGPUFrameCameraUniforms"
	);
	assert.ok(frameWrites.length > 0);
	const data = frameWrites[frameWrites.length - 1][2];
	const offset = WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT.byteOffsetOf(field) / 4;
	const length = WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT.byteSizeOf(field) / 4;
	return Array.from(data.slice(offset, offset + length));
}

function assertArrayNearlyEqual(actual, expected, epsilon = 1e-6) {
	assert.equal(actual.length, expected.length);
	for (let i = 0; i < actual.length; i++) {
		nearlyEqual(actual[i], expected[i], epsilon);
	}
}

async function testWebGPUPrepareFrameTemporalStateModes() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const resources = new WebGPURenderResources(backend);
	await resources.init();

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
		},
		{
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			ssao: false,
			taa: true,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	const postProcess = createResolvedPostProcess(
		{ taa: { enabled: true, options: { jitterScale: 1 } } },
		"webgpu"
	);
	const width = 16;
	const height = 8;
	const previousViewProjection = new Matrix4([
		[1, 0, 0, 10],
		[0, 1, 0, 20],
		[0, 0, 1, 30],
		[0, 0, 0, 1],
	]);
	const currentViewProjection = new Matrix4([
		[2, 0, 0, 40],
		[0, 2, 0, 50],
		[0, 0, 2, 60],
		[0, 0, 0, 1],
	]);
	const captureViewProjection = new Matrix4([
		[3, 0, 0, 70],
		[0, 3, 0, 80],
		[0, 0, 3, 90],
		[0, 0, 0, 1],
	]);

	const previousFrame = createFrame(packet);
	previousFrame.camera.viewProjectionMatrix = previousViewProjection;
	const currentFrame = createFrame(packet);
	currentFrame.camera.viewProjectionMatrix = currentViewProjection;
	const captureFrame = createFrame(packet);
	captureFrame.camera.viewProjectionMatrix = captureViewProjection;

	const previousContext = createWebGPUFrameContextForTemporalTest(
		previousFrame,
		features,
		postProcess,
		width,
		height
	);
	resources.prepareFrame(previousContext, createMainFrameOptions());
	const firstJitter = computeHaltonJitterNDC(0, width, height, 1);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
		[firstJitter[0], firstJitter[1], 0, 0]
	);

	const currentContext = createWebGPUFrameContextForTemporalTest(
		currentFrame,
		features,
		postProcess,
		width,
		height
	);
	resources.prepareFrame(currentContext, createMainFrameOptions());
	const secondJitter = computeHaltonJitterNDC(1, width, height, 1);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "prevViewProjection"),
		Array.from(packMatrix4ForWGSL(previousViewProjection))
	);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
		[secondJitter[0], secondJitter[1], firstJitter[0], firstJitter[1]]
	);

	const captureContext = createWebGPUFrameContextForTemporalTest(
		captureFrame,
		features,
		postProcess,
		width,
		height,
		true
	);
	resources.prepareFrame(
		captureContext,
		createMainFrameOptions({
			scopeKey: "capture",
			temporalStateMode: "disabled",
		})
	);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "prevViewProjection"),
		Array.from(packMatrix4ForWGSL(captureViewProjection))
	);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
		[0, 0, 0, 0]
	);

	resources.prepareFrame(
		currentContext,
		createMainFrameOptions({ temporalStateMode: "reuse" })
	);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "prevViewProjection"),
		Array.from(packMatrix4ForWGSL(previousViewProjection))
	);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
		[secondJitter[0], secondJitter[1], firstJitter[0], firstJitter[1]]
	);

	resources.prepareFrame(currentContext, createMainFrameOptions());
	const thirdJitter = computeHaltonJitterNDC(2, width, height, 1);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
		[thirdJitter[0], thirdJitter[1], secondJitter[0], secondJitter[1]]
	);

	const resetContext = createWebGPUFrameContextForTemporalTest(
		currentFrame,
		features,
		postProcess,
		width,
		height,
		true
	);
	resources.prepareFrame(resetContext, createMainFrameOptions());
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "prevViewProjection"),
		Array.from(packMatrix4ForWGSL(currentViewProjection))
	);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
		[firstJitter[0], firstJitter[1], 0, 0]
	);
}

async function testSceneFrameBindingLayoutMatchesFallbackEnvironmentContract() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(backend);
	await resources.init();

	const sceneLayout = backend.device.bindGroupLayouts.find(
		(layout) => layout.desc.label === "WebGPUSceneFrameBindGroupLayout"
	);
	assert.ok(sceneLayout);
	assert.equal(sceneLayout.desc.entries.length, 17);
	assert.deepEqual(
		sceneLayout.desc.entries.map((entry) => entry.binding),
		[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
	);
	assert.equal(sceneLayout.desc.entries[4].texture?.sampleType, "float");
	assert.equal(sceneLayout.desc.entries[5].sampler?.type, "filtering");
	assert.equal(sceneLayout.desc.entries[6].buffer?.type, "uniform");
	assert.equal(sceneLayout.desc.entries[7].buffer?.type, "read-only-storage");
	assert.equal(sceneLayout.desc.entries[8].texture?.sampleType, "float");
	assert.equal(sceneLayout.desc.entries[9].texture?.sampleType, "float");
	assert.equal(sceneLayout.desc.entries[10].texture?.sampleType, "float");
	assert.equal(sceneLayout.desc.entries[11].texture?.sampleType, "uint");
	assert.equal(sceneLayout.desc.entries[12].texture?.sampleType, "depth");
	assert.equal(sceneLayout.desc.entries[13].sampler?.type, "comparison");
	assert.equal(sceneLayout.desc.entries[14].buffer?.type, "uniform");
	assert.equal(sceneLayout.desc.entries[15].buffer?.type, "uniform");
	assert.equal(sceneLayout.desc.entries[16].buffer?.type, "uniform");

	resources.destroy();
}

async function testParticleShadowVolumeBufferUpdatesForDirectionalSlice() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const scene = new Scene();
	const light = new DirectionalLight();
	scene.add(light);
	scene.shadows.bind(light, scene.shadows.createSingle({
		size: 8,
		bias: {
			constant: 0,
			slope: 0,
			texel: 0,
			max: 1,
			normal: 0,
			normalMin: 0,
		},
	}));
	const shadowConfig = scene.shadows.getLegacyShadowConfig(light);
	assert.ok(shadowConfig);
	const renderSet = createShadowRenderSet(shadowConfig);
	renderSet.slices[0].shadowMap.viewProjectionMatrix = Matrix4.identity();
	frame.lights = [light];
	frame.shadowMaps = new Map([[light, renderSet]]);

	const resources = new WebGPURenderResources(backend);
	await resources.init();
	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableShadows: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			environment: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	const frameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		createMainFrameOptions()
	);
	resources.updateParticleShadowVolumes(
		frameResources,
		{
			camera: frame.camera,
			attachments: { width: 16, height: 16 },
			features,
			postProcess: createResolvedPostProcess(),
			shadowMaps: frame.shadowMaps,
			scene: { ...frame, particleSystems: [] },
			shCoeffs: SH.empty(),
			shAmbientCoeffs: SH.empty(),
			worldMatrix: Matrix4.identity(),
			transient: new Map([
				[
					PARTICLE_TRANSIENT_BATCHES_KEY,
					[
						{
							systemId: "particle-shadow-volume",
							blendMode: ParticleBlendMode.Alpha,
							texture: null,
							receiveShadows: true,
							castShadows: true,
							shadowDensity: 8,
							shadowSoftness: 1,
							particles: [
								{
									position: { x: 0, y: 0, z: 0 },
									size: 2,
									color: { r: 255, g: 255, b: 255, a: 1 },
									rotation: 0,
									depth: 1,
									uvRect: { u0: 0, v0: 0, u1: 1, v1: 1 },
								},
							],
						},
					],
				],
			]),
		}
	);

	const volumeBuffer = backend.buffers.findLast(
		(buffer) =>
			buffer.label === "WebGPUParticleShadowVolumeBuffer" &&
			!buffer.destroyed
	);
	assert.ok(volumeBuffer);
	assert.ok(volumeBuffer.size > 1024);
	assert.equal(volumeBuffer.lastWrite[16], 1);
	assert.equal(volumeBuffer.lastWrite[17], 64);
	assert.equal(volumeBuffer.lastWrite[18], 64);
	assert.equal(volumeBuffer.lastWrite[19], 32);
	assert.equal(volumeBuffer.lastWrite[20], 96);
	assert.ok(
		volumeBuffer.lastWrite.slice(96).some((value) => value > 0),
		"Particle shadow volume buffer should contain injected density"
	);

	resources.destroy();
}

async function testShadowAtlasSizeTracksShadowMapsWhenLightingDisabled() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const scene = new Scene();
	const light = new DirectionalLight();
	scene.add(light);
	scene.shadows.bind(light, scene.shadows.createSingle({ size: 1024 }));
	const shadowConfig = scene.shadows.getLegacyShadowConfig(light);
	assert.ok(shadowConfig);
	const shadowRenderSet = createShadowRenderSet(shadowConfig);
	shadowRenderSet.slices[0].shadowMap.viewProjectionMatrix = Matrix4.identity();
	frame.lights = [light];
	frame.shadowMaps = new Map([[light, shadowRenderSet]]);
	const resources = new WebGPURenderResources(backend);
	await resources.init();
	const features = resolveFeatureState(
		{
			enableLighting: false,
			enableGamma: true,
			enableShadows: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			environment: false,
			ssao: false,
			ssgi: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	const frameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		createMainFrameOptions()
	);
	const frameBinding = resources.getFrameBinding(frameResources);
	const shadowAtlasEntry = frameBinding.desc.entries.find(
		(entry) => entry.binding === 1
	);
	const shadowTransmittanceAtlasEntry = frameBinding.desc.entries.find(
		(entry) => entry.binding === 8
	);
	const shadowComparisonSamplerEntry = frameBinding.desc.entries.find(
		(entry) => entry.binding === 13
	);
	assert.ok(shadowAtlasEntry?.resource);
	assert.ok(shadowTransmittanceAtlasEntry?.resource);
	assert.ok(shadowComparisonSamplerEntry?.resource);
	assert.equal(shadowComparisonSamplerEntry.resource.desc.compare, "less-equal");
	assert.equal(
		shadowAtlasEntry.resource.width,
		1024 * WEBGPU_SHADOW_ATLAS_COLUMNS
	);
	assert.equal(
		shadowTransmittanceAtlasEntry.resource.width,
		1024 * WEBGPU_SHADOW_ATLAS_COLUMNS
	);
}

async function testParticleBindingCacheEvictsStaleSystems() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend);
	await resources.init();

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableShadows: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			environment: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	let frameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		createMainFrameOptions()
	);

	const texture = new Texture(
		new Uint8Array([255, 255, 255, 255]),
		1,
		1,
		"sRGB"
	);
	const context = {
		camera: frame.camera,
		attachments: { width: 16, height: 16 },
		features,
		postProcess: createResolvedPostProcess(),
		shadowMaps: frame.shadowMaps,
		scene: { ...frame, particleSystems: [] },
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		transient: new Map([
			[
				PARTICLE_TRANSIENT_BATCHES_KEY,
				[
					{
						systemId: "particleSystem-evict",
						blendMode: ParticleBlendMode.Alpha,
						texture,
						receiveShadows: true,
						particles: [
							{
								position: { x: 0, y: 0, z: 0 },
								size: 1,
								color: { r: 255, g: 255, b: 255, a: 1 },
								rotation: 0,
								depth: 1,
								uvRect: { u0: 0, v0: 0, u1: 1, v1: 1 },
							},
						],
					},
				],
			],
		]),
	};
	const encoder = new FakeRenderEncoder();
	const renderTarget = { width: 16, height: 16, destroy() {} };
	await resources.renderParticles(
		encoder,
		context,
		{
			label: "WebGPUParticles_TestEvict",
			colorAttachments: [
				{
					view: renderTarget,
					loadOp: "load",
					storeOp: "store",
				},
			],
			depth: renderTarget,
		},
		frameResources,
		"single"
	);

	const particleBinding = backend.bindingGroups.find(
		(binding) => binding.label === "ParticleBinding_particleSystem-evict"
	);
	assert.ok(particleBinding);
	const uvBuffer = particleBinding.desc.entries[2].resource;
	assert.ok(uvBuffer);
	assert.equal(uvBuffer.destroyed, false);

	for (let i = 0; i < 130; i++) {
		resources.beginFrameResourceLifecycle();
		frameResources = resources.prepareFrame(
			createFrameContext(frame, features),
			createMainFrameOptions()
		);
	}

	assert.equal(particleBinding.destroyed, true);
	assert.equal(uvBuffer.destroyed, true);
}

async function testRenderResourcesDestroyCleansParticleAndGeometryResources() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend);
	await resources.init();

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableShadows: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			environment: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	const frameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		createMainFrameOptions()
	);
	const draw = await resources.getDrawResources(packet, frameResources);
	assert.ok(draw && draw.length > 0);
	const geometryDraw = draw[0];

	const texture = new Texture(
		new Uint8Array([255, 255, 255, 255]),
		1,
		1,
		"sRGB"
	);
	const context = {
		camera: frame.camera,
		attachments: { width: 16, height: 16 },
		features,
		postProcess: createResolvedPostProcess(),
		shadowMaps: frame.shadowMaps,
		scene: { ...frame, particleSystems: [] },
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		transient: new Map([
			[
				PARTICLE_TRANSIENT_BATCHES_KEY,
				[
					{
						systemId: "particleSystem-destroy",
						blendMode: ParticleBlendMode.Alpha,
						texture,
						receiveShadows: true,
						particles: [
							{
								position: { x: 0, y: 0, z: 0 },
								size: 1,
								color: { r: 255, g: 255, b: 255, a: 1 },
								rotation: 0,
								depth: 1,
								uvRect: { u0: 0, v0: 0, u1: 1, v1: 1 },
							},
						],
					},
				],
			],
		]),
	};
	const encoder = new FakeRenderEncoder();
	const renderTarget = { width: 16, height: 16, destroy() {} };
	await resources.renderParticles(
		encoder,
		context,
		{
			label: "WebGPUParticles_TestDestroy",
			colorAttachments: [
				{
					view: renderTarget,
					loadOp: "load",
					storeOp: "store",
				},
			],
			depth: renderTarget,
		},
		frameResources,
		"single"
	);
	const particleBinding = backend.bindingGroups.find(
		(binding) => binding.label === "ParticleBinding_particleSystem-destroy"
	);
	assert.ok(particleBinding);
	const particleUvBuffer = particleBinding.desc.entries[2].resource;
	assert.ok(particleUvBuffer);

	resources.destroy();

	assert.equal(geometryDraw.vertexBuffer.destroyed, true);
	assert.equal(geometryDraw.indexBuffer.destroyed, true);
	assert.equal(geometryDraw.modelBinding.destroyed, true);
	assert.equal(particleBinding.destroyed, true);
	assert.equal(particleUvBuffer.destroyed, true);
	assert.ok(
		backend.buffers.some(
			(buffer) =>
				buffer.desc.label === "WebGPUParticleQuad" && buffer.destroyed === true
		)
	);
	assert.ok(
		backend.buffers.some(
			(buffer) =>
				buffer.desc.label === "WebGPUParticleInstances" &&
				buffer.destroyed === true
		)
	);
}

function testWebGPUGeometryRegistryReleaseGeometryDestroysBuffers() {
	const backend = new FakeBackend();
	const registry = new WebGPUGeometryRegistry(backend);
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const handle = registry.getGeometry(packet.primitive);

	assert.equal(handle.vertexBuffer.destroyed, false);
	assert.equal(handle.indexBuffer.destroyed, false);
	assert.equal(handle.wireframeIndexBuffer.destroyed, false);

	registry.releaseGeometry(packet.primitive);

	assert.equal(handle.vertexBuffer.destroyed, true);
	assert.equal(handle.indexBuffer.destroyed, true);
	assert.equal(handle.wireframeIndexBuffer.destroyed, true);
	registry.destroy();
}

function testDynamicTextureReuploadOnVersionChange() {
	const backend = new FakeBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture(
		new Uint8ClampedArray([10, 20, 30, 255]),
		1,
		1,
		"sRGB"
	);

	const first = registry.getTextureForSlot(
		texture,
		WEBGPU_TEXTURE_SLOT.BASE_COLOR
	);
	assert.ok(first);
	assert.equal(backend.textureWrites.length, 1);

	const second = registry.getTextureForSlot(
		texture,
		WEBGPU_TEXTURE_SLOT.BASE_COLOR
	);
	assert.equal(second, first);
	assert.equal(backend.textureWrites.length, 1);

	texture.data[0] = 200;
	texture.markNeedsUpdate();

	const third = registry.getTextureForSlot(
		texture,
		WEBGPU_TEXTURE_SLOT.BASE_COLOR
	);
	assert.equal(third, first);
	assert.equal(backend.textureWrites.length, 2);
}

function testHDRTextureUploadsAsRGBA16Float() {
	const backend = new FakeBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture(
		new Float32Array([2, 1, 0.5, 1]),
		1,
		1,
		"HDR"
	);

	registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);

	assert.equal(backend.createTextureCalls[0].format, TextureFormat.RGBA16Float);
	assert.equal(backend.textureWrites.length, 1);
	const uploaded = Uint8Array.from(backend.textureWrites[0].data);
	const view = new DataView(uploaded.buffer);
	nearlyEqual(float16BitsToFloat32(view.getUint16(0, true)), 2);
	nearlyEqual(float16BitsToFloat32(view.getUint16(2, true)), 1);
	nearlyEqual(float16BitsToFloat32(view.getUint16(4, true)), 0.5);
	nearlyEqual(float16BitsToFloat32(view.getUint16(6, true)), 1);
}

function testSamplerCacheInvalidatesWhenTextureSamplerStateChanges() {
	const backend = new FakeBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture(
		new Uint8ClampedArray([255, 255, 255, 255]),
		1,
		1,
		"sRGB"
	);
	const samplerA = registry.getSamplerForTexture(texture);
	assert.equal(backend.samplerDescs.length, 1);

	texture.wrapS = "Clamp";
	const samplerB = registry.getSamplerForTexture(texture);
	assert.equal(backend.samplerDescs.length, 2);
	assert.notEqual(samplerA, samplerB);

	const samplerC = registry.getSamplerForTexture(texture);
	assert.equal(backend.samplerDescs.length, 2);
	assert.equal(samplerC, samplerB);
}

async function run() {
	testMatrixPackingAndDepthRemap();
	testTransformComposition();
	testMaterialAdaptation();
	testFeatureGate();
	testRenderResourcesResolveComputeFacadeFromBackend();
	testRenderResourcesLeaveShaderRuntimeSubscriptionToBackend();
	testFrameExecutorConsumesComputeFacadeFromHost();
	await testSceneShaderCoverage();
	await testGBufferSlotShaderABI();
	testScenePipelineLimitConstantsMatchLayout();
	testDecalBatchLayoutHonorsStorageTextureLimit();
	testShadowDepthLayoutMatchesTransmittanceShaderBinding();
	await testParticleShaderDepthConsistency();
	await testWebGPUShaderConstantTokenInjection();
	testEnvironmentCollection();
	testEnvironmentCollectionWithCubeTextures();
	testEnvironmentCollectionUsesParentedProbeCaptureOrigin();
	testLightProbeDCAmbientFallbackWhenSHDisabled();
	testEnvironmentSynthesizesSHAmbientFromLightProbeWhenMissingFrameSH();
	testEnvironmentCollectsLocalizedLightProbesWithoutPollutingGlobalSH();
	testWebGPUShadowBiasAvoidsSlopeOffset();
	testWebGPUShadowPCSSParams();
	testWebGPUPointLightLimit();
	testWebGPUAreaLightCollection();
	testWebGPUClusteredSpotShadowBudgetFallback();
	testWebGPUFrameFeatureRegistryGatesVolumetricLighting();
	await testRenderResourcesUseCopyDstForUploads();
	await testWebGPUBlendMaterialsUseTransparentPipelineState();
	await testWebGPUTransmissionMaterialsUseTransparentPipelineState();
	await testWebGPUEarlyZPrepassOpaquePipelineHasDepthOnlyState();
	await testWebGPUEarlyZPrepassMaskPipelineUsesMaskDepthFragment();
	await testWebGPUEarlyZColorPipelineUsesReadOnlyDepthState();
	await testWebGPUEarlyZShaderMaterialDepthContract();
	await testWebGPUShaderMaterialDepthWriteFalseSkipsDepthPrepass();
	await testWebGPUShaderMaterialCustomUniformBufferBinding();
	await testWebGPUOITTransparentPipelineUsesDualTargets();
	await testWebGPUOITTransmissionMaterialsStayLegacyPipeline();
	await testWebGPUOITParticlePipelinesSplitAlphaAndAdditive();
	await testWebGPUEnvironmentCombinationsRegression();
	await testScopedSceneTargetModesUseDistinctBindings();
	await testSampleCountOverrideUsesSingleSampleCapturePipelines();
	await testReflectionProbeCaptureUsesCanvasAttachmentFormats();
	await testReflectionProbeCaptureUsesParentWorldPositionAsOrigin();
	await testParticleUVLayoutAndUniformBinding();
	await testFrameBindingReplacementDestroysOldBinding();
	await testWebGPUPrepareFrameTemporalStateModes();
	await testSceneFrameBindingLayoutMatchesFallbackEnvironmentContract();
	await testParticleShadowVolumeBufferUpdatesForDirectionalSlice();
	await testShadowAtlasSizeTracksShadowMapsWhenLightingDisabled();
	await testParticleBindingCacheEvictsStaleSystems();
	await testRenderResourcesDestroyCleansParticleAndGeometryResources();
	testWebGPUGeometryRegistryReleaseGeometryDestroysBuffers();
	testDynamicTextureReuploadOnVersionChange();
	testHDRTextureUploadsAsRGBA16Float();
	testSamplerCacheInvalidatesWhenTextureSamplerStateChanges();
	console.log("WebGPU bridge tests passed");
}

await run();
