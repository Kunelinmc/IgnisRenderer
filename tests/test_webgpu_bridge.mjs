import assert from "node:assert/strict";
import { WebGPURenderResources } from "../src/renderers/webgpu/WebGPURenderResources.ts";
import { WebGPUFrameExecutor } from "../src/renderers/webgpu/WebGPUFrameExecutor.ts";
import { getWebGPUParticleShader } from "../src/shaders/webgpu/particleShader.ts";
import { getWebGPUSceneShader } from "../src/shaders/webgpu/sceneShader.ts";
import { getWebGPUSkyboxShader } from "../src/shaders/webgpu/skyboxShader.ts";
import {
	loadClusteredLightingCullShaderComposite,
	loadPostProcessShaderPart,
} from "../src/shaders/webgpu/shaderSource.ts";
import {
	collectWebGPUEnvironment,
	collectWebGPULighting,
	createWebGPUMaterialUniformData,
	packMatrix4ForWGSL,
	remapClipSpaceDepth,
	WEBGPU_FRAME_UNIFORM_FLOATS,
} from "../src/renderers/webgpu/index.ts";
import { WebGPUReflectionProbeCapturePass } from "../src/renderers/webgpu/WebGPUReflectionProbeCapturePass.ts";
import { resolveFeatureState } from "../src/pipeline/FeatureResolver.ts";
import { BufferUsage } from "../src/renderers/types.ts";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { DirectionalLight } from "../src/lights/DirectionalLight.ts";
import { PointLight } from "../src/lights/PointLight.ts";
import { ReflectionProbe } from "../src/lights/ReflectionProbe.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { SH } from "../src/maths/SH.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { PhongMaterial } from "../src/materials/PhongMaterial.ts";
import { AlphaMode } from "../src/materials/Material.ts";
import { Texture } from "../src/core/Texture.ts";
import { CubeTexture } from "../src/core/CubeTexture.ts";
import { UnlitMaterial } from "../src/materials/UnlitMaterial.ts";
import { MeshAsset } from "../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../src/meshes/MeshInstance.ts";
import { ShadowMap } from "../src/lights/ShadowMapping.ts";
import { PARTICLE_TRANSIENT_BATCHES_KEY } from "../src/pipeline/types.ts";
import { ParticleBlendMode } from "../src/particles/types.ts";
import { WEBGPU_PARTICLE_VERTEX_LAYOUTS } from "../src/renderers/webgpu/particleLayout.ts";
import {
	WEBGPU_MAX_DIRECTIONAL_LIGHTS,
	WEBGPU_MAX_POINT_LIGHTS,
	WEBGPU_MAX_REFLECTION_PROBES,
	WEBGPU_MAX_SPOT_LIGHTS,
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_SHADOW_ATLAS_COLUMNS,
	WEBGPU_TEXTURE_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT,
} from "../src/renderers/webgpu/constants.ts";
import { WebGPUGeometryRegistry } from "../src/renderers/webgpu/WebGPUGeometryRegistry.ts";
import { WebGPUTextureRegistry } from "../src/renderers/webgpu/WebGPUTextureRegistry.ts";

globalThis.GPUShaderStage ??= {
	VERTEX: 1,
	FRAGMENT: 2,
	COMPUTE: 4,
};

import {
	FakeCommandEncoder as FakeRenderEncoder,
	FakeWebGPUBackend as FakeBackend,
} from "./helpers/test_fakes.mjs";

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
		skybox: null,
		meshInstances: [packet.meshInstance],
		shadowMaps: new Map(),
		opaquePackets: [packet],
		transparentPackets: [],
		shadowCasterPackets: [],
		shadowTransmitterPackets: [],
		reflectivePackets: [],
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
	assert.equal(pbrData.textureSlots.length, 14);

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
}

function testFeatureGate() {
	const featureState = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableSH: true,
			enableShadows: true,
			enableReflection: true,
			enableSkybox: true,
			enableSSAO: true,
			enableTAA: true,
			enableSSR: true,
			enableVolumetric: true,
			enableBloom: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			skybox: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: false,
		},
		"webgpu"
	);

	assert.equal(featureState.enableLighting, true);
	assert.equal(featureState.enableGamma, true);
	assert.equal(featureState.enableSH, false);
	assert.equal(featureState.enableShadows, true);
	assert.equal(featureState.enableReflection, false);
	assert.equal(featureState.enableSkybox, false);
	assert.equal(featureState.enableSSAO, false);
	assert.equal(featureState.enableTAA, false);
	assert.equal(featureState.enableSSR, false);
	assert.equal(featureState.enableVolumetric, false);
	assert.equal(featureState.enableBloom, false);
	assert.ok(featureState.ssaoOptions);
	assert.ok(featureState.taaOptions);
	assert.ok(featureState.ssrOptions);
	assert.ok(featureState.volumetricOptions);
	assert.ok(featureState.bloomOptions);
	assert.equal(featureState.ssaoOptions.downsample, 2);
	assert.equal(featureState.ssaoOptions.blurRadius, 2);
	assert.equal(featureState.ssaoOptions.blurSharpness, 8);
	assert.equal(featureState.taaOptions.jitterScale, 1);
	assert.equal(featureState.taaOptions.historyWeight, 0.9);
	assert.equal(featureState.taaOptions.disocclusionDepthThreshold, 0.02);
	assert.equal(featureState.taaOptions.motionFactor, 80);
	assert.equal(featureState.taaOptions.varianceClampGamma, 1);
	assert.equal(featureState.taaOptions.sharpen, 0.1);
	assert.equal(featureState.ssrOptions.downsample, 2);
	assert.equal(featureState.ssrOptions.binarySearchSteps, 6);
	assert.equal(featureState.ssrOptions.edgeFade, 0.12);
	assert.equal(featureState.ssrOptions.maxRoughness, 0.85);
	assert.equal(featureState.bloomOptions.threshold, 1);
	assert.equal(featureState.bloomOptions.softKnee, 0.5);
	assert.equal(featureState.bloomOptions.intensity, 0.8);
	assert.equal(featureState.bloomOptions.radius, 1);
	assert.ok(featureState.warnings.length >= 8);
}

async function testSceneShaderCoverage() {
	const WEBGPU_SCENE_SHADER = await getWebGPUSceneShader();
	const WEBGPU_SKYBOX_SHADER = await getWebGPUSkyboxShader();

	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"let pointCount = u32(frame.lightCounts.y + 0.5);"
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			"frame.pointLights[i].positionRange.xyz - input.worldPosition"
		)
	);
	assert.ok(WEBGPU_SCENE_SHADER.includes("sampleDirectionalShadowVisibility"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("textureLoad(shadowAtlas"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("texture_depth_2d"));
	assert.ok(!WEBGPU_SCENE_SHADER.includes("decodePackedShadowDepth"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("let shadowNormal = normal;"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("let pbrShadowNormal = pbrNormal;"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("shadowData.paramsC.x"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("calculateIrradianceFromSH"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("sampleEnvironmentSpecular"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(2)"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(2) @binding(0)"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("if (isClusteredLightingEnabled())"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("decodeClusteredLightRef"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@location(4) gMotionDepth"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("frame.prevViewProjection"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("model.prevModelMatrix"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(1) @binding(30)"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("applySkinning("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("struct SceneFragmentOITOutput"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("fn resolveOITWeight("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("fn buildSceneOITOutput("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@fragment\nfn fsMainOIT("));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@builtin(vertex_index)"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@location(8) weights1"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("@group(0) @binding(1)"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("prevViewProjection"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("taaJitterCurrentPrev"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("atan2(direction.x, direction.z)"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("frame.environmentOptionsB.z < 0.5"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("frame.options.w > 0.5"));
}

async function testParticleShaderDepthConsistency() {
	const WEBGPU_PARTICLE_SHADER = await getWebGPUParticleShader();

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
	const WEBGPU_SCENE_SHADER = await getWebGPUSceneShader();
	const WEBGPU_SKYBOX_SHADER = await getWebGPUSkyboxShader();
	const WEBGPU_PARTICLE_SHADER = await getWebGPUParticleShader();
	const WEBGPU_SSR_SHADER = await loadPostProcessShaderPart("ssr");
	const WEBGPU_CLUSTERED_CULL_SHADER =
		(await loadClusteredLightingCullShaderComposite()).code;

	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`directionalLights: array<DirectionalLightData, ${WEBGPU_MAX_DIRECTIONAL_LIGHTS}>`
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`pointLights: array<PointLightData, ${WEBGPU_MAX_POINT_LIGHTS}>`
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`spotLights: array<SpotLightData, ${WEBGPU_MAX_SPOT_LIGHTS}>`
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`shAmbientCoeffs: array<vec4<f32>, ${WEBGPU_SH_COEFFICIENT_COUNT}>`
		)
	);
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			`reflectionProbes: array<ReflectionProbeData, ${WEBGPU_MAX_REFLECTION_PROBES}>`
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
		WEBGPU_SKYBOX_SHADER.includes(
			`pointLights: array<PointLightData, ${WEBGPU_MAX_POINT_LIGHTS}>`
		)
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes(
			`directionalLights: array<vec4<f32>, ${WEBGPU_MAX_DIRECTIONAL_LIGHTS * 2}>`
		)
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes(
			`pointLights: array<vec4<f32>, ${WEBGPU_MAX_POINT_LIGHTS * 2}>`
		)
	);
	assert.ok(
		WEBGPU_PARTICLE_SHADER.includes(
			`spotLights: array<vec4<f32>, ${WEBGPU_MAX_SPOT_LIGHTS * 3}>`
		)
	);
	assert.ok(
		WEBGPU_SSR_SHADER.includes(
			`pointLights: array<PointLightData, ${WEBGPU_MAX_POINT_LIGHTS}>`
		)
	);
	assert.ok(
		WEBGPU_CLUSTERED_CULL_SHADER.includes(
			`pointLights: array<PointLightData, ${WEBGPU_MAX_POINT_LIGHTS}>`
		)
	);
	assert.ok(!WEBGPU_SCENE_SHADER.includes("__WEBGPU_"));
	assert.ok(!WEBGPU_SKYBOX_SHADER.includes("__WEBGPU_"));
	assert.ok(!WEBGPU_PARTICLE_SHADER.includes("__WEBGPU_"));
	assert.ok(!WEBGPU_SSR_SHADER.includes("__WEBGPU_"));
	assert.ok(!WEBGPU_CLUSTERED_CULL_SHADER.includes("__WEBGPU_"));
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

function testEnvironmentCollection() {
	const skybox = createTinyTexture(1);
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
			skybox,
			lights: [probeA, probeB],
		},
		true,
		sh
	);
	assert.equal(prioritized.skyboxTexture, skybox);
	assert.ok(prioritized.envSpecularTexture);
	assert.notEqual(prioritized.envSpecularTexture, skybox);
	assert.equal(prioritized.envSpecularMaxMipLevel, 2);
	assert.equal(prioritized.reflectionProbeCount, 2);
	assert.equal(prioritized.reflectionProbes.length, 2);
	assert.equal(prioritized.hasSHAmbient, true);
	assert.ok(prioritized.brdfLUTTexture);

	const fallback = collectWebGPUEnvironment(
		{
			skybox: null,
			lights: [probeA, probeB],
		},
		true,
		sh
	);
	assert.ok(fallback.skyboxTexture);
	assert.ok(fallback.envSpecularTexture);
	assert.equal(fallback.reflectionProbeCount, 2);

	const failedSkybox = createTinyTexture(1);
	failedSkybox.markAsLoadErrorFallback();
	const fallbackFromFailedSkybox = collectWebGPUEnvironment(
		{
			skybox: failedSkybox,
			lights: [probeA],
		},
		true,
		sh
	);
	assert.ok(fallbackFromFailedSkybox.skyboxTexture);
	assert.ok(fallbackFromFailedSkybox.envSpecularTexture);
	assert.equal(fallbackFromFailedSkybox.reflectionProbeCount, 1);
	assert.ok(
		fallbackFromFailedSkybox.warnings.some(
			(warning) => warning.key === "webgpu-skybox-load-error-fallback"
		)
	);

	const failedOnlySkybox = collectWebGPUEnvironment(
		{
			skybox: failedSkybox,
			lights: [],
		},
		true,
		sh
	);
	assert.equal(failedOnlySkybox.skyboxTexture, null);
	assert.equal(failedOnlySkybox.envSpecularTexture, null);
	assert.equal(failedOnlySkybox.reflectionProbeCount, 0);

	const disabledFallback = collectWebGPUEnvironment(
		{
			skybox,
			lights: [],
			allowSkyboxSpecularFallback: false,
		},
		true,
		sh
	);
	assert.equal(disabledFallback.skyboxTexture, skybox);
	assert.equal(disabledFallback.envSpecularTexture, null);
	assert.equal(disabledFallback.reflectionProbeCount, 0);
	assert.equal(disabledFallback.brdfLUTTexture, null);
	assert.equal(disabledFallback.envSpecularMaxMipLevel, 0);
}

function testEnvironmentCollectionWithCubeTextures() {
	const skybox = createTinyCubeTexture(2, 0.5);
	const probeMap = createTinyCubeTexture(3, 0.75);
	const probe = new ReflectionProbe({
		shape: "sphere",
		prefilteredMap: probeMap,
	});

	const state = collectWebGPUEnvironment(
		{
			skybox,
			lights: [probe],
		},
		false,
		null
	);
	assert.ok(state.skyboxTexture);
	assert.ok(state.envSpecularTexture);
	assert.notEqual(state.skyboxTexture, skybox);
	assert.notEqual(state.envSpecularTexture, probeMap);
	assert.equal(state.skyboxTexture.width, 4);
	assert.equal(state.skyboxTexture.height, 2);
	assert.equal(state.envSpecularTexture.width, 4);
	assert.equal(state.envSpecularTexture.height, 2);
	assert.equal(state.reflectionProbeCount, 1);
	assert.equal(state.envSpecularMaxMipLevel, 2);
}

function testLightProbeDCAmbientFallbackWhenSHDisabled() {
	const sh = SH.empty();
	sh[0] = { r: 120, g: 60, b: 30 };
	const probe = new LightProbe(sh, 0.75);
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
	const probe = new LightProbe(sh, 0.5);
	const state = collectWebGPUEnvironment(
		{
			skybox: null,
			lights: [probe],
		},
		true,
		null
	);
	assert.equal(state.enableSH, true);
	assert.equal(state.hasSHAmbient, true);
	assert.ok(state.shAmbientCoeffs);
	assert.ok(Math.abs(state.shAmbientCoeffs[0].r - 40) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[0].g - 20) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[0].b - 10) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[5].r - 1.5) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[5].g - 1.0) < 1e-6);
	assert.ok(Math.abs(state.shAmbientCoeffs[5].b - 0.5) < 1e-6);
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
}

function testWebGPUPointLightLimit() {
	const withinLimit = Array.from(
		{ length: WEBGPU_MAX_POINT_LIGHTS },
		() => new PointLight()
	);
	const withinState = collectWebGPULighting(withinLimit, true, false);
	assert.equal(withinState.pointLights.length, WEBGPU_MAX_POINT_LIGHTS);
	assert.ok(
		!withinState.warnings.some(
			(warning) => warning.key === "webgpu-point-limit"
		)
	);

	const overLimit = Array.from(
		{ length: WEBGPU_MAX_POINT_LIGHTS + 2 },
		() => new PointLight()
	);
	const overState = collectWebGPULighting(overLimit, true, false);
	assert.equal(overState.pointLights.length, WEBGPU_MAX_POINT_LIGHTS);
	assert.ok(
		overState.warnings.some((warning) => warning.key === "webgpu-point-limit")
	);
}

function testRenderResourcesRequestsComputeFacadeFromBackend() {
	const backend = new FakeBackend();
	const renderer = { logger: { warn() {} } };
	const resources = new WebGPURenderResources(renderer, backend);

	assert.equal(backend.getComputeFacadeCalls, 1);
	assert.equal(
		typeof resources._clusteredLighting._compute.createComputePipeline,
		"function"
	);

	resources.destroy();
}

function testFrameExecutorRequestsComputeFacadeFromBackend() {
	const backend = new FakeBackend();
	const resourcesStub = { sceneFrameLayout: null };
	const executor = new WebGPUFrameExecutor(backend, resourcesStub);

	assert.equal(backend.getComputeFacadeCalls, 1);
	assert.equal(
		typeof executor._postRuntime._compute.createComputePipeline,
		"function"
	);
}

async function testRenderResourcesUseCopyDstForUploads() {
	const backend = new FakeBackend();
	const renderer = {
		logger: { warn() {} },
	};
	const model = createModel([
		new PBRMaterial({
			albedo: { r: 255, g: 255, b: 255 },
		}),
	]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(renderer, backend);

	await resources.init();
	resources.prepareFrame(
		frame,
		resolveFeatureState(
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				skybox: false,
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
		)
	);

	const draw = await resources.getDrawResources(packet);

	assert.ok(draw);
	const firstDraw = draw[0];
	assert.ok(firstDraw);
	assert.equal(firstDraw.frameBinding.desc.entries.length, 5);
	assert.equal(firstDraw.modelBinding.desc.entries.length, 34);
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
	assert.ok(modelBindingIndices.includes(30));
	assert.ok(modelBindingIndices.includes(31));
	assert.ok(modelBindingIndices.includes(32));
	assert.ok(modelBindingIndices.includes(33));
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
	assert.ok(
		backend.bufferDescs.some(
			(desc) => desc.size === WEBGPU_FRAME_UNIFORM_FLOATS * 4
		)
	);
}

async function testWebGPUBlendMaterialsUseTransparentPipelineState() {
	const backend = new FakeBackend();
	const renderer = { logger: { warn() {} } };
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
	const resources = new WebGPURenderResources(renderer, backend);

	await resources.init();
	resources.prepareFrame(
		frame,
		resolveFeatureState(
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				skybox: false,
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
		)
	);

	const draw = await resources.getDrawResources(packet);
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
	const renderer = { logger: { warn() {} } };
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
	const resources = new WebGPURenderResources(renderer, backend);

	await resources.init();
	resources.prepareFrame(
		frame,
		resolveFeatureState(
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				skybox: false,
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
		)
	);

	const draw = await resources.getDrawResources(packet);
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

async function testWebGPUOITTransparentPipelineUsesDualTargets() {
	const backend = new FakeBackend();
	const renderer = { logger: { warn() {} } };
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
	const resources = new WebGPURenderResources(renderer, backend);

	await resources.init();
	resources.prepareFrame(
		frame,
		resolveFeatureState(
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
				skybox: false,
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
		)
	);

	const draw = await resources.getDrawResources(packet, {
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
	const renderer = { logger: { warn() {} } };
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
	const resources = new WebGPURenderResources(renderer, backend);

	await resources.init();
	resources.prepareFrame(
		frame,
		resolveFeatureState(
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
				skybox: false,
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
		)
	);

	const draw = await resources.getDrawResources(packet, {
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
	const renderer = { logger: { warn() {} } };
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(renderer, backend);
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
			skybox: false,
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
	resources.prepareFrame(frame, features);

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
	const renderer = { logger: { warn() {} } };
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const baseScene = createFrame(packet);
	const resources = new WebGPURenderResources(renderer, backend);
	await resources.init();

	const caps = {
		sh: true,
		shadows: true,
		reflection: false,
		skybox: true,
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
			skybox: createTinyTexture(1),
			lights: [probe],
			enableSH: true,
			expectSkybox: true,
		},
		{
			skybox: null,
			lights: [probe],
			enableSH: true,
			expectSkybox: true,
		},
		{
			skybox: null,
			lights: [],
			enableSH: false,
			expectSkybox: false,
		},
	];

	for (const scenario of cases) {
		const scene = {
			...baseScene,
			skybox: scenario.skybox,
			lights: scenario.lights,
		};
		const features = resolveFeatureState(
			{
				enableLighting: true,
				enableGamma: true,
				enableSH: scenario.enableSH,
				enableShadows: true,
				enableSkybox: true,
			},
			caps,
			"webgpu"
		);
		resources.prepareFrame({
			camera: scene.camera,
			attachments: { width: 16, height: 16 },
			features,
			shadowMaps: scene.shadowMaps,
			scene,
			shCoeffs: SH.empty(),
			shAmbientCoeffs: scenario.enableSH ? shAmbient : SH.empty(),
			worldMatrix: Matrix4.identity(),
			transient: new Map(),
		});

		const skyboxResources = await resources.getSkyboxResources();
		assert.equal(!!skyboxResources, scenario.expectSkybox);
		const draw = await resources.getDrawResources(packet);
		assert.ok(draw);
	}
}

async function testReflectionProbeCaptureUsesCanvasAttachmentFormats() {
	const backend = new FakeBackend();
	backend.canvasFormat = "bgra8unorm";
	const renderer = { logger: { warn() {} } };
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const preparedScene = {
		...createFrame(packet),
		particleSystems: [],
		hasActiveAnimations: false,
		allowSkyboxSpecularFallback: false,
		spatialIndex: null,
	};
	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableClusteredLighting: true,
			enableSkybox: false,
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
			skybox: false,
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
		setSceneTargetMode() {},
		prepareFrame() {},
		async buildClusteredLighting() {},
		async getSkyboxResources() {
			return null;
		},
		async getDrawResources() {
			return null;
		},
		async renderParticles() {
			return 0;
		},
	};
	const probe = new ReflectionProbe({
		includeMeshes: false,
		includeSkybox: false,
		includeTransparent: false,
		includeParticles: false,
		includeShadows: false,
	});
	const capturePass = new WebGPUReflectionProbeCapturePass(backend, resources);
	const result = await capturePass.captureFace({
		frameContext,
		probe,
		faceIndex: 0,
		faceSize: 1,
		includeSkybox: false,
		includeTransparent: false,
		includeParticles: false,
		includeShadows: false,
	});

	assert.ok(result);
	assert.equal(result.length, 4);
	assert.equal(backend.createTextureCalls.length >= 2, true);
	assert.equal(backend.createTextureCalls[0].format, backend.canvasFormat);
	assert.equal(backend.createTextureCalls[1].format, backend.canvasDepthFormat);
}

async function testParticleUVLayoutAndUniformBinding() {
	const backend = new FakeBackend();
	const renderer = { logger: { warn() {} } };
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(renderer, backend);

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
			skybox: false,
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

	resources.prepareFrame(frame, features);

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
	const renderer = { logger: { warn() {} } };
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(renderer, backend);
	await resources.init();

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableSkybox: true,
		},
		{
			sh: false,
			shadows: false,
			reflection: false,
			skybox: true,
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

	resources.prepareFrame({ ...frame, skybox: createTinyTexture(1) }, features);
	const firstSkybox = await resources.getSkyboxResources();
	assert.ok(firstSkybox);
	const firstBinding = firstSkybox.frameBinding;
	assert.equal(firstBinding.destroyed, false);

	resources.prepareFrame({ ...frame, skybox: createTinyTexture(1) }, features);
	assert.equal(firstBinding.destroyed, true);

	const secondSkybox = await resources.getSkyboxResources();
	assert.ok(secondSkybox);
	assert.notEqual(secondSkybox.frameBinding, firstBinding);
}

async function testShadowAtlasSizeTracksShadowMapsWhenLightingDisabled() {
	const backend = new FakeBackend();
	const renderer = { logger: { warn() {} } };
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const light = new DirectionalLight();
	const shadowMap = new ShadowMap(1024);
	shadowMap.viewProjectionMatrix = Matrix4.identity();
	frame.lights = [light];
	frame.shadowMaps = new Map([[light, shadowMap]]);
	const resources = new WebGPURenderResources(renderer, backend);
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
			skybox: false,
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
	resources.prepareFrame(frame, features);
	const frameBinding = resources.getFrameBinding();
	const shadowAtlasEntry = frameBinding.desc.entries.find(
		(entry) => entry.binding === 1
	);
	assert.ok(shadowAtlasEntry?.resource);
	assert.equal(
		shadowAtlasEntry.resource.width,
		1024 * WEBGPU_SHADOW_ATLAS_COLUMNS
	);
}

async function testParticleBindingCacheEvictsStaleSystems() {
	const backend = new FakeBackend();
	const renderer = { logger: { warn() {} } };
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(renderer, backend);
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
			skybox: false,
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
	resources.prepareFrame(frame, features);

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
		resources.prepareFrame(frame, features);
	}

	assert.equal(particleBinding.destroyed, true);
	assert.equal(uvBuffer.destroyed, true);
}

async function testRenderResourcesDestroyCleansParticleAndGeometryResources() {
	const backend = new FakeBackend();
	const renderer = { logger: { warn() {} } };
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(renderer, backend);
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
			skybox: false,
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
	resources.prepareFrame(frame, features);
	const draw = await resources.getDrawResources(packet);
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
	testRenderResourcesRequestsComputeFacadeFromBackend();
	testFrameExecutorRequestsComputeFacadeFromBackend();
	await testSceneShaderCoverage();
	await testParticleShaderDepthConsistency();
	await testWebGPUShaderConstantTokenInjection();
	testEnvironmentCollection();
	testEnvironmentCollectionWithCubeTextures();
	testLightProbeDCAmbientFallbackWhenSHDisabled();
	testEnvironmentSynthesizesSHAmbientFromLightProbeWhenMissingFrameSH();
	testWebGPUShadowBiasAvoidsSlopeOffset();
	testWebGPUPointLightLimit();
	await testRenderResourcesUseCopyDstForUploads();
	await testWebGPUBlendMaterialsUseTransparentPipelineState();
	await testWebGPUTransmissionMaterialsUseTransparentPipelineState();
	await testWebGPUOITTransparentPipelineUsesDualTargets();
	await testWebGPUOITTransmissionMaterialsStayLegacyPipeline();
	await testWebGPUOITParticlePipelinesSplitAlphaAndAdditive();
	await testWebGPUEnvironmentCombinationsRegression();
	await testReflectionProbeCaptureUsesCanvasAttachmentFormats();
	await testParticleUVLayoutAndUniformBinding();
	await testFrameBindingReplacementDestroysOldBinding();
	await testShadowAtlasSizeTracksShadowMapsWhenLightingDisabled();
	await testParticleBindingCacheEvictsStaleSystems();
	await testRenderResourcesDestroyCleansParticleAndGeometryResources();
	testWebGPUGeometryRegistryReleaseGeometryDestroysBuffers();
	testDynamicTextureReuploadOnVersionChange();
	testSamplerCacheInvalidatesWhenTextureSamplerStateChanges();
	console.log("WebGPU bridge tests passed");
}

await run();
