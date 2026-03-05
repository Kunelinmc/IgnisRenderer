import assert from "node:assert/strict";
import { WebGPURenderResources } from "../src/core/backend/webgpu/WebGPURenderResources.ts";
import { WEBGPU_SCENE_SHADER } from "../src/shaders/webgpu/sceneShader.ts";
import { WEBGPU_SKYBOX_SHADER } from "../src/shaders/webgpu/skyboxShader.ts";
import {
	collectWebGPUEnvironment,
	collectWebGPULighting,
	createWebGPUMaterialUniformData,
	packMatrix4ForWGSL,
	remapClipSpaceDepth,
	WEBGPU_FRAME_UNIFORM_FLOATS,
} from "../src/core/backend/webgpu/index.ts";
import { resolveFeatureState } from "../src/core/pipeline/FeatureResolver.ts";
import { BufferUsage } from "../src/core/backend/types.ts";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { SH } from "../src/maths/SH.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { PhongMaterial } from "../src/materials/PhongMaterial.ts";
import { Texture } from "../src/core/Texture.ts";
import { UnlitMaterial } from "../src/materials/UnlitMaterial.ts";
import { SimpleModel } from "../src/models/SimpleModel.ts";

globalThis.GPUShaderStage ??= {
	VERTEX: 1,
	FRAGMENT: 2,
};

class FakeDevice {
	constructor() {
		this.bufferDescs = [];
		this.pipelineLayouts = [];
	}

	createBindGroupLayout(desc) {
		return { desc };
	}

	createPipelineLayout(desc) {
		const layout = { desc };
		this.pipelineLayouts.push(layout);
		this.lastPipelineLayout = layout;
		return layout;
	}
}

class FakeBackend {
	constructor() {
		this.type = "webgpu";
		this.canvasFormat = "rgba8unorm";
		this.bufferDescs = [];
		this.device = new FakeDevice();
	}

	createBuffer(desc) {
		this.bufferDescs.push(desc);
		return {
			size: desc.size,
			desc,
			destroy() {},
		};
	}

	createTexture(desc) {
		return {
			width: desc.width,
			height: desc.height,
			desc,
			destroy() {},
		};
	}

	createSampler(desc) {
		return { label: desc.label, desc };
	}

	async createShaderModule(desc) {
		return { label: desc.label, desc };
	}

	createPipeline(desc) {
		return { label: desc.label, desc };
	}

	createBindingGroup(desc) {
		return { label: desc.label, desc };
	}

	writeBuffer(buffer, data) {
		buffer.lastWrite = data;
	}

	writeTexture(texture, data, layout, size) {
		texture.lastWrite = { data, layout, size };
	}
}

function createModel(materials) {
	return SimpleModel.fromFaces(
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
}

function createPacket(model) {
	const primitive = model.primitives[0];
	const worldMatrix = Matrix4.fromTransform(model.transform);
	return {
		id: `${model.id}:${primitive.id}`,
		model,
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
	return {
		sceneBounds: packet.model.boundingSphere,
		lights: [],
		camera: {
			viewProjectionMatrix: Matrix4.identity(),
			viewMatrix: Matrix4.identity(),
			position: { x: 0, y: 0, z: 5 },
			fov: 60,
			aspectRatio: 1,
			type: "perspective",
		},
		skybox: null,
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

	assert.deepEqual(Matrix4.fromTransform(transform).elements, expected.elements);
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
			enableVolumetric: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			skybox: false,
			ssao: false,
			volumetric: false,
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
	assert.equal(featureState.enableVolumetric, false);
	assert.ok(featureState.warnings.length >= 5);
}

function testSceneShaderCoverage() {
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
	assert.ok(WEBGPU_SCENE_SHADER.includes("calculateIrradianceFromSH"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("sampleEnvironmentSpecular"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(2)"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("@group(0) @binding(1)"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("atan2(direction.x, direction.z)"));
}

function createTinyTexture(mips = 1) {
	const texture = new Texture(
		new Float32Array([1, 1, 1, 1]),
		1,
		1,
		"HDR"
	);
	texture.mipmaps = Array.from({ length: mips }, () => new Float32Array([1, 1, 1, 1]));
	return texture;
}

function testEnvironmentCollection() {
	const skybox = createTinyTexture(1);
	const probeMap = createTinyTexture(3);
	const sh = SH.empty();
	sh[0] = { r: 10, g: 10, b: 10 };
	const probeA = new LightProbe(SH.empty(), 1.0, probeMap);
	const probeB = new LightProbe(SH.empty(), 1.0, createTinyTexture(2));

	const prioritized = collectWebGPUEnvironment(
		{
			skybox,
			lights: [probeA, probeB],
		},
		true,
		sh
	);
	assert.equal(prioritized.skyboxTexture, skybox);
	assert.equal(prioritized.envSpecularTexture, probeMap);
	assert.equal(prioritized.envSpecularMaxMipLevel, 2);
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
	assert.equal(fallback.skyboxTexture, probeMap);
	assert.equal(fallback.envSpecularTexture, probeMap);
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

async function testRenderResourcesUseCopyDstForUploads() {
	const backend = new FakeBackend();
	const renderer = {
		warnOnce() {},
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
				volumetric: false,
			},
			"webgpu"
		)
	);

	const draw = await resources.getDrawResources(packet);

	assert.ok(draw);
	assert.equal(draw.frameBinding.desc.entries.length, 4);
	assert.equal(draw.modelBinding.desc.entries.length, 29);
	assert.equal(draw.pipeline.desc.layout, backend.device.pipelineLayouts[0]);
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

async function testWebGPUEnvironmentCombinationsRegression() {
	const backend = new FakeBackend();
	const renderer = { warnOnce() {} };
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
		volumetric: false,
	};

	const shAmbient = SH.empty();
	shAmbient[0] = { r: 12, g: 12, b: 12 };
	const probeMap = createTinyTexture(2);
	const probe = new LightProbe(SH.empty(), 1.0, probeMap);

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

async function run() {
	testMatrixPackingAndDepthRemap();
	testTransformComposition();
	testMaterialAdaptation();
	testFeatureGate();
	testSceneShaderCoverage();
	testEnvironmentCollection();
	testLightProbeDCAmbientFallbackWhenSHDisabled();
	await testRenderResourcesUseCopyDstForUploads();
	await testWebGPUEnvironmentCombinationsRegression();
	console.log("WebGPU bridge tests passed");
}

await run();
