import assert from "node:assert/strict";
import { RenderResources } from "../src/core/resources/RenderResources.ts";
import { WEBGPU_SCENE_SHADER } from "../src/shaders/webgpu/sceneShader.ts";
import {
	createWebGPUMaterialUniformData,
	packMatrix4ForWGSL,
	remapClipSpaceDepth,
	WEBGPU_FRAME_UNIFORM_FLOATS,
} from "../src/core/bridge/webgpu";
import { resolveFeatureState } from "../src/core/pipeline/FeatureResolver.ts";
import { BufferUsage } from "../src/core/ral/types.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { PhongMaterial } from "../src/materials/PhongMaterial.ts";
import { UnlitMaterial } from "../src/materials/UnlitMaterial.ts";
import { SimpleModel } from "../src/models/SimpleModel.ts";
import { getModelMatrix } from "../src/core/modelMatrix.ts";

globalThis.GPUShaderStage ??= {
	VERTEX: 1,
	FRAGMENT: 2,
};

class FakeDevice {
	constructor() {
		this.bufferDescs = [];
	}

	createBindGroupLayout(desc) {
		return { desc };
	}

	createPipelineLayout(desc) {
		this.lastPipelineLayout = { desc };
		return this.lastPipelineLayout;
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
	return {
		id: `${model.id}:${primitive.id}`,
		model,
		primitive,
		material: primitive.material,
		geometry: primitive.geometry,
		worldMatrix: getModelMatrix(model),
		normalMatrix: Matrix4.normalMatrix(getModelMatrix(model)),
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
			position: { x: 0, y: 0, z: 5 },
		},
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
	assert.ok(WEBGPU_SCENE_SHADER.includes("textureLoad(directionalShadowAtlas"));
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
	const resources = new RenderResources(renderer, backend);

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
	assert.equal(draw.frameBinding.desc.entries.length, 3);
	assert.equal(draw.modelBinding.desc.entries.length, 29);
	assert.equal(draw.pipeline.desc.layout, backend.device.lastPipelineLayout);
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

async function run() {
	testMatrixPackingAndDepthRemap();
	testMaterialAdaptation();
	testFeatureGate();
	testSceneShaderCoverage();
	await testRenderResourcesUseCopyDstForUploads();
	console.log("WebGPU bridge tests passed");
}

await run();
