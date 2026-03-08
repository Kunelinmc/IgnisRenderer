import assert from "node:assert/strict";
import { WebGPURenderResources } from "../src/renderers/webgpu/WebGPURenderResources.ts";
import { getWebGPUParticleShader } from "../src/shaders/webgpu/particleShader.ts";
import { getWebGPUSceneShader } from "../src/shaders/webgpu/sceneShader.ts";
import { getWebGPUSkyboxShader } from "../src/shaders/webgpu/skyboxShader.ts";
import {
	collectWebGPUEnvironment,
	collectWebGPULighting,
	createWebGPUMaterialUniformData,
	packMatrix4ForWGSL,
	remapClipSpaceDepth,
	WEBGPU_FRAME_UNIFORM_FLOATS,
} from "../src/renderers/webgpu/index.ts";
import { resolveFeatureState } from "../src/pipeline/FeatureResolver.ts";
import { BufferUsage } from "../src/renderers/types.ts";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { SH } from "../src/maths/SH.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { PhongMaterial } from "../src/materials/PhongMaterial.ts";
import { Texture } from "../src/core/Texture.ts";
import { UnlitMaterial } from "../src/materials/UnlitMaterial.ts";
import { MeshAsset } from "../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../src/meshes/MeshInstance.ts";
import { PARTICLE_TRANSIENT_BATCHES_KEY } from "../src/pipeline/types.ts";
import { ParticleBlendMode } from "../src/particles/types.ts";
import { WEBGPU_PARTICLE_VERTEX_LAYOUTS } from "../src/renderers/webgpu/particleLayout.ts";

globalThis.GPUShaderStage ??= {
	VERTEX: 1,
	FRAGMENT: 2,
};

class FakeDevice {
	constructor() {
		this.bufferDescs = [];
		this.pipelineLayouts = [];
		this.bindGroupLayouts = [];
	}

	createBindGroupLayout(desc) {
		const layout = { desc };
		this.bindGroupLayouts.push(layout);
		return layout;
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
		this.pipelines = [];
		this.bindingGroups = [];
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
		const pipeline = { label: desc.label, desc };
		this.pipelines.push(pipeline);
		return pipeline;
	}

	createBindingGroup(desc) {
		const bindingGroup = { label: desc.label, desc };
		this.bindingGroups.push(bindingGroup);
		return bindingGroup;
	}

	writeBuffer(buffer, data) {
		buffer.lastWrite = data;
	}

	writeTexture(texture, data, layout, size) {
		texture.lastWrite = { data, layout, size };
	}
}

class FakeRenderEncoder {
	beginRenderPass() {}
	setBindingGroup() {}
	setVertexBuffer() {}
	setPipeline() {}
	draw() {}
	endRenderPass() {}
}

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
	assert.ok(featureState.ssaoOptions);
	assert.ok(featureState.taaOptions);
	assert.ok(featureState.ssrOptions);
	assert.ok(featureState.volumetricOptions);
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
	assert.ok(featureState.warnings.length >= 7);
}

async function testSceneShaderCoverage() {
	const WEBGPU_SCENE_SHADER = await getWebGPUSceneShader()
	const WEBGPU_SKYBOX_SHADER = await getWebGPUSkyboxShader()

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
	assert.ok(WEBGPU_SCENE_SHADER.includes("calculateIrradianceFromSH"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("sampleEnvironmentSpecular"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@group(0) @binding(2)"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("@location(4) gMotionDepth"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("frame.prevViewProjection"));
	assert.ok(WEBGPU_SCENE_SHADER.includes("model.prevModelMatrix"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("@group(0) @binding(1)"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("prevViewProjection"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("taaJitterCurrentPrev"));
	assert.ok(WEBGPU_SKYBOX_SHADER.includes("atan2(direction.x, direction.z)"));
}

async function testParticleShaderDepthConsistency() {
	const WEBGPU_PARTICLE_SHADER = await getWebGPUParticleShader()

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
		WEBGPU_PARTICLE_SHADER.includes(
			"let currentDepth = ndc.z * 0.5 + 0.5;"
		)
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
		WEBGPU_PARTICLE_SHADER.includes("rotatedUV + particleUVTransform.transformA.zw")
	);
}

function createTinyTexture(mips = 1) {
	const texture = new Texture(new Float32Array([1, 1, 1, 1]), 1, 1, "HDR");
	texture.mipmaps = Array.from(
		{ length: mips },
		() => new Float32Array([1, 1, 1, 1])
	);
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
				taa: false,
				ssr: false,
				volumetric: false,
			},
			"webgpu"
		)
	);

	const draw = await resources.getDrawResources(packet);

	assert.ok(draw);
	const firstDraw = draw[0];
	assert.ok(firstDraw);
	assert.equal(firstDraw.frameBinding.desc.entries.length, 4);
	assert.equal(firstDraw.modelBinding.desc.entries.length, 29);
	assert.equal(firstDraw.pipeline.desc.layout, backend.device.pipelineLayouts[0]);
	assert.equal(firstDraw.pipeline.desc.fragment.targets.length, 5);
	assert.deepEqual(
		firstDraw.pipeline.desc.fragment.targets.map((target) => target.format),
		["rgba16float", "rgba8unorm", "rgba16float", "rgba16float", "rgba16float"]
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
		taa: false,
		ssr: false,
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

async function testParticleUVLayoutAndUniformBinding() {
	const backend = new FakeBackend();
	const renderer = { warnOnce() {} };
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
		},
		"webgpu"
	);

	resources.prepareFrame(frame, features);

	const texture = new Texture(new Uint8Array([255, 255, 255, 255]), 1, 1, "sRGB");
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
		{ color: renderTarget, depth: renderTarget },
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
	assert.deepEqual(particlePipeline.desc.vertex.buffers, WEBGPU_PARTICLE_VERTEX_LAYOUTS);

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

async function run() {
	testMatrixPackingAndDepthRemap();
	testTransformComposition();
	testMaterialAdaptation();
	testFeatureGate();
	await testSceneShaderCoverage();
	await testParticleShaderDepthConsistency();
	testEnvironmentCollection();
	testLightProbeDCAmbientFallbackWhenSHDisabled();
	await testRenderResourcesUseCopyDstForUploads();
	await testWebGPUEnvironmentCombinationsRegression();
	await testParticleUVLayoutAndUniformBinding();
	console.log("WebGPU bridge tests passed");
}

await run();
