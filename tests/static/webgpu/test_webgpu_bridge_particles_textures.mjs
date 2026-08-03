import assert from "node:assert/strict";
import {
	WebGPUFrameServiceOwner as WebGPURenderResources
} from "../../../src/backends/webgpu/WebGPUFrameServiceOwner.ts";
import {
	ShaderSource
} from "../../../src/shaders/ShaderSource.ts";
import {
	resolveFeatureState
} from "../../../src/pipeline/FeatureResolver.ts";
import {
	TextureFormat
} from "../../../src/backends/types.ts";
import {
	DirectionalLight
} from "../../../src/lights/DirectionalLight.ts";
import {
	Matrix4
} from "../../../src/maths/Matrix4.ts";
import {
	SH
} from "../../../src/maths/SH.ts";
import {
	PBRMaterial
} from "../../../src/materials/PBRMaterial.ts";
import {
	Texture
} from "../../../src/core/Texture.ts";
import {
	float16BitsToFloat32
} from "../../../src/foundation/Float16.ts";
import {
	Logger
} from "../../../src/foundation/Logger.ts";
import {
	Scene
} from "../../../src/core/Scene.ts";
import {
	createShadowRenderSet
} from "../../../src/lights/shadows/ShadowMapping.ts";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY
} from "../../../src/pipeline/types.ts";
import {
	ParticleBlendMode
} from "../../../src/particles/types.ts";
import {
	WEBGPU_PARTICLE_VERTEX_LAYOUTS
} from "../../../src/backends/webgpu/bufferLayouts.ts";
import {
	WEBGPU_SHADOW_ATLAS_COLUMNS,
	WEBGPU_TEXTURE_SLOT
} from "../../../src/backends/webgpu/constants.ts";
import {
	WebGPUGeometryRegistry
} from "../../../src/backends/webgpu/WebGPUGeometryRegistry.ts";
import {
	WebGPUTextureRegistry
} from "../../../src/backends/webgpu/WebGPUTextureRegistry.ts";
import {
	createWebGPUComputeFacade
} from "../../../src/backends/webgpu/ComputeFacade.ts";
import {
	createResolvedPostProcess
} from "../../helpers/postprocess.mjs";

import {
	FakeCommandEncoder as FakeRenderEncoder,
	FakeWebGPUBackend as FakeBackend
} from "../../helpers/fakes.mjs";

import {
	createFrame,
	createFrameContext,
	createMainFrameOptions,
	createModel,
	createPacket,
	nearlyEqual
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

async function testParticleUVLayoutAndUniformBinding() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

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

	const texture = new Texture({
		data: new Uint8Array([255, 255, 255, 255]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	});
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
	await resources.getParticleBillboardRenderer().renderParticles(
		encoder,
		context,
		{
			label: "WebGPUParticles_TestUV",
			sampleCount: 1,
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

	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));
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
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));
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
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));
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

	const texture = new Texture({
		data: new Uint8Array([255, 255, 255, 255]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	});
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
	await resources.getParticleBillboardRenderer().renderParticles(
		encoder,
		context,
		{
			label: "WebGPUParticles_TestEvict",
			sampleCount: 1,
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
	resources.commitTemporalFrame();

	for (let i = 0; i < 130; i++) {
		resources.beginFrameResourceLifecycle();
		frameResources = resources.prepareFrame(
			createFrameContext(frame, features),
			createMainFrameOptions()
		);
		resources.commitTemporalFrame();
	}

	assert.equal(particleBinding.destroyed, true);
	assert.equal(uvBuffer.destroyed, true);
}

async function testRenderResourcesDestroyCleansParticleAndGeometryResources() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));
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

	const texture = new Texture({
		data: new Uint8Array([255, 255, 255, 255]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	});
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
	await resources.getParticleBillboardRenderer().renderParticles(
		encoder,
		context,
		{
			label: "WebGPUParticles_TestDestroy",
			sampleCount: 1,
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
	const registry = new WebGPUTextureRegistry(backend, backend);
	const texture = new Texture({
		data: new Uint8ClampedArray([10, 20, 30, 255]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	});

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
	const registry = new WebGPUTextureRegistry(backend, backend);
	const texture = new Texture({
		data: new Float32Array([2, 1, 0.5, 1]),
		width: 1,
		height: 1,
		colorSpace: "HDR",
	});

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
	const registry = new WebGPUTextureRegistry(backend, backend);
	const texture = new Texture({
		data: new Uint8ClampedArray([255, 255, 255, 255]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	});
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
	try {
		await testParticleUVLayoutAndUniformBinding();
		await testParticleShadowVolumeBufferUpdatesForDirectionalSlice();
		await testShadowAtlasSizeTracksShadowMapsWhenLightingDisabled();
		await testParticleBindingCacheEvictsStaleSystems();
		await testRenderResourcesDestroyCleansParticleAndGeometryResources();
		await testWebGPUGeometryRegistryReleaseGeometryDestroysBuffers();
		await testDynamicTextureReuploadOnVersionChange();
		await testHDRTextureUploadsAsRGBA16Float();
		await testSamplerCacheInvalidatesWhenTextureSamplerStateChanges();
		console.log("WebGPU bridge particles/textures tests passed");
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
