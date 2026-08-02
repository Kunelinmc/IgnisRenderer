import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { Texture } from "../../../src/core/Texture.ts";
import { Material, AlphaMode } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { ParticleBlendMode } from "../../../src/particles/types.ts";
import {
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	PARTICLE_TRANSIENT_BATCHES_KEY,
	createTransientStore,
} from "../../../src/pipeline/types.ts";
import { createWebGPUComputeFacade } from "../../../src/backends/webgpu/ComputeFacade.ts";
import { WebGPUFrameServiceOwner } from "../../../src/backends/webgpu/WebGPUFrameServiceOwner.ts";
import { WebGPUParticleRenderResources } from "../../../src/backends/webgpu/WebGPUParticleRenderResources.ts";
import { WEBGPU_PARTICLE_DRAW_BATCHES_KEY } from "../../../src/backends/webgpu/types.ts";
import {
	FakeCommandEncoder,
	FakeWebGPUBackend,
} from "../../helpers/fakes.mjs";

globalThis.GPUShaderStage ??= {
	VERTEX: 1,
	FRAGMENT: 2,
	COMPUTE: 4,
};

function createTriangleMesh(material) {
	return MeshAsset.fromFaces([
		{
			material,
			vertices: [
				{ x: 0, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 1, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 0, y: 1, z: 0, normal: { x: 0, y: 0, z: 1 } },
			],
		},
	]);
}

function createMeshBatch({
	systemId,
	templateIndex,
	templateId,
	mesh,
	material,
	position,
	size,
	depth,
}) {
	return {
		kind: "mesh",
		systemId,
		templateIndex,
		templateId,
		mesh,
		primitive: mesh.primitives[0],
		material,
		receiveShadows: true,
		castShadows: true,
		shadowDensity: 1,
		shadowSoftness: 1,
		particles: [
			{
				templateIndex,
				position,
				previousPosition: { ...position, x: position.x - 1 },
				size,
				color: { r: 255, g: 255, b: 255, a: 1 },
				rotation: 0.25,
				previousRotation: 0,
				depth,
			},
		],
	};
}

function testOwnerExposesStableNarrowParticleProvider() {
	const backend = new FakeWebGPUBackend();
	const owner = new WebGPUFrameServiceOwner(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);
	const first = owner.getParticleRenderProvider();
	const second = owner.getParticleRenderProvider();
	assert.ok(first instanceof WebGPUParticleRenderResources);
	assert.equal(first, second);
	owner.destroy();
	owner.destroy();
}

function testMeshParticlePacketConstruction() {
	const backend = new FakeWebGPUBackend();
	const owner = new WebGPUFrameServiceOwner(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);
	const particles = owner.getParticleRenderProvider();
	const opaqueMaterial = new Material({ name: "particle-opaque" });
	const transparentMaterial = new Material({
		name: "particle-transparent",
		alphaMode: AlphaMode.Blend,
		opacity: 0.5,
	});
	const opaqueMesh = createTriangleMesh(opaqueMaterial);
	const transparentMesh = createTriangleMesh(transparentMaterial);
	const context = { transient: createTransientStore() };
	context.transient.set(PARTICLE_MESH_TRANSIENT_BATCHES_KEY, [
		createMeshBatch({
			systemId: "particle-system",
			templateIndex: 0,
			templateId: "opaque-shard",
			mesh: opaqueMesh,
			material: opaqueMaterial,
			position: { x: 1, y: 2, z: 3 },
			size: 2,
			depth: 4,
		}),
		createMeshBatch({
			systemId: "particle-system",
			templateIndex: 1,
			templateId: "transparent-shard",
			mesh: transparentMesh,
			material: transparentMaterial,
			position: { x: 0, y: 0, z: 0 },
			size: 1,
			depth: 2,
		}),
	]);

	const opaquePackets = particles.buildParticleMeshDrawPackets(context, {
		includeOpaque: true,
		includeTransparent: false,
	});
	const transparentPackets = particles.buildParticleMeshDrawPackets(context, {
		includeOpaque: false,
		includeTransparent: true,
	});
	const shadowPackets = particles.buildParticleMeshDrawPackets(context, {
		includeOpaque: false,
		includeTransparent: false,
		includeShadowCasters: true,
		includeShadowTransmitters: true,
	});

	assert.equal(opaquePackets.length, 1);
	assert.equal(opaquePackets[0].material, opaqueMaterial);
	assert.equal(
		opaquePackets[0].worldBounds.radius,
		opaqueMesh.primitives[0].boundingSphere.radius * 2,
	);
	assert.ok(
		(opaquePackets[0].passFlags & DRAW_PACKET_FLAG_SHADOW_CASTER) !== 0,
	);
	assert.equal(transparentPackets.length, 1);
	assert.ok(
		(transparentPackets[0].passFlags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0,
	);
	assert.ok(
		(transparentPackets[0].passFlags & DRAW_PACKET_FLAG_SHADOW_TRANSMITTER) !== 0,
	);
	assert.equal(shadowPackets.length, 2);
	owner.destroy();
}

function createBillboardBatch(systemId, blendMode, texture) {
	return {
		systemId,
		blendMode,
		texture,
		receiveShadows: true,
		particles: [
			{
				position: { x: 1, y: 2, z: 3 },
				size: 2,
				color: { r: 255, g: 128, b: 64, a: 0.5 },
				rotation: 0.25,
				depth: 4,
				uvRect: { u0: 0, v0: 0, u1: 1, v1: 1 },
			},
		],
	};
}

function createBillboardContext(transient) {
	return {
		attachments: { width: 100, height: 100 },
		incremental: {
			enabled: true,
			forceFullFrame: false,
			dirtyRects: [{ x: 10, y: 20, width: 30, height: 40 }],
		},
		transient,
	};
}

function createParticleTargets() {
	const target = { width: 50, height: 25, destroy() {} };
	return {
		label: "WebGPUParticles_ResourcesTest",
		colorAttachments: [
			{
				view: target,
				loadOp: "load",
				storeOp: "store",
			},
		],
		depth: target,
	};
}

async function testBillboardPipelinesBindingsScissorAndLifecycle() {
	const backend = new FakeWebGPUBackend();
	const owner = new WebGPUFrameServiceOwner(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);
	await owner.init();
	const particles = owner.getParticleRenderProvider();
	const texture = new Texture({
		data: new Uint8Array([255, 255, 255, 255]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	});
	texture.repeat = { x: 2, y: 3 };
	texture.offset = { x: 0.25, y: -0.5 };
	texture.rotation = Math.PI / 4;
	const transient = createTransientStore();
	transient.set(PARTICLE_TRANSIENT_BATCHES_KEY, [
		createBillboardBatch("particle-alpha", ParticleBlendMode.Alpha, texture),
		createBillboardBatch("particle-additive", ParticleBlendMode.Additive, texture),
	]);
	const context = createBillboardContext(transient);
	const targets = createParticleTargets();
	const frameResources = { frameBinding: { label: "frame-binding" } };
	const encoder = new FakeCommandEncoder(backend);

	const renderedCount = await particles.renderParticles(
		encoder,
		context,
		targets,
		frameResources,
		"single",
	);
	assert.equal(renderedCount, 2);
	assert.ok(
		backend.pipelines.some(
			(pipeline) => pipeline.label === "WebGPUParticlePipeline_alpha_single",
		),
	);
	assert.ok(
		backend.pipelines.some(
			(pipeline) => pipeline.label === "WebGPUParticlePipeline_additive_single",
		),
	);
	assert.deepEqual(
		encoder.calls.filter((call) => call[0] === "setScissorRect"),
		[
			["setScissorRect", 5, 5, 15, 10],
			["setScissorRect", 5, 5, 15, 10],
		],
	);

	const alphaBinding = backend.bindingGroups.find(
		(binding) => binding.label === "ParticleBinding_particle-alpha",
	);
	assert.ok(alphaBinding);
	const uvBuffer = alphaBinding.desc.entries[2].resource;
	const uvTransform = Array.from(uvBuffer.lastWrite);
	assert.deepEqual(uvTransform.slice(0, 4), [2, 3, 0.25, -0.5]);
	assert.ok(Math.abs(uvTransform[4] - Math.cos(Math.PI / 4)) < 1e-6);
	assert.ok(Math.abs(uvTransform[5] - Math.sin(Math.PI / 4)) < 1e-6);
	assert.deepEqual(uvTransform.slice(6), [0, 0]);

	const oitEncoder = new FakeCommandEncoder(backend);
	await particles.renderParticles(
		oitEncoder,
		context,
		{
			...targets,
			colorAttachments: [targets.colorAttachments[0], targets.colorAttachments[0]],
		},
		frameResources,
		"single",
		{
			includeBlendModes: [ParticleBlendMode.Alpha],
			pipelineMode: "oit",
		},
	);
	const oitPipeline = backend.pipelines.find(
		(pipeline) => pipeline.label === "WebGPUParticlePipeline_oit-alpha_single",
	);
	assert.ok(oitPipeline);
	assert.equal(oitPipeline.desc.fragment.entryPoint, "fsMainOIT");
	assert.equal(oitPipeline.desc.fragment.targets.length, 2);

	particles.onShaderRuntimeChanged();
	assert.equal(alphaBinding.destroyed, true);
	assert.equal(uvBuffer.destroyed, true);
	const pipelinesBeforeRebuild = backend.pipelines.length;
	await particles.renderParticles(
		new FakeCommandEncoder(backend),
		context,
		targets,
		frameResources,
		"single",
	);
	assert.ok(backend.pipelines.length > pipelinesBeforeRebuild);

	const rebuiltBinding = backend.bindingGroups.findLast(
		(binding) => binding.label === "ParticleBinding_particle-alpha",
	);
	const quadBuffer = backend.buffers.find(
		(buffer) => buffer.desc.label === "WebGPUParticleQuad",
	);
	const instanceBuffer = backend.buffers.findLast(
		(buffer) => buffer.desc.label === "WebGPUParticleInstances",
	);
	owner.destroy();
	assert.equal(rebuiltBinding.destroyed, true);
	assert.equal(quadBuffer.destroyed, true);
	assert.equal(instanceBuffer.destroyed, true);
}

async function testGPUIndirectDrawAndFallback() {
	const backend = new FakeWebGPUBackend();
	const owner = new WebGPUFrameServiceOwner(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);
	await owner.init();
	const particles = owner.getParticleRenderProvider();
	const texture = new Texture({
		data: new Uint8Array([255, 255, 255, 255]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	});
	const instanceBuffer = backend.createBuffer({
		size: 64,
		usage: 1,
		label: "ParticleGPUInstances_Test",
	});
	const indirectBuffer = backend.createBuffer({
		size: 16,
		usage: 1,
		label: "ParticleGPUIndirect_Test",
	});
	const transient = createTransientStore();
	transient.set(WEBGPU_PARTICLE_DRAW_BATCHES_KEY, [
		{
			systemId: "particle-gpu",
			blendMode: ParticleBlendMode.Alpha,
			texture,
			instanceBuffer,
			indirectBuffer,
			indirectOffset: 4,
			instanceCount: 3,
		},
	]);
	const context = createBillboardContext(transient);
	const targets = createParticleTargets();
	const frameResources = { frameBinding: { label: "frame-binding" } };
	const indirectEncoder = new FakeCommandEncoder(backend);

	assert.equal(
		await particles.renderParticles(
			indirectEncoder,
			context,
			targets,
			frameResources,
			"single",
		),
		3,
	);
	assert.deepEqual(
		indirectEncoder.calls.find((call) => call[0] === "drawIndirect").slice(1),
		[indirectBuffer, 4],
	);

	const fallbackEncoder = new FakeCommandEncoder(backend);
	fallbackEncoder.drawIndirect = undefined;
	await particles.renderParticles(
		fallbackEncoder,
		context,
		targets,
		frameResources,
		"single",
	);
	assert.deepEqual(
		fallbackEncoder.calls.find((call) => call[0] === "draw"),
		["draw", 6, 3, 0, 0],
	);
	owner.destroy();
}

async function testParticleBindingCacheEviction() {
	const backend = new FakeWebGPUBackend();
	const owner = new WebGPUFrameServiceOwner(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);
	await owner.init();
	const texture = new Texture({
		data: new Uint8Array([255, 255, 255, 255]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	});
	const transient = createTransientStore();
	transient.set(PARTICLE_TRANSIENT_BATCHES_KEY, [
		createBillboardBatch("particle-evict", ParticleBlendMode.Alpha, texture),
	]);
	await owner.getParticleRenderProvider().renderParticles(
		new FakeCommandEncoder(backend),
		createBillboardContext(transient),
		createParticleTargets(),
		{ frameBinding: { label: "frame-binding" } },
		"single",
	);
	const binding = backend.bindingGroups.find(
		(entry) => entry.label === "ParticleBinding_particle-evict",
	);
	const uvBuffer = binding.desc.entries[2].resource;
	for (let frame = 0; frame <= 120; frame++) {
		owner.beginFrameResourceLifecycle();
	}
	assert.equal(binding.destroyed, true);
	assert.equal(uvBuffer.destroyed, true);
	owner.destroy();
}

function testOwnerSourceContainsNoParticleRenderingImplementation() {
	const ownerSource = readFileSync(
		new URL(
			"../../../src/backends/webgpu/WebGPUFrameServiceOwner.ts",
			import.meta.url,
		),
		"utf8",
	);
	assert.doesNotMatch(ownerSource, /_particlePipelineAlpha/);
	assert.doesNotMatch(ownerSource, /public async renderParticles/);
	assert.doesNotMatch(ownerSource, /ShaderSource\.load\("webgpu\.particle\.composite"\)/);
}

testOwnerExposesStableNarrowParticleProvider();
testMeshParticlePacketConstruction();
await testBillboardPipelinesBindingsScissorAndLifecycle();
await testGPUIndirectDrawAndFallback();
await testParticleBindingCacheEviction();
testOwnerSourceContainsNoParticleRenderingImplementation();
console.log("WebGPU particle render resources tests passed");
