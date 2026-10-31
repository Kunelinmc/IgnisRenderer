import assert from "node:assert/strict";

import { Texture } from "../../../src/core/Texture.ts";
import { ParticleBlendMode } from "../../../src/particles/types.ts";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
} from "../../../src/pipeline/types.ts";
import { createTransientStore } from "../../../src/foundation/TransientStore.ts";
import { createWebGPUComputeFacade } from "../../../src/backends/webgpu/ComputeFacade.ts";
import { WebGPUFrameServiceOwner } from "../../../src/backends/webgpu/WebGPUFrameServiceOwner.ts";
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

function createBillboardBatch(systemId, blendMode, texture) {
	return {
		systemId,
		blendMode,
		texture,
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

function createParticleTargets(sampleCount = 1) {
	const target = { width: 50, height: 25, destroy() {} };
	return {
		label: "WebGPUParticles_ResourcesTest",
		sampleCount,
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
		createWebGPUComputeFacade(backend),
	);
	const particles = owner.getParticleBillboardRenderer();
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
	const frameResources = {
		frameBinding: { label: "scene-frame-binding" },
		particleFrameBinding: { label: "particle-frame-binding" },
	};
	const encoder = new FakeCommandEncoder(backend);

	const renderedCount = await particles.renderParticles(
		encoder,
		context,
		targets,
		frameResources,
		"single",
	);
	assert.equal(renderedCount, 2);
	assert.deepEqual(
		encoder.calls.filter((call) => call[0] === "setBindGroup" && call[1] === 0),
		[["setBindGroup", 0, frameResources.particleFrameBinding]],
		"CPU billboard draws must bind the camera without scene shadow resources",
	);
	const instanceData = backend.buffers.find(
		(buffer) => buffer.desc.label === "WebGPUParticleInstances",
	).lastWrite;
	assert.deepEqual(Array.from(instanceData).slice(13, 16), [0, 0, 0],
		"particle instance alignment uses zero padding without shadow flags");
	assert.ok(
		backend.pipelines.some(
			(pipeline) => pipeline.label === "WebGPUParticlePipeline_alpha_single",
		),
	);
	const additivePipeline = backend.pipelines.find(
		(pipeline) => pipeline.label === "WebGPUParticlePipeline_additive_single",
	);
	assert.ok(additivePipeline);
	assert.equal(additivePipeline.desc.fragment.entryPoint, "fsMain");
	assert.equal(additivePipeline.desc.fragment.targets.length, 1);
	assert.equal(additivePipeline.desc.fragment.targets[0].blend.color.dstFactor, "one");
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
	const oitCount = await particles.renderParticles(
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
	assert.equal(oitCount, 1);
	const oitPipeline = backend.pipelines.find(
		(pipeline) => pipeline.label === "WebGPUParticlePipeline_oit-alpha_single",
	);
	assert.ok(oitPipeline);
	assert.equal(oitPipeline.desc.fragment.entryPoint, "fsMainOIT");
	assert.equal(oitPipeline.desc.fragment.targets.length, 2);
	assert.equal(oitPipeline.desc.fragment.targets[1].format, "r8unorm");
	const additiveCount = await particles.renderParticles(
		new FakeCommandEncoder(backend),
		context,
		targets,
		frameResources,
		"single",
		{ includeBlendModes: [ParticleBlendMode.Additive], pipelineMode: "legacy" },
	);
	assert.equal(additiveCount, 1);

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
		createWebGPUComputeFacade(backend),
	);
	const particles = owner.getParticleBillboardRenderer();
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
	const frameResources = {
		frameBinding: { label: "scene-frame-binding" },
		particleFrameBinding: { label: "particle-frame-binding" },
	};
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
	assert.deepEqual(
		indirectEncoder.calls.find((call) => call[0] === "setBindGroup" && call[1] === 0),
		["setBindGroup", 0, frameResources.particleFrameBinding],
		"GPU billboard draws must bind the camera without scene shadow resources",
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

async function testParticlePipelinesUsePassTargetSampleCount() {
	const backend = new FakeWebGPUBackend();
	const owner = new WebGPUFrameServiceOwner(
		backend,
		createWebGPUComputeFacade(backend),
	);
	const transient = createTransientStore();
	transient.set(PARTICLE_TRANSIENT_BATCHES_KEY, [
		createBillboardBatch(
			"particle-msaa",
			ParticleBlendMode.Alpha,
			new Texture({
				data: new Uint8Array([255, 255, 255, 255]),
				width: 1,
				height: 1,
				colorSpace: "sRGB",
			}),
		),
	]);

	await owner.getParticleBillboardRenderer().renderParticles(
		new FakeCommandEncoder(backend),
		createBillboardContext(transient),
		createParticleTargets(4),
		{ particleFrameBinding: { label: "particle-frame-binding" } },
		"mrt",
	);

	const pipeline = backend.pipelines.find(
		(entry) => entry.label === "WebGPUParticlePipeline_alpha_mrt",
	);
	assert.ok(pipeline);
	assert.equal(pipeline.desc.sampleCount, 4);
	owner.destroy();
}

async function testParticleBindingCacheEviction() {
	const backend = new FakeWebGPUBackend();
	const owner = new WebGPUFrameServiceOwner(
		backend,
		createWebGPUComputeFacade(backend),
	);
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
	await owner.getParticleBillboardRenderer().renderParticles(
		new FakeCommandEncoder(backend),
		createBillboardContext(transient),
		createParticleTargets(),
		{ particleFrameBinding: { label: "particle-frame-binding" } },
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

await testBillboardPipelinesBindingsScissorAndLifecycle();
await testParticlePipelinesUsePassTargetSampleCount();
await testGPUIndirectDrawAndFallback();
await testParticleBindingCacheEviction();
console.log("WebGPU particle render resources tests passed");
