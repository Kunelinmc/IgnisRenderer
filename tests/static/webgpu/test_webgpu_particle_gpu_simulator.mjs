import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { SH } from "../../../src/maths/SH.ts";
import { resolveFeatureState } from "../../../src/pipeline/FeatureResolver.ts";
import {
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	PARTICLE_TRANSIENT_BATCHES_KEY,
} from "../../../src/pipeline/types.ts";
import { Material, AlphaMode } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { ParticleSystem } from "../../../src/particles/ParticleSystem.ts";
import { ParticleBlendMode } from "../../../src/particles/types.ts";
import { BufferUsage } from "../../../src/backends/types.ts";
import { WebGPUFrameServiceOwner as WebGPURenderResources } from "../../../src/backends/webgpu/WebGPUFrameServiceOwner.ts";
import { WebGPUParticleSimulator } from "../../../src/simulation/particles/WebGPUParticleSimulator.ts";
import { WEBGPU_PARTICLE_DRAW_BATCHES_KEY } from "../../../src/backends/webgpu/particleTransient.ts";
import {
	FakeCommandEncoder as FakeRenderEncoder,
	FakeWebGPUBackend as FakeBackend,
} from "../../helpers/fakes.mjs";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

globalThis.GPUShaderStage ??= {
	VERTEX: 1,
	FRAGMENT: 2,
	COMPUTE: 4,
};

function createMainFrameOptions(options = {}) {
	return {
		scopeKey: "main",
		sceneTargetMode: "mrt",
		...options,
	};
}

function createFeatures() {
	return resolveFeatureState(
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
			ssgi: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			colorFilter: false,
			clusteredLighting: true,
		},
		"webgpu",
	);
}

function createContext(particleSystems = []) {
	const camera = new Camera();
	camera.position.set(0, 0, 8);
	camera.updateMatrices();
	const features = createFeatures();
	const scene = {
		sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 10 },
		lights: [],
		particleSystems,
		hasActiveAnimations: false,
		camera,
		environment: null,
		meshInstances: [],
		shadowMaps: new Map(),
		opaquePackets: [],
		transparentPackets: [],
		shadowCasterPackets: [],
		shadowTransmitterPackets: [],
		reflectivePackets: [],
		decalPackets: [],
		spatialIndex: null,
	};
	return {
		viewCamera: camera,
		attachments: {
			width: 64,
			height: 64,
		},
		features,
		postProcess: createResolvedPostProcess(),
		shadowMaps: scene.shadowMaps,
		scene,
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width: 64, height: 64 }],
			dirtyTileSize: 64,
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			reasonMask: 0,
			temporalHistoryReset: false,
		},
		transient: new Map(),
	};
}

function createTriangleMesh(material = new Material({ name: "ParticleMesh" })) {
	return MeshAsset.fromFaces([
		{
			material,
			vertices: [
				{
					x: 0,
					y: 0,
					z: 0,
					u: 0,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 1,
					y: 0,
					z: 0,
					u: 1,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 0,
					y: 1,
					z: 0,
					u: 0,
					v: 1,
					normal: { x: 0, y: 0, z: 1 },
				},
			],
		},
	]);
}

function testWebGPUParticleSimulatorPublishesDrawBatches() {
	const backend = new FakeBackend();
	const simulator = new WebGPUParticleSimulator({
		backend,
		backendTag: "webgpu-test",
		maxParticlesPerSystem: 1024,
	});
	const system = new ParticleSystem({
		maxParticles: 64,
		emit: {
			rate: 0,
			bursts: [{ time: 0, count: 3 }],
			lifetimeRange: [5, 5],
			speedRange: [0, 0],
			sizeRange: [1, 1],
		},
	});
	const context = createContext([system]);

	simulator.beginFrame(context);
	simulator.simulate(context, 0.016);
	simulator.emitRenderBatches(context);

	const cpuBatches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) ?? [];
	const gpuBatches = context.transient.get(WEBGPU_PARTICLE_DRAW_BATCHES_KEY) ?? [];
	assert.ok(cpuBatches.length > 0);
	assert.equal(gpuBatches.length, cpuBatches.length);
	assert.ok(gpuBatches[0].instanceCount > 0);
	assert.ok(
		backend.buffers.some((buffer) =>
			String(buffer.label).startsWith("WebGPUParticleIndirect_"),
		),
	);

	simulator.destroy();
}

async function testWebGPUParticleSimulatorDispatchesComputeSimulation() {
	const backend = new FakeBackend();
	const simulator = new WebGPUParticleSimulator({
		backend,
		backendTag: "webgpu-test",
		maxParticlesPerSystem: 1024,
	});
	const system = new ParticleSystem({
		maxParticles: 64,
		blendMode: ParticleBlendMode.Additive,
		emit: {
			rate: 0,
			bursts: [{ time: 0, count: 3 }],
			lifetimeRange: [5, 5],
			speedRange: [0, 0],
			sizeRange: [1, 1],
		},
	});
	const context = createContext([system]);

	simulator.beginFrame(context);
	await simulator.simulateAndEmitRenderBatches(context, 0.016);

	const cpuBatches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) ?? [];
	const gpuBatches = context.transient.get(WEBGPU_PARTICLE_DRAW_BATCHES_KEY) ?? [];
	assert.equal(cpuBatches.length, 0);
	assert.equal(gpuBatches.length, 1);
	assert.equal(gpuBatches[0].systemId, system.id);
	assert.equal(gpuBatches[0].blendMode, ParticleBlendMode.Additive);
	assert.equal(gpuBatches[0].instanceCount, 64);
	assert.ok(backend.dispatches.length >= 3);
	assert.ok(
		backend.shaderModules.some(
			(module) => module.label === "WebGPUParticleSimulateModule",
		),
	);

	const instanceBuffer = backend.buffers.find(
		(buffer) => buffer.label === `WebGPUParticleInstances_${system.id}`,
	);
	assert.ok(instanceBuffer);
	assert.ok(instanceBuffer.usage & BufferUsage.Storage);
	assert.ok(instanceBuffer.usage & BufferUsage.Vertex);
	assert.equal(instanceBuffer.lastWrite, null);

	const indirectBuffer = backend.buffers.find(
		(buffer) => buffer.label === `WebGPUParticleIndirect_${system.id}`,
	);
	assert.ok(indirectBuffer);
	assert.ok(indirectBuffer.usage & BufferUsage.Storage);
	assert.ok(indirectBuffer.usage & BufferUsage.Indirect);

	simulator.destroy();
}

async function testWebGPUParticleSimulatorMixesComputeAndCpuFallbackBatches() {
	const backend = new FakeBackend();
	const simulator = new WebGPUParticleSimulator({
		backend,
		backendTag: "webgpu-test",
		maxParticlesPerSystem: 1024,
	});
	const computeSystem = new ParticleSystem({
		maxParticles: 64,
		blendMode: ParticleBlendMode.Additive,
		emit: {
			rate: 0,
			bursts: [{ time: 0, count: 2 }],
			lifetimeRange: [5, 5],
			speedRange: [0, 0],
			sizeRange: [1, 1],
		},
	});
	const fallbackSystem = new ParticleSystem({
		maxParticles: 64,
		blendMode: ParticleBlendMode.Alpha,
		emit: {
			rate: 0,
			bursts: [{ time: 0, count: 2 }],
			lifetimeRange: [5, 5],
			speedRange: [0, 0],
			sizeRange: [1, 1],
		},
	});
	const context = createContext([computeSystem, fallbackSystem]);

	simulator.beginFrame(context);
	await simulator.simulateAndEmitRenderBatches(context, 0.016);

	const cpuBatches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) ?? [];
	const gpuBatches = context.transient.get(WEBGPU_PARTICLE_DRAW_BATCHES_KEY) ?? [];
	assert.equal(cpuBatches.length, 1);
	assert.equal(cpuBatches[0].systemId, fallbackSystem.id);
	assert.equal(gpuBatches.length, 2);
	assert.ok(
		gpuBatches.some(
			(batch) =>
				batch.systemId === computeSystem.id &&
				batch.blendMode === ParticleBlendMode.Additive,
		),
	);
	assert.ok(
		gpuBatches.some(
			(batch) =>
				batch.systemId === fallbackSystem.id &&
				batch.blendMode === ParticleBlendMode.Alpha &&
				batch.instanceCount === 2,
		),
	);

	simulator.destroy();
}

async function testRenderResourcesPrefersGPUDrawBatches() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(backend);
	await resources.init();

	const context = createContext([]);
	const frameResources = resources.prepareFrame(context, createMainFrameOptions());

	const instanceBuffer = backend.createBuffer({
		size: 3 * 64,
		usage: BufferUsage.Vertex | BufferUsage.CopyDst,
		label: "TestParticleInstances",
	});
	const indirectBuffer = backend.createBuffer({
		size: Uint32Array.BYTES_PER_ELEMENT * 4,
		usage: BufferUsage.CopyDst | BufferUsage.Indirect,
		label: "TestParticleIndirect",
	});
	backend.writeBuffer(indirectBuffer, new Uint32Array([6, 3, 0, 0]));
	const texture = new Texture(new Uint8Array([255, 255, 255, 255]), 1, 1, "sRGB");
	context.transient.set(WEBGPU_PARTICLE_DRAW_BATCHES_KEY, [
		{
			systemId: "particleSystem-gpu",
			blendMode: ParticleBlendMode.Alpha,
			texture,
			receiveShadows: true,
			instanceBuffer,
			instanceCount: 3,
			indirectBuffer,
			indirectOffset: 0,
		},
	]);

	const encoder = new FakeRenderEncoder();
	const renderTarget = { width: 16, height: 16, destroy() {} };
	const renderedCount = await resources.renderParticles(
		encoder,
		context,
		{
			label: "WebGPUParticlesGPUIndirect_Test",
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
			pipelineMode: "legacy",
		},
	);

	assert.equal(renderedCount, 3);
	assert.ok(encoder.calls.some((call) => call[0] === "drawIndirect"));
}

function testRenderResourcesBuildsParticleMeshDrawPackets() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(backend);
	const opaqueMaterial = new Material({ name: "particle-opaque" });
	const transparentMaterial = new Material({
		name: "particle-transparent",
		alphaMode: AlphaMode.Blend,
		opacity: 0.5,
	});
	const opaqueMesh = createTriangleMesh(opaqueMaterial);
	const transparentMesh = createTriangleMesh(transparentMaterial);
	const context = createContext([]);
	context.transient.set(PARTICLE_MESH_TRANSIENT_BATCHES_KEY, [
		{
			kind: "mesh",
			systemId: "particleSystem-mesh",
			templateIndex: 0,
			templateId: "opaque-shard",
			mesh: opaqueMesh,
			primitive: opaqueMesh.primitives[0],
			material: opaqueMaterial,
			receiveShadows: true,
			castShadows: true,
			shadowDensity: 1,
			shadowSoftness: 1,
			particles: [
				{
					templateIndex: 0,
					position: { x: 1, y: 2, z: 3 },
					previousPosition: { x: 0, y: 2, z: 3 },
					size: 2,
					color: { r: 255, g: 255, b: 255, a: 1 },
					rotation: 0.25,
					previousRotation: 0,
					depth: 4,
				},
			],
		},
		{
			kind: "mesh",
			systemId: "particleSystem-mesh",
			templateIndex: 1,
			templateId: "transparent-shard",
			mesh: transparentMesh,
			primitive: transparentMesh.primitives[0],
			material: transparentMaterial,
			receiveShadows: true,
			castShadows: true,
			shadowDensity: 1,
			shadowSoftness: 1,
			particles: [
				{
					templateIndex: 1,
					position: { x: 0, y: 0, z: 0 },
					previousPosition: { x: 0, y: 0, z: 0 },
					size: 1,
					color: { r: 255, g: 255, b: 255, a: 1 },
					rotation: 0,
					previousRotation: 0,
					depth: 2,
				},
			],
		},
	]);

	const opaquePackets = resources.buildParticleMeshDrawPackets(context, {
		includeOpaque: true,
		includeTransparent: false,
	});
	const transparentPackets = resources.buildParticleMeshDrawPackets(context, {
		includeOpaque: false,
		includeTransparent: true,
	});
	const shadowPackets = resources.buildParticleMeshDrawPackets(context, {
		includeOpaque: false,
		includeTransparent: false,
		includeShadowCasters: true,
	});

	assert.equal(opaquePackets.length, 1);
	assert.equal(opaquePackets[0].material, opaqueMaterial);
	assert.equal(
		opaquePackets[0].worldBounds.radius,
		opaqueMesh.primitives[0].boundingSphere.radius * 2
	);
	assert.ok(opaquePackets[0].previousWorldMatrix instanceof Matrix4);
	assert.equal(
		opaquePackets[0].passFlags & DRAW_PACKET_FLAG_SHADOW_CASTER,
		DRAW_PACKET_FLAG_SHADOW_CASTER
	);
	assert.equal(transparentPackets.length, 1);
	assert.equal(
		transparentPackets[0].passFlags & DRAW_PACKET_FLAG_TRANSPARENT,
		DRAW_PACKET_FLAG_TRANSPARENT
	);
	assert.equal(shadowPackets.length, 1);
}

async function run() {
	testWebGPUParticleSimulatorPublishesDrawBatches();
	await testWebGPUParticleSimulatorDispatchesComputeSimulation();
	await testWebGPUParticleSimulatorMixesComputeAndCpuFallbackBatches();
	await testRenderResourcesPrefersGPUDrawBatches();
	testRenderResourcesBuildsParticleMeshDrawPackets();
	console.log("WebGPU particle GPU simulator tests passed");
}

await run();
