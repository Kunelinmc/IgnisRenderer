import assert from "node:assert/strict";

import { Material, AlphaMode } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import {
	getWebGPUParticleMeshFramePackets,
	prepareWebGPUParticleMeshFramePackets,
} from "../../../src/backends/webgpu/particleMeshFramePackets.ts";
import { WebGPUBackendPassDispatcher } from "../../../src/backends/webgpu/WebGPUBackendPassDispatcher.ts";
import {
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	createTransientStore,
} from "../../../src/pipeline/types.ts";

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

function testMeshParticleFramePreparation() {
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

	const packets = prepareWebGPUParticleMeshFramePackets(context);
	assert.strictEqual(prepareWebGPUParticleMeshFramePackets(context), packets);
	assert.strictEqual(getWebGPUParticleMeshFramePackets(context), packets);
	assert.equal(packets.all.length, 2);
	assert.equal(packets.opaque.length, 1);
	assert.equal(packets.transparent.length, 1);
	assert.equal(packets.shadowCasters.length, 1);
	assert.equal(packets.shadowTransmitters.length, 1);
	assert.strictEqual(packets.all[0], packets.opaque[0]);
	assert.strictEqual(packets.all[1], packets.transparent[0]);
	assert.strictEqual(packets.opaque[0], packets.shadowCasters[0]);
	assert.strictEqual(packets.transparent[0], packets.shadowTransmitters[0]);
	assert.equal(packets.opaque[0].material, opaqueMaterial);
	assert.equal(
		packets.opaque[0].worldBounds.radius,
		opaqueMesh.primitives[0].boundingSphere.radius * 2,
	);
	assert.ok(
		(packets.opaque[0].passFlags & DRAW_PACKET_FLAG_SHADOW_CASTER) !== 0,
	);
	assert.ok(
		(packets.transparent[0].passFlags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0,
	);
	assert.ok(
		(packets.transparent[0].passFlags & DRAW_PACKET_FLAG_SHADOW_TRANSMITTER) !== 0,
	);
	const emptyViewA = { transient: createTransientStore() };
	const emptyViewB = { transient: createTransientStore() };
	assert.notStrictEqual(
		prepareWebGPUParticleMeshFramePackets(emptyViewA),
		prepareWebGPUParticleMeshFramePackets(emptyViewB),
	);
}

async function testSimulationSealsFrameAfterBatchEmission() {
	const events = [];
	const context = { transient: createTransientStore() };
	context.transient.set(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 0.25);
	const dispatcher = new WebGPUBackendPassDispatcher({
		particleSimulator: {
			async simulateAndEmitRenderBatches(_context, deltaTimeSeconds) {
				assert.equal(deltaTimeSeconds, 0.25);
				events.push("simulate");
			},
		},
		frameOrchestrator: {
			sealParticleSimulation(value) {
				assert.strictEqual(value, context);
				events.push("seal");
			},
		},
		postProcessRuntime: {},
	});
	await dispatcher.executePass(
		{
			stage: "particle-sim",
			executor: "backend",
			enabled: true,
			dependsOn: [],
		},
		context,
	);
	assert.deepEqual(events, ["simulate", "seal"]);
}

testMeshParticleFramePreparation();
await testSimulationSealsFrameAfterBatchEmission();
console.log("WebGPU particle mesh frame packet tests passed");
