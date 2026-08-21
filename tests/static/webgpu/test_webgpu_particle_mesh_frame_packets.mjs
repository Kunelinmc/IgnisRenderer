import assert from "node:assert/strict";

import { Camera } from "../../../src/cameras/Camera.ts";
import { Material, AlphaMode } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import {
	prepareFramePackets,
} from "../../../src/pipeline/FramePackets.ts";
import { WebGPUBackend } from "../../../src/backends/webgpu/WebGPUBackend.ts";
import {
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
} from "../../../src/pipeline/types.ts";
import { createTransientStore } from "../../../src/foundation/TransientStore.ts";

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
				rotation: 0.25,
				previousRotation: 0,
				depth,
			},
		],
	};
}

function createPacketContext() {
	return {
		transient: createTransientStore(),
		viewCamera: new Camera(),
		scene: {
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
		},
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
	const context = createPacketContext();
	context.transient.set(PARTICLE_MESH_TRANSIENT_BATCHES_KEY, [
		createMeshBatch({
			systemId: "particle-system",
			templateIndex: 0,
			templateId: "opaque-shard",
			mesh: opaqueMesh,
			material: opaqueMaterial,
			position: { x: 1, y: 2, z: -3 },
			size: 2,
			depth: 4,
		}),
		createMeshBatch({
			systemId: "particle-system",
			templateIndex: 1,
			templateId: "transparent-shard",
			mesh: transparentMesh,
			material: transparentMaterial,
			position: { x: 0, y: 0, z: -2 },
			size: 1,
			depth: 2,
		}),
	]);

	const packets = prepareFramePackets(context, "main");
	assert.strictEqual(prepareFramePackets(context, "main"), packets);
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
	const emptyViewA = createPacketContext();
	const emptyViewB = createPacketContext();
	assert.notStrictEqual(
		prepareFramePackets(emptyViewA, "main"),
		prepareFramePackets(emptyViewB, "main"),
	);
	const probePackets = prepareFramePackets(context, "probe-capture");
	assert.equal(probePackets.all.length, 2);
	assert.notStrictEqual(probePackets.all[0], packets.all[0]);
	assert.equal(prepareFramePackets(context, "planar-reflection").all.length, 0);
}

async function testSimulationSealsFrameAfterBatchEmission() {
	const events = [];
	const context = { transient: createTransientStore() };
	context.transient.set(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 0.25);
	const backend = new WebGPUBackend();
	backend._state = "ready";
	backend._device = {};
	backend._queue = {};
	backend._activeFrameTransaction = {
		isOpen: true,
		assertRecordingContext(value) {
			assert.strictEqual(value, context);
		},
	};
	backend._particleSimulator = {
		async simulateAndEmitRenderBatches(_context, deltaTimeSeconds) {
			assert.equal(deltaTimeSeconds, 0.25);
			events.push("simulate");
		},
	};
	backend._frameOrchestrator = {
		recordOpaqueGraphStage(stage) {
			assert.equal(stage, "particle-sim");
			events.push("record-opaque");
		},
		sealParticleSimulation(value) {
			assert.strictEqual(value, context);
			events.push("seal");
		},
		executePass() {
			assert.fail("particle simulation must not execute through the frame graph");
		},
	};
	await backend.executePass(
		{
			stage: "particle-sim",
			executor: "backend",
			enabled: true,
			dependsOn: [],
		},
		context,
	);
	assert.deepEqual(events, ["record-opaque", "simulate", "seal"]);
}

testMeshParticleFramePreparation();
await testSimulationSealsFrameAfterBatchEmission();
console.log("WebGPU particle mesh frame packet tests passed");
