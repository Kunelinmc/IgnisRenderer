import assert from "node:assert/strict";

import {
	createBaselineFramePacketSet,
	prepareFramePackets,
} from "../../../src/pipeline/FramePackets.ts";
import {
	DRAW_PACKET_FLAG_REFLECTIVE,
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
} from "../../../src/pipeline/types.ts";
import { createTransientStore } from "../../../src/foundation/TransientStore.ts";
import { Material, AlphaMode } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { Camera } from "../../../src/cameras/Camera.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function createPacket(id, passFlags = 0) {
	return createTestDrawPacket({ id, passFlags });
}

function createContext({ transient = createTransientStore(), camera = new Camera(), scene } = {}) {
	return {
		transient,
		viewCamera: camera,
		scene: scene ?? {
			opaquePackets: [createPacket("scene-opaque")],
			transparentPackets: [
				createPacket("scene-transparent", DRAW_PACKET_FLAG_TRANSPARENT),
			],
			shadowCasterPackets: [
				createPacket("scene-shadow", DRAW_PACKET_FLAG_SHADOW_CASTER),
			],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
		},
	};
}

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

function testBaselinePacketSetCreation() {
	const context = createContext();
	const baseline = createBaselineFramePacketSet(context);

	assert.deepEqual(baseline.all.map((p) => p.submission.id), [
		"scene-opaque",
		"scene-transparent",
	]);
	assert.deepEqual(baseline.opaque.map((p) => p.submission.id), ["scene-opaque"]);
	assert.deepEqual(baseline.transparent.map((p) => p.submission.id), ["scene-transparent"]);
	assert.deepEqual(baseline.shadowCasters.map((p) => p.submission.id), ["scene-shadow"]);
	assert.deepEqual(baseline.shadowTransmitters, []);
	assert.deepEqual(baseline.reflective, []);
}

function testFramePacketPreparationAndCaching() {
	const opaqueMaterial = new Material({ name: "particle-opaque" });
	const transparentMaterial = new Material({
		name: "particle-transparent",
		alphaMode: AlphaMode.Blend,
		opacity: 0.5,
		reflectivity: 0.8,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, distance: 0 },
	});
	const opaqueMesh = createTriangleMesh(opaqueMaterial);
	const transparentMesh = createTriangleMesh(transparentMaterial);
	const context = createContext();
	context.transient.set(PARTICLE_MESH_TRANSIENT_BATCHES_KEY, [
		createMeshBatch({
			systemId: "particles",
			templateIndex: 0,
			templateId: "shard-opaque",
			mesh: opaqueMesh,
			material: opaqueMaterial,
			position: { x: 0, y: 0, z: -5 },
			size: 1,
			depth: 5,
		}),
		createMeshBatch({
			systemId: "particles",
			templateIndex: 1,
			templateId: "shard-transparent",
			mesh: transparentMesh,
			material: transparentMaterial,
			position: { x: 0, y: 0, z: -2 },
			size: 1,
			depth: 2,
		}),
	]);

	const packets = prepareFramePackets(context, "main");
	assert.strictEqual(prepareFramePackets(context, "main"), packets);

	assert.equal(packets.all.length, 4);
	assert.equal(packets.all[0].submission.id, "scene-opaque");
	assert.equal(packets.all[1].submission.id, "scene-transparent");
	assert.ok(packets.all[2].submission.id.startsWith("particleMesh:particles:0:"));
	assert.ok(packets.all[3].submission.id.startsWith("particleMesh:particles:1:"));
	assert.equal(packets.opaque.length, 2);
	assert.equal(packets.transparent.length, 2);
	assert.equal(packets.shadowCasters.length, 2);
	assert.equal(packets.shadowTransmitters.length, 1);
	assert.equal(packets.reflective.length, 1);

	// Test view purpose isolation: planar reflection excludes mesh particles
	const planarPackets = prepareFramePackets(context, "planar-reflection");
	assert.notStrictEqual(planarPackets, packets);
	assert.equal(planarPackets.all.length, 2);
	assert.deepEqual(planarPackets.all.map((p) => p.submission.id), [
		"scene-opaque",
		"scene-transparent",
	]);

	// Test probe capture calculates depth for capture camera
	const captureContext = createContext({
		transient: createTransientStore(context.transient),
		camera: new Camera(),
	});
	const probePackets = prepareFramePackets(captureContext, "probe-capture");
	assert.notStrictEqual(probePackets, packets);
	assert.equal(probePackets.all.length, 4);
	assert.strictEqual(
		prepareFramePackets(captureContext, "probe-capture"),
		probePackets,
	);
}

testBaselinePacketSetCreation();
testFramePacketPreparationAndCaching();
console.log("Frame packet preparation tests passed");
