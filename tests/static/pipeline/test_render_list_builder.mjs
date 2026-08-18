import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { PreparedSceneBuilder } from "../../../src/pipeline/PreparedSceneBuilder.ts";
import { Material } from "../../../src/materials/Material.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { normalizeOcclusionCullingOptions } from "../../../src/pipeline/OcclusionCulling.ts";
import { DRAW_PACKET_FLAG_SHADOW_RECEIVER } from "../../../src/pipeline/types.ts";

function createTriangleMesh(material) {
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

function run() {
	const scene = new Scene();
	const camera = new Camera();
	camera.position.set(0, 0, 5);
	scene.add(camera);
	scene.updateWorldMatrices();
	camera.updateMatrices();

	const opaqueMaterial = new Material({
		name: "Opaque",
		alphaMode: "OPAQUE",
	});
	const transparentMaterial = new Material({
		name: "Transparent",
		alphaMode: "BLEND",
	});
	const reflectiveMaterial = new Material({
		name: "Reflective",
		alphaMode: "OPAQUE",
		reflectivity: 0.7,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	});
	const transmissiveMaterial = new PBRMaterial({
		name: "Transmissive",
		albedo: { r: 220, g: 220, b: 220 },
		roughness: 0.05,
		metalness: 0,
		transmissionFactor: 1,
		ior: 1.52,
	});

	const opaqueMesh = createTriangleMesh(opaqueMaterial);
	const transparentMesh = createTriangleMesh(transparentMaterial);
	const reflectiveMesh = createTriangleMesh(reflectiveMaterial);
	const transmissiveMesh = createTriangleMesh(transmissiveMaterial);

	const nearOpaque = scene.add(
		new MeshInstance({ mesh: opaqueMesh, name: "nearOpaque" })
	);
	nearOpaque.position.z = 0;
	const farOpaque = scene.add(
		new MeshInstance({ mesh: opaqueMesh, name: "farOpaque" })
	);
	farOpaque.position.z = -4;
	const nearTransparent = scene.add(
		new MeshInstance({ mesh: transparentMesh, name: "nearTransparent" })
	);
	nearTransparent.position.z = -1;
	const farTransparent = scene.add(
		new MeshInstance({ mesh: transparentMesh, name: "farTransparent" })
	);
	farTransparent.position.z = -6;
	const transmissive = scene.add(
		new MeshInstance({ mesh: transmissiveMesh, name: "transmissive" })
	);
	transmissive.position.z = -3;
	const reflective = scene.add(
		new MeshInstance({ mesh: reflectiveMesh, name: "reflective" })
	);
	reflective.position.x = 3;

	scene.updateWorldMatrices();
	camera.updateMatrices();

	const frame = PreparedSceneBuilder.build({
		scene,
		camera,
		shadowMaps: new Map(),
		hasActiveAnimations: false,
	});

	assert.equal(frame.opaquePackets.length, 3);
	assert.equal(frame.transparentPackets.length, 3);
	assert.equal(frame.reflectivePackets.length, 1);
	assert.equal(frame.shadowCasterPackets.length, 3);
	assert.equal(frame.shadowTransmitterPackets.length, 3);

	assert.equal(frame.opaquePackets[0].meshInstance.id, nearOpaque.id);
	assert.equal(frame.opaquePackets[1].meshInstance.id, farOpaque.id);
	assert.equal(frame.transparentPackets[0].meshInstance.id, farTransparent.id);
	assert.equal(frame.transparentPackets[1].meshInstance.id, transmissive.id);
	assert.equal(frame.transparentPackets[2].meshInstance.id, nearTransparent.id);
	assert.equal(frame.reflectivePackets[0].meshInstance.id, reflective.id);

		const occludedFrame = PreparedSceneBuilder.build(
		{
			scene,
			camera,
			shadowMaps: new Map(),
			hasActiveAnimations: false,
		},
		{
			viewportWidth: 800,
			viewportHeight: 600,
			occlusionCullingOptions: normalizeOcclusionCullingOptions({}),
			occlusionVisibilityProvider: {
				sourceFrameIndex: 4,
				isPacketVisible(candidate) {
					return candidate.packet.meshInstance.id !== farOpaque.id;
				},
			},
		}
	);

	assert.equal(occludedFrame.occlusion.enabled, true);
	assert.equal(occludedFrame.occlusion.sourceFrameIndex, 4);
	assert.equal(occludedFrame.occlusion.culledPacketIds.length, 1);
	assert.equal(occludedFrame.occlusion.culledPacketIds[0].startsWith(farOpaque.id), true);
	assert.equal(
		occludedFrame.opaquePackets.some(
			(packet) => packet.meshInstance.id === farOpaque.id
		),
		false
	);
	assert.equal(
		occludedFrame.shadowCasterPackets.some(
			(packet) => packet.meshInstance.id === farOpaque.id
		),
		true
	);
	assert.equal(occludedFrame.transparentPackets.length, 3);

	testRebuildForCameraUsesOverrideFrustum();
	testTransparentSortUsesDeformedCenter();

	console.log("Render list builder tests passed");
}

function testRebuildForCameraUsesOverrideFrustum() {
	const scene = new Scene();
	const mainCamera = new Camera();
	mainCamera.position.set(0, 0, 5);
	scene.add(mainCamera);

	const material = new Material({
		name: "OverrideCameraOpaque",
		alphaMode: "OPAQUE",
	});
	const mesh = createTriangleMesh(material);
	const mainVisible = scene.add(
		new MeshInstance({ mesh, name: "mainVisible" })
	);
	mainVisible.position.z = 0;
	const overrideVisible = scene.add(
		new MeshInstance({ mesh, name: "overrideVisible" })
	);
	overrideVisible.position.x = 80;
	overrideVisible.position.z = 0;

	scene.updateWorldMatrices();
	mainCamera.updateMatrices();
	const overridePacketId = `${overrideVisible.id}:${mesh.primitives[0].id}`;
	const deformationStates = new Map([
		[
			overridePacketId,
			{
				packetId: overridePacketId,
				revision: 42,
				localBounds: {
					center: { x: 1.5, y: 0.5, z: 0 },
					radius: 0.75,
				},
			},
		],
	]);

	const mainFrame = PreparedSceneBuilder.build({
		scene,
		camera: mainCamera,
		shadowMaps: new Map(),
		hasActiveAnimations: false,
		deformationStates,
	});

	assert.equal(
		mainFrame.opaquePackets.some(
			(packet) => packet.meshInstance.id === overrideVisible.id
		),
		false
	);
	assert.equal(mainFrame.meshInstances.length, 2);

	const overrideCamera = new Camera();
	overrideCamera.position.set(80, 0, 5);
	overrideCamera.updateMatrices();
	const rebuiltFrame = PreparedSceneBuilder.rebuildForCamera(
		mainFrame,
		overrideCamera
	);

	assert.equal(
		rebuiltFrame.opaquePackets.some(
			(packet) => packet.meshInstance.id === overrideVisible.id
		),
		true
	);
	assert.equal(
		rebuiltFrame.opaquePackets.some(
			(packet) => packet.meshInstance.id === mainVisible.id
		),
		false
	);
	const rebuiltOverridePacket = rebuiltFrame.opaquePackets.find(
		(packet) => packet.meshInstance.id === overrideVisible.id
	);
	assert.ok(rebuiltOverridePacket);
	assert.equal(rebuiltOverridePacket.deformationRevision, 42);
	assert.equal(rebuiltOverridePacket.worldBounds.center.x, 81.5);
	assert.equal(rebuiltOverridePacket.worldBounds.radius, 0.75);
	assert.equal(
		(rebuiltOverridePacket.passFlags & DRAW_PACKET_FLAG_SHADOW_RECEIVER) !== 0,
		true
	);
	assert.strictEqual(rebuiltFrame.environment, mainFrame.environment);
	assert.strictEqual(
		rebuiltFrame.shadowCasterPackets,
		mainFrame.shadowCasterPackets
	);
	assert.strictEqual(
		rebuiltFrame.shadowTransmitterPackets,
		mainFrame.shadowTransmitterPackets
	);
}

function testTransparentSortUsesDeformedCenter() {
	const scene = new Scene();
	const camera = new Camera();
	camera.position.set(0, 0, 5);
	scene.add(camera);
	const material = new Material({
		name: "DeformedTransparent",
		alphaMode: "BLEND",
	});
	const mesh = createTriangleMesh(material);
	const near = scene.add(new MeshInstance({ mesh, name: "deformedNear" }));
	const far = scene.add(new MeshInstance({ mesh, name: "deformedFar" }));
	scene.updateWorldMatrices();
	camera.updateMatrices();

	const primitiveId = mesh.primitives[0].id;
	const deformationStates = new Map([
		[
			`${near.id}:${primitiveId}`,
			{
				packetId: `${near.id}:${primitiveId}`,
				revision: 1,
				localBounds: { center: { x: 0, y: 0, z: 0 }, radius: 1 },
			},
		],
		[
			`${far.id}:${primitiveId}`,
			{
				packetId: `${far.id}:${primitiveId}`,
				revision: 1,
				localBounds: { center: { x: 0, y: 0, z: -4 }, radius: 1 },
			},
		],
	]);

	const frame = PreparedSceneBuilder.build({
		scene,
		camera,
		hasActiveAnimations: true,
		deformationStates,
	});
	assert.deepEqual(
		frame.transparentPackets.map((packet) => packet.meshInstance.id),
		[far.id, near.id],
	);
}

run();
