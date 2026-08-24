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

function testMultiPrimitivePacketsShareResolvedInstanceBindings() {
	const material = new Material();
	const createPrimitive = (id, offsetX) => ({
		id,
		geometryVersion: 0,
		topology: "triangle-list",
		material,
		geometry: {
			positions: new Float32Array([
				offsetX, 0, -2,
				offsetX + 1, 0, -2,
				offsetX, 1, -2,
			]),
			indices: new Uint32Array([0, 1, 2]),
		},
		boundingSphere: {
			center: { x: offsetX + 0.5, y: 0.5, z: -2 },
			radius: 1,
		},
		boundingBox: {
			min: { x: offsetX, y: 0, z: -2 },
			max: { x: offsetX + 1, y: 1, z: -2 },
		},
		visible: true,
		castShadows: true,
		receiveShadows: true,
	});
	const mesh = new MeshAsset([
		createPrimitive("primitive:a", -1),
		createPrimitive("primitive:b", 1),
	]);
	const scene = new Scene();
	const camera = scene.add(new Camera());
	const instance = scene.add(new MeshInstance({ mesh }));
	instance.renderLayers = 3;
	scene.updateWorldMatrices();
	camera.updateMatrices();

	const frame = PreparedSceneBuilder.build({
		scene,
		camera,
		hasActiveAnimations: false,
	});
	assert.equal(frame.opaquePackets.length, 2);
	const [first, second] = frame.opaquePackets;
	assert.strictEqual(first.submission.source, second.submission.source);
	assert.strictEqual(first.submission.instance, second.submission.instance);
	assert.notStrictEqual(
		first.submission.geometry.resourceKey,
		second.submission.geometry.resourceKey,
	);
	assert.equal(first.submission.instance.renderLayers, 3);
}

function run() {
	testMultiPrimitivePacketsShareResolvedInstanceBindings();
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

	assert.equal(frame.opaquePackets[0].submission.source.instanceId, nearOpaque.id);
	assert.equal(frame.opaquePackets[1].submission.source.instanceId, farOpaque.id);
	assert.equal(
		frame.transparentPackets[0].submission.source.instanceId,
		farTransparent.id,
	);
	assert.equal(
		frame.transparentPackets[1].submission.source.instanceId,
		transmissive.id,
	);
	assert.equal(
		frame.transparentPackets[2].submission.source.instanceId,
		nearTransparent.id,
	);
	assert.equal(
		frame.reflectivePackets[0].submission.source.instanceId,
		reflective.id,
	);

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
					return candidate.packet.submission.source.instanceId !== farOpaque.id;
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
			(packet) => packet.submission.source.instanceId === farOpaque.id
		),
		false
	);
	assert.equal(
		occludedFrame.shadowCasterPackets.some(
			(packet) => packet.submission.source.instanceId === farOpaque.id
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
			(packet) => packet.submission.source.instanceId === overrideVisible.id
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
			(packet) => packet.submission.source.instanceId === overrideVisible.id
		),
		true
	);
	assert.equal(
		rebuiltFrame.opaquePackets.some(
			(packet) => packet.submission.source.instanceId === mainVisible.id
		),
		false
	);
	const rebuiltOverridePacket = rebuiltFrame.opaquePackets.find(
		(packet) => packet.submission.source.instanceId === overrideVisible.id
	);
	assert.ok(rebuiltOverridePacket);
	assert.equal(rebuiltOverridePacket.submission.deformation.revision, 42);
	assert.equal(rebuiltOverridePacket.submission.worldBounds.center.x, 81.5);
	assert.equal(rebuiltOverridePacket.submission.worldBounds.radius, 0.75);
	assert.equal(
		(rebuiltOverridePacket.submission.passFlags & DRAW_PACKET_FLAG_SHADOW_RECEIVER) !== 0,
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
		frame.transparentPackets.map((packet) => packet.submission.source.instanceId),
		[far.id, near.id],
	);
}

run();
