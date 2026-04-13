import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Scene } from "../src/core/Scene.ts";
import { PreparedSceneBuilder } from "../src/pipeline/PreparedSceneBuilder.ts";
import { Material } from "../src/materials/Material.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { MeshAsset } from "../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../src/meshes/MeshInstance.ts";

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
		animationSystem: {
			hasActiveActions() {
				return false;
			},
		},
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

	console.log("Render list builder tests passed");
}

run();
