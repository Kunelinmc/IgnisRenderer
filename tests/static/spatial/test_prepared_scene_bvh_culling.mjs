import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { PreparedSceneBuilder } from "../../../src/pipeline/PreparedSceneBuilder.ts";

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

	const mesh = createTriangleMesh(
		new Material({
			name: "Opaque",
			alphaMode: "OPAQUE",
		})
	);
	const visibleMesh = scene.add(new MeshInstance({ mesh, name: "visibleMesh" }));
	visibleMesh.position.z = 0;

	const culledMesh = scene.add(new MeshInstance({ mesh, name: "culledMesh" }));
	culledMesh.position.y = 250;
	culledMesh.position.z = 0;

	scene.updateWorldMatrices();
	camera.updateMatrices();

	const originalIsSphereInFrustum = camera.isSphereInFrustum.bind(camera);
	camera.isSphereInFrustum = () => {
		throw new Error("PreparedSceneBuilder should not call camera.isSphereInFrustum");
	};

	let frame;
	try {
		frame = PreparedSceneBuilder.build({
			scene,
			camera,
			shadowMaps: new Map(),
			hasActiveAnimations: false,
		});
	} finally {
		camera.isSphereInFrustum = originalIsSphereInFrustum;
	}

	assert.equal(frame.opaquePackets.length, 1);
	assert.equal(frame.opaquePackets[0].meshInstance.id, visibleMesh.id);
	assert.equal(frame.shadowCasterPackets.length, 2);
	assert.ok(
		frame.shadowCasterPackets.some(
			(packet) => packet.meshInstance.id === culledMesh.id
		)
	);
	assert.ok(scene.spatial);
	console.log("Prepared scene BVH culling tests passed");
}

run();
