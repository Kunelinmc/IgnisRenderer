import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { BVH } from "../../../src/spatial/BVH.ts";
import { HybridSpatialIndex } from "../../../src/spatial/HybridSpatialIndex.ts";

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

	const mesh = createTriangleMesh(new Material({ name: "SceneSpatialMode" }));
	const visible = scene.add(new MeshInstance({ mesh, name: "visible" }));
	visible.position.z = -2;
	const dynamicMesh = scene.add(
		new MeshInstance({ mesh, name: "dynamic", skeleton: {} })
	);
	dynamicMesh.position.x = 0.7;
	dynamicMesh.position.z = -2;

	scene.updateWorldMatrices();
	camera.updateMatrices();

	assert.equal(scene.spatialIndexMode, "bvh");
	let hits = scene.queryMeshInstancesInFrustum(camera, [visible, dynamicMesh]);
	assert.equal(hits.length, 2);
	assert.ok(scene.spatial instanceof BVH);

	scene.setSpatialIndexMode("hybrid");
	assert.equal(scene.spatial, null);

	hits = scene.queryMeshInstancesInFrustum(camera, [visible, dynamicMesh]);
	assert.equal(hits.length, 2);
	assert.ok(scene.spatial instanceof HybridSpatialIndex);

	scene.spatialIndexMode = "bvh";
	hits = scene.queryMeshInstancesInFrustum(camera, [visible, dynamicMesh]);
	assert.equal(hits.length, 2);
	assert.ok(scene.spatial instanceof BVH);

	console.log("Scene spatial hybrid mode tests passed");
}

run();
