import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Material } from "../src/materials/Material.ts";
import { MeshAsset } from "../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../src/meshes/MeshInstance.ts";
import { BVH } from "../src/spatial/BVH.ts";

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
	const camera = new Camera();
	camera.updateMatrices();

	const mesh = createTriangleMesh(new Material({ name: "BVHTest" }));
	const visibleMesh = new MeshInstance({ mesh, name: "visibleMesh" });
	visibleMesh.position.z = -3;
	visibleMesh.updateWorldMatrix();

	const culledMesh = new MeshInstance({ mesh, name: "culledMesh" });
	culledMesh.position.y = 120;
	culledMesh.position.z = -3;
	culledMesh.updateWorldMatrix();

	const bvh = new BVH([visibleMesh, culledMesh]);
	const hits = bvh.queryFrustum(camera.frustum);

	assert.equal(hits.length, 1);
	assert.equal(hits[0].id, visibleMesh.id);
	assert.ok(bvh.root);
	console.log("Spatial BVH frustum tests passed");
}

run();
