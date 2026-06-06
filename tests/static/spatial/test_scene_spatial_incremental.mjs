import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";

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
	const mesh = createTriangleMesh(new Material({ name: "SpatialIncremental" }));
	const first = scene.add(new MeshInstance({ mesh, name: "first" }));
	first.position.z = 0;

	scene.updateWorldMatrices();
	camera.updateMatrices();

	const initialHits = scene.queryMeshInstancesInFrustum(camera, [first]);
	assert.equal(initialHits.length, 1);
	assert.ok(scene.spatial, "Expected scene.spatial to be initialized");

	const spatial = scene.spatial;
	let rebuildCalls = 0;
	const originalRebuild = spatial.rebuild.bind(spatial);
	spatial.rebuild = (...args) => {
		rebuildCalls++;
		return originalRebuild(...args);
	};

	const stableHits = scene.queryMeshInstancesInFrustum(camera, [first]);
	assert.equal(stableHits.length, 1);
	assert.equal(
		rebuildCalls,
		0,
		"Expected unchanged frame to skip BVH full rebuild"
	);

	first.position.y = 240;
	scene.updateWorldMatrices();
	camera.updateMatrices();

	const movedHits = scene.queryMeshInstancesInFrustum(camera, [first]);
	assert.equal(movedHits.length, 0);
	assert.equal(
		rebuildCalls,
		0,
		"Expected transform-only update to use BVH refit path"
	);

	const second = scene.add(new MeshInstance({ mesh, name: "second" }));
	second.position.z = 0;
	scene.updateWorldMatrices();
	camera.updateMatrices();

	const structuralHits = scene.queryMeshInstancesInFrustum(camera, [first, second]);
	assert.equal(structuralHits.length, 1);
	assert.equal(structuralHits[0].id, second.id);
	assert.equal(
		rebuildCalls,
		1,
		"Expected structural changes to trigger one BVH rebuild"
	);

	console.log("Scene spatial incremental tests passed");
}

run();
