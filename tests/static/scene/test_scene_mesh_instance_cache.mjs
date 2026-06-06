import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";

function createMesh() {
	return MeshAsset.fromFaces([
		{
			material: new Material(),
			vertices: [
				{ x: -1, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 1, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 0, y: 1, z: 0, normal: { x: 0, y: 0, z: 1 } },
			],
		},
	]);
}

function run() {
	const scene = new Scene();
	const mesh = createMesh();
	const firstMesh = scene.add(new MeshInstance({ mesh, name: "first" }));
	scene.updateWorldMatrices();

	const initial = scene.getMeshInstances();
	const repeated = scene.getMeshInstances();
	assert.strictEqual(
		repeated,
		initial,
		"Expected stable getMeshInstances cache between repeated reads"
	);

	firstMesh.position.x = 3;
	scene.updateWorldMatrices();
	const afterTransform = scene.getMeshInstances();
	assert.strictEqual(
		afterTransform,
		initial,
		"Expected transform-only updates to preserve mesh list cache identity"
	);

	scene.add(new MeshInstance({ mesh, name: "second" }));
	scene.updateWorldMatrices();
	const afterAdd = scene.getMeshInstances();
	assert.notStrictEqual(
		afterAdd,
		initial,
		"Expected structural add to refresh mesh list cache"
	);
	assert.equal(afterAdd.length, 2);

	console.log("Scene mesh instance cache tests passed");
}

run();
