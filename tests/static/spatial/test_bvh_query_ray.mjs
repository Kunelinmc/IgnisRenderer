import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { MeshFactory } from "../../../src/meshes/MeshFactory.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { BVH } from "../../../src/spatial/BVH.ts";

function run() {
	const scene = new Scene();
	const material = new PBRMaterial();
	const nearBox = MeshFactory.createBox(
		{ x: 0, y: 0, z: -5 },
		2,
		2,
		2,
		material
	);
	const farBox = MeshFactory.createBox(
		{ x: 0, y: 0, z: -10 },
		2,
		2,
		2,
		material
	);
	scene.add(nearBox);
	scene.add(farBox);
	scene.updateWorldMatrices();

	const bvh = new BVH(scene.getMeshInstances());
	const hits = bvh.queryRayDetailed(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: -1 },
		{ maxDistance: 100 }
	);
	assert.ok(hits.length >= 2, `Expected at least 2 hits, got ${hits.length}`);
	assert.equal(hits[0].meshInstance, nearBox);
	assert.equal(hits[1].meshInstance, farBox);
	assert.ok(hits[0].distance < hits[1].distance);

	const clampedHits = bvh.queryRayDetailed(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: -1 },
		{ maxDistance: 3 }
	);
	assert.equal(clampedHits.length, 0);

	console.log("BVH ray query tests passed");
}

run();
