import assert from "node:assert/strict";
import { Scene } from "../src/core/Scene.ts";
import { Material } from "../src/materials/Material.ts";
import { MeshAsset } from "../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../src/meshes/MeshInstance.ts";
import { getDefaultIncrementalRegistry } from "../src/pipeline/incremental.ts";

function createMeshInstance(x) {
	const mesh = MeshAsset.fromFaces([
		{
			material: new Material(),
			vertices: [
				{ x: -1, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 1, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 0, y: 1, z: 0, normal: { x: 0, y: 0, z: 1 } },
			],
		},
	]);
	const meshInstance = new MeshInstance({ mesh });
	meshInstance.position.x = x;
	return meshInstance;
}

function run() {
	const scene = new Scene();
	const initialVersion = scene.version;
	const firstBounds = scene.getBounds();
	assert.equal(firstBounds.radius, 100);

	const left = createMeshInstance(-10);
	const right = createMeshInstance(10);
	scene.add(left);
	scene.add(right);
	scene.updateWorldMatrices();
	assert.ok(scene.version > initialVersion);

	const expanded = scene.getBounds();
	const repeated = scene.getBounds();
	assert.deepEqual(expanded, repeated);
	assert.ok(expanded.radius > 0);

	scene.remove(right);
	scene.updateWorldMatrices();
	const reduced = scene.getBounds();
	assert.ok(reduced.radius < expanded.radius);

	scene.clear();
	scene.updateWorldMatrices();
	const cleared = scene.getBounds();
	assert.equal(cleared.radius, 100);

	const registry = getDefaultIncrementalRegistry();
	registry.registerDirtyReason({ id: "custom-no-bounds-test" });
	registry.registerDirtyReason({
		id: "custom-bounds-test",
		invalidatesSceneBounds: true,
	});
	try {
		scene.getBounds();
		assert.equal(scene._boundsDirty, false);
		scene.invalidate("custom-no-bounds-test");
		assert.equal(scene._boundsDirty, false);
		scene.invalidate("custom-bounds-test");
		assert.equal(scene._boundsDirty, true);
	} finally {
		registry.unregisterDirtyReason("custom-no-bounds-test");
		registry.unregisterDirtyReason("custom-bounds-test");
	}

	console.log("Scene bounds tests passed");
}

run();
