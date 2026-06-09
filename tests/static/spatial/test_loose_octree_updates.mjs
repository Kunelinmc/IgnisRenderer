import assert from "node:assert/strict";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { LooseOctree } from "../../../src/spatial/LooseOctree.ts";

function createTriangleMesh(material) {
	return MeshAsset.fromFaces([
		{
			material,
			vertices: [
				{
					x: -0.5,
					y: -0.5,
					z: 0,
					u: 0,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 0.5,
					y: -0.5,
					z: 0,
					u: 1,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 0,
					y: 0.5,
					z: 0,
					u: 0.5,
					v: 1,
					normal: { x: 0, y: 0, z: 1 },
				},
			],
		},
	]);
}

function createInstance(mesh, x, y, z) {
	const instance = new MeshInstance({ mesh });
	instance.position.set(x, y, z);
	instance.updateWorldMatrix();
	return instance;
}

function queryUnitBounds(index, centerX) {
	return index.queryBounds({
		min: { x: centerX - 1, y: -1, z: -1 },
		max: { x: centerX + 1, y: 1, z: 1 },
	});
}

function testInPlaceDirtyUpdateWithinLooseNode() {
	const mesh = createTriangleMesh(new Material({ name: "OctreeInPlace" }));
	const instance = createInstance(mesh, 0, 0, 0);
	const index = new LooseOctree([instance], {
		leafCapacity: 1,
		looseness: 2,
	});

	const entryBefore = index._entriesByMeshInstance.get(instance);
	assert.ok(entryBefore);
	instance.position.x = 0.1;
	instance.updateWorldMatrix();
	index.markDirty(instance);

	const entryAfter = index._entriesByMeshInstance.get(instance);
	assert.ok(entryAfter);
	assert.equal(entryAfter.node, entryBefore.node);
	assert.equal(queryUnitBounds(index, 0.1).includes(instance), true);
	assert.equal(queryUnitBounds(index, 50).includes(instance), false);
}

function testDirtyUpdateReinsertsWhenLeavingLooseNode() {
	const mesh = createTriangleMesh(new Material({ name: "OctreeReinsert" }));
	const instance = createInstance(mesh, 0, 0, 0);
	const index = new LooseOctree([instance], {
		leafCapacity: 1,
		looseness: 1.25,
	});

	instance.position.x = 50;
	instance.updateWorldMatrix();
	index.markDirty(instance);

	assert.equal(queryUnitBounds(index, 0).includes(instance), false);
	assert.equal(queryUnitBounds(index, 50).includes(instance), true);
}

function testSwapRemoveMaintainsMovedEntryIndex() {
	const mesh = createTriangleMesh(new Material({ name: "OctreeSwapRemove" }));
	const first = createInstance(mesh, 0, 0, 0);
	const second = createInstance(mesh, 0.2, 0, 0);
	const index = new LooseOctree([first, second], {
		leafCapacity: 4,
		maxDepth: 0,
	});

	assert.equal(index.remove(first), true);
	const secondEntry = index._entriesByMeshInstance.get(second);
	assert.ok(secondEntry);
	assert.equal(secondEntry.objectIndex, 0);
	assert.equal(secondEntry.node.objects[0], second);
	assert.equal(queryUnitBounds(index, 0.2).includes(second), true);
}

function testRayValidationRunsOnEmptyIndex() {
	assert.throws(
		() =>
			new LooseOctree([]).queryRayDetailedInto(
				{ x: 0, y: 0, z: 0 },
				{ x: 0, y: 0, z: 0 },
				[]
			),
		/direction must be non-zero/
	);
}

function run() {
	testInPlaceDirtyUpdateWithinLooseNode();
	testDirtyUpdateReinsertsWhenLeavingLooseNode();
	testSwapRemoveMaintainsMovedEntryIndex();
	testRayValidationRunsOnEmptyIndex();
	console.log("Loose octree update tests passed");
}

run();
