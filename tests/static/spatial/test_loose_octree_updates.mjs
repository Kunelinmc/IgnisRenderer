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

function testSplitPlaneClusterUsesLooseChildren() {
	const mesh = MeshAsset.fromFaces([
		{
			material: new Material({ name: "OctreeSplitPlanes" }),
			vertices: [
				{ x: -0.5, y: -0.5, z: -0.5 },
				{ x: 0.5, y: 0.5, z: 0.5 },
				{ x: -0.5, y: 0.5, z: 0.5 },
			],
		},
	]);
	const anchors = [createInstance(mesh, -8, -8, -8), createInstance(mesh, 8, 8, 8)];
	const cluster = Array.from({ length: 64 }, (_, i) =>
		createInstance(
			mesh,
			(i % 4 - 1.5) * 0.1,
			(Math.floor(i / 4) % 4 - 1.5) * 0.1,
			(Math.floor(i / 16) - 1.5) * 0.1
		)
	);
	const instances = [...anchors, ...cluster];
	for (const [looseness, maxDepth] of [[1.5, 1], [1, 1], [1.5, 0]]) {
		const index = new LooseOctree(instances, { looseness, maxDepth, leafCapacity: 1 });
		const shouldDescend = looseness > 1 && maxDepth > 0;
		for (const instance of cluster) {
			assert.equal(
				index._entriesByMeshInstance.get(instance).node !== index._root,
				shouldDescend,
				"Split-plane objects should descend only when a loose child can contain them"
			);
		}
		const stored = [];
		const stack = [index._root];
		while (stack.length > 0) {
			const node = stack.pop();
			stored.push(...node.objects);
			for (const child of node.children ?? []) {
				if (child) stack.push(child);
			}
		}
		assert.equal(stored.length, instances.length);
		assert.equal(new Set(stored).size, instances.length);
		// The query straddles all tight child planes and must not duplicate hits.
		const hits = index.queryBounds({
			min: { x: -0.2, y: -0.2, z: -0.2 },
			max: { x: 0.2, y: 0.2, z: 0.2 },
		});
		assert.equal(hits.length, cluster.length);
		assert.deepEqual(new Set(hits), new Set(cluster));
	}

	const target = createInstance(mesh, 0.1, 0.1, 0.1);
	const index = new LooseOctree([...anchors, target], {
		looseness: 1.5,
		maxDepth: 1,
		leafCapacity: 1,
	});
	const node = index._entriesByMeshInstance.get(target).node;
	assert.notEqual(node, index._root);
	// Moving across tight split planes still fits the original loose child.
	target.position.set(-0.1, -0.1, -0.1);
	target.updateWorldMatrix();
	index.markDirty(target);
	assert.equal(index._entriesByMeshInstance.get(target).node, node);
	assert.deepEqual(
		index.queryBounds({
			min: { x: -0.5, y: -0.5, z: -0.5 },
			max: { x: -0.4, y: -0.4, z: -0.4 },
		}),
		[target]
	);
	const rayHits = index.queryRayDetailed(
		{ x: -0.45, y: -0.45, z: 2 },
		{ x: 0, y: 0, z: -1 },
		{ maxResults: 1 }
	);
	assert.equal(rayHits.length, 1);
	assert.equal(rayHits[0].meshInstance, target);

	const oversized = createInstance(mesh, 0, 0, 0);
	oversized.scale.set(20, 20, 20);
	oversized.updateWorldMatrix();
	index.upsert(oversized);
	assert.equal(index._entriesByMeshInstance.get(oversized).node, index._root);
	const allHits = index.queryBounds({
		min: { x: -10, y: -10, z: -10 },
		max: { x: 10, y: 10, z: 10 },
	});
	assert.equal(allHits.filter((instance) => instance === oversized).length, 1);
}

function testRedistributionUsesLooseChildren() {
	const mesh = createTriangleMesh(new Material({ name: "OctreeRedistribution" }));
	const resized = createInstance(mesh, 0, 0, 0);
	resized.scale.set(20, 20, 20);
	resized.updateWorldMatrix();
	const index = new LooseOctree([resized], {
		looseness: 1.5,
		maxDepth: 1,
		leafCapacity: 1,
	});
	assert.equal(index._entriesByMeshInstance.get(resized).node, index._root);
	resized.scale.set(1, 1, 1);
	resized.updateWorldMatrix();
	index.markDirty(resized);
	// Splitting the leaf must move its now-smaller resident into a loose child.
	const added = createInstance(mesh, 1, 1, 0);
	index.upsert(added);
	const entry = index._entriesByMeshInstance.get(resized);
	assert.notEqual(entry.node, index._root);
	assert.equal(entry.node.objects[entry.objectIndex], resized);
	assert.deepEqual(
		index.queryBounds({
			min: { x: -0.4, y: -0.4, z: -0.1 },
			max: { x: -0.3, y: -0.3, z: 0.1 },
		}),
		[resized]
	);
}

function run() {
	testSplitPlaneClusterUsesLooseChildren();
	testRedistributionUsesLooseChildren();
	testInPlaceDirtyUpdateWithinLooseNode();
	testDirtyUpdateReinsertsWhenLeavingLooseNode();
	testSwapRemoveMaintainsMovedEntryIndex();
	testRayValidationRunsOnEmptyIndex();
	console.log("Loose octree update tests passed");
}

run();
