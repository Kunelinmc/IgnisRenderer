import assert from "node:assert/strict";
import { Node } from "../src/core/Node.ts";
import { Material } from "../src/materials/Material.ts";
import { MeshAsset } from "../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../src/meshes/MeshInstance.ts";

function createUnitBoundsMesh() {
	return MeshAsset.fromFaces([
		{
			material: new Material(),
			vertices: [
				{ x: -1, y: -1, z: -1, normal: { x: 0, y: 0, z: 1 } },
				{ x: 1, y: 1, z: 1, normal: { x: 0, y: 0, z: 1 } },
				{ x: 1, y: -1, z: 1, normal: { x: 0, y: 0, z: 1 } },
			],
		},
	]);
}

function testWorldBoundsOutBoxForHierarchy() {
	const mesh = createUnitBoundsMesh();
	const root = new Node();
	root.position.copy({ x: 10, y: 0, z: 0 });

	const left = new MeshInstance({ mesh });
	left.position.copy({ x: -2, y: 0, z: 0 });
	const right = new MeshInstance({ mesh });
	right.position.copy({ x: 1, y: 0, z: 0 });

	root.addChild(left);
	root.addChild(right);
	root.updateWorldMatrix();

	const out = {
		min: { x: 999, y: 999, z: 999 },
		max: { x: -999, y: -999, z: -999 },
	};
	const minRef = out.min;
	const maxRef = out.max;
	const result = root.getWorldBoundingBox(out);

	assert.equal(result, out);
	assert.equal(out.min, minRef);
	assert.equal(out.max, maxRef);
	assert.deepEqual(out, {
		min: { x: 7, y: -1, z: -1 },
		max: { x: 12, y: 1, z: 1 },
	});

	left.position.x = -5;
	root.updateWorldMatrix();
	const updated = root.getWorldBoundingBox(out);
	assert.equal(updated, out);
	assert.equal(out.min, minRef);
	assert.equal(out.max, maxRef);
	assert.deepEqual(out, {
		min: { x: 4, y: -1, z: -1 },
		max: { x: 12, y: 1, z: 1 },
	});
}

function testWorldBoundsOutBoxWithoutGeometry() {
	const node = new Node();
	node.position.copy({ x: 2, y: -3, z: 5 });
	node.updateWorldMatrix();

	const out = {
		min: { x: 0, y: 0, z: 0 },
		max: { x: 0, y: 0, z: 0 },
	};

	const result = node.getWorldBoundingBox(out);
	assert.equal(result, out);
	assert.deepEqual(out, {
		min: { x: 2, y: -3, z: 5 },
		max: { x: 2, y: -3, z: 5 },
	});
}

function run() {
	testWorldBoundsOutBoxForHierarchy();
	testWorldBoundsOutBoxWithoutGeometry();
	console.log("Node world bounds out-parameter tests passed");
}

run();
