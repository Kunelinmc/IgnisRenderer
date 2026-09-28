import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { Quaternion } from "../../../src/maths/Quaternion.ts";
import { Vector3 } from "../../../src/maths/Vector3.ts";

function testNodeSetters() {
	const node = new Node();

	// Test setPosition with three numbers
	node.setPosition(1, 2, 3);
	assert.equal(node.position.x, 1);
	assert.equal(node.position.y, 2);
	assert.equal(node.position.z, 3);

	// Test setPosition with IVector3 object
	node.setPosition({ x: 4, y: 5, z: 6 });
	assert.equal(node.position.x, 4);
	assert.equal(node.position.y, 5);
	assert.equal(node.position.z, 6);

	// Test setScale with three numbers
	node.setScale(2, 3, 4);
	assert.equal(node.scale.x, 2);
	assert.equal(node.scale.y, 3);
	assert.equal(node.scale.z, 4);

	// Test setScale with IVector3 object
	node.setScale({ x: 0.5, y: 1.5, z: 2.5 });
	assert.equal(node.scale.x, 0.5);
	assert.equal(node.scale.y, 1.5);
	assert.equal(node.scale.z, 2.5);

	// Test setRotation with four numbers (quaternion components)
	node.setRotation(0, 0, 0, 1);
	assert.equal(node.quaternion.x, 0);
	assert.equal(node.quaternion.y, 0);
	assert.equal(node.quaternion.z, 0);
	assert.equal(node.quaternion.w, 1);

	// Test setRotation with Quaternion object
	const quat = new Quaternion(0, 1, 0, 0).normalize();
	node.setRotation(quat);
	assert.ok(Math.abs(node.quaternion.y - 1) < 1e-6);

	// Test method chaining
	const chainedNode = node.setPosition(10, 11, 12)
		.setRotation(0, 0, 1, 0)
		.setScale(5, 5, 5);

	assert.equal(chainedNode, node);
	assert.equal(node.position.x, 10);
	assert.equal(node.scale.z, 5);
	assert.ok(Math.abs(node.quaternion.z - 1) < 1e-6);
}

function recordMatrixWork(update) {
	const compose = Matrix4.compose;
	const multiply = Matrix4.multiply;
	const work = { composed: [], multiplied: [] };
	Matrix4.compose = (...args) => {
		work.composed.push(args[3]);
		return compose(...args);
	};
	Matrix4.multiply = (...args) => {
		work.multiplied.push(args[2]);
		return multiply(...args);
	};
	try {
		update();
	} finally {
		Matrix4.compose = compose;
		Matrix4.multiply = multiply;
	}
	return work;
}

function testUnchangedTransformsSkipMatrixWork() {
	const root = new Node({ position: { x: 10, y: 0, z: 0 } });
	const child = root.addChild(new Node({ position: { x: 2, y: 0, z: 0 } }));
	root.updateWorldMatrix();
	const revisions = [root.worldTransformRevision, child.worldTransformRevision];
	const changed = [];
	const work = recordMatrixWork(() => {
		for (let frame = 0; frame < 3; frame++) {
			root.updateLocalMatrix();
			root.updateWorldMatrix(undefined, changed);
		}
	});
	assert.deepEqual(work, { composed: [], multiplied: [] },
		"Unchanged transforms must not repeat composition or world multiplication");
	assert.equal(child.worldMatrix.elements[0][3], 12);
	assert.deepEqual([root.worldTransformRevision, child.worldTransformRevision], revisions);
	assert.deepEqual(changed, []);
}

function testDirectEditsBelowCleanAncestors() {
	const root = new Node({ position: { x: 10, y: 0, z: 0 } });
	const child = root.addChild(new Node({ position: { x: 2, y: 0, z: 0 } }));
	const grandchild = child.addChild(new Node({ position: { x: 3, y: 0, z: 0 } }));
	const sibling = root.addChild(new Node({ position: { x: 4, y: 0, z: 0 } }));
	root.updateWorldMatrix();
	const siblingRevision = sibling.worldTransformRevision;
	const childRevision = child.worldTransformRevision;
	const changed = [];
	child.position.x = 5;
	const work = recordMatrixWork(() => root.updateWorldMatrix(undefined, changed));
	assert.deepEqual(work.composed, [child.localMatrix]);
	assert.deepEqual(work.multiplied, [child.worldMatrix, grandchild.worldMatrix]);
	assert.equal(child.worldMatrix.elements[0][3], 15);
	assert.equal(grandchild.worldMatrix.elements[0][3], 18);
	assert.equal(sibling.worldMatrix.elements[0][3], 14);
	assert.equal(child.worldTransformRevision, childRevision + 1);
	assert.equal(sibling.worldTransformRevision, siblingRevision);
	assert.deepEqual(changed, [child, grandchild]);

	child.quaternion = new Quaternion(0, 0, 1, 0);
	child.scale.set(2, 3, 0);
	root.updateWorldMatrix();
	assert.equal(child.worldMatrix.elements[0][0], -2);
	assert.equal(child.worldMatrix.elements[1][1], -3);
	assert.equal(child.worldMatrix.elements[2][2], 0);
	assert.equal(grandchild.worldMatrix.elements[0][3], 9);

	child.position = new Vector3(1, 2, 3);
	child.scale = new Vector3(-1, 2, 3);
	child.quaternion.set(0, 0, 0, 1);
	root.updateWorldMatrix();
	assert.equal(grandchild.worldMatrix.elements[0][3], 8);
	assert.equal(grandchild.worldMatrix.elements[1][3], 2);
	assert.equal(grandchild.worldMatrix.elements[2][3], 3);
}

function testParentChangesReuseLocalMatrices() {
	const root = new Node();
	const child = root.addChild(new Node({ position: { x: 2, y: 0, z: 0 } }));
	const grandchild = child.addChild(new Node({ position: { x: 3, y: 0, z: 0 } }));
	root.updateWorldMatrix();
	const changed = [];
	root.setPosition(10, 0, 0);
	const work = recordMatrixWork(() => root.updateWorldMatrix(undefined, changed));
	assert.deepEqual(work.composed, [], "A setter's resolved local matrix must be reused");
	assert.deepEqual(work.multiplied, [child.worldMatrix, grandchild.worldMatrix]);
	assert.equal(grandchild.worldMatrix.elements[0][3], 15);
	assert.deepEqual(changed, [root, child, grandchild]);
}

function testEveryTransformComponentRemainsObservable() {
	const cases = [
		["position", "x", 2, 0, 3, 2],
		["position", "y", 3, 1, 3, 3],
		["position", "z", 4, 2, 3, 4],
		["scale", "x", 2, 0, 0, 2],
		["scale", "y", 3, 1, 1, 3],
		["scale", "z", 4, 2, 2, 4],
		["quaternion", "x", 1, 1, 1, -1],
		["quaternion", "y", 1, 0, 0, -1],
		["quaternion", "z", 1, 0, 0, -1],
		["quaternion", "w", 1, 1, 2, -2],
	];
	for (const [field, component, value, row, column, expected] of cases) {
		const node = new Node();
		if (field === "quaternion" && component === "w") node.quaternion.set(1, 0, 0, 0);
		node.updateWorldMatrix();
		node[field][component] = value;
		node.updateWorldMatrix();
		assert.equal(node.worldMatrix.elements[row][column], expected, `${field}.${component}`);
	}
}

function testExternalParentMatrixMutationAndRemoval() {
	const node = new Node({ position: { x: 2, y: 0, z: 0 } });
	const parent = Matrix4.fromTranslation([10, 0, 0]);
	node.updateWorldMatrix(parent);
	assert.equal(node.worldMatrix.elements[0][3], 12);
	const revision = node.worldTransformRevision;
	assert.deepEqual(recordMatrixWork(() => node.updateWorldMatrix(parent)), {
		composed: [], multiplied: [],
	});
	parent.elements[0][3] = 20;
	node.updateWorldMatrix(parent);
	assert.equal(node.worldMatrix.elements[0][3], 22);
	assert.equal(node.worldTransformRevision, revision + 1);
	node.updateWorldMatrix(Matrix4.fromTranslation([-5, 0, 0]));
	assert.equal(node.worldMatrix.elements[0][3], -3);
	node.updateWorldMatrix();
	assert.equal(node.worldMatrix.elements[0][3], 2);
}

function testReparentingAndDetachedClone() {
	const first = new Node({ position: { x: 10, y: 0, z: 0 } });
	const second = new Node({ position: { x: -10, y: 0, z: 0 } });
	const child = first.addChild(new Node());
	const grandchild = child.addChild(new Node({ position: { x: 3, y: 0, z: 0 } }));
	first.updateWorldMatrix();
	const clone = child.clone();
	clone.updateWorldMatrix();
	assert.equal(clone.worldMatrix.elements[0][3], 0);
	assert.equal(clone.children[0].worldMatrix.elements[0][3], 3);
	assert.equal(child.worldMatrix.elements[0][3], 10);

	second.addChild(child);
	second.updateWorldMatrix();
	assert.equal(child.worldMatrix.elements[0][3], -10);
	assert.equal(grandchild.worldMatrix.elements[0][3], -7);
	second.removeChild(child);
	child.updateWorldMatrix();
	assert.equal(grandchild.worldMatrix.elements[0][3], 3);
}

function testParentMatrixAliasingWorldOutput() {
	const cases = [
		["same matrix", (node) => node.worldMatrix],
		["same elements", (node) => new Matrix4(node.worldMatrix.elements)],
		["shared rows", (node) => new Matrix4([...node.worldMatrix.elements])],
		["one shared row", (node) => new Matrix4(
			node.worldMatrix.elements.map((row, index) => index === 0 ? row : [...row]),
		)],
	];
	for (const [name, createParent] of cases) {
		const node = new Node({ position: { x: 1, y: 0, z: 0 } });
		const parent = createParent(node);
		for (const expectedX of [2, 3, 4]) {
			const revision = node.worldTransformRevision;
			const changed = [];
			node.updateWorldMatrix(parent, changed);
			assert.equal(node.worldMatrix.elements[0][3], expectedX,
				`${name}: the next update must use the parent modified by the previous output`);
			assert.equal(node.worldTransformRevision, revision + 1, name);
			assert.deepEqual(changed, [node], name);
		}
	}
}

function testDerivedMatrixEditsAreRestored() {
	const node = new Node({ position: { x: 2, y: 3, z: 4 } });
	const parent = Matrix4.fromTranslation([10, 0, 0]);
	node.updateWorldMatrix(parent);
	const revisions = [node.localTransformRevision, node.worldTransformRevision];
	node.localMatrix.elements[0][3] = 99;
	node.worldMatrix.elements[1][3] = 99;
	const changed = [];
	node.updateWorldMatrix(parent, changed);
	assert.equal(node.localMatrix.elements[0][3], 2);
	assert.equal(node.worldMatrix.elements[0][3], 12);
	assert.equal(node.worldMatrix.elements[1][3], 3);
	assert.deepEqual([node.localTransformRevision, node.worldTransformRevision], revisions);
	assert.deepEqual(changed, []);

	node.localMatrix = Matrix4.identity();
	node.worldMatrix = Matrix4.identity();
	node.updateWorldMatrix(parent);
	assert.equal(node.localMatrix.elements[2][3], 4);
	assert.equal(node.worldMatrix.elements[0][3], 12);
	assert.equal(node.worldMatrix.elements[2][3], 4);
}

function testCustomLocalTransformOverrides() {
	class OffsetNode extends Node {
		offset = 0;
		updateLocalMatrix() {
			super.updateLocalMatrix();
			this.localMatrix.elements[0][3] += this.offset ?? 0;
		}
	}
	const node = new OffsetNode();
	const child = node.addChild(new Node({ position: { x: 1, y: 0, z: 0 } }));
	node.updateWorldMatrix();
	const revision = node.worldTransformRevision;
	node.offset = 10;
	const changed = [];
	node.updateWorldMatrix(undefined, changed);
	assert.equal(node.worldMatrix.elements[0][3], 10);
	assert.equal(child.worldMatrix.elements[0][3], 11);
	assert.equal(node.worldTransformRevision, revision + 1);
	assert.deepEqual(changed, [node, child]);
	changed.length = 0;
	node.updateWorldMatrix(undefined, changed);
	assert.equal(child.worldMatrix.elements[0][3], 11);
	assert.deepEqual(changed, []);
}

function testSignedZeroOutputRestoration() {
	const node = new Node();
	node.updateWorldMatrix();
	node.position.y = -0;
	node.updateWorldMatrix();
	assert.ok(Object.is(node.worldMatrix.elements[1][3], -0));
	node.localMatrix.elements[1][3] = 0;
	node.updateLocalMatrix();
	assert.ok(Object.is(node.localMatrix.elements[1][3], -0));
	node.position.y = 0;
	node.updateWorldMatrix();
	assert.ok(Object.is(node.worldMatrix.elements[1][3], 0));
}

testNodeSetters();
testUnchangedTransformsSkipMatrixWork();
testDirectEditsBelowCleanAncestors();
testParentChangesReuseLocalMatrices();
testEveryTransformComponentRemainsObservable();
testExternalParentMatrixMutationAndRemoval();
testReparentingAndDetachedClone();
testParentMatrixAliasingWorldOutput();
testDerivedMatrixEditsAreRestored();
testCustomLocalTransformOverrides();
testSignedZeroOutputRestoration();
console.log("All Node setters tests passed successfully!");
