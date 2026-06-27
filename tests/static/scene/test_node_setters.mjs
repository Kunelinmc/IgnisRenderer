import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { Quaternion } from "../../../src/maths/Quaternion.ts";
import { Vector3 } from "../../../src/maths/Vector3";

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

testNodeSetters();
console.log("All Node setters tests passed successfully!");
