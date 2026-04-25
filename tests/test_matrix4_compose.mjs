import assert from "node:assert/strict";
import { Node } from "../src/core/Node.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { Quaternion } from "../src/maths/Quaternion.ts";

function testComposeMatchesTRSMultiplication() {
	const position = { x: 3.5, y: -2.25, z: 8.75 };
	const quaternion = Quaternion.fromEuler(0.41, -0.27, 1.13).normalize();
	const scale = { x: 2, y: 3, z: 4 };

	const composed = Matrix4.compose(position, quaternion, scale);
	const expected = Matrix4.multiply(
		Matrix4.fromTranslation([position.x, position.y, position.z]),
		Matrix4.multiply(
			Matrix4.fromQuaternion([
				quaternion.x,
				quaternion.y,
				quaternion.z,
				quaternion.w,
			]),
			Matrix4.fromScale([scale.x, scale.y, scale.z])
		)
	);

	assert.deepEqual(composed.elements, expected.elements);
}

function testComposeSupportsOutMatrixReuse() {
	const position = { x: -1, y: 5, z: 0.25 };
	const quaternion = Quaternion.fromEuler(-0.5, 0.75, -1.25).normalize();
	const scale = { x: 0.5, y: 1.5, z: 2.5 };
	const reused = Matrix4.identity();
	const result = Matrix4.compose(position, quaternion, scale, reused);

	assert.equal(result, reused);
	assert.deepEqual(result.elements[3], [0, 0, 0, 1]);
}

function testNodeLocalMatrixUsesComposeContract() {
	const node = new Node({
		position: { x: 11, y: -7, z: 2 },
		quaternion: Quaternion.fromEuler(0.2, 0.4, -0.6),
		scale: { x: 1.2, y: 0.75, z: 3.1 },
	});
	node.updateLocalMatrix();

	const expected = Matrix4.compose(node.position, node.quaternion, node.scale);
	assert.deepEqual(node.localMatrix.elements, expected.elements);
}

function run() {
	testComposeMatchesTRSMultiplication();
	testComposeSupportsOutMatrixReuse();
	testNodeLocalMatrixUsesComposeContract();
	console.log("Matrix4 compose tests passed");
}

run();
