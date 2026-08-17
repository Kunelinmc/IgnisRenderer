import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { Quaternion } from "../../../src/maths/Quaternion.ts";

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

function testInverseReturnsMultiplicativeIdentity() {
	const matrix = Matrix4.multiply(
		Matrix4.fromTranslation([2, -3, 4]),
		Matrix4.multiply(
			Matrix4.fromQuaternion([
				0.10259783520851541,
				0.20519567041703082,
				-0.3077935056255462,
				0.9233805168766387,
			]),
			Matrix4.fromScale([2, 3, 4])
		)
	);
	const inverse = Matrix4.inverse(matrix);

	assert.ok(inverse);
	assertMatrixApproximatelyIdentity(Matrix4.multiply(matrix, inverse));
	assertMatrixApproximatelyIdentity(Matrix4.multiply(inverse, matrix));
}

function testInverseReturnsNullForSingularMatrix() {
	const matrix = Matrix4.fromScale([1, 0, 1]);
	assert.equal(Matrix4.inverse(matrix), null);
}

function testColumnMajorArrayPacking() {
	const matrix = new Matrix4([
		[1, 2, 3, 4],
		[5, 6, 7, 8],
		[9, 10, 11, 12],
		[13, 14, 15, 16],
	]);
	const expected = [
		1, 5, 9, 13,
		2, 6, 10, 14,
		3, 7, 11, 15,
		4, 8, 12, 16,
	];
	assert.deepEqual(Array.from(Matrix4.toColumnMajorArray(matrix)), expected);
	assert.deepEqual(
		Array.from(Matrix4.toColumnMajorArray({ elements: matrix.elements })),
		expected
	);
}

function testFiniteMatrixValidation() {
	const matrix = Matrix4.identity();
	assert.equal(Matrix4.isFinite(matrix), true);
	matrix.elements[2][1] = Number.NaN;
	assert.equal(Matrix4.isFinite(matrix), false);
	matrix.elements[2][1] = Number.POSITIVE_INFINITY;
	assert.equal(Matrix4.isFinite(matrix), false);
}

function assertMatrixApproximatelyIdentity(matrix) {
	const identity = Matrix4.identity().elements;
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			assert.ok(
				Math.abs(matrix.elements[row][col] - identity[row][col]) < 1e-9,
				`Expected identity at [${row}, ${col}], got ${matrix.elements[row][col]}`
			);
		}
	}
}

function run() {
	testComposeMatchesTRSMultiplication();
	testComposeSupportsOutMatrixReuse();
	testNodeLocalMatrixUsesComposeContract();
	testInverseReturnsMultiplicativeIdentity();
	testInverseReturnsNullForSingularMatrix();
	testColumnMajorArrayPacking();
	testFiniteMatrixValidation();
	console.log("Matrix4 compose tests passed");
}

run();
