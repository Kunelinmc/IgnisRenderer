import assert from "node:assert/strict";
import { Matrix3 } from "../../../src/maths/Matrix3.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { Quaternion } from "../../../src/maths/Quaternion.ts";

function testFromQuaternionMatchesMatrix4RotationBlock() {
	const quaternion = Quaternion.fromEuler(0.41, -0.27, 1.13).normalize();
	const source = [
		quaternion.x,
		quaternion.y,
		quaternion.z,
		quaternion.w,
	];
	const matrix3 = Matrix3.fromQuaternion(source);
	const matrix4 = Matrix4.fromQuaternion(source);

	assertUpperLeft3x3ApproximatelyEqual(matrix3, matrix4);
}

function testFromQuaternionSupportsOutReuseAndObjectSource() {
	const quaternion = Quaternion.fromEuler(-0.5, 0.75, -1.25).normalize();
	const reused = Matrix3.identity();
	const result = Matrix3.fromQuaternion(quaternion, reused);

	assert.equal(result, reused);
	assertUpperLeft3x3ApproximatelyEqual(
		reused,
		Matrix4.fromQuaternion([
			quaternion.x,
			quaternion.y,
			quaternion.z,
			quaternion.w,
		])
	);
}

function assertUpperLeft3x3ApproximatelyEqual(matrix3, matrix4) {
	for (let row = 0; row < 3; row++) {
		for (let col = 0; col < 3; col++) {
			assert.ok(
				Math.abs(matrix3.elements[row][col] - matrix4.elements[row][col]) <
					1e-12,
				`Expected [${row}, ${col}] to match Matrix4 rotation block`
			);
		}
	}
}

function run() {
	testFromQuaternionMatchesMatrix4RotationBlock();
	testFromQuaternionSupportsOutReuseAndObjectSource();
	console.log("Matrix3 quaternion tests passed");
}

run();
