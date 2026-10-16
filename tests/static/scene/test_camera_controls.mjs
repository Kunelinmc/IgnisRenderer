import assert from "node:assert/strict";

import { Camera } from "../../../src/cameras/Camera.ts";
import { FPSCamera } from "../../../src/cameras/FPSCamera.ts";
import { OrbitCamera } from "../../../src/cameras/OrbitCamera.ts";
import { OrthographicCamera } from "../../../src/cameras/OrthographicCamera.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";

function nearlyEqual(actual, expected, epsilon = 1e-9) {
	assert.ok(
		Math.abs(actual - expected) <= epsilon,
		`expected ${actual} to be within ${epsilon} of ${expected}`
	);
}

function testFPSCameraUsesCallerScaledRotationDeltas() {
	const camera = new FPSCamera();

	assert.equal("moveSpeed" in camera, false);
	assert.equal("lookSensitivity" in camera, false);

	camera.rotate(0.25, 0.125);

	nearlyEqual(camera.yaw, -0.25);
	nearlyEqual(camera.pitch, -0.125);
}

function testOrbitCameraUsesCallerScaledControlDeltas() {
	const camera = new OrbitCamera({ x: 0, y: 0, z: 0 }, 100);

	assert.equal("lookSensitivity" in camera, false);
	assert.equal("zoomSensitivity" in camera, false);

	camera.rotate(0.25, 0.125);
	camera.zoom(-10);

	nearlyEqual(camera.theta, -0.25);
	nearlyEqual(camera.phi, Math.PI / 3 - 0.125);
	nearlyEqual(camera.distance, 90);
}

function testCameraInversesRecomputeOnAccess() {
	const camera = new Camera();
	const inverse = Matrix4.inverse;
	let calls = 0;
	Matrix4.inverse = (...args) => { calls++; return inverse(...args); };
	try {
		const view = camera.inverseViewMatrix;
		assert.ok(view instanceof Matrix4);
		assert.equal(calls, 1, "only the requested inverse is computed");
		camera.viewMatrix = camera.viewMatrix.clone();
		assert.equal(camera.inverseViewMatrix, view);
		assert.equal(calls, 2, "every access recomputes even for equal source values");
		camera.viewMatrix.elements[0][3] = 7;
		assert.equal(camera.inverseViewMatrix, view);
		assert.equal(calls, 3);
		nearlyEqual(view.elements[0][3], -7);
		const projection = camera.inverseProjectionMatrix;
		assert.equal(calls, 4);
		camera.projectionMatrix.elements[0][0] *= 2;
		assert.equal(camera.inverseProjectionMatrix, projection);
		assert.equal(calls, 5, "each requested source is inverted again");
		camera.position.set(2, 3, 4);
		camera.updateMatrices();
		assert.equal(camera.inverseViewMatrix, view);
		nearlyEqual(view.elements[0][3], 2);
		nearlyEqual(view.elements[1][3], 3);
		nearlyEqual(view.elements[2][3], 4);
	} finally {
		Matrix4.inverse = inverse;
	}
}

function testCameraInverseFailuresAndRecovery() {
	for (const [source, result] of [
		["viewMatrix", "inverseViewMatrix"],
		["projectionMatrix", "inverseProjectionMatrix"],
	]) {
		const camera = new Camera();
		camera[source] = Matrix4.identity();
		const output = camera[result];
		assert.ok(output instanceof Matrix4);
		for (const invalid of [0, Infinity, NaN]) {
			camera[source].elements[0][0] = invalid;
			assert.throws(() => camera[result], /singular|finite/i);
		}
		camera[source] = new Matrix4([
			[1, 1e308, 0, 0], [0, 1, 1e308, 0], [0, 0, 1, 1e308], [0, 0, 0, 1],
		]);
		assert.throws(() => camera[result], /inverse.*finite/i);
		camera[source] = Matrix4.identity();
		assert.equal(camera[result], output);
		assert.deepEqual(output.elements, Matrix4.identity().elements);
		camera[source].elements[0][3] = 1e40;
		assert.equal(camera[result].elements[0][3], -1e40,
			"Camera math does not impose a backend's f32 range");
	}
}

function testCustomProjectionAndCloneInverses() {
	for (const camera of [new Camera(), new OrthographicCamera(10)]) {
		camera.position.set(3, 4, 5);
		camera.updateMatrices();
		camera.projectionMatrix.applyObliqueClipping({
			normal: { x: 0, y: -1, z: 0 }, constant: -1,
		});
		camera.viewProjectionMatrix = Matrix4.multiply(camera.projectionMatrix, camera.viewMatrix);
		for (const [source, result] of [
			["viewMatrix", "inverseViewMatrix"],
			["projectionMatrix", "inverseProjectionMatrix"],
		]) {
			const product = Matrix4.multiply(camera[source], camera[result]);
			for (let row = 0; row < 4; row++) {
				for (let column = 0; column < 4; column++) {
					nearlyEqual(product.elements[row][column], row === column ? 1 : 0);
				}
			}
		}
		const clone = camera.clone(false);
		assert.notEqual(clone.inverseViewMatrix, camera.inverseViewMatrix);
		assert.notEqual(clone.inverseProjectionMatrix, camera.inverseProjectionMatrix);
		assert.deepEqual(clone.inverseProjectionMatrix.elements, camera.inverseProjectionMatrix.elements);
		clone.viewMatrix.elements[0][3] = -12;
		nearlyEqual(clone.inverseViewMatrix.elements[0][3], 12);
		nearlyEqual(camera.inverseViewMatrix.elements[0][3], 3);
	}
}

function run() {
	testCameraInversesRecomputeOnAccess();
	testCameraInverseFailuresAndRecovery();
	testCustomProjectionAndCloneInverses();
	testFPSCameraUsesCallerScaledRotationDeltas();
	testOrbitCameraUsesCallerScaledControlDeltas();
	console.log("Camera control tests passed");
}

run();
