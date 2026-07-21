import assert from "node:assert/strict";

import { FPSCamera } from "../../../src/cameras/FPSCamera.ts";
import { OrbitCamera } from "../../../src/cameras/OrbitCamera.ts";

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

function run() {
	testFPSCameraUsesCallerScaledRotationDeltas();
	testOrbitCameraUsesCallerScaledControlDeltas();
	console.log("Camera control tests passed");
}

run();
