import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { OrthographicCamera } from "../src/cameras/OrthographicCamera.ts";
import { screenToWorldRay } from "../src/interaction/screenToWorldRay.ts";

function assertAlmost(value, expected, epsilon = 1e-4) {
	assert.ok(
		Math.abs(value - expected) <= epsilon,
		`Expected ${value} to be within ${epsilon} of ${expected}`
	);
}

function testPerspectiveCenterRay() {
	const camera = new Camera();
	camera.updateMatrices();
	const ray = screenToWorldRay(camera, {
		screenX: 49.5,
		screenY: 49.5,
		viewportWidth: 100,
		viewportHeight: 100,
	});
	assertAlmost(ray.origin.x, 0);
	assertAlmost(ray.origin.y, 0);
	assertAlmost(ray.origin.z, 0);
	assertAlmost(ray.direction.x, 0);
	assertAlmost(ray.direction.y, 0);
	assertAlmost(ray.direction.z, -1);
}

function testOrthographicRayOriginShift() {
	const camera = new OrthographicCamera(10);
	camera.aspectRatio = 1;
	camera.near = 1;
	camera.updateMatrices();

	const centerRay = screenToWorldRay(camera, {
		screenX: 49.5,
		screenY: 49.5,
		viewportWidth: 100,
		viewportHeight: 100,
	});
	assertAlmost(centerRay.origin.x, 0, 0.05);
	assertAlmost(centerRay.origin.y, 0, 0.05);
	assertAlmost(centerRay.direction.z, -1);

	const rightRay = screenToWorldRay(camera, {
		screenX: 99.5,
		screenY: 49.5,
		viewportWidth: 100,
		viewportHeight: 100,
	});
	assert.ok(rightRay.origin.x > 4.9, `Expected right origin x > 4.9, got ${rightRay.origin.x}`);
}

function run() {
	testPerspectiveCenterRay();
	testOrthographicRayOriginShift();
	console.log("screenToWorldRay tests passed");
}

run();
