import assert from "node:assert/strict";
import { OrthographicCamera } from "../../../src/cameras/OrthographicCamera.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { GLTFLoader } from "../../../src/loaders/GLTFLoader.ts";

function nearlyEqual(a, b, epsilon = 1e-6) {
	return Math.abs(a - b) <= epsilon;
}

function assertMatrixNearlyEqual(left, right, epsilon = 1e-6) {
	for (let r = 0; r < 4; r++) {
		for (let c = 0; c < 4; c++) {
			assert.ok(
				nearlyEqual(left[r][c], right[r][c], epsilon),
				`matrix mismatch at [${r}, ${c}] -> ${left[r][c]} vs ${right[r][c]}`
			);
		}
	}
}

function testDefaultBoundsAreSizeDerived() {
	const camera = new OrthographicCamera(40);
	camera.aspectRatio = 2;

	const bounds = camera.getBounds();
	assert.deepEqual(bounds, {
		left: -40,
		right: 40,
		bottom: -20,
		top: 20,
	});
}

function testExplicitBoundsOverrideSizeBasedPlanes() {
	const camera = new OrthographicCamera(40);
	camera.aspectRatio = 2;
	camera.setBounds(-2, 6, -1, 3);

	const bounds = camera.getBounds();
	assert.deepEqual(bounds, {
		left: -2,
		right: 6,
		bottom: -1,
		top: 3,
	});

	const projection = camera.calculateProjectionMatrix();
	const expected = Matrix4.ortho(-2, 6, -1, 3, camera.near, camera.far);
	assertMatrixNearlyEqual(projection.elements, expected.elements);
}

function testClearBoundsReturnsToSizeMode() {
	const camera = new OrthographicCamera(60);
	camera.aspectRatio = 1.5;
	camera.setBounds(-4, 8, -2, 6);
	camera.clearBounds();

	const bounds = camera.getBounds();
	assert.deepEqual(bounds, {
		left: -45,
		right: 45,
		bottom: -30,
		top: 30,
	});
}

function testInvalidBoundsThrow() {
	const camera = new OrthographicCamera(20);
	camera.setBounds(1, 1, -5, 5);
	assert.throws(() => camera.calculateProjectionMatrix(), /right != left/);
}

function testGLTFOrthographicCameraUsesXmagAndYmag() {
	const loader = new GLTFLoader();
	const camera = loader.parseCamera({
		type: "orthographic",
		orthographic: {
			xmag: 3,
			ymag: 2,
			znear: 0.5,
			zfar: 250,
		},
	});
	const ortho = camera;

	assert.equal(ortho.type, "orthographic");
	assert.equal(ortho.near, 0.5);
	assert.equal(ortho.far, 250);
	assert.deepEqual(ortho.getBounds(), {
		left: -3,
		right: 3,
		bottom: -2,
		top: 2,
	});
}

function run() {
	testDefaultBoundsAreSizeDerived();
	testExplicitBoundsOverrideSizeBasedPlanes();
	testClearBoundsReturnsToSizeMode();
	testInvalidBoundsThrow();
	testGLTFOrthographicCameraUsesXmagAndYmag();
	console.log("Orthographic camera bounds tests passed");
}

run();
