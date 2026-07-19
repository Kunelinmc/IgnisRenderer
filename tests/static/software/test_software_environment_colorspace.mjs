import assert from "node:assert/strict";
import { CameraType } from "../../../src/cameras/Camera.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { sRGBToLinear } from "../../../src/maths/Common.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { EnvironmentBackgroundRenderer } from "../../../src/backends/software/EnvironmentRenderer.ts";

const DEFAULT_BACKGROUND_OPTIONS = {
	strength: 1,
	tintLinear: { r: 1, g: 1, b: 1 },
	exposure: 1,
};

function createCamera() {
	return {
		viewMatrix: Matrix4.identity(),
		type: CameraType.Perspective,
		fov: 60,
		aspectRatio: 1,
	};
}

function testEnvironmentRendererDecodesSRGBToLinear() {
	const environment = new Texture(
		new Uint8ClampedArray([128, 64, 32, 255]),
		1,
		1,
		"sRGB"
	);
	const pixels = new Uint8ClampedArray(4);

	EnvironmentBackgroundRenderer.render(
		environment,
		DEFAULT_BACKGROUND_OPTIONS,
		pixels,
		createCamera(),
		1,
		1
	);

	const expectedR = Math.round(sRGBToLinear(128 / 255) * 255);
	const expectedG = Math.round(sRGBToLinear(64 / 255) * 255);
	const expectedB = Math.round(sRGBToLinear(32 / 255) * 255);
	assert.ok(Math.abs(pixels[0] - expectedR) <= 1);
	assert.ok(Math.abs(pixels[1] - expectedG) <= 1);
	assert.ok(Math.abs(pixels[2] - expectedB) <= 1);
	assert.equal(pixels[3], 255);
}

function testEnvironmentRendererPreservesLinearTextureValues() {
	const environment = new Texture(
		new Uint8ClampedArray([128, 64, 32, 255]),
		1,
		1,
		"Linear"
	);
	const pixels = new Uint8ClampedArray(4);

	EnvironmentBackgroundRenderer.render(
		environment,
		DEFAULT_BACKGROUND_OPTIONS,
		pixels,
		createCamera(),
		1,
		1
	);

	assert.deepEqual(Array.from(pixels), [128, 64, 32, 255]);
}

function testEnvironmentRendererPreservesHDRTextureValues() {
	const environment = new Texture(new Float32Array([1, 0.5, 0.25, 1]), 1, 1, "HDR");
	const pixels = new Uint8ClampedArray(4);

	EnvironmentBackgroundRenderer.render(
		environment,
		DEFAULT_BACKGROUND_OPTIONS,
		pixels,
		createCamera(),
		1,
		1
	);

	assert.deepEqual(Array.from(pixels), [255, 128, 64, 255]);
}

function run() {
	testEnvironmentRendererDecodesSRGBToLinear();
	testEnvironmentRendererPreservesLinearTextureValues();
	testEnvironmentRendererPreservesHDRTextureValues();
	console.log("Software environment color-space tests passed");
}

run();
