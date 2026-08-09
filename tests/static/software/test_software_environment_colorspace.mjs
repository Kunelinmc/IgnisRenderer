import assert from "node:assert/strict";
import { CameraType } from "../../../src/cameras/Camera.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { sRGBToLinear } from "../../../src/maths/Common.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { SkyboxRenderer } from "../../../src/backends/software/SkyboxRenderer.ts";

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

function testSkyboxRendererDecodesSRGBToLinear() {
	const environment = new Texture({
		data: new Uint8ClampedArray([128, 64, 32, 255]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	});
	const pixels = new Uint8ClampedArray(4);

	SkyboxRenderer.render(
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

function testSkyboxRendererPreservesLinearTextureValues() {
	const environment = new Texture({
		data: new Uint8ClampedArray([128, 64, 32, 255]),
		width: 1,
		height: 1,
		colorSpace: "Linear",
	});
	const pixels = new Uint8ClampedArray(4);

	SkyboxRenderer.render(
		environment,
		DEFAULT_BACKGROUND_OPTIONS,
		pixels,
		createCamera(),
		1,
		1
	);

	assert.deepEqual(Array.from(pixels), [128, 64, 32, 255]);
}

function testSkyboxRendererPreservesHDRTextureValues() {
	const environment = new Texture({
		data: new Float32Array([1, 0.5, 0.25, 1]),
		width: 1,
		height: 1,
		colorSpace: "HDR",
	});
	const pixels = new Uint8ClampedArray(4);

	SkyboxRenderer.render(
		environment,
		DEFAULT_BACKGROUND_OPTIONS,
		pixels,
		createCamera(),
		1,
		1
	);

	assert.deepEqual(Array.from(pixels), [255, 128, 64, 255]);
}

function testSkyboxRendererClipsToDirtyRegions() {
	const environment = new Texture({
		data: new Uint8ClampedArray([128, 64, 32, 255]),
		width: 1,
		height: 1,
		colorSpace: "Linear",
	});
	const pixels = new Uint8ClampedArray(4 * 4 * 4);

	SkyboxRenderer.render(
		environment,
		DEFAULT_BACKGROUND_OPTIONS,
		pixels,
		createCamera(),
		4,
		4,
		[{ minX: 1, minY: 1, maxXExclusive: 3, maxYExclusive: 3 }],
	);

	assert.deepEqual(Array.from(pixels.slice(0, 4)), [0, 0, 0, 0]);
	const dirtyPixel = (1 * 4 + 1) * 4;
	assert.deepEqual(
		Array.from(pixels.slice(dirtyPixel, dirtyPixel + 4)),
		[128, 64, 32, 255],
	);
}

function run() {
	testSkyboxRendererDecodesSRGBToLinear();
	testSkyboxRendererPreservesLinearTextureValues();
	testSkyboxRendererPreservesHDRTextureValues();
	testSkyboxRendererClipsToDirtyRegions();
	console.log("Software environment color-space tests passed");
}

run();
