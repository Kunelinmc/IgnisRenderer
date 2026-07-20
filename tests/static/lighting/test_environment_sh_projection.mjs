import assert from "node:assert/strict";
import { Texture } from "../../../src/core/Texture.ts";
import { projectEnvironmentTextureToSH } from "../../../src/lights/ibl/EnvironmentSH.ts";

function createTestTexture(width = 32, height = 16) {
	const data = new Uint8ClampedArray(width * height * 4);
	for (let i = 0; i < data.length; i += 4) {
		const pixelIndex = i >> 2;
		data[i] = (pixelIndex * 17) % 255;
		data[i + 1] = (pixelIndex * 31) % 255;
		data[i + 2] = (pixelIndex * 47) % 255;
		data[i + 3] = 255;
	}
	return new Texture({ data: data, width: width, height: height, colorSpace: "sRGB" });
}

function testProjectsEnvironmentTextureToSH() {
	const texture = createTestTexture();
	const sh = projectEnvironmentTextureToSH(texture);
	assert.equal(sh.length, 16);
	assert.ok(sh[0].r > 0);
	assert.ok(sh[0].g > 0);
	assert.ok(sh[0].b > 0);
}

function testProjectionHonorsSampleLimits() {
	const texture = new Texture({
		data: new Float32Array([4, 2, 1, 1]),
		width: 1,
		height: 1,
		colorSpace: "HDR",
	});
	const sh = projectEnvironmentTextureToSH(texture, {
		maxSampleWidth: 1,
		maxSampleHeight: 1,
	});
	assert.equal(sh.length, 16);
	assert.ok(sh[0].r > sh[0].g);
	assert.ok(sh[0].g > sh[0].b);
}

function testProjectionSupportsAbortSignal() {
	const texture = createTestTexture();
	const controller = new AbortController();
	controller.abort();
	assert.throws(
		() =>
			projectEnvironmentTextureToSH(texture, {
				signal: controller.signal,
			}),
		(error) => error instanceof Error && error.name === "AbortError"
	);
}

function testProjectionRejectsInvalidTexture() {
	assert.throws(
		() => projectEnvironmentTextureToSH(new Texture({
			data: null,
			width: 0,
			height: 0,
			colorSpace: "sRGB",
		})),
		/valid environment texture/
	);
}

function run() {
	testProjectsEnvironmentTextureToSH();
	testProjectionHonorsSampleLimits();
	testProjectionSupportsAbortSignal();
	testProjectionRejectsInvalidTexture();
	console.log("Environment SH projection tests passed");
}

run();
