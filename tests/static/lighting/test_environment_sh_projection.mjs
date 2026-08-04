import assert from "node:assert/strict";
import { Texture } from "../../../src/core/Texture.ts";
import { projectEnvironmentTextureToSH } from "../../../src/lights/ibl/EnvironmentSH.ts";
import { SH } from "../../../src/maths/SH.ts";

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

function testProjectionPreservesEquirectangularDirection() {
	const width = 64;
	const height = 32;
	const data = new Uint8ClampedArray(width * height * 4);
	for (let y = 0; y < height; y++) {
		const theta = ((y + 0.5) / height) * Math.PI;
		const sinTheta = Math.sin(theta);
		for (let x = 0; x < width; x++) {
			const phi = ((x + 0.5) / width) * Math.PI * 2 - Math.PI;
			const value = Math.round(Math.max(0, sinTheta * Math.sin(phi)) * 255);
			const index = (y * width + x) * 4;
			data[index] = value;
			data[index + 1] = value;
			data[index + 2] = value;
			data[index + 3] = 255;
		}
	}

	const sh = projectEnvironmentTextureToSH(new Texture({
		data,
		width,
		height,
		colorSpace: "sRGB",
	}), {
		maxSampleWidth: width,
		maxSampleHeight: height,
	});
	const positiveX = SH.calculateIrradiance({ x: 1, y: 0, z: 0 }, sh).r;
	const negativeX = SH.calculateIrradiance({ x: -1, y: 0, z: 0 }, sh).r;
	assert.ok(
		positiveX > negativeX * 10,
		`Expected +X environment irradiance to dominate, got +X=${positiveX}, -X=${negativeX}`
	);
}

function testProjectionNormalizesTextureStorageTypes() {
	const linearByte = projectEnvironmentTextureToSH(new Texture({
		data: new Uint8Array([128, 128, 128, 255]),
		width: 1,
		height: 1,
		colorSpace: "Linear",
	}));
	const linearFloat = projectEnvironmentTextureToSH(new Texture({
		data: new Float32Array([128 / 255, 128 / 255, 128 / 255, 1]),
		width: 1,
		height: 1,
		colorSpace: "Linear",
	}));
	assert.ok(Math.abs(linearByte[0].r - linearFloat[0].r) < 1e-4);

	const srgbByte = projectEnvironmentTextureToSH(new Texture({
		data: new Uint8Array([128, 128, 128, 255]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	}));
	const srgbFloat = projectEnvironmentTextureToSH(new Texture({
		data: new Float32Array([128 / 255, 128 / 255, 128 / 255, 1]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	}));
	assert.ok(Math.abs(srgbByte[0].r - srgbFloat[0].r) < 1e-4);
}

function run() {
	testProjectsEnvironmentTextureToSH();
	testProjectionHonorsSampleLimits();
	testProjectionSupportsAbortSignal();
	testProjectionRejectsInvalidTexture();
	testProjectionPreservesEquirectangularDirection();
	testProjectionNormalizesTextureStorageTypes();
	console.log("Environment SH projection tests passed");
}

run();
