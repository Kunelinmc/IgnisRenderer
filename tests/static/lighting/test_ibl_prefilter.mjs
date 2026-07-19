import assert from "node:assert/strict";
import { Texture } from "../../../src/core/Texture.ts";
import {
	IBLPrefilter,
	prefilterEnvironmentIBL,
} from "../../../src/lights/ibl/IBLPrefilter.ts";
import { sampleEnvironmentTextureSpecular } from "../../../src/lights/runtime/environmentMapRuntime.ts";
import { TextureFormat } from "../../../src/backends/types.ts";

import { FakeWebGPUBackend } from "../../helpers/fakes.mjs";

function nearlyEqual(actual, expected, epsilon = 1e-4) {
	assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected}`);
}

function createTestTexture(width = 16, height = 8) {
	const data = new Uint8ClampedArray(width * height * 4);
	for (let i = 0; i < data.length; i += 4) {
		const pixelIndex = i >> 2;
		data[i] = (pixelIndex * 17) % 255;
		data[i + 1] = (pixelIndex * 31) % 255;
		data[i + 2] = (pixelIndex * 47) % 255;
		data[i + 3] = 255;
	}
	return new Texture(data, width, height, "sRGB");
}

async function testClassPrefiltersOnCPU() {
	const prefilter = new IBLPrefilter();
	const texture = createTestTexture();
	const result = await prefilter.prefilter(texture, {
		acceleration: "cpu",
		maxMipLevels: 3,
	});
	assert.ok(result instanceof Texture);
	assert.equal(result.colorSpace, "HDR");
	assert.equal(result.mipmaps.length, 3);
}

async function testHelperPreservesHDRRadiance() {
	const texture = new Texture(new Float32Array([4, 2, 1, 1]), 1, 1, "HDR");
	const result = await prefilterEnvironmentIBL(texture, {
		acceleration: "cpu",
		maxSampleWidth: 1,
		maxSampleHeight: 1,
		maxMipLevels: 1,
	});
	const mip0 = result.mipmaps[0];
	assert.ok(mip0 instanceof Float32Array);
	nearlyEqual(mip0[0], 4);
	nearlyEqual(mip0[1], 2);
	nearlyEqual(mip0[2], 1);

	const sample = sampleEnvironmentTextureSpecular(
		result,
		{ x: 0, y: 0, z: 1 },
		0
	);
	nearlyEqual(sample.r, 4);
	nearlyEqual(sample.g, 2);
	nearlyEqual(sample.b, 1);
}

async function testExplicitWebGPURejectsNonWebGPUBackend() {
	const texture = createTestTexture();
	const prefilter = new IBLPrefilter({
		backend: {
			type: "webgl",
		},
	});
	await assert.rejects(
		prefilter.prefilter(texture, {
			acceleration: "webgpu",
		}),
		(error) =>
			error instanceof Error &&
			error.message.includes("no WebGPU backend or compute source")
	);
}

async function testAutoFallsBackWhenWebGPUPathFails() {
	const texture = createTestTexture();
	const result = await prefilterEnvironmentIBL(texture, {
		acceleration: "auto",
		computeSource: { type: "webgpu" },
		maxMipLevels: 2,
	});
	assert.equal(result.mipmaps.length, 2);
}

async function testWebGPUPrefilterUsesRGBA16FloatForHDR() {
	const texture = new Texture(new Float32Array([4, 2, 1, 1]), 1, 1, "HDR");
	const backend = new FakeWebGPUBackend();
	const prefilter = new IBLPrefilter(backend);
	const result = await prefilter.prefilter(texture, {
		acceleration: "webgpu",
		maxSampleWidth: 1,
		maxSampleHeight: 1,
		maxMipLevels: 1,
	});
	assert.ok(result.mipmaps[0] instanceof Float32Array);

	const inputTexture = backend.createTextureCalls.find(
		(call) => call.label === "IBLPrefilterInputTexture"
	);
	const outputTexture = backend.createTextureCalls.find(
		(call) => call.label === "IBLPrefilterOutput_mip0"
	);
	assert.equal(inputTexture?.format, TextureFormat.RGBA16Float);
	assert.equal(outputTexture?.format, TextureFormat.RGBA16Float);
}

async function testAbortSignal() {
	const texture = createTestTexture();
	const controller = new AbortController();
	controller.abort();
	await assert.rejects(
		prefilterEnvironmentIBL(texture, {
			acceleration: "cpu",
			signal: controller.signal,
		}),
		(error) => error instanceof Error && error.name === "AbortError"
	);
}

async function run() {
	await testClassPrefiltersOnCPU();
	await testHelperPreservesHDRRadiance();
	await testExplicitWebGPURejectsNonWebGPUBackend();
	await testAutoFallsBackWhenWebGPUPathFails();
	await testWebGPUPrefilterUsesRGBA16FloatForHDR();
	await testAbortSignal();
	console.log("IBL prefilter tests passed");
}

await run();
