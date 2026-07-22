import assert from "node:assert/strict";
import { Texture } from "../../../src/core/Texture.ts";
import { Platform } from "../../../src/foundation/Platform.ts";
import {
	IBLPrefilter,
	prefilterEnvironmentIBL,
} from "../../../src/lights/ibl/IBLPrefilter.ts";
import { sampleEnvironmentTextureSpecular } from "../../../src/lights/runtime/environmentMapRuntime.ts";
import { WEBGPU_COMPUTE_EXTENSION } from "../../../src/backends/BackendExtensions.ts";
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
	return new Texture({ data: data, width: width, height: height, colorSpace: "sRGB" });
}

function createRenderBackend(computeFacade) {
	return {
		profile: { id: computeFacade ? "webgpu" : "software" },
		attach() {},
		async initialize() {},
		extensions: {
			getBackendExtension(key) {
				return key.id === WEBGPU_COMPUTE_EXTENSION.id ? computeFacade : null;
			},
		},
	};
}

async function testClassPrefiltersOnSingleThread() {
	const prefilter = new IBLPrefilter();
	const texture = createTestTexture();
	const result = await prefilter.prefilter(texture, {
		acceleration: "single-thread",
		maxMipLevels: 3,
	});
	assert.ok(result instanceof Texture);
	assert.equal(result.colorSpace, "HDR");
	assert.equal(result.mipmaps.length, 3);
}

async function testLegacyAccelerationValuesAreRejected() {
	await assert.rejects(
		new IBLPrefilter().prefilter(createTestTexture(), {
			acceleration: "cpu",
		}),
		(error) =>
			error instanceof Error &&
			error.message.includes('Unsupported IBL prefilter acceleration "cpu"')
	);
}

async function testMultiThreadRequiresWorkerAPI() {
	const originalHasWorker = Platform.hasWorker;
	Platform.hasWorker = () => false;
	try {
		await assert.rejects(
			new IBLPrefilter().prefilter(createTestTexture(), {
				acceleration: "multi-thread",
			}),
			(error) =>
				error instanceof Error &&
				error.message.includes("Worker API is unavailable")
		);
	} finally {
		Platform.hasWorker = originalHasWorker;
	}
}

async function testHelperPreservesHDRRadiance() {
	const texture = new Texture({
		data: new Float32Array([4, 2, 1, 1]),
		width: 1,
		height: 1,
		colorSpace: "HDR",
	});
	const result = await prefilterEnvironmentIBL(texture, {
		acceleration: "single-thread",
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
	const prefilter = new IBLPrefilter({ type: "webgl" });
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
	const texture = new Texture({
		data: new Float32Array([4, 2, 1, 1]),
		width: 1,
		height: 1,
		colorSpace: "HDR",
	});
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

async function testIRenderBackendResolvesWebGPUComputeExtension() {
	const texture = new Texture({
		data: new Float32Array([4, 2, 1, 1]),
		width: 1,
		height: 1,
		colorSpace: "HDR",
	});
	const computeFacade = new FakeWebGPUBackend();
	const prefilter = new IBLPrefilter(createRenderBackend(computeFacade));
	const result = await prefilter.prefilter(texture, {
		acceleration: "webgpu",
		maxSampleWidth: 1,
		maxSampleHeight: 1,
		maxMipLevels: 1,
	});

	assert.ok(result.mipmaps[0] instanceof Float32Array);
	assert.ok(
		computeFacade.createTextureCalls.some(
			(call) => call.label === "IBLPrefilterOutput_mip0"
		)
	);
}

async function testIRenderBackendRejectsUnavailableWebGPUState() {
	const texture = createTestTexture();
	const computeFacade = new FakeWebGPUBackend();
	computeFacade.device = null;
	computeFacade.queue = null;
	const prefilter = new IBLPrefilter(createRenderBackend(computeFacade));

	await assert.rejects(
		prefilter.prefilter(texture, { acceleration: "webgpu" }),
		(error) =>
			error instanceof Error &&
			error.message.includes("device or queue is unavailable")
	);
}

async function testIRenderBackendRejectsMissingWebGPUExtension() {
	const prefilter = new IBLPrefilter(createRenderBackend(null));

	await assert.rejects(
		prefilter.prefilter(createTestTexture(), { acceleration: "webgpu" }),
		(error) =>
			error instanceof Error &&
			error.message.includes("does not expose the WebGPU compute extension")
	);
}

async function testIRenderBackendAutoFallsBackWhenWebGPUIsUnavailable() {
	const texture = createTestTexture();
	const computeFacade = new FakeWebGPUBackend();
	computeFacade.device = null;
	computeFacade.queue = null;
	const result = await new IBLPrefilter(
		createRenderBackend(computeFacade)
	).prefilter(texture, {
		acceleration: "auto",
		maxSampleWidth: 1,
		maxSampleHeight: 1,
		maxMipLevels: 1,
	});

	assert.equal(result.mipmaps.length, 1);
	assert.equal(computeFacade.createTextureCalls.length, 0);
}

async function testAbortSignal() {
	const texture = createTestTexture();
	const controller = new AbortController();
	controller.abort();
	await assert.rejects(
		prefilterEnvironmentIBL(texture, {
			acceleration: "single-thread",
			signal: controller.signal,
		}),
		(error) => error instanceof Error && error.name === "AbortError"
	);
}

async function run() {
	await testClassPrefiltersOnSingleThread();
	await testLegacyAccelerationValuesAreRejected();
	await testMultiThreadRequiresWorkerAPI();
	await testHelperPreservesHDRRadiance();
	await testExplicitWebGPURejectsNonWebGPUBackend();
	await testAutoFallsBackWhenWebGPUPathFails();
	await testWebGPUPrefilterUsesRGBA16FloatForHDR();
	await testIRenderBackendResolvesWebGPUComputeExtension();
	await testIRenderBackendRejectsUnavailableWebGPUState();
	await testIRenderBackendRejectsMissingWebGPUExtension();
	await testIRenderBackendAutoFallsBackWhenWebGPUIsUnavailable();
	await testAbortSignal();
	console.log("IBL prefilter tests passed");
}

await run();
