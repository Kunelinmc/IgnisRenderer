import assert from "node:assert/strict";

import { Texture } from "../../../src/core/Texture.ts";
import { Platform } from "../../../src/foundation/Platform.ts";
import {
	IBLPrefilter,
	prefilterEnvironmentIBL,
} from "../../../src/lights/ibl/IBLPrefilter.ts";
import { WEBGL_AUXILIARY_RASTER_EXTENSION } from "../../../src/backends/webgl/WebGLAuxiliaryRaster.ts";
import {
	directionFromEquirectUV,
	sampleEnvironmentTextureSpecular,
} from "../../../src/lights/runtime/environmentMapRuntime.ts";

function nearlyEqual(actual, expected, epsilon = 1e-4) {
	assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected}`);
}

function createTestTexture(width = 16, height = 8) {
	const data = new Uint8ClampedArray(width * height * 4);
	for (let index = 0; index < data.length; index += 4) {
		const pixelIndex = index >> 2;
		data[index] = (pixelIndex * 17) % 255;
		data[index + 1] = (pixelIndex * 31) % 255;
		data[index + 2] = (pixelIndex * 47) % 255;
		data[index + 3] = 255;
	}
	return new Texture({ data, width, height, colorSpace: "sRGB" });
}

function createExecutor(id, options = {}) {
	let calls = 0;
	return {
		id,
		get calls() {
			return calls;
		},
		getAvailability() {
			return options.availability ?? {
				state: "ready",
				acceptsRequests: true,
				reason: null,
			};
		},
		async execute() {
			calls++;
			if (options.error) throw options.error;
			return [{
				level: 0,
				width: 1,
				height: 1,
				data: new Float32Array(4).fill(1),
			}];
		},
	};
}

function createRenderBackend(executor = null) {
	return {
		profile: { id: executor?.id ?? "software" },
		attach() {},
		async initialize() {},
		extensions: {
			getBackendExtension(key) {
				if (
					executor?.id === "webgl" &&
					key.id === WEBGL_AUXILIARY_RASTER_EXTENSION.id
				) {
					return {
						getAvailability: () => executor.getAvailability(),
						execute: () => executor.execute(),
					};
				}
				return null;
			},
		},
	};
}

async function testClassPrefiltersOnSingleThread() {
	const result = await new IBLPrefilter().prefilter(createTestTexture(), {
		acceleration: "single-thread",
		maxMipLevels: 3,
	});
	assert.ok(result instanceof Texture);
	assert.equal(result.colorSpace, "HDR");
	assert.equal(result.mipmaps.length, 3);
	assert.equal(result.minFilter, "LinearMipmapLinear");
}

async function testNaturalMipCountCapsWorkAndProgress() {
	const progress = [];
	const result = await new IBLPrefilter().prefilter(
		new Texture({
			data: new Float32Array([1, 1, 1, 1]),
			width: 1,
			height: 1,
			colorSpace: "HDR",
		}),
		{
			acceleration: "single-thread",
			maxMipLevels: 8,
			onProgress: (entry) => progress.push(entry),
		},
	);
	assert.equal(result.mipmaps.length, 1);
	assert.equal(result.minFilter, "Linear");
	assert.deepEqual(progress.map(({ completed, total }) => [completed, total]), [
		[1, 1],
	]);
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
	nearlyEqual(mip0[0], 4);
	nearlyEqual(mip0[1], 2);
	nearlyEqual(mip0[2], 1);
	const sample = sampleEnvironmentTextureSpecular(
		result,
		{ x: 0, y: 0, z: 1 },
		0,
	);
	nearlyEqual(sample.r, 4);
	nearlyEqual(sample.g, 2);
	nearlyEqual(sample.b, 1);
}

async function testDirectionAndWrapModes() {
	const width = 16;
	const height = 8;
	const data = new Float32Array(width * height * 4);
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const direction = directionFromEquirectUV(
				(x + 0.5) / width,
				(y + 0.5) / height,
			);
			const index = (y * width + x) * 4;
			data[index] = direction.x * 0.5 + 0.5;
			data[index + 1] = direction.y * 0.5 + 0.5;
			data[index + 2] = direction.z * 0.5 + 0.5;
			data[index + 3] = 1;
		}
	}
	const result = await prefilterEnvironmentIBL(
		new Texture({ data, width, height, colorSpace: "HDR" }),
		{
			acceleration: "single-thread",
			maxSampleWidth: width,
			maxSampleHeight: height,
			maxMipLevels: 1,
		},
	);
	const plusZ = sampleEnvironmentTextureSpecular(
		result,
		{ x: 0, y: 0, z: 1 },
		0,
	);
	const minusZ = sampleEnvironmentTextureSpecular(
		result,
		{ x: 0, y: 0, z: -1 },
		0,
	);
	assert.ok(plusZ.b > 0.9);
	assert.ok(minusZ.b < 0.1);
	assert.equal(result.wrapS, "Repeat");
	assert.equal(result.wrapT, "Clamp");
}

async function testBackendExecutorSelectionAndNestedService() {
	const executor = createExecutor("webgl");
	const result = await prefilterEnvironmentIBL(createTestTexture(1, 1), {
		service: { backend: createRenderBackend(executor) },
		acceleration: "webgl",
		maxMipLevels: 1,
	});
	assert.equal(executor.calls, 1);
	assert.equal(result.mipmaps.length, 1);
}

async function testAutoUsesReadyBackendBeforeCPU() {
	const executor = createExecutor("webgl");
	await new IBLPrefilter({ backend: createRenderBackend(executor) }).prefilter(
		createTestTexture(1, 1),
		{ acceleration: "auto", maxMipLevels: 1 },
	);
	assert.equal(executor.calls, 1);
}

async function testAutoSkipsTemporarilyUnavailableBackend() {
	const executor = createExecutor("webgl", {
		availability: {
			state: "temporarily-unavailable",
			acceptsRequests: true,
			reason: "context lost",
		},
	});
	const originalHasWorker = Platform.hasWorker;
	Platform.hasWorker = () => false;
	try {
		const result = await new IBLPrefilter({
			backend: createRenderBackend(executor),
		}).prefilter(createTestTexture(1, 1), {
			acceleration: "auto",
			maxMipLevels: 1,
		});
		assert.equal(result.mipmaps.length, 1);
		assert.equal(executor.calls, 0);
	} finally {
		Platform.hasWorker = originalHasWorker;
	}
}

async function testExplicitExecutorHonorsAcceptsRequests() {
	const waiting = createExecutor("webgl", {
		availability: {
			state: "temporarily-unavailable",
			acceptsRequests: true,
			reason: "context lost",
		},
	});
	await new IBLPrefilter({ backend: createRenderBackend(waiting) }).prefilter(
		createTestTexture(1, 1),
		{ acceleration: "webgl", maxMipLevels: 1 },
	);
	assert.equal(waiting.calls, 1);

	const rejected = createExecutor("webgl", {
		availability: {
			state: "temporarily-unavailable",
			acceptsRequests: false,
			reason: "device unavailable",
		},
	});
	await assert.rejects(
		new IBLPrefilter({ backend: createRenderBackend(rejected) }).prefilter(
			createTestTexture(1, 1),
			{ acceleration: "webgl", maxMipLevels: 1 },
		),
		/device unavailable/,
	);
}

async function testExecutorFailureDoesNotFallback() {
	const executor = createExecutor("webgl", { error: new Error("gpu failed") });
	await assert.rejects(
		new IBLPrefilter({ backend: createRenderBackend(executor) }).prefilter(
			createTestTexture(1, 1),
			{ acceleration: "auto", maxMipLevels: 1 },
		),
		/gpu failed/,
	);
	assert.equal(executor.calls, 1);
}

async function testValidationAndCancellation() {
	assert.throws(
		() => new IBLPrefilter(createRenderBackend()),
		/IBLPrefilterServiceOptions object/,
	);
	await assert.rejects(
		new IBLPrefilter().prefilter(createTestTexture(), { acceleration: "cpu" }),
		/Unsupported IBL prefilter acceleration/,
	);
	const originalHasWorker = Platform.hasWorker;
	Platform.hasWorker = () => false;
	try {
		await assert.rejects(
			new IBLPrefilter().prefilter(createTestTexture(), {
				acceleration: "multi-thread",
			}),
			/Worker API is unavailable/,
		);
	} finally {
		Platform.hasWorker = originalHasWorker;
	}
	const controller = new AbortController();
	controller.abort();
	await assert.rejects(
		prefilterEnvironmentIBL(createTestTexture(), {
			acceleration: "single-thread",
			signal: controller.signal,
		}),
		(error) => error instanceof Error && error.name === "AbortError",
	);
}

await testClassPrefiltersOnSingleThread();
await testNaturalMipCountCapsWorkAndProgress();
await testHelperPreservesHDRRadiance();
await testDirectionAndWrapModes();
await testBackendExecutorSelectionAndNestedService();
await testAutoUsesReadyBackendBeforeCPU();
await testAutoSkipsTemporarilyUnavailableBackend();
await testExplicitExecutorHonorsAcceptsRequests();
await testExecutorFailureDoesNotFallback();
await testValidationAndCancellation();

console.log("IBL prefilter tests passed");
