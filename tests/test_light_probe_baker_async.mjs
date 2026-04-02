import assert from "node:assert/strict";
import { Texture } from "../src/core/Texture.ts";
import { bakeEnvironmentIBLFromEnvironmentMap } from "../src/pipeline/EnvironmentIBLBaker.ts";

function createTestTexture(width = 32, height = 16) {
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

async function testBakeReturnsLightProbeWithPrefilteredMap() {
	const texture = createTestTexture();
	const probe = await bakeEnvironmentIBLFromEnvironmentMap(texture, {
		acceleration: "cpu",
	});
	assert.ok(probe);
	assert.ok(probe.prefilteredMap);
	assert.equal(probe.sh.length, 16);
	assert.equal(probe.prefilteredMap?.mipmaps.length, 5);
}

async function testBakeSupportsAbortSignal() {
	const texture = createTestTexture();
	const controller = new AbortController();
	controller.abort();
	await assert.rejects(
		bakeEnvironmentIBLFromEnvironmentMap(texture, {
			acceleration: "cpu",
			signal: controller.signal,
		}),
		(error) => error instanceof Error && error.name === "AbortError"
	);
}

async function testExplicitWorkerModeThrowsWhenWorkersAreUnavailable() {
	const texture = createTestTexture();
	await assert.rejects(
		bakeEnvironmentIBLFromEnvironmentMap(texture, {
			acceleration: "worker",
		}),
		(error) =>
			error instanceof Error &&
			(error.message.includes("environment IBL baking") ||
				error.message.includes("Worker constructor is unavailable"))
	);
}

async function testExplicitWebGPUModeRequiresSource() {
	const texture = createTestTexture();
	await assert.rejects(
		bakeEnvironmentIBLFromEnvironmentMap(texture, {
			acceleration: "webgpu",
		}),
		(error) =>
			error instanceof Error &&
			error.message.includes("no webgpuSource was provided")
	);
}

async function testAutoFallsBackWhenWebGPUPathFails() {
	const texture = createTestTexture();
	const probe = await bakeEnvironmentIBLFromEnvironmentMap(texture, {
		acceleration: "auto",
		webgpuSource: { type: "webgpu" },
	});
	assert.ok(probe.prefilteredMap);
	assert.equal(probe.prefilteredMap?.mipmaps.length, 5);
}

async function testBakeReportsProgressMonotonically() {
	const texture = createTestTexture(16, 8);
	const progressEvents = [];
	await bakeEnvironmentIBLFromEnvironmentMap(texture, {
		acceleration: "cpu",
		onProgress: (event) => progressEvents.push(event),
	});

	assert.ok(progressEvents.length >= 3);
	assert.equal(progressEvents[0].phase, "project-sh");
	assert.equal(progressEvents[progressEvents.length - 1].phase, "finalize");

	const total = progressEvents[0].total;
	let previousCompleted = -1;
	for (const event of progressEvents) {
		assert.equal(event.total, total);
		assert.ok(event.completed > previousCompleted);
		previousCompleted = event.completed;
	}
	assert.equal(previousCompleted, total);
}

async function run() {
	await testBakeReturnsLightProbeWithPrefilteredMap();
	await testBakeSupportsAbortSignal();
	await testExplicitWorkerModeThrowsWhenWorkersAreUnavailable();
	await testExplicitWebGPUModeRequiresSource();
	await testAutoFallsBackWhenWebGPUPathFails();
	await testBakeReportsProgressMonotonically();
	console.log("Environment IBL baker async tests passed");
}

await run();
