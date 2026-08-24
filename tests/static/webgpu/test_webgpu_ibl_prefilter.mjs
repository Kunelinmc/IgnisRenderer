import assert from "node:assert/strict";

import { Texture } from "../../../src/core/Texture.ts";
import {
	captureIBLPrefilterSourceRevision,
} from "../../../src/lights/ibl/IBLPrefilterExecutor.ts";
import { createWebGPUComputeFacade } from "../../../src/backends/webgpu/ComputeFacade.ts";
import { WebGPUPrefilterExecutor } from "../../../src/lights/ibl/WebGPUPrefilterExecutor.ts";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";

import { FakeWebGPUBackend } from "../../helpers/fakes.mjs";

function createTexture() {
	return new Texture({
		data: new Float32Array([4, 2, 1, 1]),
		width: 1,
		height: 1,
		colorSpace: "HDR",
	});
}

function createRequest(texture) {
	return {
		envMap: texture,
		plan: {
			baseWidth: 1,
			baseHeight: 1,
			mipLevels: [
				{ level: 0, width: 1, height: 1, roughness: 0 },
			],
		},
		sourceRevision: captureIBLPrefilterSourceRevision(texture),
	};
}

async function testExecutesRGBA16FloatPrefilter() {
	const backend = new FakeWebGPUBackend();
	const executor = new WebGPUPrefilterExecutor(
		createWebGPUComputeFacade(backend),
	);
	const result = await executor.execute(createRequest(createTexture()));
	assert.equal(result.length, 1);
	assert.ok(result[0].data instanceof Float32Array);
	assert.equal(
		backend.createTextureCalls.find(
			(call) => call.label === "IBLPrefilterInputTexture",
		)?.format,
		TextureFormat.RGBA16Float,
	);
	assert.equal(
		backend.createTextureCalls.find(
			(call) => call.label === "IBLPrefilterOutput_mip0",
		)?.format,
		TextureFormat.RGBA16Float,
	);
}

async function testUnavailableDeviceRejectsRequests() {
	const backend = new FakeWebGPUBackend();
	const executor = new WebGPUPrefilterExecutor(
		createWebGPUComputeFacade(backend),
	);
	backend.device = null;
	backend.queue = null;
	assert.deepEqual(executor.getAvailability(), {
		state: "temporarily-unavailable",
		acceptsRequests: false,
		reason:
			"WebGPU IBL prefilter executor requires an initialized device and queue.",
	});
	await assert.rejects(
		executor.execute(createRequest(createTexture())),
		/initialized device and queue/,
	);
}

async function testSourceRevisionIsValidated() {
	const backend = new FakeWebGPUBackend();
	const executor = new WebGPUPrefilterExecutor(
		createWebGPUComputeFacade(backend),
	);
	const texture = createTexture();
	const request = createRequest(texture);
	texture.markNeedsUpdate();
	await assert.rejects(
		executor.execute(request),
		/source changed while waiting/,
	);
}

await testExecutesRGBA16FloatPrefilter();
await testUnavailableDeviceRejectsRequests();
await testSourceRevisionIsValidated();

console.log("WebGPU IBL prefilter executor tests passed");
