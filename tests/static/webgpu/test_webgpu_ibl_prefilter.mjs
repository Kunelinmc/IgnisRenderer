import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

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
	const texture = new Texture({
		colorSpace: "HDR",
		levels: [
			{ width: 4, height: 2, data: new Float32Array(32).fill(1) },
			{ width: 2, height: 1, data: new Float32Array(8).fill(1) },
			{ width: 1, height: 1, data: new Float32Array(4).fill(1) },
		],
	});
	const request = createRequest(texture);
	request.plan.mipLevels[0].roughness = 0.5;
	const result = await executor.execute(request);
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
	const params = backend.buffers.find((buffer) => buffer.label === "IBLPrefilterParams_mip0");
	assert.equal(params.size, 32);
	const data = params.lastWrite.data;
	assert.equal(data.byteLength, 32);
	const view = new DataView(data);
	assert.deepEqual([0, 4, 8, 12].map((offset) => view.getUint32(offset, true)),
		[1, 1, 4, 2]);
	assert.equal(view.getFloat32(16, true), 0.5);
	assert.equal(view.getUint32(20, true), 152);
	assert.equal(view.getUint32(24, true), 3);
	assert.equal(view.getUint32(28, true), 0);
	const shader = readFileSync(new URL(
		"../../../src/shaders/webgpu/environment/iblPrefilter.wgsl", import.meta.url,
	), "utf8");
	const fields = shader.match(/struct PrefilterParams\s*\{([^}]+)\}/)[1]
		.match(/\w+\s*:\s*(?:f32|u32)\s*,/g);
	assert.equal(fields.length * 4, 32);
	assert.match(fields[6], /^sourceMipLevelCount\s*:/);
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

async function testSrgbSourceRejectsNativeFormatFallback() {
	const backend = new FakeWebGPUBackend();
	const createTextureResource = backend.createTexture.bind(backend);
	backend.createTexture = (descriptor) => createTextureResource({
		...descriptor,
		format: descriptor.label === "IBLPrefilterInputTexture"
			? TextureFormat.RGBA8Unorm : descriptor.format,
	});
	const executor = new WebGPUPrefilterExecutor(createWebGPUComputeFacade(backend));
	const texture = new Texture({
		width: 1, height: 1, data: new Uint8Array([128, 64, 32, 128]),
		colorSpace: "sRGB",
	});
	await assert.rejects(executor.execute(createRequest(texture)), /sRGB.*format/);
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
await testSrgbSourceRejectsNativeFormatFallback();
await testUnavailableDeviceRejectsRequests();
await testSourceRevisionIsValidated();

console.log("WebGPU IBL prefilter executor tests passed");
