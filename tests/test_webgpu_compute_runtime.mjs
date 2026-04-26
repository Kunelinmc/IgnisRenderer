import assert from "node:assert/strict";
import {
	BufferUsage,
	TextureFormat,
	TextureUsage,
} from "../src/renderers/types.ts";
import { ComputeRuntime } from "../src/renderers/webgpu/ComputeRuntime.ts";

import { FakeWebGPUBackend } from "./helpers/test_fakes.mjs";

function toUint8Array(data) {
	if (data instanceof Uint8Array) {
		return data;
	}
	return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

function nearlyEqual(actual, expected, epsilon = 1e-6) {
	assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected}`);
}

async function testReadTextureSkipsPaddingInNormalizedConversion() {
	const runtime = new ComputeRuntime(new FakeWebGPUBackend());
	const texture = runtime.createTexture({
		width: 3,
		height: 2,
		format: TextureFormat.RGBA8Unorm,
		usage: TextureUsage.CopyDst | TextureUsage.CopySrc | TextureUsage.TextureBinding,
		label: "PaddingTexture",
	});
	const bytesPerRow = 256;
	const upload = new Uint8Array(bytesPerRow * 2);
	upload.fill(199);
	upload.set([10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255], 0);
	upload.set([100, 110, 120, 255, 130, 140, 150, 255, 160, 170, 180, 255], bytesPerRow);
	runtime.writeTexture(
		texture,
		upload,
		{ bytesPerRow, rowsPerImage: 2 },
		{ width: 3, height: 2, depthOrArrayLayers: 1 }
	);

	const readback = await runtime.readTexture({
		texture,
		format: TextureFormat.RGBA8Unorm,
	});
	assert.equal(readback.bytesPerRow, 256);
	const normalized = readback.toNormalizedRGBA8Float32();
	assert.equal(normalized.length, 24);
	nearlyEqual(normalized[0], 10 / 255);
	nearlyEqual(normalized[4], 40 / 255);
	nearlyEqual(normalized[12], 100 / 255);
	nearlyEqual(normalized[20], 160 / 255);
	runtime.destroy();
}

async function testReadTextureNormalizesBGRA8Unorm() {
	const runtime = new ComputeRuntime(new FakeWebGPUBackend());
	const texture = runtime.createTexture({
		width: 1,
		height: 1,
		format: TextureFormat.BGRA8Unorm,
		usage:
			TextureUsage.CopyDst |
			TextureUsage.CopySrc |
			TextureUsage.TextureBinding,
		label: "BgraTexture",
	});
	const upload = new Uint8Array([30, 20, 10, 255]);
	runtime.writeTexture(
		texture,
		upload,
		{ bytesPerRow: 4, rowsPerImage: 1 },
		{ width: 1, height: 1, depthOrArrayLayers: 1 }
	);

	const readback = await runtime.readTexture({
		texture,
		format: TextureFormat.BGRA8Unorm,
	});
	const normalized = readback.toNormalizedRGBA8Float32();
	assert.equal(normalized.length, 4);
	nearlyEqual(normalized[0], 10 / 255);
	nearlyEqual(normalized[1], 20 / 255);
	nearlyEqual(normalized[2], 30 / 255);
	nearlyEqual(normalized[3], 255 / 255);
	runtime.destroy();
}

async function testKernelSchemaValidation() {
	const runtime = new ComputeRuntime(new FakeWebGPUBackend());
	await assert.rejects(
		runtime.createKernel({
			code: "@compute @workgroup_size(1) fn csMain() {}",
			bindings: [
				{ key: "a", binding: 0, type: "buffer" },
				{ key: "a", binding: 1, type: "texture" },
			],
			workgroupSize: { x: 1 },
		}),
		/duplicate key/
	);
	await assert.rejects(
		runtime.createKernel({
			code: "@compute @workgroup_size(1) fn csMain() {}",
			bindings: [{ key: "a", binding: 0, type: "invalid" }],
			workgroupSize: { x: 1 },
		}),
		/unsupported type/
	);
	runtime.destroy();
}

async function testDispatchValidationRules() {
	const runtime = new ComputeRuntime(new FakeWebGPUBackend());
	const kernel = await runtime.createKernel({
		code: "@compute @workgroup_size(1) fn csMain() {}",
		bindings: [{ key: "params", binding: 0, type: "buffer" }],
		workgroupSize: { x: 8, y: 8, z: 1 },
	});
	const texture = runtime.createTexture({
		width: 1,
		height: 1,
		format: TextureFormat.RGBA8Unorm,
		usage: TextureUsage.TextureBinding,
	});
	assert.throws(
		() =>
			kernel.dispatch({
				resources: { params: texture },
				dispatch: { x: 1, y: 1, z: 1 },
			}),
		/expects "buffer"/
	);
	const paramsBuffer = runtime.createBuffer({
		size: 16,
		usage: BufferUsage.Uniform | BufferUsage.CopyDst,
	});
	assert.throws(
		() =>
			kernel.dispatch({
				resources: { params: paramsBuffer },
				dispatch: { x: 1, y: 1, z: 1 },
				dispatch2D: { width: 1, height: 1 },
			}),
		/cannot include both dispatch and dispatch2D/
	);
	assert.throws(
		() =>
			kernel.dispatch({
				resources: { params: paramsBuffer },
				dispatch: { x: 1, y: 1, z: 1 },
				extraBindGroups: [{ index: 0, group: { label: "illegal" } }],
			}),
		/cannot target index 0/
	);
	runtime.destroy();
}

async function testRuntimeOwnedResourceDestroyIsDeferredUntilDispatchDone() {
	const runtime = new ComputeRuntime(new FakeWebGPUBackend());
	const kernel = await runtime.createKernel({
		code: "@compute @workgroup_size(1) fn csMain() {}",
		bindings: [{ key: "params", binding: 0, type: "buffer" }],
		workgroupSize: { x: 1, y: 1, z: 1 },
	});
	const paramsBuffer = runtime.createBuffer({
		size: 16,
		usage: BufferUsage.Uniform | BufferUsage.CopyDst,
	});
	const ticket = kernel.dispatch({
		resources: { params: paramsBuffer },
		dispatch: { x: 1, y: 1, z: 1 },
	});
	paramsBuffer.destroy();
	assert.equal(paramsBuffer._gpuResource.destroyed, false);
	await ticket.done;
	assert.equal(paramsBuffer._gpuResource.destroyed, true);
	runtime.destroy();
}

await testReadTextureSkipsPaddingInNormalizedConversion();
await testReadTextureNormalizesBGRA8Unorm();
await testKernelSchemaValidation();
await testDispatchValidationRules();
await testRuntimeOwnedResourceDestroyIsDeferredUntilDispatchDone();
console.log("WebGPU compute runtime tests passed");
