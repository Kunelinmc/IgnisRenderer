import assert from "node:assert/strict";
import {
	BufferUsage,
	TextureFormat,
	TextureUsage,
} from "../src/renderers/types.ts";
import { ComputeRuntime } from "../src/renderers/webgpu/ComputeRuntime.ts";

class FakeGPUBuffer {
	constructor(size) {
		this.size = size;
		this.destroyed = false;
		this._bytes = new Uint8Array(size);
		this._bytes.fill(231);
	}

	get [Symbol.toStringTag]() {
		return "GPUBuffer";
	}

	mapAsync(_mode, _offset = 0, _size = this.size) {
		return Promise.resolve();
	}

	getMappedRange(offset = 0, size = this.size - offset) {
		const start = Math.max(0, Math.floor(offset));
		const end = Math.min(this._bytes.length, start + Math.floor(size));
		return this._bytes.buffer.slice(start, end);
	}

	unmap() {}

	destroy() {
		this.destroyed = true;
	}
}

class FakeGPUTextureView {
	constructor(texture) {
		this.texture = texture;
	}

	get [Symbol.toStringTag]() {
		return "GPUTextureView";
	}
}

class FakeGPUTexture {
	constructor(width, height, bytesPerPixel) {
		this.width = width;
		this.height = height;
		this.bytesPerPixel = bytesPerPixel;
		this.destroyed = false;
		this._bytes = new Uint8Array(width * height * bytesPerPixel);
	}

	get [Symbol.toStringTag]() {
		return "GPUTexture";
	}

	createView(_desc) {
		return new FakeGPUTextureView(this);
	}

	destroy() {
		this.destroyed = true;
	}
}

class FakeNativeCommandEncoder {
	constructor() {
		this._ops = [];
	}

	copyTextureToBuffer(src, dst, size) {
		this._ops.push(() => {
			const texture = src.texture;
			const buffer = dst.buffer;
			const width = Math.max(1, Math.floor(size.width));
			const height = Math.max(1, Math.floor(size.height));
			const bytesPerPixel = 4;
			const srcBytesPerRow = width * bytesPerPixel;
			const dstBytesPerRow = Math.max(srcBytesPerRow, dst.bytesPerRow);
			for (let y = 0; y < height; y++) {
				const srcOffset = y * srcBytesPerRow;
				const dstOffset = y * dstBytesPerRow;
				buffer._bytes.set(
					texture._bytes.subarray(srcOffset, srcOffset + srcBytesPerRow),
					dstOffset
				);
			}
		});
	}

	copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
		this._ops.push(() => {
			const copied = src._bytes.subarray(srcOffset, srcOffset + size);
			dst._bytes.set(copied, dstOffset);
		});
	}

	finish() {
		const ops = [...this._ops];
		return {
			execute() {
				for (const op of ops) {
					op();
				}
			},
		};
	}
}

class FakeComputeCommandEncoder {
	constructor(backend) {
		this._backend = backend;
	}

	beginComputePass(_desc) {}
	setComputePipeline(_pipeline) {}
	setBindingGroup(_index, _group) {}
	dispatchWorkgroups(x, y = 1, z = 1) {
		this._backend.dispatches.push([x, y, z]);
	}
	endComputePass() {}
	finish() {
		return { execute() {} };
	}
}

class FakeGPUQueue {
	writeTexture(dst, data, layout, size) {
		const targetTexture = dst.texture;
		const src = toUint8Array(data);
		const width = Math.max(1, Math.floor(size.width));
		const height = Math.max(1, Math.floor(size.height));
		const bytesPerPixel = 4;
		const rowSize = width * bytesPerPixel;
		const bytesPerRow = layout.bytesPerRow ?? rowSize;
		for (let y = 0; y < height; y++) {
			const srcOffset = y * bytesPerRow;
			const dstOffset = y * rowSize;
			targetTexture._bytes.set(
				src.subarray(srcOffset, srcOffset + rowSize),
				dstOffset
			);
		}
	}

	submit(commands) {
		for (const command of commands) {
			command.execute?.();
		}
	}

	onSubmittedWorkDone() {
		return new Promise((resolve) => {
			setTimeout(resolve, 0);
		});
	}
}

class FakeWebGPUBackend {
	constructor() {
		this.type = "webgpu";
		this.queue = new FakeGPUQueue();
		this.dispatches = [];
		this.device = {
			createCommandEncoder: () => new FakeNativeCommandEncoder(),
			createBuffer: (desc) => new FakeGPUBuffer(desc.size),
			createBindGroupLayout: (_desc) => ({}),
			createPipelineLayout: (_desc) => ({}),
		};
	}

	createSampler(desc) {
		return {
			label: desc.label,
			destroy() {},
			_gpuResource: { [Symbol.toStringTag]: "GPUSampler" },
		};
	}

	async createShaderModule(desc) {
		return { label: desc.label, destroy() {} };
	}

	createComputePipeline(desc) {
		return {
			label: desc.label,
			destroy() {},
			_gpuResource: {
				getBindGroupLayout() {
					return {};
				},
			},
		};
	}

	createBuffer(desc) {
		return {
			size: desc.size,
			destroy() {
				this._gpuResource.destroy();
			},
			_gpuResource: new FakeGPUBuffer(desc.size),
		};
	}

	createTexture(desc) {
		const texture = new FakeGPUTexture(desc.width, desc.height, 4);
		return {
			width: desc.width,
			height: desc.height,
			destroy() {
				this._gpuResource.destroy();
			},
			_gpuResource: texture,
			_webgpuTexture: {
				texture,
				view: texture.createView(),
			},
		};
	}

	createBindingGroup(desc) {
		return {
			label: desc.label,
			entries: desc.entries,
			destroy() {},
		};
	}

	createTextureView(texture, desc) {
		const gpuTexture = texture._gpuResource;
		return gpuTexture.createView(desc);
	}

	createCommandEncoder() {
		return new FakeComputeCommandEncoder(this);
	}

	submit(commands) {
		this.queue.submit(commands);
	}

	writeBuffer(buffer, data, offset = 0) {
		const view = toUint8Array(data);
		buffer._gpuResource._bytes.set(view, offset);
	}

	getTextureForSlot(texture, _slotIndex) {
		return texture;
	}

	registerExternalTexture(_texture, _resource, _uploadedVersion, _mipLevelCount) {}
	unregisterExternalTexture(_texture) {}
}

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
await testKernelSchemaValidation();
await testDispatchValidationRules();
await testRuntimeOwnedResourceDestroyIsDeferredUntilDispatchDone();
console.log("WebGPU compute runtime tests passed");
