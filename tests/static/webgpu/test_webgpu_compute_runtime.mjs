import assert from "node:assert/strict";

import { createWebGPUResourceManager } from "../../helpers/fakes.mjs";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";
import {
	BufferUsage,
	TextureUsage,
} from "../../../src/backends/types.ts";
import { ComputeRuntime } from "../../../src/backends/webgpu/ComputeRuntime.ts";
import { createWebGPUComputeFacade } from "../../../src/backends/webgpu/ComputeFacade.ts";
import { WebGPUCommandEncoder } from "../../../src/backends/webgpu/WebGPUCommandEncoder.ts";
import { float32ToFloat16Bits } from "../../../src/foundation/Float16.ts";

import { FakeWebGPUBackend } from "../../helpers/fakes.mjs";

function createRuntime() {
	const backend = new FakeWebGPUBackend();
	return new ComputeRuntime(createWebGPUComputeFacade(backend));
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
	const runtime = createRuntime();
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
	const rgba = readback.toRGBAFloat32();
	assert.equal(normalized.length, 24);
	assert.equal(rgba.length, 24);
	nearlyEqual(normalized[0], 10 / 255);
	nearlyEqual(normalized[4], 40 / 255);
	nearlyEqual(normalized[12], 100 / 255);
	nearlyEqual(normalized[20], 160 / 255);
	nearlyEqual(rgba[0], 10 / 255);
	nearlyEqual(rgba[4], 40 / 255);
	nearlyEqual(rgba[12], 100 / 255);
	nearlyEqual(rgba[20], 160 / 255);
	runtime.destroy();
}

async function testReadTextureNormalizesBGRA8Unorm() {
	const runtime = createRuntime();
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
	const rgba = readback.toRGBAFloat32();
	assert.equal(normalized.length, 4);
	nearlyEqual(normalized[0], 10 / 255);
	nearlyEqual(normalized[1], 20 / 255);
	nearlyEqual(normalized[2], 30 / 255);
	nearlyEqual(normalized[3], 255 / 255);
	nearlyEqual(rgba[0], 10 / 255);
	nearlyEqual(rgba[1], 20 / 255);
	nearlyEqual(rgba[2], 30 / 255);
	nearlyEqual(rgba[3], 255 / 255);
	runtime.destroy();
}

async function testReadTextureDecodesRGBA16FloatWithPadding() {
	const runtime = createRuntime();
	const texture = runtime.createTexture({
		width: 3,
		height: 2,
		format: TextureFormat.RGBA16Float,
		usage:
			TextureUsage.CopyDst |
			TextureUsage.CopySrc |
			TextureUsage.TextureBinding,
		label: "HalfFloatTexture",
	});
	const bytesPerRow = 256;
	const upload = new Uint8Array(bytesPerRow * 2);
	const view = new DataView(upload.buffer);
	const values = [
		2, 1, 0.5, 1,
		4, 2, 1, 1,
		8, 4, 2, 1,
		16, 8, 4, 1,
		32, 16, 8, 1,
		64, 32, 16, 1,
	];
	for (let pixel = 0; pixel < 6; pixel++) {
		const row = Math.floor(pixel / 3);
		const column = pixel % 3;
		const base = row * bytesPerRow + column * 8;
		for (let component = 0; component < 4; component++) {
			view.setUint16(
				base + component * 2,
				float32ToFloat16Bits(values[pixel * 4 + component]),
				true
			);
		}
	}
	runtime.writeTexture(
		texture,
		upload,
		{ bytesPerRow, rowsPerImage: 2 },
		{ width: 3, height: 2, depthOrArrayLayers: 1 }
	);

	const readback = await runtime.readTexture({
		texture,
		format: TextureFormat.RGBA16Float,
	});
	assert.equal(readback.bytesPerRow, 256);
	const rgba = readback.toRGBAFloat32();
	assert.equal(rgba.length, 24);
	nearlyEqual(rgba[0], 2);
	nearlyEqual(rgba[4], 4);
	nearlyEqual(rgba[12], 16);
	nearlyEqual(rgba[20], 64);
	runtime.destroy();
}

async function testKernelSchemaValidation() {
	const runtime = createRuntime();
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
	const runtime = createRuntime();
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

async function testBindingKindsIgnoreNativeLookingMetadata() {
	const backend = new FakeWebGPUBackend();
	const runtime = new ComputeRuntime(createWebGPUComputeFacade(backend));
	const buffer = runtime.createBuffer({ size: 16, usage: BufferUsage.Uniform });
	const sampler = runtime.createSampler({});
	// Extra metadata must not override the declared renderer resource kind.
	Object.assign(buffer, { width: 1, height: 1, [Symbol.toStringTag]: "GPUTexture" });
	Object.assign(sampler, { size: 16, createView() {} });
	const kernel = await runtime.createKernel({
		code: "@compute @workgroup_size(1) fn csMain() {}",
		bindings: [
			{ key: "params", binding: 0, type: "buffer" },
			{ key: "sampler", binding: 1, type: "sampler" },
		],
		workgroupSize: { x: 1 },
	});
	const ticket = kernel.dispatch({
		resources: { params: buffer, sampler }, dispatch: { x: 1 },
	});
	assert.equal(backend.bindingGroups.at(-1).entries[0].resource, buffer);
	assert.equal(backend.bindingGroups.at(-1).entries[1].resource, sampler);
	await ticket.done;
	runtime.destroy();
}

async function testSchemaBindingsRequireLiveResourcesFromTheRuntimeStore() {
	const backend = new FakeWebGPUBackend();
	const foreign = new FakeWebGPUBackend();
	const runtime = new ComputeRuntime(createWebGPUComputeFacade(backend));
	const make = {
		buffer: (owner) => owner.createBuffer({ size: 16, usage: BufferUsage.Uniform }),
		texture: (owner) => owner.createTexture({
			width: 1, height: 1, format: TextureFormat.RGBA8Unorm,
			usage: TextureUsage.TextureBinding,
		}),
		sampler: (owner) => owner.createSampler({}),
	};
	const retired = Object.fromEntries(Object.entries(make).map(([kind, create]) =>
		[kind, create(backend)]));
	backend.resourceManager.reset();
	const owned = Object.fromEntries(Object.entries(make).map(([kind, create]) =>
		[kind, create(backend)]));
	for (const kind of Object.keys(make)) {
		const kernel = await runtime.createKernel({
			code: "@compute @workgroup_size(1) fn csMain() {}",
			bindings: [{ key: "value", binding: 0, type: kind }],
			workgroupSize: { x: 1 },
		});
		const destroyed = make[kind](backend);
		destroyed.destroy();
		const otherKind = kind === "buffer" ? owned.texture : owned.buffer;
		const forged = { ...owned[kind] };
		const native = kind === "buffer" ? backend.resourceManager.getBuffer(owned[kind]) :
			kind === "texture" ? backend.resourceManager.getTexture(owned[kind]).texture :
			backend.resourceManager.getSampler(owned[kind]);
		const initial = [backend.createBindingGroupCalls, backend.createCommandEncoderCalls,
			backend.submits];
		const invalidResources = [
			make[kind](foreign), destroyed, retired[kind], forged, native, otherKind,
		];
		if (kind === "texture") {
			invalidResources.push(backend.resourceManager.getTexture(owned.texture).view);
		}
		for (const invalid of invalidResources) {
			for (const override of [false, true]) {
				assert.throws(() => kernel.dispatch({
					resources: { value: override ? owned[kind] : invalid },
					overrideEntries: override ? [{ binding: 0, resource: invalid }] : undefined,
					dispatch: { x: 1 },
				}), /expects|owned|not live|renderer resource/i,
				`${kind} schema entries and overrides must reject invalid resources`);
			}
		}
		assert.deepEqual([backend.createBindingGroupCalls, backend.createCommandEncoderCalls,
			backend.submits], initial, "invalid bindings must fail before native work is created");
		const ticket = kernel.dispatch({ resources: { value: owned[kind] }, dispatch: { x: 1 } });
		await ticket.done;
		kernel.destroy();
	}
	for (const resource of Object.values(owned)) resource.destroy();
	foreign.resourceManager.reset();
	runtime.destroy();
}

async function testBufferRangesPreserveBindingsAndRetainTheirUnderlyingBuffer() {
	const backend = new FakeWebGPUBackend();
	const foreign = new FakeWebGPUBackend();
	const runtime = new ComputeRuntime(createWebGPUComputeFacade(backend));
	const kernel = await runtime.createKernel({
		code: "@compute @workgroup_size(1) fn csMain() {}",
		bindings: [
			{ key: "first", binding: 0, type: "buffer" },
			{ key: "second", binding: 1, type: "buffer" },
		],
		workgroupSize: { x: 1 },
	});
	const buffer = runtime.createBuffer({ size: 64, usage: BufferUsage.Uniform });
	const native = runtime._resourceManager.getBuffer(buffer);
	const first = { buffer, offset: 0, size: 16 };
	const second = { buffer, offset: 32, size: 16 };
	const dead = backend.createBuffer({ size: 64, usage: BufferUsage.Uniform });
	dead.destroy();
	for (const invalidBuffer of [
		foreign.createBuffer({ size: 64, usage: BufferUsage.Uniform }), dead, native,
		{ ...buffer }, backend.createTexture({
			width: 1, height: 1, format: TextureFormat.RGBA8Unorm, usage: TextureUsage.TextureBinding,
		}),
	]) {
		assert.throws(() => kernel.dispatch({
			resources: { first, second },
			overrideEntries: [{ binding: 0, resource: { buffer: invalidBuffer, offset: 0, size: 16 } }],
			dispatch: { x: 1 },
		}), /expects|owned|not live|renderer resource/i);
	}
	assert.equal(backend.createBindingGroupCalls, 0);
	assert.equal(backend.createCommandEncoderCalls, 0);
	const ticket = kernel.dispatch({
		resources: { first: { buffer, offset: 16, size: 16 }, second },
		overrideEntries: [{ binding: 0, resource: first }], dispatch: { x: 1 },
	});
	assert.deepEqual(backend.bindingGroups.at(-1).entries,
		[{ binding: 0, resource: first }, { binding: 1, resource: second }]);
	assert.equal(runtime.getResourceStats().inflightReferenceCount, 3,
		"two ranges retain one buffer reference plus the kernel's module and pipeline");
	buffer.destroy();
	assert.equal(native.destroyed, false, "ranges keep their underlying buffer alive in flight");
	await ticket.done;
	assert.equal(native.destroyed, true);
	assert.equal(runtime.getResourceStats().inflightReferenceCount, 0);
	foreign.resourceManager.reset();
	backend.resourceManager.reset();
	runtime.destroy();
}

async function testFailedDispatchReleasesBindingsAndRetainedResources() {
	for (const stage of ["foreign group", "destroyed group", "encoder", "recording", "finish",
		"submission", "cleanup", "completion cleanup"]) {
		const backend = new FakeWebGPUBackend();
		const foreign = new FakeWebGPUBackend();
		const failure = new Error(`dispatch failure: ${stage}`);
		backend.createCommandEncoder = () => {
			if (stage === "encoder") throw failure;
			return new WebGPUCommandEncoder({
				beginComputePass() {
					return {
						setPipeline() {},
						setBindGroup() {},
						dispatchWorkgroups() {
							if (stage === "recording" || stage === "cleanup") throw failure;
						},
						end() {},
					};
				},
				finish() {
					if (stage === "finish") throw failure;
					return {};
				},
			}, { resourceManager: backend.resourceManager, createPassTimestampWrites() {} }, {});
		};
		backend.submit = () => { if (stage === "submission") throw failure; };
		if (stage.endsWith("cleanup")) {
			const createBindingGroup = backend.createBindingGroup.bind(backend);
			backend.createBindingGroup = (desc) => {
				const group = createBindingGroup(desc);
				const destroy = group.destroy;
				group.destroy = () => { destroy(); throw new Error("binding cleanup failure"); };
				return group;
			};
		}
		const runtime = new ComputeRuntime(createWebGPUComputeFacade(backend));
		const kernel = await runtime.createKernel({
			code: "@compute @workgroup_size(1) fn csMain() {}",
			bindings: [{ key: "params", binding: 0, type: "buffer" }],
			workgroupSize: { x: 1 },
		});
		const buffer = runtime.createBuffer({ size: 16, usage: BufferUsage.Uniform });
		const native = backend.resourceManager.getBuffer(buffer);
		let extraBindGroups;
		if (stage === "foreign group" || stage === "destroyed group") {
			const owner = stage === "foreign group" ? foreign : backend;
			const group = owner.resourceManager.registerBindingGroup({});
			if (stage === "destroyed group") group.destroy();
			extraBindGroups = [{ index: 1, group }];
		}
		const dispatch = () => kernel.dispatch({
			resources: { params: buffer }, dispatch: { x: 1 }, extraBindGroups,
		});
		if (stage === "completion cleanup") {
			await assert.rejects(dispatch().done, /binding cleanup failure/);
		} else {
			const expectedError = stage.endsWith("group") ? /owned|not live/i :
				stage === "cleanup" ? /binding cleanup failure/ : (error) => error === failure;
			assert.throws(dispatch, expectedError);
		}
		assert.equal(backend.bindingGroups.at(-1).destroyed, true,
			`${stage}: a failed dispatch must release its transient binding group`);
		assert.equal(runtime.getResourceStats().inflightReferenceCount, 0,
			`${stage}: a failed dispatch must release all acquired references`);
		assert.equal(backend.resourceManager.getBuffer(buffer), native,
			`${stage}: failure must leave caller-owned resources usable`);
		runtime.destroy();
		assert.equal(native.destroyed, true,
			`${stage}: destruction must not wait for a ticket that was never returned`);
		assert.equal(runtime.getResourceStats().ownedResourceCount, 0);
		foreign.resourceManager.reset();
		backend.resourceManager.reset();
	}
}

async function testRuntimeOwnedResourceDestroyIsDeferredUntilDispatchDone() {
	const backend = new FakeWebGPUBackend();
	let complete;
	backend.queue.onSubmittedWorkDone = () => new Promise((resolve) => { complete = resolve; });
	const facade = createWebGPUComputeFacade(backend);
	const runtime = new ComputeRuntime(facade);
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
	const nativeBuffer = runtime._resourceManager.getBuffer(paramsBuffer);
	paramsBuffer.destroy();
	assert.equal(nativeBuffer.destroyed, false);
	const bindingCount = backend.createBindingGroupCalls;
	const encoderCount = backend.createCommandEncoderCalls;
	assert.throws(() => kernel.dispatch({
		resources: { params: paramsBuffer }, dispatch: { x: 1 },
	}), /not live|destroyed/i);
	assert.throws(() => facade.createBindingGroup({
		entries: [{ binding: 0, resource: { buffer: paramsBuffer } }],
	}), /not live|destroyed/i);
	assert.throws(() => runtime.writeBuffer(paramsBuffer, new Uint8Array(4)), /not live|destroyed/i);
	await assert.rejects(() => runtime.readBuffer({ buffer: paramsBuffer, size: 0 }),
		/not live|destroyed/i);
	assert.equal(backend.createBindingGroupCalls, bindingCount);
	assert.equal(backend.createCommandEncoderCalls, encoderCount);
	assert.equal(nativeBuffer.destroyed, false);
	complete();
	await ticket.done;
	assert.equal(nativeBuffer.destroyed, true);
	runtime.destroy();
}

async function testRuntimeDestroyInvalidatesInflightTextureAndSampler() {
	const backend = new FakeWebGPUBackend();
	let complete;
	backend.queue.onSubmittedWorkDone = () => new Promise((resolve) => { complete = resolve; });
	const facade = createWebGPUComputeFacade(backend);
	const runtime = new ComputeRuntime(facade);
	const kernel = await runtime.createKernel({
		code: "@compute @workgroup_size(1) fn csMain() {}",
		bindings: [
			{ key: "image", binding: 0, type: "texture" },
			{ key: "sampler", binding: 1, type: "sampler" },
		],
		workgroupSize: { x: 1 },
	});
	const texture = runtime.createTexture({
		width: 1, height: 1, format: TextureFormat.RGBA8Unorm,
		usage: TextureUsage.CopySrc | TextureUsage.CopyDst | TextureUsage.TextureBinding,
	});
	const sampler = runtime.createSampler({});
	const nativeTexture = runtime._resourceManager.getTexture(texture).texture;
	const ticket = kernel.dispatch({ resources: { image: texture, sampler }, dispatch: { x: 1 } });
	runtime.destroy();
	assert.equal(nativeTexture.destroyed, false);
	assert.equal(backend.samplerDestroyCalls, 0);
	for (const resource of [texture, sampler]) {
		assert.throws(() => facade.createBindingGroup({ entries: [{ binding: 0, resource }] }),
			/not live|destroyed/i);
	}
	assert.throws(() => facade.writeTexture(texture, new Uint8Array(4),
		{ bytesPerRow: 4 }, { width: 1, height: 1 }), /not live|destroyed/i);
	texture.destroy();
	sampler.destroy();
	complete();
	await ticket.done;
	assert.equal(nativeTexture.destroyed, true);
	assert.equal(backend.samplerDestroyCalls, 1);
}

async function testReadbacksRequireTheRuntimeBackendOwner() {
	const backendA = new FakeWebGPUBackend();
	const backendB = new FakeWebGPUBackend();
	const runtimeA = new ComputeRuntime(createWebGPUComputeFacade(backendA));
	const runtimeB = new ComputeRuntime(createWebGPUComputeFacade(backendB));
	const foreignBuffer = runtimeA.createBuffer({
		size: 16, usage: BufferUsage.CopySrc | BufferUsage.CopyDst,
	});
	const foreignTexture = runtimeA.createTexture({
		width: 1, height: 1, format: TextureFormat.RGBA8Unorm,
		usage: TextureUsage.CopySrc | TextureUsage.CopyDst,
	});
	const initialBufferCount = backendB.buffers.length;
	const initialEncoderCount = backendB.createCommandEncoderCalls;
	await assert.rejects(() => runtimeB.readBuffer({ buffer: foreignBuffer }),
		/owned|backend|WebGPU/i, "buffer readback must reject another backend's resource");
	await assert.rejects(() => runtimeB.readBuffer({ buffer: foreignBuffer, size: 0 }),
		/owned|backend|WebGPU/i, "empty reads still require valid resource ownership");
	await assert.rejects(() => runtimeB.readTexture({
		texture: foreignTexture, format: TextureFormat.RGBA8Unorm,
	}), /owned|backend|WebGPU/i,
	"texture readback must reject another backend's resource");
	assert.equal(backendB.buffers.length, initialBufferCount,
		"foreign readbacks cannot allocate staging buffers");
	assert.equal(backendB.createCommandEncoderCalls, initialEncoderCount,
		"foreign readbacks cannot record native copies");
	assert.equal(backendB.submits, 0);

	// Backend-created resources remain borrowable by a runtime on that same backend.
	const borrowedBuffer = backendB.createBuffer({
		size: 4, usage: BufferUsage.CopySrc | BufferUsage.CopyDst,
	});
	backendB.writeBuffer(borrowedBuffer, new Uint8Array([1, 2, 3, 4]));
	const beforeRawRead = backendB.buffers.length;
	await assert.rejects(() => runtimeB.readBuffer({
		buffer: backendB.resourceManager.getBuffer(borrowedBuffer),
	}),
		/owned|backend|WebGPU/i, "raw native buffers cannot bypass facade ownership");
	assert.equal(backendB.buffers.length, beforeRawRead);
	const readback = await runtimeB.readBuffer({ buffer: borrowedBuffer });
	assert.deepEqual(Array.from(readback.bytes), [1, 2, 3, 4]);
	const nativeBorrowed = backendB.resourceManager.getBuffer(borrowedBuffer);
	runtimeB.destroy();
	assert.equal(nativeBorrowed.destroyed, false,
		"runtime readback does not acquire ownership of a borrowed source buffer");
	borrowedBuffer.destroy();
	runtimeA.destroy();
}

async function testGetResourceStatsForDebugging() {
	const runtime = createRuntime();
	const kernel = await runtime.createKernel({
		code: "@compute @workgroup_size(1) fn csMain() {}",
		bindings: [{ key: "params", binding: 0, type: "buffer" }],
		workgroupSize: { x: 1, y: 1, z: 1 },
	});
	const paramsBuffer = runtime.createBuffer({
		size: 16,
		usage: BufferUsage.Uniform | BufferUsage.CopyDst,
	});

	const beforeDispatch = runtime.getResourceStats();
	assert.equal(beforeDispatch.destroyed, false);
	assert.equal(beforeDispatch.kernelCount, 1);
	assert.equal(beforeDispatch.ownedResourceCount, 3);
	assert.equal(beforeDispatch.activeResourceCount, 3);
	assert.equal(beforeDispatch.destroyRequestedResourceCount, 0);
	assert.equal(beforeDispatch.inflightReferenceCount, 0);
	assert.equal(beforeDispatch.byKind.buffer, 1);
	assert.equal(beforeDispatch.byKind.shaderModule, 1);
	assert.equal(beforeDispatch.byKind.computePipeline, 1);

	const ticket = kernel.dispatch({
		resources: { params: paramsBuffer },
		dispatch: { x: 1, y: 1, z: 1 },
	});
	paramsBuffer.destroy();

	const pendingDestroy = runtime.getResourceStats();
	assert.equal(pendingDestroy.ownedResourceCount, 3);
	assert.equal(pendingDestroy.activeResourceCount, 2);
	assert.equal(pendingDestroy.destroyRequestedResourceCount, 1);
	assert.equal(pendingDestroy.inflightReferenceCount, 3);
	assert.equal(pendingDestroy.byKind.buffer, 1);

	await ticket.done;

	const afterDispatch = runtime.getResourceStats();
	assert.equal(afterDispatch.ownedResourceCount, 2);
	assert.equal(afterDispatch.activeResourceCount, 2);
	assert.equal(afterDispatch.destroyRequestedResourceCount, 0);
	assert.equal(afterDispatch.inflightReferenceCount, 0);
	assert.equal(afterDispatch.byKind.buffer, 0);
	assert.equal(afterDispatch.byKind.shaderModule, 1);
	assert.equal(afterDispatch.byKind.computePipeline, 1);

	kernel.destroy();
	const afterKernelDestroy = runtime.getResourceStats();
	assert.equal(afterKernelDestroy.kernelCount, 0);
	assert.equal(afterKernelDestroy.ownedResourceCount, 0);

	runtime.destroy();
	const afterRuntimeDestroy = runtime.getResourceStats();
	assert.equal(afterRuntimeDestroy.destroyed, true);
	assert.equal(afterRuntimeDestroy.ownedResourceCount, 0);
}

await testReadTextureSkipsPaddingInNormalizedConversion();
await testReadTextureNormalizesBGRA8Unorm();
await testReadTextureDecodesRGBA16FloatWithPadding();
await testKernelSchemaValidation();
await testDispatchValidationRules();
await testBufferRangesPreserveBindingsAndRetainTheirUnderlyingBuffer();
await testBindingKindsIgnoreNativeLookingMetadata();
await testSchemaBindingsRequireLiveResourcesFromTheRuntimeStore();
await testFailedDispatchReleasesBindingsAndRetainedResources();
await testRuntimeOwnedResourceDestroyIsDeferredUntilDispatchDone();
await testRuntimeDestroyInvalidatesInflightTextureAndSampler();
await testReadbacksRequireTheRuntimeBackendOwner();
await testGetResourceStatsForDebugging();
console.log("WebGPU compute runtime tests passed");
