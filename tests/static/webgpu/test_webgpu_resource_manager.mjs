import assert from "node:assert/strict";

import { createWebGPUResourceManager, FakeWebGPUBackend } from "../../helpers/fakes.mjs";
import { Texture } from "../../../src/core/Texture.ts";
import { WebGPUResourceManager } from "../../../src/backends/webgpu/WebGPUResourceManager.ts";
import { BufferUsage, TextureUsage } from "../../../src/backends/types.ts";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";
import { WebGPUCanvasTargetManager } from "../../../src/backends/webgpu/WebGPUCanvasTargetManager.ts";

const store = createWebGPUResourceManager();
const foreign = createWebGPUResourceManager();
let releases = 0;
const native = { destroy() { releases++; } };
const buffer = store.registerBuffer(native, { size: 256 }, () => native.destroy());
assert.equal(buffer.kind, "buffer");
assert.equal(store.getBuffer(buffer), native);
assert.throws(() => foreign.getBuffer(buffer), /owned|owner|foreign/i);
assert.throws(() => store.getBuffer({ ...buffer }), /owned|owner|foreign/i);
const prototypeForgery = Object.create(Object.getPrototypeOf(buffer));
assert.throws(() => store.getBuffer(prototypeForgery), /owned|owner|foreign/i,
	"copying a facade prototype cannot forge private registration records");
assert.throws(() => store.getBuffer(new Proxy(buffer, {})), /owned|owner|foreign/i,
	"unregistered proxies cannot transfer private resource ownership");
assert.throws(() => store.getBuffer({ _gpuResource: native }), /WebGPU|owned/i);
assert.equal(Reflect.ownKeys(buffer).some((key) =>
	typeof key === "string" && key.startsWith("_")), false);
assert.equal(Object.values(buffer).includes(native), false);
buffer.destroy();
buffer.destroy();
assert.equal(releases, 1);
assert.throws(() => store.getBuffer(buffer), /live|destroyed|retired/i);

let borrowedDestroyed = 0;
const texture = { destroy() { borrowedDestroyed++; } };
const view = {};
const borrowed = store.registerTexture({ texture, view }, { width: 4, height: 4 });
const live = store.registerBuffer(native, { size: 32 }, () => native.destroy());
store.reset();
assert.equal(releases, 2, "store retirement disposes each remaining allocation once");
assert.equal(borrowedDestroyed, 0, "borrowed textures are never destroyed by the store");
assert.throws(() => store.getTexture(borrowed), /live|destroyed|retired/i);
assert.throws(() => store.getBuffer(live), /live|destroyed|retired/i);
live.destroy();
assert.equal(releases, 2);
const restored = store.registerBuffer(native, { size: 64 });
assert.equal(store.getBuffer(restored), native);
assert.throws(() => store.getBuffer(live), /live|destroyed|retired/i);
store.reset();

globalThis.GPUBufferUsage ??= { UNIFORM: 1 };
globalThis.GPUTextureUsage ??= { TEXTURE_BINDING: 1 };
const registered = new Set();
const uploads = [];
let allocationsDestroyed = 0;
let borrowedAllocationsDestroyed = 0;
const nativeManagedBuffer = { destroy() { allocationsDestroyed++; } };
const manager = new WebGPUResourceManager({
	device: {
		features: new Set(),
		createBuffer() {
			return nativeManagedBuffer;
		},
		createTexture() {
			return { createView() { return {}; }, destroy() { allocationsDestroyed++; } };
		},
	},
	queue: {
		writeBuffer(...args) { uploads.push(["buffer", ...args]); },
		writeTexture(...args) { uploads.push(["texture", ...args]); },
	},
	assertDeviceOperational() {},
	resolveSupportedSampleCount(value) { return value; },
	runValidationScope(_label, operation) { return operation(); },
	createManagedDestroy(target, { dispose }) {
		registered.add(target);
		return () => {
			assert.equal(registered.delete(target), true,
				"managed registration must be removed exactly once");
			dispose();
		};
	},
});
const managedBuffer = manager.createBuffer({ size: 16, usage: BufferUsage.Uniform });
const managedTexture = manager.createTexture({
	width: 2, height: 2, format: TextureFormat.RGBA8Unorm,
	usage: TextureUsage.TextureBinding,
});
assert.equal(manager.getBuffer(managedBuffer), nativeManagedBuffer,
	"native allocation and resource resolution must share one manager owner");
const nativeBorrowedTexture = { destroy() { borrowedAllocationsDestroyed++; } };
const borrowedTexture = manager.registerTexture({ texture: nativeBorrowedTexture, view: {} },
	{ width: 2, height: 2 });
assert.equal(manager.getTexture(borrowedTexture).texture, nativeBorrowedTexture,
	"already-created native resources register into the same owner as allocated resources");
assert.equal(manager.createTextureView(managedTexture), manager.getTexture(managedTexture).view);
const data = new Uint8Array([1, 2, 3, 4]);
manager.writeBuffer(managedBuffer, data, 4);
const textureData = new Uint8Array(16);
manager.writeTexture(managedTexture, textureData, { bytesPerRow: 8 }, { width: 2, height: 2 });
assert.deepEqual(uploads[0], ["buffer", nativeManagedBuffer, 4, data]);
assert.equal(uploads[1][1].texture, manager.getTexture(managedTexture).texture);
assert.equal(uploads[1][2], textureData);
assert.deepEqual(uploads[1][3], { offset: 0, bytesPerRow: 8, rowsPerImage: undefined });
assert.deepEqual(uploads[1][4], { width: 2, height: 2 });
assert.equal(registered.size, 2);
manager.reset();
assert.equal(registered.size, 0, "backend teardown clears managed finalization registrations");
assert.equal(allocationsDestroyed, 2);
assert.equal(borrowedAllocationsDestroyed, 0);
assert.throws(() => manager.getTexture(borrowedTexture), /live|destroyed|retired/i);
managedBuffer.destroy();
managedTexture.destroy();
assert.equal(allocationsDestroyed, 2);

const presentationStore = createWebGPUResourceManager();
const targets = new WebGPUCanvasTargetManager(presentationStore);
let presentedDestroyed = 0;
const presented = { createView() { return {}; }, destroy() { presentedDestroyed++; } };
const context = { getCurrentTexture() { return presented; } };
const canvas = { width: 8, height: 4 };
const firstPresentation = targets.getCanvasColorTexture(context, canvas);
assert.equal(targets.getCanvasColorTexture(context, canvas), firstPresentation,
	"presentation facades are stable while their acquired texture is current");
targets.resetCurrentCanvasTargets();
assert.throws(() => presentationStore.getTexture(firstPresentation), /live|destroyed|retired/i);
const nextPresentation = targets.getCanvasColorTexture(context, canvas);
assert.notEqual(nextPresentation, firstPresentation);
assert.equal(presentationStore.getTexture(nextPresentation).texture, presented);
targets.release();
assert.throws(() => presentationStore.getTexture(nextPresentation), /live|destroyed|retired/i);
assert.equal(presentedDestroyed, 0, "releasing presentation facades preserves browser ownership");

for (const retire of [
	(resource) => resource.destroy(),
	(resource) => presentationStore.invalidate(resource),
	() => presentationStore.reset(),
]) {
	const previous = targets.getCanvasColorTexture(context, canvas);
	retire(previous);
	const replacement = targets.getCanvasColorTexture(context, canvas);
	assert.notEqual(replacement, previous, "an unavailable canvas facade is recreated");
	assert.equal(presentationStore.getTexture(replacement).texture, presented);
	assert.throws(() => presentationStore.getTexture(previous), /live|destroyed|retired/i);
}
targets.release();
presentationStore.reset();
assert.equal(presentedDestroyed, 0, "canvas facade recovery never destroys browser-owned textures");

async function testManagedBufferCollectionPreservesExplicitTeardown() {
	const registry = new FinalizationRegistry(() => {});
	let unmaps = 0;
	let destroyed = 0;
	const nativeBuffer = {
		mapState: "mapped",
		unmap() { unmaps++; this.mapState = "unmapped"; },
		destroy() { destroyed++; },
	};
	const resources = createWebGPUResourceManager({
		device: { createBuffer() { return nativeBuffer; } },
		assertDeviceOperational() {},
		createManagedDestroy(target, { label, dispose }) {
			const token = {};
			registry.register(target, label, token);
			return () => {
				registry.unregister(token);
				dispose();
			};
		},
	});
	function createWeakBuffer() {
		const resource = resources.createBuffer({ size: 16, usage: BufferUsage.Uniform });
		resource.unmap();
		return new WeakRef(resource);
	}
	const reference = createWeakBuffer();
	assert.equal(unmaps, 1, "live buffer facades must still unmap their native allocation");
	for (let attempt = 0; attempt < 30; attempt++) {
		// WeakRef targets stay alive within one job; collect only after a job boundary.
		await new Promise((resolve) => setTimeout(resolve, 0));
		Bun.gc(true);
	}
	assert.equal(reference.deref(), undefined,
		"managed disposal must not retain an otherwise unreachable buffer facade");
	assert.equal(destroyed, 0, "facade collection does not replace explicit native disposal");
	resources.reset();
	resources.reset();
	assert.equal(destroyed, 1, "teardown must release allocations whose facades were collected");
}

function testSourceUploadsValidateOwnershipBeforeReadingPixels() {
	const backend = new FakeWebGPUBackend();
	const other = new FakeWebGPUBackend();
	const source = new Texture({ width: 1, height: 1 });
	let reads = 0;
	source.readPixelData = () => { reads++; return new Uint8Array(4); };
	const target = other.createTexture({
		width: 1, height: 1, format: TextureFormat.RGBA8Unorm, usage: TextureUsage.CopyDst,
	});
	assert.throws(() => backend.resourceManager.writeTexture(target, source), /owned|owner/i);
	assert.equal(reads, 0);
	target.destroy();
}

function testSourceUploadsPreserveExplicitRowsAndRejectBeforePartialWrites() {
	const backend = new FakeWebGPUBackend();
	const target = backend.createTexture({
		width: 2, height: 2, format: TextureFormat.RGBA8Unorm, usage: TextureUsage.CopyDst,
		mipLevelCount: 2,
	});
	const data = new Uint8Array([
		1, 2, 3, 4, 5, 6, 7, 8, 199, 199, 199, 199,
		21, 22, 23, 24, 25, 26, 27, 28,
	]);
	const source = new Texture({
		format: TextureFormat.RGBA8UnormSrgb,
		levels: [{ width: 2, height: 2, data, bytesPerRow: 12 }],
	});
	backend.resourceManager.writeTexture(target, source);
	assert.deepEqual(Array.from(backend.resourceManager.getTexture(target).texture._bytes),
		[1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28]);
	backend.textureWrites.length = 0;
	const invalidMip = new Texture({
		format: TextureFormat.RGBA8Unorm,
		levels: [
			{ width: 2, height: 2, data: new Uint8Array(16) },
			{ width: 1, height: 1, data: new Uint8Array(3), bytesPerRow: 4 },
		],
	});
	assert.throws(() => backend.resourceManager.writeTexture(target, invalidMip), /payload/i);
	assert.equal(backend.textureWrites.length, 0, "all mips must validate before the first write");
	target.destroy();
}

function testTextureWritesRequireCompleteArgumentForms() {
	const backend = new FakeWebGPUBackend();
	const target = backend.createTexture({
		width: 1, height: 1, format: TextureFormat.RGBA8Unorm, usage: TextureUsage.CopyDst,
	});
	const source = new Texture({ width: 1, height: 1, data: new Uint8Array(4) });
	const bytes = new Uint8Array(4);
	const layout = { bytesPerRow: 4 };
	const size = { width: 1, height: 1 };
	for (const args of [
		[bytes], [bytes, layout], [bytes, undefined, size],
		[source, layout], [source, undefined, size], [source, layout, size],
		[null], [{}], [new ArrayBuffer(4)], [new DataView(new ArrayBuffer(4))],
	]) {
		assert.throws(() => backend.resourceManager.writeTexture(target, ...args),
			/Texture|layout|size/i);
	}
	assert.equal(backend.textureWrites.length, 0);
	assert.equal(target.destroyed, false);
	target.destroy();
}

function testUnsupportedRawConversionRejectsBeforeAnyMipWrite() {
	const backend = new FakeWebGPUBackend();
	const target = backend.createTexture({ width: 2, height: 2,
		format: TextureFormat.RGBA16Float, usage: TextureUsage.CopyDst, mipLevelCount: 2 });
	for (const explicitLayout of [false, true]) {
		const source = new Texture({ format: TextureFormat.RGBA8Unorm, levels: [
			{ width: 2, height: 2, data: new Uint8ClampedArray(16).fill(255) },
			{ width: 1, height: 1, data: new Uint8Array(4),
				...(explicitLayout ? { bytesPerRow: 4 } : {}) },
		] });
		assert.throws(() => backend.resourceManager.writeTexture(target, source),
			/raw.*conversion|conversion.*raw|raw.*format|format.*compatible/i);
		assert.equal(backend.textureWrites.length, 0);
		assert.equal(backend.textureDestroyCalls, 0);
		assert.equal(target.destroyed, false);
		assert.equal(backend.createTextureCalls.length, 1);
	}
	target.destroy();
}

function testBytesWritesPreserveViewsAndExplicitCopyParameters() {
	const backend = new FakeWebGPUBackend();
	const target = backend.createTexture({
		width: 2, height: 2, format: TextureFormat.RGBA8Unorm, usage: TextureUsage.CopyDst,
		mipLevelCount: 2,
	});
	const backing = new Uint8Array([91, 92, 1, 2, 3, 4, 93, 94]);
	const bytes = new DataView(backing.buffer, 1, 6);
	const writes = [];
	backend.queue.writeTexture = (...args) => writes.push(args);
	backend.resourceManager.writeTexture(target, bytes,
		{ offset: 1, bytesPerRow: 4, rowsPerImage: 1, mipLevel: 1 },
		{ width: 1, height: 1, depthOrArrayLayers: 1 });
	assert.equal(writes.length, 1);
	assert.equal(writes[0][0].texture, backend.resourceManager.getTexture(target).texture);
	assert.equal(writes[0][0].mipLevel, 1);
	assert.equal(writes[0][1], bytes);
	assert.deepEqual(writes[0][2], { offset: 1, bytesPerRow: 4, rowsPerImage: 1 });
	assert.deepEqual(writes[0][3], { width: 1, height: 1, depthOrArrayLayers: 1 });
	assert.deepEqual(Array.from(backing), [91, 92, 1, 2, 3, 4, 93, 94]);
	target.destroy();
}

function testSourceWritesValidateTargetsAndAllMipRanges() {
	for (const { targetDesc, source, error } of [
		{ targetDesc: { usage: TextureUsage.TextureBinding },
			source: new Texture({ width: 2, height: 2, data: new Uint8Array(16) }),
			error: /CopyDst|COPY_DST/i },
		{ targetDesc: { usage: TextureUsage.RenderAttachment, sampleCount: 4 },
			source: new Texture({ width: 2, height: 2, data: new Uint8Array(16) }),
			error: /sample/i },
		{ targetDesc: {},
			source: new Texture({ width: 3, height: 2, data: new Uint8Array(24) }),
			error: /range|extent/i },
		{ targetDesc: {}, source: new Texture({ levels: [
			{ width: 2, height: 2, data: new Uint8Array(16) },
			{ width: 1, height: 1, data: new Uint8Array(4) },
		] }), error: /mip|range/i },
		{ targetDesc: { width: 4, height: 4, mipLevelCount: 2 },
			source: new Texture({ levels: [
				{ width: 4, height: 4, data: new Uint8Array(64) },
				{ width: 3, height: 3, data: new Uint8Array(36) },
			] }), error: /range|extent/i },
		{ targetDesc: { width: 4, height: 4, mipLevelCount: 2 },
			source: new Texture({ levels: [
				{ width: 4, height: 4, data: new Uint8Array(64) },
				{ width: NaN, height: 1, data: new Uint8Array(4) },
			] }), error: /range|extent/i },
		{ targetDesc: { width: 4, height: 4, mipLevelCount: 2 },
			source: new Texture({ levels: [
				{ width: 4, height: 4, data: new Uint8Array(64) },
				{ width: 1, height: NaN, data: new Uint8Array(4) },
			] }), error: /range|extent/i },
	]) {
		const backend = new FakeWebGPUBackend();
		const target = backend.createTexture({
			width: 2, height: 2, format: TextureFormat.RGBA8Unorm,
			usage: TextureUsage.CopyDst, ...targetDesc,
		});
		assert.throws(() => backend.resourceManager.writeTexture(target, source), error);
		assert.equal(backend.textureWrites.length, 0);
		assert.equal(backend.copyCalls.length, 0);
		assert.equal(target.destroyed, false);
		assert.equal(backend.createTextureCalls.length, 1);
		target.destroy();
	}
}

function testSourceWriteFailurePreservesExistingTarget() {
	const backend = new FakeWebGPUBackend();
	const target = backend.createTexture({
		width: 2, height: 2, format: TextureFormat.RGBA8Unorm, usage: TextureUsage.CopyDst,
	});
	const source = new Texture({
		width: 2, height: 2, format: TextureFormat.RGBA8Unorm, data: new Uint8Array(16).fill(29),
	});
	const write = backend.queue.writeTexture;
	backend.queue.writeTexture = () => { throw new Error("simulated write failure"); };
	assert.throws(() => backend.resourceManager.writeTexture(target, source), /write failure/);
	assert.equal(target.destroyed, false);
	assert.equal(backend.textureDestroyCalls, 0);
	backend.queue.writeTexture = write;
	backend.resourceManager.writeTexture(target, source);
	assert.deepEqual(backend.resourceManager.getTexture(target).texture._bytes, source.data);
	assert.equal(backend.createTextureCalls.length, 1);
	target.destroy();
}

function testExternalSourceWritesUseNativeCopyOrCpuFallback() {
	for (const mode of ["native", "usage fallback", "copy failure", "oversized",
		"integer format", "snorm format"]) {
		const backend = new FakeWebGPUBackend();
		const format = mode === "integer format" ? TextureFormat.RGBA8Uint :
			mode === "snorm format" ? TextureFormat.RGBA8Snorm : TextureFormat.RGBA8Unorm;
		const target = backend.createTexture({
			width: 1, height: 1, format,
			usage: TextureUsage.CopyDst |
				(mode === "usage fallback" ? 0 : TextureUsage.RenderAttachment),
		});
		const source = new Texture({
			width: mode === "oversized" ? 2 : 1, height: 1, format,
		});
		const image = { width: source.width, height: 1 };
		let reads = 0;
		source.getUploadSource = () => image;
		source.readPixelData = () => { reads++; return new Uint8Array([11, 22, 33, 44]); };
		if (mode === "copy failure") {
			backend.queue.copyExternalImageToTexture = () => { throw new Error("copy failed"); };
		}
		if (mode === "oversized") {
			assert.throws(() => backend.resourceManager.writeTexture(target, source), /range|extent/i);
			assert.equal(backend.copyCalls.length, 0);
			assert.equal(reads, 0);
		} else {
			backend.resourceManager.writeTexture(target, source);
			assert.equal(backend.copyCalls.length, mode === "native" ? 1 : 0);
			assert.equal(reads, mode === "native" ? 0 : 1);
			if (mode !== "native") {
				assert.deepEqual(Array.from(backend.resourceManager.getTexture(target).texture._bytes),
					[11, 22, 33, 44]);
			}
		}
		assert.equal(target.destroyed, false);
		target.destroy();
	}
}

function testSourceWritesValidateDepthUploadSupportBeforeReadingPixels() {
	for (const format of [TextureFormat.Depth16Unorm, TextureFormat.Stencil8,
		TextureFormat.Depth24Plus, TextureFormat.Depth32Float,
		TextureFormat.Depth24PlusStencil8, TextureFormat.Depth32FloatStencil8]) {
		const backend = new FakeWebGPUBackend();
		const target = backend.createTexture({ width: 1, height: 1, format,
			usage: TextureUsage.CopyDst });
		const supported = format === TextureFormat.Depth16Unorm || format === TextureFormat.Stencil8;
		const data = new Uint8Array(format === TextureFormat.Stencil8 ? 1 : 2).fill(17);
		const source = new Texture({ width: 1, height: 1, format, data });
		let reads = 0;
		source.getUploadSource = () => { reads++; return data; };
		if (supported) {
			backend.resourceManager.writeTexture(target, source);
			assert.equal(backend.textureWrites.length, 1);
			assert.deepEqual(backend.textureWrites[0].data, Array.from(data));
		} else {
			assert.throws(() => backend.resourceManager.writeTexture(target, source),
				/unsupported.*source.*upload/i);
			assert.equal(reads, 0);
			assert.equal(backend.textureWrites.length, 0);
			// Explicit byte writes retain their existing delegated validation behavior.
			backend.resourceManager.writeTexture(target, data, { bytesPerRow: 2 },
				{ width: 1, height: 1 });
			assert.equal(backend.textureWrites.length, 1);
		}
		assert.equal(target.destroyed, false);
		target.destroy();
	}
}

testSourceUploadsValidateOwnershipBeforeReadingPixels();
testSourceUploadsPreserveExplicitRowsAndRejectBeforePartialWrites();
testTextureWritesRequireCompleteArgumentForms();
testUnsupportedRawConversionRejectsBeforeAnyMipWrite();
testBytesWritesPreserveViewsAndExplicitCopyParameters();
testSourceWritesValidateTargetsAndAllMipRanges();
testSourceWriteFailurePreservesExistingTarget();
testExternalSourceWritesUseNativeCopyOrCpuFallback();
testSourceWritesValidateDepthUploadSupportBeforeReadingPixels();
await testManagedBufferCollectionPreservesExplicitTeardown();

console.log("WebGPU resource manager tests passed");
