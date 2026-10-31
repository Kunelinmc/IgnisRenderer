import assert from "node:assert/strict";

import { createWebGPUResourceManager } from "../../helpers/fakes.mjs";
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
assert.equal(foreign.tryGetBuffer(buffer), null);
assert.throws(() => foreign.getBuffer(buffer), /owned|owner|foreign/i);
assert.throws(() => store.getBuffer({ ...buffer }), /owned|owner|foreign/i);
const prototypeForgery = Object.create(Object.getPrototypeOf(buffer));
assert.equal(store.tryGetBuffer(prototypeForgery), null,
	"copying a facade prototype cannot forge the private owner brand");
assert.equal(store.tryGetBuffer(new Proxy(buffer, {})), null,
	"unregistered proxies cannot transfer private resource ownership");
assert.throws(() => store.getBuffer({ _gpuResource: native }), /WebGPU|owned/i);
assert.equal(Reflect.ownKeys(buffer).some((key) =>
	typeof key === "string" && key.startsWith("_")), false);
assert.equal(Object.values(buffer).includes(native), false);
buffer.destroy();
buffer.destroy();
assert.equal(releases, 1);
assert.equal(store.tryGetBuffer(buffer), null);
assert.throws(() => store.getBuffer(buffer), /live|destroyed|retired/i);

let borrowedDestroyed = 0;
const texture = { destroy() { borrowedDestroyed++; } };
const view = {};
const borrowed = store.registerTexture({ texture, view }, { width: 4, height: 4 });
const live = store.registerBuffer(native, { size: 32 }, () => native.destroy());
store.reset();
assert.equal(releases, 2, "store retirement disposes each remaining allocation once");
assert.equal(borrowedDestroyed, 0, "borrowed textures are never destroyed by the store");
assert.equal(store.tryGetTexture(borrowed), null);
assert.equal(store.tryGetBuffer(live), null);
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
manager.writeTexture(managedTexture, data, { bytesPerRow: 8 }, { width: 2, height: 2 });
assert.deepEqual(uploads[0], ["buffer", nativeManagedBuffer, 4, data]);
assert.equal(uploads[1][1].texture, manager.getTexture(managedTexture).texture);
assert.equal(registered.size, 2);
manager.reset();
assert.equal(registered.size, 0, "backend teardown clears managed finalization registrations");
assert.equal(allocationsDestroyed, 2);
assert.equal(borrowedAllocationsDestroyed, 0);
assert.equal(manager.tryGetTexture(borrowedTexture), null);
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
assert.equal(presentationStore.tryGetTexture(firstPresentation), null);
const nextPresentation = targets.getCanvasColorTexture(context, canvas);
assert.notEqual(nextPresentation, firstPresentation);
assert.equal(presentationStore.getTexture(nextPresentation).texture, presented);
targets.release();
assert.equal(presentationStore.tryGetTexture(nextPresentation), null);
assert.equal(presentedDestroyed, 0, "releasing presentation facades preserves browser ownership");

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

await testManagedBufferCollectionPreservesExplicitTeardown();

console.log("WebGPU resource manager tests passed");
