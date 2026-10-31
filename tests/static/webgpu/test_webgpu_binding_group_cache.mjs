import assert from "node:assert/strict";

import { createWebGPUResourceManager } from "../../helpers/fakes.mjs";
import { WebGPUBindingGroupCache } from "../../../src/backends/webgpu/WebGPUBindingGroupCache.ts";
import { WebGPUObjectIdentity } from "../../../src/backends/webgpu/WebGPUObjectIdentity.ts";

function createHost() {
	const created = [];
	const managedDisposals = [];
	const device = {
		createBindGroup(desc) {
			const group = { desc, destroy() {} };
			created.push(group);
			return group;
		},
	};
	return {
		created,
		managedDisposals,
		host: {
			device,
			resourceManager: createWebGPUResourceManager(),
			frameSerial: 1,
			objectIdentity: new WebGPUObjectIdentity(),
			assertDeviceOperational() {},
			createManagedDestroy(target, options) {
				return () => {
					managedDisposals.push(target);
					options.dispose();
				};
			},
			runValidationScope(_label, operation) {
				return operation();
			},
		},
	};
}

function createArena(store) {
	const native = { label: "native-buffer", destroy() {} };
	const arena = store.registerBuffer(native, { size: 1024 * 256 });
	return { arena, native };
}

function testObjectIdentityTracksSuppliedObjects() {
	const { host } = createHost();
	const native = { destroy() {} };
	const first = host.resourceManager.registerBuffer(native, { size: 16 });
	const alias = host.resourceManager.registerBuffer(native, { size: 16 });
	const identity = host.objectIdentity;
	assert.notEqual(identity.getCacheToken(first), identity.getCacheToken(alias),
		"object identity must distinguish supplied facades even when their native allocation matches");
	const nativeToken = identity.getCacheToken(native);
	assert.equal(identity.getCacheToken(host.resourceManager.getBuffer(first)), nativeToken);
	assert.equal(identity.getCacheToken(host.resourceManager.getBuffer(alias)), nativeToken);
	first.destroy();
	assert.doesNotThrow(() => identity.getCacheToken(first),
		"tracking object identity must not read a retired facade's native state");
	assert.throws(() => host.resourceManager.getBuffer(first), /not live/);
	alias.destroy();
}

function testRangeResourcesMapToNativeBufferRanges() {
	const { host } = createHost();
	const cache = new WebGPUBindingGroupCache(host);
	const { arena, native } = createArena(host.resourceManager);
	assert.deepEqual(
		cache.mapBindingResource({ buffer: arena, offset: 0, size: 208 }),
		{ buffer: native, offset: 0, size: 208 },
		"high-level arena wrappers must map to an explicit native sub-range",
	);
	assert.deepEqual(
		cache.mapBindingResource({ buffer: arena, offset: 256, size: 208 }),
		{ buffer: native, offset: 256, size: 208 },
	);
	assert.deepEqual(
		cache.mapBindingResource({ buffer: native, offset: 512 }),
		{ buffer: native, offset: 512 },
		"raw native buffer bindings keep their range",
	);
}

function testBindingSizeParticipatesInGroupCacheIdentity() {
	const { host, created } = createHost();
	const cache = new WebGPUBindingGroupCache(host);
	const { arena, native } = createArena(host.resourceManager);
	const layout = { label: "layout" };
	const desc = (size) => ({
		label: "ModelBinding_pbr",
		layout,
		entries: [{ binding: 0, resource: { buffer: arena, offset: 0, size } }],
	});
	const first = cache.createBindingGroup(desc(208));
	const repeated = cache.createBindingGroup(desc(208));
	assert.equal(host.resourceManager.getBindingGroup(repeated), host.resourceManager.getBindingGroup(first),
		"identical arena ranges reuse the native bind group");
	assert.equal(created.length, 1);
	assert.deepEqual(created[0].desc.entries[0].resource,
		{ buffer: native, offset: 0, size: 208 });
	const resized = cache.createBindingGroup(desc(416));
	assert.notEqual(host.resourceManager.getBindingGroup(resized), host.resourceManager.getBindingGroup(first),
		"binding size participates in cache identity");
	assert.equal(created.length, 2);
}

function testForgedResourceHandlesAreRejected() {
	const { host } = createHost();
	const cache = new WebGPUBindingGroupCache(host);
	assert.throws(
		() => cache.mapBindingResource({ _gpuResource: { destroy() {} } }),
		/unsupported|invalid|owned/i,
		"an arbitrary object must not smuggle a native handle through the resource contract",
	);
}

function testForeignDestroyedAndCopiedResourcesAreRejected() {
	const { host } = createHost();
	const cache = new WebGPUBindingGroupCache(host);
	const { arena } = createArena(host.resourceManager);
	const { arena: foreign } = createArena(createWebGPUResourceManager());
	assert.throws(() => cache.mapBindingResource(foreign), /owned|invalid|live/i);
	assert.throws(() => cache.mapBindingResource({ ...arena }), /owned|invalid|live/i);
	assert.throws(() => cache.mapBindingResource({ buffer: foreign, size: 208 }), /owned|invalid|live/i);
	arena.destroy();
	assert.throws(() => cache.mapBindingResource(arena), /owned|invalid|live/i);
	assert.throws(() => cache.mapBindingResource({ buffer: arena, size: 208 }), /owned|invalid|live/i);
}

function testNativeAliasesAndTextureViewsControlCacheIdentity() {
	const { host, created } = createHost();
	const cache = new WebGPUBindingGroupCache(host);
	const native = { destroy() {} };
	const bufferA = host.resourceManager.registerBuffer(native, { size: 1024 });
	const bufferB = host.resourceManager.registerBuffer(native, { size: 1024 });
	const samplerNative = {};
	const samplerA = host.resourceManager.registerSampler(samplerNative);
	const samplerB = host.resourceManager.registerSampler(samplerNative);
	const nativeTexture = {};
	const view = {};
	const textureA = host.resourceManager.registerTexture(
		{ texture: nativeTexture, view }, { width: 1, height: 1 },
	);
	const textureB = host.resourceManager.registerTexture(
		{ texture: nativeTexture, view }, { width: 1, height: 1 },
	);
	const textureOtherView = host.resourceManager.registerTexture(
		{ texture: nativeTexture, view: {} }, { width: 1, height: 1 },
	);
	const layout = {};
	const desc = (buffer, sampler, texture) => ({
		layout,
		entries: [buffer, sampler, texture].map((resource, binding) => ({ binding, resource })),
	});
	const first = cache.createBindingGroup(desc(bufferA, samplerA, textureA));
	const repeated = cache.createBindingGroup(desc(bufferB, samplerB, textureB));
	const firstNative = host.resourceManager.getBindingGroup(first);
	assert.equal(host.resourceManager.getBindingGroup(repeated), firstNative,
		"different engine references to identical native resources reuse the binding");
	assert.equal(created.length, 1);
	const otherView = cache.createBindingGroup(desc(bufferA, samplerA, textureOtherView));
	assert.notEqual(host.resourceManager.getBindingGroup(otherView), firstNative,
		"distinct views of one texture require distinct bindings");
	assert.equal(created.length, 2);
	repeated.destroy();
	first.destroy();
	assert.throws(() => host.resourceManager.getBindingGroup(first), /owned|invalid|live/i);
	const replacement = cache.createBindingGroup(desc(bufferA, samplerA, textureA));
	assert.notEqual(host.resourceManager.getBindingGroup(replacement), firstNative,
		"destroying every acquisition releases its cache entry and permits a fresh binding");
}

function testStoreResetCompletesManagedBindingDisposal() {
	const { host, managedDisposals } = createHost();
	const cache = new WebGPUBindingGroupCache(host);
	const { arena } = createArena(host.resourceManager);
	const desc = {
		layout: {},
		entries: [{ binding: 0, resource: arena }],
	};
	const group = cache.createBindingGroup(desc);
	const alias = cache.createBindingGroup(desc);
	host.resourceManager.reset();
	assert.equal(managedDisposals.length, 2,
		"device reset must complete managed disposal and unregister forgotten-destroy diagnostics");
	assert.equal(cache.getDebugStats().entryCount, 0);
	group.destroy();
	alias.destroy();
	assert.equal(managedDisposals.length, 2, "a later facade destroy must remain idempotent");
}

function testCachedBindingAcquisitionsOwnIndependentReferences() {
	const { host, created } = createHost();
	const cache = new WebGPUBindingGroupCache(host);
	const { arena } = createArena(host.resourceManager);
	const desc = { layout: {}, entries: [{ binding: 0, resource: arena }] };
	const first = cache.createBindingGroup(desc);
	const second = cache.createBindingGroup(desc);
	assert.equal(created.length, 1, "equivalent acquisitions share one native binding");
	first.destroy();
	first.destroy();
	assert.doesNotThrow(() => host.resourceManager.getBindingGroup(second),
		"releasing one acquisition must preserve another caller's live binding");
	assert.deepEqual(cache.getDebugStats().refCounts, [1],
		"a facade releases its own reference exactly once");
	second.destroy();
	assert.equal(cache.getDebugStats().entryCount, 0,
		"the last acquisition releases the native binding cache entry");
	assert.throws(() => host.resourceManager.getBindingGroup(second), /owned|invalid|live/i);
}

testCachedBindingAcquisitionsOwnIndependentReferences();
testStoreResetCompletesManagedBindingDisposal();
testForgedResourceHandlesAreRejected();
testForeignDestroyedAndCopiedResourcesAreRejected();
testNativeAliasesAndTextureViewsControlCacheIdentity();
testObjectIdentityTracksSuppliedObjects();
testRangeResourcesMapToNativeBufferRanges();
testBindingSizeParticipatesInGroupCacheIdentity();
console.log("WebGPU binding group cache tests passed");
