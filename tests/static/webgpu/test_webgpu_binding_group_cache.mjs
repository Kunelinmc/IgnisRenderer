import assert from "node:assert/strict";

import { WebGPUBindingGroupCache } from "../../../src/backends/webgpu/WebGPUBindingGroupCache.ts";
import { WebGPUObjectIdentity } from "../../../src/backends/webgpu/WebGPUObjectIdentity.ts";

function createHost() {
	const created = [];
	const device = {
		createBindGroup(desc) {
			const group = { desc, destroy() {} };
			created.push(group);
			return group;
		},
	};
	return {
		created,
		host: {
			device,
			frameSerial: 1,
			objectIdentity: new WebGPUObjectIdentity(),
			assertDeviceOperational() {},
			createManagedDestroy() {
				return () => {};
			},
			runValidationScope(_label, operation) {
				return operation();
			},
		},
	};
}

function createArena() {
	const native = { label: "native-buffer", destroy() {} };
	const arena = {
		label: "WebGPUObjectUniformArena_0",
		size: 1024 * 256,
		_gpuResource: native,
		destroy() {},
	};
	return { arena, native };
}

function testRangeResourcesMapToNativeBufferRanges() {
	const { host } = createHost();
	const cache = new WebGPUBindingGroupCache(host);
	const { arena, native } = createArena();
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
	const { arena, native } = createArena();
	const layout = { label: "layout" };
	const desc = (size) => ({
		label: "ModelBinding_pbr",
		layout,
		entries: [{ binding: 0, resource: { buffer: arena, offset: 0, size } }],
	});
	const first = cache.createBindingGroup(desc(208));
	const repeated = cache.createBindingGroup(desc(208));
	assert.equal(repeated, first, "identical arena ranges reuse the bind group");
	assert.equal(created.length, 1);
	assert.deepEqual(created[0].desc.entries[0].resource,
		{ buffer: native, offset: 0, size: 208 });
	const resized = cache.createBindingGroup(desc(416));
	assert.notEqual(resized, first, "binding size participates in cache identity");
	assert.equal(created.length, 2);
}

testRangeResourcesMapToNativeBufferRanges();
testBindingSizeParticipatesInGroupCacheIdentity();
console.log("WebGPU binding group cache tests passed");
