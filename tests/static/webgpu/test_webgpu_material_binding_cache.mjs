import assert from "node:assert/strict";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { WebGPUDrawResourceCache } from "../../../src/backends/webgpu/WebGPUDrawResourceCache.ts";
import { WEBGPU_TEXTURE_SLOT_COUNT } from "../../../src/backends/webgpu/constants.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function createBackendStub() {
	const backend = {
		bufferDestroyCalls: 0,
		bindingGroupDestroyCalls: 0,
		createBindingGroupCalls: 0,
		writeBufferCalls: 0,
		writeCalls: [],
		bufferLabels: [],
		createBuffer(desc = {}) {
			backend.bufferLabels.push(desc.label ?? "");
			let destroyed = false;
			return {
				label: desc.label,
				destroy() {
					if (destroyed) return;
					destroyed = true;
					backend.bufferDestroyCalls++;
				},
			};
		},
		writeBuffer(buffer, data, offset = 0) {
			backend.writeBufferCalls++;
			backend.writeCalls.push({ buffer, data, offset });
		},
		createBindingGroup(desc) {
			let destroyed = false;
			backend.createBindingGroupCalls++;
			return {
				desc,
				destroy() {
					if (destroyed) return;
					destroyed = true;
					backend.bindingGroupDestroyCalls++;
				},
			};
		},
	};
	return backend;
}

function createPacket(id = "meshInstance_8:primitive_8") {
	return createTestDrawPacket({ id, worldMatrix: Matrix4.identity(), normalMatrix: Matrix4.identity() });
}

function createAnimationState() {
	return { generation: 0, paramsBuffer: {}, jointMatricesBuffer: {}, morphWeightsBuffer: {} };
}

function createCache(backend, textures = Array.from({ length: WEBGPU_TEXTURE_SLOT_COUNT }, () => ({}))) {
	const fallback = {};
	const sampler = {};
	return new WebGPUDrawResourceCache(backend,
		{ modelBindGroupLayouts: { pbr: {}, phong: {}, flat: {}, unlit: {} } },
		{ getFallbackStorageBuffer: () => fallback }, {
			async getTextureForSlotAsync(_texture, index) { return textures[index]; },
			getSamplerForTexture() { return sampler; },
		});
}

function getWriteCountForLabel(backend, prefix) {
	return backend.writeCalls.filter(({ buffer }) => buffer.label?.startsWith(prefix)).length;
}

async function testBudgetedCacheRetainsModelBindingGroup() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const snapshot = await cache.resolveMaterial(new PBRMaterial(), false);
	cache.beginFrame();
	cache.getMaterialBinding(createPacket(), snapshot, createAnimationState());
	for (let i = 0; i < 6; i++) cache.beginFrame();
	assert.equal(backend.bindingGroupDestroyCalls, 0);
	cache.destroy();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
}

async function testTextureRebindRetainsPreparedBindingUntilFrameEnd() {
	const backend = createBackendStub();
	const textures = Array.from({ length: WEBGPU_TEXTURE_SLOT_COUNT }, () => ({}));
	const cache = createCache(backend, textures);
	const snapshot = await cache.resolveMaterial(new PBRMaterial(), false);
	textures[WEBGPU_TEXTURE_SLOT_COUNT - 1] = {};
	const replacement = await cache.resolveMaterial(new PBRMaterial(), false);
	const packet = createPacket();
	const animation = createAnimationState();
	cache.beginFrame();
	const first = cache.getMaterialBinding(packet, snapshot, animation);
	const second = cache.getMaterialBinding(packet, replacement, animation);
	assert.notEqual(first.group, second.group);
	assert.equal(backend.bindingGroupDestroyCalls, 0, "prepared bindings survive the active frame");
	cache.beginFrame();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
	cache.destroy();
}

async function testSnapshotReuseDoesNotUploadOrRebind() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const snapshot = await cache.resolveMaterial(new PBRMaterial(), false);
	const packet = createPacket();
	const animation = createAnimationState();
	cache.beginFrame();
	const first = cache.getMaterialBinding(packet, snapshot, animation);
	const writes = backend.writeBufferCalls;
	const repeated = cache.getMaterialBinding(packet, snapshot, animation);
	assert.equal(repeated.group, first.group);
	assert.deepEqual(repeated.dynamicOffsets, first.dynamicOffsets);
	assert.equal(backend.createBindingGroupCalls, 1);
	assert.equal(backend.writeBufferCalls, writes);
	assert.equal(getWriteCountForLabel(backend, "ModelJointMatrices_"), 0);
	assert.equal(getWriteCountForLabel(backend, "ModelMorphWeights_"), 0);
	assert.equal(getWriteCountForLabel(backend, "ModelAnimationParams_"), 0);
	cache.destroy();
}

async function testCompatibleDrawsShareOneSceneBinding() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const snapshot = await cache.resolveMaterial(new PBRMaterial(), false);
	const animation = createAnimationState();
	cache.beginFrame();
	const first = cache.getMaterialBinding(createPacket("first"), snapshot, animation);
	const second = cache.getMaterialBinding(createPacket("second"), snapshot, animation);
	assert.equal(first.group, second.group, "compatible draws reuse one model group");
	assert.deepEqual(first.dynamicOffsets, [0]);
	assert.deepEqual(second.dynamicOffsets, [256]);
	assert.equal(backend.createBindingGroupCalls, 1);
	const objectWrites = backend.writeCalls.filter(
		({ buffer }) => buffer.label?.startsWith("WebGPUObjectUniformArena"),
	);
	assert.deepEqual(objectWrites.map(({ offset }) => offset), [0, 256],
		"each draw owns a distinct arena slot");
	assert.deepEqual(cache.getDebugStats().materialBuffers, { entries: 1, references: 1 });
	cache.destroy();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
	assert.equal(snapshot.references, 0);
}

async function testMaterialSnapshotChangeRebindsWithoutObjectUpload() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const material = new PBRMaterial();
	const snapshot = await cache.resolveMaterial(material, false);
	const packet = createPacket();
	const animation = createAnimationState();
	cache.beginFrame();
	const first = cache.getMaterialBinding(packet, snapshot, animation);
	const objectWrites = getWriteCountForLabel(backend, "WebGPUObjectUniformArena");
	material.roughness = 0.2;
	cache.beginFrame();
	const next = await cache.resolveMaterial(material, false);
	const second = cache.getMaterialBinding(packet, next, animation);
	assert.notEqual(first.group, second.group);
	assert.equal(getWriteCountForLabel(backend, "WebGPUObjectUniformArena"), objectWrites);
	assert.equal(backend.createBindingGroupCalls, 2);
	cache.destroy();
}

async function testPayloadGenerationRebuildsBindingWithoutOwningPayloadBuffers() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const snapshot = await cache.resolveMaterial(new PBRMaterial(), false);
	const packet = createPacket();
	const animation = createAnimationState();
	cache.beginFrame();
	const first = cache.getMaterialBinding(packet, snapshot, animation);
	const destroyCount = backend.bufferDestroyCalls;
	animation.generation++;
	animation.jointMatricesBuffer = { destroy() { throw new Error("borrowed buffer"); } };
	const second = cache.getMaterialBinding(packet, snapshot, animation);
	assert.notEqual(first.group, second.group);
	assert.equal(backend.createBindingGroupCalls, 2);
	assert.equal(backend.bufferDestroyCalls, destroyCount);
	cache.destroy();
}

async function testMaterialBuffersDestroyAfterLastReference() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const snapshot = await cache.resolveMaterial(new PBRMaterial(), false);
	assert.equal(backend.bufferLabels.length, 0, "resolving a material leaves buffers unallocated");
	const first = cache.retainMaterial(snapshot);
	const second = cache.retainMaterial(snapshot);
	assert.equal(first, second,
		"retaining one material returns its shared buffers without per-reference handles");
	assert.deepEqual(cache.getDebugStats().materialBuffers, { entries: 1, references: 2 });
	cache.releaseMaterial(snapshot);
	assert.deepEqual(cache.getDebugStats().materialBuffers, { entries: 1, references: 1 });
	const destroyed = backend.bufferDestroyCalls;
	cache.releaseMaterial(snapshot);
	assert.deepEqual(cache.getDebugStats().materialBuffers, { entries: 0, references: 0 });
	assert.equal(backend.bufferDestroyCalls, destroyed + 2);
	assert.throws(() => cache.releaseMaterial(snapshot), /no retained reference/);
	const revived = cache.retainMaterial(snapshot);
	assert.notEqual(revived, first, "a retained CPU snapshot can rebuild its buffers");
	cache.releaseMaterial(snapshot);
	cache.destroy();
}

async function testCacheTeardownReleasesOutstandingMaterialReferences() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const snapshot = await cache.resolveMaterial(new PBRMaterial(), false);
	cache.retainMaterial(snapshot);
	cache.destroy();
	assert.equal(backend.bufferDestroyCalls, backend.bufferLabels.length);
	assert.equal(snapshot.references, 0);
	assert.equal(snapshot.buffers, null);
	cache.releaseMaterial(snapshot); cache.releaseMaterial(snapshot); cache.destroy();
	assert.equal(backend.bufferDestroyCalls, backend.bufferLabels.length);
	assert.throws(() => cache.retainMaterial(snapshot), /destroyed/);
}

async function testShaderUniformsBelongToTheMaterialSnapshot() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const material = new ShaderMaterial({ uniformBindings: [{ name: "time", type: "f32", value: 1 }] });
	const snapshot = await cache.resolveMaterial(material, false);
	const animation = createAnimationState();
	const first = cache.getMaterialBinding(createPacket("first"), snapshot, animation);
	const second = cache.getMaterialBinding(createPacket("second"), snapshot, animation);
	const uniform = (binding) => binding.group.desc.entries.find((entry) => entry.binding === 39).resource;
	assert.equal(uniform(first), uniform(second), "draws share the snapshot's custom uniform buffer");
	assert.equal(getWriteCountForLabel(backend, "ShaderMaterialUniform_"), 1);
	material.setUniform("time", 2);
	const next = await cache.resolveMaterial(material, false);
	const changed = cache.getMaterialBinding(createPacket("first"), next, animation);
	assert.notEqual(uniform(changed), uniform(first), "new revisions preserve old snapshot buffers");
	assert.equal(getWriteCountForLabel(backend, "ShaderMaterialUniform_"), 2);
	cache.destroy();
}

async function testFailedMaterialAllocationReleasesPartialBuffers() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const material = new ShaderMaterial({ uniformBindings: [{ name: "time", type: "f32", value: 1 }] });
	const snapshot = await cache.resolveMaterial(material, false);
	const writeBuffer = backend.writeBuffer;
	backend.writeBuffer = (buffer, data) => {
		if (buffer.label.startsWith("ShaderMaterialUniform_")) throw new Error("upload failed");
		writeBuffer(buffer, data);
	};
	assert.throws(() => cache.retainMaterial(snapshot), /upload failed/);
	assert.equal(backend.bufferDestroyCalls, backend.bufferLabels.length,
		"failed allocation releases the entire partial material buffer set");
	assert.deepEqual(cache.getDebugStats().materialBuffers, { entries: 0, references: 0 });
	backend.writeBuffer = writeBuffer;
	cache.retainMaterial(snapshot);
	cache.releaseMaterial(snapshot);
	cache.destroy();
}

async function testFailedBindingPreparationReleasesItsMaterialReference() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const snapshot = await cache.resolveMaterial(new PBRMaterial(), false);
	const writeBuffer = backend.writeBuffer;
	backend.writeBuffer = (buffer, data) => {
		if (buffer.label === "WebGPUFallbackShaderUniformBuffer") throw new Error("fallback failed");
		writeBuffer(buffer, data);
	};
	assert.throws(() => cache.getMaterialBinding(createPacket(), snapshot, createAnimationState()),
		/fallback failed/);
	assert.deepEqual(cache.getDebugStats().materialBuffers, { entries: 0, references: 0 });
	backend.writeBuffer = writeBuffer;
	cache.getMaterialBinding(createPacket(), snapshot, createAnimationState());
	cache.destroy();
	assert.equal(backend.bufferDestroyCalls, backend.bufferLabels.length);
}

await testFailedMaterialAllocationReleasesPartialBuffers();
await testFailedBindingPreparationReleasesItsMaterialReference();
await testBudgetedCacheRetainsModelBindingGroup();
await testTextureRebindRetainsPreparedBindingUntilFrameEnd();
await testSnapshotReuseDoesNotUploadOrRebind();
await testCompatibleDrawsShareOneSceneBinding();
await testMaterialSnapshotChangeRebindsWithoutObjectUpload();
await testPayloadGenerationRebuildsBindingWithoutOwningPayloadBuffers();
await testMaterialBuffersDestroyAfterLastReference();
await testCacheTeardownReleasesOutstandingMaterialReferences();
await testShaderUniformsBelongToTheMaterialSnapshot();
console.log("WebGPU draw material resource tests passed");
