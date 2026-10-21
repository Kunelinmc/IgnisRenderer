import assert from "node:assert/strict";
import { AlphaMode } from "../../../src/materials/Material.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { DRAW_PACKET_FLAG_SHADOW_RECEIVER } from "../../../src/pipeline/types.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";
import { packObjectUniformData } from "../../../src/backends/webgpu/packing.ts";
import { WebGPUTextureRegistry } from "../../../src/backends/webgpu/WebGPUTextureRegistry.ts";

import { WebGPUDrawResourceCache } from "../../../src/backends/webgpu/WebGPUDrawResourceCache.ts";
import { WebGPUDepthMaterialResources } from "../../../src/backends/webgpu/WebGPUDepthMaterialResources.ts";

function backendStub() {
	return {
		buffers: [], groups: [], writes: [],
		createBuffer(desc) {
			const buffer = { ...desc, _gpuResource: {}, destroys: 0,
				destroy() { this.destroys++; } };
			this.buffers.push(buffer);
			return buffer;
		},
		writeBuffer(buffer, data, offset = 0) {
			this.writes.push({ buffer, data: new Float32Array(data), offset });
		},
		createBindingGroup(desc) {
			const group = { desc, destroys: 0, destroy() { this.destroys++; } };
			this.groups.push(group);
			return group;
		},
	};
}
const textures = {
	async getTextureForSlotAsync() { return {}; },
	getSamplerForTexture() { return {}; },
};
const layouts = {
	depthMaterialBindGroupLayout: {}, depthModelBindGroupLayout: {},
	modelBindGroupLayouts: { pbr: {}, flat: {}, phong: {}, unlit: {} },
};

async function testDepthProjectionAndRetirement() {
	const backend = backendStub();
	const white = { _gpuResource: {} };
	let nativeTexture = { _gpuResource: {} };
	let nativeSampler = { _gpuResource: {} };
	const calls = [];
	const registry = {
		async getTextureForSlotAsync(map, slot) {
			calls.push({ map, slot });
			return map ? nativeTexture : white;
		},
		getSamplerForTexture() { return nativeSampler; },
		isTextureBindingCurrent(_map, resource, sampler) {
			return resource === nativeTexture && sampler === nativeSampler;
		},
	};
	const cache = new WebGPUDepthMaterialResources(backend, layouts, registry);
	cache.beginFrame();
	const map = new Texture();
	map.offset.x = 0.2;
	map.repeat.y = 2;
	map.rotation = 0.3;
	const material = new PBRMaterial({ map, alphaMode: AlphaMode.Mask, opacity: 0.7 });
	material.albedoMapUV = 2;
	Object.defineProperty(material, "normalMap", { get() { throw new Error("unrelated texture read"); } });
	const first = await cache.resolve(material);
	assert.deepEqual(first.binding.desc.entries.map(({ binding }) => binding), [0, 1, 2]);
	assert.deepEqual(Array.from(backend.writes.at(-1).data),
		Array.from(new Float32Array([0.7, 0.5, 0, 0, 0.2, 0, 1, 2, 0.3, 2, 0, 0])));
	assert.ok(calls.every(({ map: resolved, slot }) => resolved === map && slot === 0));
	material.roughness = 0.9;
	const lightingEdit = await cache.resolve(material);
	assert.equal(lightingEdit, first);
	material.alphaCutoff = 0.6;
	const alphaEdit = await cache.resolve(material);
	assert.notEqual(alphaEdit.binding, first.binding);
	assert.equal(first.binding.destroys, 0, "prepared old binding remains alive in its frame");
	nativeTexture._gpuResource = {};
	const nativeEdit = await cache.resolve(material);
	assert.notEqual(nativeEdit.binding, alphaEdit.binding);
	nativeSampler = { _gpuResource: {} };
	const samplerEdit = await cache.resolve(material);
	assert.notEqual(samplerEdit.binding, nativeEdit.binding);
	map.rotation = 0.8;
	assert.notEqual((await cache.resolve(material)).binding, samplerEdit.binding);
	const opaque = new PBRMaterial({ map });
	const opaqueFirst = await cache.resolve(opaque);
	assert.equal(calls.at(-1).map, null, "opaque uses the white fallback without uploading its map");
	opaque.opacity = 0.2;
	assert.equal(await cache.resolve(opaque), opaqueFirst);
	cache.beginFrame();
	assert.equal(first.binding.destroys, 1);
	cache.destroy();
}

async function testObjectSharingAndDepthBindings() {
	for (const depthFirst of [false, true]) {
		const backend = backendStub();
		const fallback = { _gpuResource: {} };
		const pool = { getFallbackStorageBuffer: () => fallback };
		const animation = { generation: 1, paramsBuffer: fallback,
			jointMatricesBuffer: fallback, morphWeightsBuffer: fallback };
		const cache = new WebGPUDrawResourceCache(backend, layouts, pool, textures);
		const packet = createTestDrawPacket({ id: "shared", worldMatrix: Matrix4.identity(),
			normalMatrix: Matrix4.identity() });
		cache.beginFrame();
		const material = await cache.resolveMaterial(new PBRMaterial(), false);
		let sceneGroup, depthGroup;
		const resolveScene = () => sceneGroup = cache.getMaterialBinding(packet, material, animation);
		const resolveDepth = () => depthGroup = cache.getDepthBinding(packet, animation, null);
		if (depthFirst) { resolveDepth(); resolveScene(); } else { resolveScene(); resolveDepth(); }
		const object = depthGroup.group.desc.entries[0].resource;
		assert.equal(object.buffer, sceneGroup.group.desc.entries[0].resource.buffer);
		assert.equal(object.offset, 0);
		assert.equal(object.size, 208);
		assert.deepEqual(sceneGroup.dynamicOffsets, [0]);
		assert.deepEqual(depthGroup.dynamicOffsets, [0]);
		assert.equal(backend.writes.filter((write) => write.buffer === object.buffer).length, 1);
		assert.deepEqual(depthGroup.group.desc.entries.map(({ binding }) => binding), [0, 34, 35, 36, 37, 40]);
		cache.destroy();
		cache.destroy();
		assert.equal(sceneGroup.group.destroys, 1);
		assert.equal(depthGroup.group.destroys, 1);
		assert.ok(backend.buffers.every((buffer) => buffer.destroys === 1),
			"one teardown releases every owned buffer exactly once");
		const bufferCount = backend.buffers.length;
		assert.throws(resolveScene, /destroyed/);
		assert.throws(resolveDepth, /destroyed/);
		assert.equal(backend.buffers.length, bufferCount, "teardown is terminal");
	}
}

function testStaticBorrowingAndNativeInvalidation() {
	const backend = backendStub();
	const borrowed = { _gpuResource: {}, size: 64, destroy() { throw new Error("borrowed"); } };
	const animation = { generation: 1, paramsBuffer: borrowed,
		jointMatricesBuffer: borrowed, morphWeightsBuffer: borrowed };
	const cache = new WebGPUDrawResourceCache(backend, layouts,
		{ getFallbackStorageBuffer: () => borrowed }, {});
	cache.beginFrame();
	const state = { objectBuffer: borrowed, instanceBuffer: borrowed };
	const first = cache.getDepthBinding({ submission: { id: "a" } }, animation, null, state);
	assert.equal(cache.getDepthBinding({ submission: { id: "b" } }, animation, null, state).group,
		first.group);
	assert.equal(backend.buffers.length, 0, "static depth borrows buffers without scene allocation");
	animation.generation++;
	assert.equal(cache.getDepthBinding({ submission: { id: "a" } }, animation, null, state).group,
		first.group, "generation changes do not change the buffers bound by a depth draw");
	borrowed._gpuResource = {};
	const next = cache.getDepthBinding({ submission: { id: "a" } }, animation, null, state);
	assert.notEqual(next.group, first.group);
	assert.equal(first.group.destroys, 0);
	cache.beginFrame();
	assert.equal(first.group.destroys, 1);
	cache.destroy();
	assert.equal(next.group.destroys, 1);
}

async function testSharedObjectFrameHistory() {
	const backend = backendStub();
	const fallback = {};
	const animation = { generation: 1, paramsBuffer: fallback,
		jointMatricesBuffer: fallback, morphWeightsBuffer: fallback };
	const cache = new WebGPUDrawResourceCache(backend, layouts,
		{ getFallbackStorageBuffer: () => fallback }, textures);
	const material = await cache.resolveMaterial(new PBRMaterial(), false);
	const world = Matrix4.identity();
	const normal = Matrix4.identity();
	const packet = createTestDrawPacket({ id: "history", worldMatrix: world, normalMatrix: normal });
	const scene = () => cache.getMaterialBinding(packet, material, animation);
	const depth = () => cache.getDepthBinding(packet, animation, null);
	const objectWrites = () => backend.writes.filter(
		({ buffer }) => buffer.label?.startsWith("WebGPUObjectUniformArena"),
	);
	cache.beginFrame();
	depth(); scene();
	assert.equal(objectWrites().length, 1);
	const previous = Matrix4.identity();
	world.elements[0][3] = 3;
	cache.beginFrame();
	scene(); depth();
	assert.deepEqual(objectWrites().at(-1).data,
		packObjectUniformData(world, normal, previous, 1, false));
	const changedWorld = new Matrix4(world.elements.map((row) => row.slice()));
	world.elements[0][3] = 5;
	depth();
	assert.deepEqual(objectWrites().at(-1).data,
		packObjectUniformData(world, normal, previous, 1, false),
		"second preparation in a frame preserves its previous transform");
	cache.beginFrame();
	depth();
	const latestPrevious = new Matrix4(world.elements.map((row) => row.slice()));
	assert.deepEqual(objectWrites().at(-1).data,
		packObjectUniformData(world, normal, latestPrevious, 1, false));
	packet.submission.instance.renderLayers = 4;
	scene();
	assert.deepEqual(objectWrites().at(-1).data,
		packObjectUniformData(world, normal, latestPrevious, 4, false));
	packet.submission.instance.previousWorldMatrix = changedWorld;
	cache.beginFrame();
	depth();
	assert.deepEqual(objectWrites().at(-1).data,
		packObjectUniformData(world, normal, changedWorld, 4, false),
		"explicit previous transforms override owner history");
	cache.destroy();
	assert.equal(objectWrites()[0].buffer.destroys, 1);
}

function testObjectUploadRetryPreservesFrameHistory() {
	const backend = backendStub();
	const fallback = {};
	const animation = { generation: 0, paramsBuffer: fallback,
		jointMatricesBuffer: fallback, morphWeightsBuffer: fallback };
	const cache = new WebGPUDrawResourceCache(backend, layouts,
		{ getFallbackStorageBuffer: () => fallback }, textures);
	const world = Matrix4.identity();
	const normal = Matrix4.identity();
	const previous = Matrix4.identity();
	const packet = createTestDrawPacket({ id: "retry", worldMatrix: world, normalMatrix: normal });
	cache.beginFrame();
	cache.getDepthBinding(packet, animation, null);
	cache.beginFrame();
	world.elements[0][3] = 7;
	const writeBuffer = backend.writeBuffer.bind(backend);
	backend.writeBuffer = () => { throw new Error("object upload failed"); };
	assert.throws(() => cache.getDepthBinding(packet, animation, null), /object upload failed/);
	backend.writeBuffer = writeBuffer;
	cache.getDepthBinding(packet, animation, null);
	assert.equal(backend.writes.length, 2, "failed uploads remain pending on same-frame retry");
	assert.deepEqual(backend.writes.at(-1).data,
		packObjectUniformData(world, normal, previous, 1, false));
	cache.destroy();
}

async function testConcurrentResolutionAndAsyncDestruction() {
	const backend = backendStub();
	const nativeTexture = { _gpuResource: {} };
	const nativeSampler = { _gpuResource: {} };
	let releaseTexture;
	let uploaded = new Promise((resolve) => { releaseTexture = resolve; });
	const registry = {
		getTextureForSlotAsync() { return uploaded; },
		getSamplerForTexture() { return nativeSampler; },
		isTextureBindingCurrent(_map, resource, sampler) {
			return resource === nativeTexture && sampler === nativeSampler;
		},
	};
	const cache = new WebGPUDepthMaterialResources(backend, layouts, registry);
	const material = new PBRMaterial({ alphaMode: AlphaMode.Mask });
	const first = cache.resolve(material);
	const second = cache.resolve(material);
	material.opacity = 0.4;
	const changed = cache.resolve(material);
	releaseTexture(nativeTexture);
	const [firstSnapshot, secondSnapshot, changedSnapshot] = await Promise.all([first, second, changed]);
	assert.equal(firstSnapshot, secondSnapshot, "concurrent equivalent projections share resources");
	assert.notEqual(changedSnapshot, firstSnapshot);
	assert.equal(backend.writes[0].data[0], 1, "pending projection stays immutable during authored edits");
	assert.equal(backend.writes[1].data[0], Math.fround(0.4));
	assert.equal(backend.buffers.length, 2);
	assert.equal(typeof cache.isCurrent, "function", "assembler needs alpha-only snapshot validation");
	assert.equal(cache.isCurrent(material, firstSnapshot), false, "pending opacity edit is stale");
	assert.equal(cache.isCurrent(material, changedSnapshot), true);
	material.roughness = 0.8;
	assert.equal(cache.isCurrent(material, changedSnapshot), true, "lighting edits keep alpha current");
	for (const mutate of [
		() => { material.opacity = 0.5; },
		() => { material.alphaCutoff = 0.3; },
		() => { material.albedoMapUV = 2; },
		() => { material.map = new Texture(); },
	]) {
		let finish;
		uploaded = new Promise((resolve) => { finish = resolve; });
		const preparing = cache.resolve(material);
		mutate();
		finish(nativeTexture);
		const snapshot = await preparing;
		assert.equal(cache.isCurrent(material, snapshot), false,
			"alpha, cutoff, UV or map edits during upload invalidate prepared snapshots");
	}
	uploaded = new Promise((resolve) => { releaseTexture = resolve; });
	const bufferCount = backend.buffers.length;
	const pending = cache.resolve(material);
	cache.destroy();
	releaseTexture(nativeTexture);
	await assert.rejects(pending, /destroyed/);
	assert.equal(backend.buffers.length, bufferCount, "destroying during upload cannot create leaked resources");
	assert.ok(backend.buffers.every((buffer) => buffer.destroys === 1));
}

function testReadOnlyTextureBindingValidation() {
	let allocations = 0;
	const backend = {
		createSampler() { allocations++; return { _gpuResource: {} }; },
		createTexture(desc) { allocations++; return { ...desc, _gpuResource: {}, destroy() {} }; },
	};
	const registry = new WebGPUTextureRegistry(backend, { writeTexture() {} });
	const map = new Texture();
	const resource = { width: map.width, height: map.height, _gpuResource: {} };
	registry.registerExternalTexture(map, resource);
	const sampler = registry.getSamplerForTexture(map);
	const initialAllocations = allocations;
	assert.equal(registry.isTextureBindingCurrent(map, resource, sampler), true);
	map.wrapS = "Clamp";
	assert.equal(registry.isTextureBindingCurrent(map, resource, sampler), false);
	map.wrapS = "Repeat";
	assert.equal(registry.isTextureBindingCurrent(map, resource, sampler), true);
	map.version++;
	assert.equal(registry.isTextureBindingCurrent(map, resource, sampler), false);
	const replacement = { ...resource, _gpuResource: {} };
	registry.registerExternalTexture(map, replacement);
	assert.equal(registry.isTextureBindingCurrent(map, resource, sampler), false);
	assert.equal(registry.isTextureBindingCurrent(map, replacement, sampler), true);
	assert.equal(allocations, initialAllocations, "read-only validation cannot allocate GPU handles");
	const white = registry.getWhiteTexture();
	const whiteSampler = registry.getWhiteSampler();
	assert.equal(registry.isTextureBindingCurrent(null, white, whiteSampler), true);
	assert.equal(registry.isTextureBindingCurrent(null, replacement, sampler), false);
}

async function testActiveFrameDrawRecordsSurviveBudgetOverflow() {
	const backend = backendStub();
	const fallback = {};
	const pool = { getFallbackStorageBuffer: () => fallback };
	const cache = new WebGPUDrawResourceCache(backend, layouts, pool, textures);
	const animation = { generation: 1, paramsBuffer: fallback,
		jointMatricesBuffer: fallback, morphWeightsBuffer: fallback };
	const data = await cache.resolveMaterial(new PBRMaterial(), false);
	cache.beginFrame();
	let first, firstDepth;
	for (let i = 0; i <= 16_384; i++) {
		const packet = createTestDrawPacket({ id: "active:" + i });
		const group = cache.getMaterialBinding(packet, data, animation);
		const depthGroup = cache.getDepthBinding(packet, animation, null);
		if (i === 0) { first = group; firstDepth = depthGroup; }
	}
	assert.equal(first.group.destroys, 0, "prepared scene binding cannot be evicted in its active frame");
	assert.equal(firstDepth.group.destroys, 0, "prepared depth binding remains live above its budget");
	assert.equal(first.group.desc.entries[0].resource.buffer.destroys, 0);
	cache.beginFrame();
	assert.equal(firstDepth.group.destroys, 1, "one frame boundary trims the inactive depth binding");
	assert.equal(first.group.destroys, 0,
		"compatible draws keep the shared scene binding alive after one record is evicted");
	assert.equal(first.group.desc.entries[0].resource.buffer.destroys, 0,
		"eviction returns the arena slot without destroying the shared arena");
	cache.destroy();
	assert.equal(first.group.destroys, 1, "teardown releases the last shared scene reference");
	assert.equal(first.group.desc.entries[0].resource.buffer.destroys, 1,
		"teardown destroys the shared arena");
	assert.ok(backend.buffers.every((buffer) => buffer.destroys === 1));
}

async function testEitherPassRetainsTheWholeDrawRecord() {
	for (const useDepth of [false, true]) {
		const backend = backendStub();
		const fallback = {};
		const animation = { generation: 1, paramsBuffer: fallback,
			jointMatricesBuffer: fallback, morphWeightsBuffer: fallback };
		const cache = new WebGPUDrawResourceCache(backend, layouts,
			{ getFallbackStorageBuffer: () => fallback }, textures);
		const snapshot = await cache.resolveMaterial(new PBRMaterial(), false);
		const packet = createTestDrawPacket({ id: "retained" });
		cache.beginFrame();
		const depth = cache.getDepthBinding(packet, animation, null);
		const scene = cache.getMaterialBinding(packet, snapshot, animation);
		const object = depth.group.desc.entries[0].resource;
		cache.beginFrame();
		if (useDepth) cache.getDepthBinding(packet, animation, null);
		else cache.getMaterialBinding(packet, snapshot, animation);
		for (let i = 0; i < 16_384; i++) {
			const filler = createTestDrawPacket({ id: "filler:" + i });
			cache.getDepthBinding(filler, animation, null);
		}
		assert.equal(depth.group.destroys, 0);
		assert.equal(scene.group.destroys, 0, "using either pass retains the active draw as a unit");
		assert.equal(object.buffer.destroys, 0);
		cache.beginFrame();
		assert.equal(depth.group.destroys, 1);
		assert.equal(scene.group.destroys, 1, "inactive record eviction releases both bindings together");
		assert.equal(object.buffer.destroys, 0, "the shared arena outlives individual draw records");
		cache.destroy();
		assert.equal(object.buffer.destroys, 1);
	}
}

async function testSceneSlotsReplaceObsoleteBindingsWithoutLosingObjectHistory() {
	const backend = backendStub();
	const fallback = {};
	const animation = { generation: 1, paramsBuffer: fallback,
		jointMatricesBuffer: fallback, morphWeightsBuffer: fallback };
	const cache = new WebGPUDrawResourceCache(backend, layouts,
		{ getFallbackStorageBuffer: () => fallback }, textures);
	const material = new PBRMaterial();
	const solid = await cache.resolveMaterial(material, false);
	const wire = await cache.resolveMaterial(material, true);
	const packet = createTestDrawPacket({ id: "variants" });
	cache.beginFrame();
	const original = cache.getMaterialBinding(packet, solid, animation);
	const wireframe = cache.getMaterialBinding(packet, wire, animation);
	const depth = cache.getDepthBinding(packet, animation, null);
	const object = depth.group.desc.entries[0].resource;
	const maskedMaterial = new PBRMaterial();
	maskedMaterial.alphaMode = AlphaMode.Mask;
	const masked = await cache.resolveMaterial(maskedMaterial, false);
	const maskedBinding = cache.getMaterialBinding(packet, masked, animation);
	const blendedMaterial = new PBRMaterial();
	blendedMaterial.alphaMode = AlphaMode.Blend;
	const blended = await cache.resolveMaterial(blendedMaterial, false);
	const latest = cache.getMaterialBinding(packet, blended, animation);
	assert.equal(original.group.destroys, 0, "replaced solid bindings survive prepared draws in this frame");
	assert.equal(maskedBinding.group.destroys, 0);
	assert.equal(cache.getMaterialBinding(packet, wire, animation).group, wireframe.group,
		"replacing the solid slot retains the wireframe slot");
	cache.beginFrame();
	assert.equal(original.group.destroys, 1);
	assert.equal(maskedBinding.group.destroys, 1,
		"obsolete pipeline variants are released without a separate expiry policy");
	for (let i = 0; i < 3; i++) {
		cache.getDepthBinding(packet, animation, null);
		cache.beginFrame();
	}
	assert.equal(cache.getMaterialBinding(packet, blended, animation).group, latest.group);
	assert.equal(cache.getMaterialBinding(packet, wire, animation).group, wireframe.group);
	assert.equal(object.buffer.destroys, 0, "scene slots share the draw's lifetime and transform history");
	cache.destroy();
	assert.ok(backend.buffers.every((buffer) => buffer.destroys === 1));
}

function testStaticArenaOwnsTheDepthBindingSlot() {
	const backend = backendStub();
	const arena = { size: 64, _gpuResource: {} };
	const firstObject = { size: 208, _gpuResource: {} };
	const secondObject = { size: 208, _gpuResource: {} };
	const animation = { generation: 0, paramsBuffer: arena,
		jointMatricesBuffer: arena, morphWeightsBuffer: arena };
	const cache = new WebGPUDrawResourceCache(backend, layouts,
		{ getFallbackStorageBuffer: () => arena }, textures);
	const packet = { submission: { id: "static" } };
	cache.beginFrame();
	const first = cache.getDepthBinding(packet, animation, null,
		{ instanceBuffer: arena, objectBuffer: firstObject });
	const replacement = cache.getDepthBinding(packet, animation, null,
		{ instanceBuffer: arena, objectBuffer: secondObject });
	assert.notEqual(first.group, replacement.group);
	const alias = { ...secondObject };
	assert.equal(cache.getDepthBinding(packet, animation, null,
		{ instanceBuffer: arena, objectBuffer: alias }).group, replacement.group,
		"another wrapper for the same native buffer does not change a binding");
	cache.beginFrame();
	assert.equal(first.group.destroys, 1, "one arena retains only its latest depth binding");
	alias.size += 16;
	const resized = cache.getDepthBinding(packet, animation, null,
		{ instanceBuffer: arena, objectBuffer: alias });
	assert.notEqual(resized.group, replacement.group, "buffer size still participates in invalidation");
	cache.destroy();
}

async function testObjectNonHistoricalFieldsUsePackedValues() {
	const backend = backendStub();
	const fallback = {};
	const animation = { generation: 0, paramsBuffer: fallback,
		jointMatricesBuffer: fallback, morphWeightsBuffer: fallback };
	const cache = new WebGPUDrawResourceCache(backend, layouts,
		{ getFallbackStorageBuffer: () => fallback }, textures);
	const normal = Matrix4.identity();
	const packet = createTestDrawPacket({ id: "packed", normalMatrix: normal });
	cache.beginFrame();
	cache.getDepthBinding(packet, animation, null);
	normal.elements[0][0] = 1 + 1e-10;
	cache.getDepthBinding(packet, animation, null);
	assert.equal(backend.writes.length, 1, "normal changes below float32 precision reuse the payload");
	packet.submission.instance.worldMatrix.elements[0][0] = 1 + 1e-10;
	cache.getDepthBinding(packet, animation, null);
	assert.equal(backend.writes.length, 1, "model comparison also uses the existing float32 payload");
	normal.elements[1][2] = 0.75;
	packet.submission.instance.renderLayers = 19;
	packet.submission.passFlags |= DRAW_PACKET_FLAG_SHADOW_RECEIVER;
	cache.getDepthBinding(packet, animation, null);
	assert.deepEqual(backend.writes.at(-1).data, packObjectUniformData(
		packet.submission.instance.worldMatrix, normal, packet.submission.instance.worldMatrix,
		19, true,
	));
	cache.destroy();
}

await testDepthProjectionAndRetirement();
testObjectUploadRetryPreservesFrameHistory();
await testObjectSharingAndDepthBindings();
testStaticBorrowingAndNativeInvalidation();
await testSharedObjectFrameHistory();
await testConcurrentResolutionAndAsyncDestruction();
testReadOnlyTextureBindingValidation();
await testActiveFrameDrawRecordsSurviveBudgetOverflow();
await testEitherPassRetainsTheWholeDrawRecord();
await testSceneSlotsReplaceObsoleteBindingsWithoutLosingObjectHistory();
testStaticArenaOwnsTheDepthBindingSlot();
await testObjectNonHistoricalFieldsUsePackedValues();
console.log("WebGPU depth resource tests passed");
