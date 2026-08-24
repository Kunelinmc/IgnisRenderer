import assert from "node:assert/strict";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { WebGPUMaterialBindingCache } from "../../../src/backends/webgpu/WebGPUMaterialBindingCache.ts";
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
		writeBuffer(buffer, data) {
			backend.writeBufferCalls++;
			backend.writeCalls.push({ buffer, data });
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

function createMaterialData(pipelineKey = "none-opaque-solid") {
	const vec4 = [0, 0, 0, 0];
	return {
		baseColorFactor: vec4,
		emissiveFactor: vec4,
		surfaceParams0: vec4,
		surfaceParams1: vec4,
		surfaceParams2: vec4,
		surfaceParams3: vec4,
		specularColorFactor: vec4,
		phongAmbientShininess: vec4,
		phongSpecularShading: vec4,
		sheenColorClearcoatNormalScale: vec4,
		attenuationColor: vec4,
		anisotropyParams: vec4,
		materialFlags: vec4,
		pbrMasks: [0, 0, 0, 0],
		textureSlots: Array.from({ length: WEBGPU_TEXTURE_SLOT_COUNT }, () => ({
			map: null,
			transformA: vec4,
			transformB: vec4,
		})),
		shaderUniforms: {
			cacheKey: "",
			byteLength: 0,
			valueRevision: 0,
			data: null,
		},
		pipelineKey,
		warnings: [],
	};
}

function createPacket(id = "meshInstance_8:primitive_8") {
	return createTestDrawPacket({
		id,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
	});
}

function createAnimationState() {
	return {
		generation: 0,
		paramsBuffer: { id: "animation:params" },
		jointMatricesBuffer: { id: "animation:joints" },
		morphWeightsBuffer: { id: "animation:morph" },
		jointCount: 0,
		morphCount: 0,
	};
}

function createAnimationPayloadPoolStub() {
	const fallbackStorage = { id: "animation:fallback" };
	return {
		getFallbackStorageBuffer() {
			return fallbackStorage;
		},
	};
}

function createCache(backend) {
	return new WebGPUMaterialBindingCache(
		backend,
		createLayoutsStub(),
		createAnimationPayloadPoolStub()
	);
}

function createLayoutsStub() {
	return {
		modelBindGroupLayout: { id: "layout:model" },
	};
}

function getWriteCountForLabel(backend, labelPrefix) {
	return backend.writeCalls.filter((call) =>
		call.buffer?.label?.startsWith(labelPrefix)
	).length;
}

function testBudgetedCacheRetainsModelBindingGroup() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const packet = createPacket();
	const materialData = createMaterialData();
	const animation = createAnimationState();
	cache.beginFrame();
	cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		[],
		[],
		animation
	);
	assert.equal(backend.bindingGroupDestroyCalls, 0);

	for (let i = 0; i < 6; i++) {
		cache.beginFrame();
	}

	assert.equal(backend.bindingGroupDestroyCalls, 0);
	cache.destroy();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
}

function testTextureRebindDestroysPreviousModelBindingGroup() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const packet = createPacket();
	const materialData = createMaterialData();
	const animation = createAnimationState();
	cache.beginFrame();
	const firstTextures = Array.from(
		{ length: WEBGPU_TEXTURE_SLOT_COUNT },
		(_, index) => ({ id: `texture:${index}` })
	);
	const secondTextures = firstTextures.slice();
	secondTextures[WEBGPU_TEXTURE_SLOT_COUNT - 1] = {
		id: "texture:anisotropy:next",
	};
	const samplers = Array.from(
		{ length: WEBGPU_TEXTURE_SLOT_COUNT },
		(_, index) => ({ id: `sampler:${index}` })
	);
	const firstGroup = cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		firstTextures,
		samplers,
		animation
	);
	const secondGroup = cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		secondTextures,
		samplers,
		animation
	);

	assert.notEqual(firstGroup, secondGroup);
	assert.equal(backend.bindingGroupDestroyCalls, 1);
}

function testPipelineChangeReusesModelBindingGroup() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const packet = createPacket();
	const materialData = createMaterialData();
	const animation = createAnimationState();
	cache.beginFrame();
	const firstGroup = cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		[],
		[],
		animation
	);
	const writeCount = backend.writeBufferCalls;
	const secondGroup = cache.getBinding(
		packet,
		{ id: "pipeline:b" },
		materialData,
		[],
		[],
		animation
	);

	assert.equal(firstGroup, secondGroup);
	assert.equal(backend.createBindingGroupCalls, 1);
	assert.equal(backend.bindingGroupDestroyCalls, 0);
	assert.equal(backend.writeBufferCalls, writeCount);
}

function testStaticMeshDoesNotWriteAnimationPayloads() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const packet = createPacket();
	const materialData = createMaterialData();
	const animation = createAnimationState();
	cache.beginFrame();
	cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		[],
		[],
		animation
	);
	cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		[],
		[],
		animation
	);

	assert.equal(getWriteCountForLabel(backend, "ModelJointMatrices_"), 0);
	assert.equal(getWriteCountForLabel(backend, "ModelMorphWeights_"), 0);
	assert.equal(getWriteCountForLabel(backend, "ModelAnimationParams_"), 0);
	assert.equal(
		backend.bufferLabels.some((label) => label.startsWith("ModelAnimationParams_")),
		false
	);
}

function testMaskMutationUpdatesUniformWithoutRebinding() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const packet = createPacket();
	const materialData = createMaterialData();
	const animation = createAnimationState();
	cache.beginFrame();
	const firstGroup = cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		[],
		[],
		animation
	);
	const modelWrites = getWriteCountForLabel(backend, "ModelUniform_");
	materialData.pbrMasks = [1, 2, 0, 0];
	const secondGroup = cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		[],
		[],
		animation
	);

	assert.equal(firstGroup, secondGroup);
	assert.equal(
		getWriteCountForLabel(backend, "ModelUniform_"),
		modelWrites + 1
	);
	assert.equal(backend.createBindingGroupCalls, 1);
}

function testPayloadGenerationRebuildsBindingWithoutOwningPayloadBuffers() {
	const backend = createBackendStub();
	const cache = createCache(backend);
	const packet = createPacket();
	const materialData = createMaterialData();
	const animation = {
		...createAnimationState(),
		generation: 1,
		paramsBuffer: { id: "params:1" },
		jointMatricesBuffer: { id: "joint:1" },
	};
	cache.beginFrame();
	const first = cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		[],
		[],
		animation,
		null,
		null
	);
	const payloadDestroyCount = backend.bufferDestroyCalls;
	animation.generation = 2;
	animation.jointMatricesBuffer = { id: "joint:2" };
	const rebuilt = cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		[],
		[],
		animation,
		null,
		null
	);

	assert.notStrictEqual(rebuilt, first);
	assert.equal(backend.createBindingGroupCalls, 2);
	assert.equal(backend.bufferDestroyCalls, payloadDestroyCount);
}

function run() {
	testBudgetedCacheRetainsModelBindingGroup();
	testTextureRebindDestroysPreviousModelBindingGroup();
	testPipelineChangeReusesModelBindingGroup();
	testStaticMeshDoesNotWriteAnimationPayloads();
	testMaskMutationUpdatesUniformWithoutRebinding();
	testPayloadGenerationRebuildsBindingWithoutOwningPayloadBuffers();
	console.log("WebGPU material binding cache tests passed");
}

run();
