import assert from "node:assert/strict";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { WebGPUMaterialBindingCache } from "../src/renderers/webgpu/WebGPUMaterialBindingCache.ts";

function createBackendStub() {
	const backend = {
		bufferDestroyCalls: 0,
		bindingGroupDestroyCalls: 0,
		createBindingGroupCalls: 0,
		writeBufferCalls: 0,
		writeCalls: [],
		createBuffer(desc = {}) {
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
		materialFlags: vec4,
		textureSlots: [],
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
	return {
		id,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
	};
}

function createAnimationState() {
	return {
		jointMatrices: null,
		morphWeights: null,
		morphTargetCount: 0,
		morphPositionBuffer: null,
		morphNormalBuffer: null,
	};
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

function testEvictionDestroysModelBindingGroup() {
	const backend = createBackendStub();
	const cache = new WebGPUMaterialBindingCache(backend, createLayoutsStub());
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

	assert.equal(backend.bindingGroupDestroyCalls, 1);
}

function testTextureRebindDestroysPreviousModelBindingGroup() {
	const backend = createBackendStub();
	const cache = new WebGPUMaterialBindingCache(backend, createLayoutsStub());
	const packet = createPacket();
	const materialData = createMaterialData();
	const animation = createAnimationState();
	cache.beginFrame();
	const firstGroup = cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		[{ id: "texture:a" }],
		[{ id: "sampler:a" }],
		animation
	);
	const secondGroup = cache.getBinding(
		packet,
		{ id: "pipeline:a" },
		materialData,
		[{ id: "texture:b" }],
		[{ id: "sampler:a" }],
		animation
	);

	assert.notEqual(firstGroup, secondGroup);
	assert.equal(backend.bindingGroupDestroyCalls, 1);
}

function testPipelineChangeReusesModelBindingGroup() {
	const backend = createBackendStub();
	const cache = new WebGPUMaterialBindingCache(backend, createLayoutsStub());
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
	const cache = new WebGPUMaterialBindingCache(backend, createLayoutsStub());
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
	assert.equal(getWriteCountForLabel(backend, "ModelAnimationParams_"), 1);
}

function run() {
	testEvictionDestroysModelBindingGroup();
	testTextureRebindDestroysPreviousModelBindingGroup();
	testPipelineChangeReusesModelBindingGroup();
	testStaticMeshDoesNotWriteAnimationPayloads();
	console.log("WebGPU material binding cache tests passed");
}

run();
