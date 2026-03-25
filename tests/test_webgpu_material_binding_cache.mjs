import assert from "node:assert/strict";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { WebGPUMaterialBindingCache } from "../src/renderers/webgpu/WebGPUMaterialBindingCache.ts";

function createBackendStub() {
	const backend = {
		bufferDestroyCalls: 0,
		bindingGroupDestroyCalls: 0,
		createBuffer() {
			let destroyed = false;
			return {
				destroy() {
					if (destroyed) return;
					destroyed = true;
					backend.bufferDestroyCalls++;
				},
			};
		},
		writeBuffer() {},
		createBindingGroup(desc) {
			let destroyed = false;
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

function testRebindDestroysPreviousModelBindingGroup() {
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
	const secondGroup = cache.getBinding(
		packet,
		{ id: "pipeline:b" },
		materialData,
		[],
		[],
		animation
	);

	assert.notEqual(firstGroup, secondGroup);
	assert.equal(backend.bindingGroupDestroyCalls, 1);
}

function run() {
	testEvictionDestroysModelBindingGroup();
	testRebindDestroysPreviousModelBindingGroup();
	console.log("WebGPU material binding cache tests passed");
}

run();
