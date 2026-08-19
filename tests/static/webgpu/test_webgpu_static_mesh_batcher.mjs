import assert from "node:assert/strict";

import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { createWebGPUMaterialUniformData } from "../../../src/backends/webgpu/material.ts";
import { WebGPUStaticMeshBatcher } from "../../../src/backends/webgpu/WebGPUStaticMeshBatcher.ts";

const writes = [];
const backend = {
	createBuffer(desc) {
		return {
			...desc,
			destroyed: false,
			destroy() {
				this.destroyed = true;
			},
		};
	},
	writeBuffer(buffer, data, offset = 0) {
		writes.push({ buffer, data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset });
	},
	createBindingGroup(desc) {
		return { desc, destroy() {} };
	},
};
const fallbackStorage = { size: 256, destroy() {} };
const fallbackParams = { size: 32, destroy() {} };
const animations = {
	getFallbackStorageBuffer() {
		return fallbackStorage;
	},
	getStaticScenePayload() {
		return {
			generation: 0,
			paramsBuffer: fallbackParams,
			jointMatricesBuffer: fallbackStorage,
			morphWeightsBuffer: fallbackStorage,
		};
	},
};
const material = new PBRMaterial();
const geometryData = {
	positions: new Float32Array(9),
	indices: new Uint32Array([0, 1, 2]),
};
function packet(id, x) {
	const worldMatrix = Matrix4.identity();
	worldMatrix.elements[0][3] = x;
	const previousWorldMatrix = Matrix4.identity();
	previousWorldMatrix.elements[0][3] = x - 1;
	return {
		id,
		material,
		worldMatrix,
		previousWorldMatrix,
		normalMatrix: Matrix4.identity(),
		meshInstance: { id: `instance:${id}`, skeleton: null, renderLayers: 1 },
		primitive: {
			geometry: geometryData,
			receiveShadows: true,
		},
	};
}

const packets = [packet("a", 1), packet("b", 2)];
const batcher = new WebGPUStaticMeshBatcher(
	backend,
	{ modelBindGroupLayout: { id: "model-layout" } },
	animations,
);
batcher.beginFrame();
batcher.preparePackets(packets);
const materialData = createWebGPUMaterialUniformData(material, false);
const snapshot = {
	revision: material.revision,
	data: materialData,
	textures: Array.from({ length: 16 }, (_, index) => ({ id: `texture:${index}` })),
	samplers: Array.from({ length: 16 }, (_, index) => ({ id: `sampler:${index}` })),
	anisotropyTexture: { id: "anisotropy" },
};
const geometry = {
	indexBuffer: { id: "index" },
	indexFormat: "uint16",
	layoutKey: "layout",
	skinProfile: "static",
	morphTargetCount: 0,
};
const pipeline = { id: "pipeline" };
const first = batcher.getDrawState(packets[0], pipeline, geometry, snapshot, "default");
const second = batcher.getDrawState(packets[1], pipeline, geometry, snapshot, "default");
assert.ok(first);
assert.ok(second);
assert.equal(first.modelBinding, second.modelBinding);
assert.equal(first.batchKey, second.batchKey);
assert.equal(first.firstInstance, 0);
assert.equal(second.firstInstance, 1);
assert.ok(writes.some((write) => write.data.byteLength === 2 * 52 * 4));

const instanceWrite = writes.find((write) => write.data.byteLength === 2 * 52 * 4);
const instanceFloats = new Float32Array(
	instanceWrite.data.buffer,
	instanceWrite.data.byteOffset,
	instanceWrite.data.byteLength,
);
assert.equal(instanceFloats[16 + 12], 0);
assert.equal(instanceFloats[52 + 16 + 12], 1);

batcher.commitFrame();
batcher.destroy();
console.log("WebGPU static mesh batcher tests passed");
