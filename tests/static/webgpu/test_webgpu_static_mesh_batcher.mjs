import assert from "node:assert/strict";

import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { createWebGPUMaterialUniformData } from "../../../src/backends/webgpu/material.ts";
import { WebGPUStaticMeshBatcher } from "../../../src/backends/webgpu/WebGPUStaticMeshBatcher.ts";
import { WebGPUMaterialBufferCache } from "../../../src/backends/webgpu/WebGPUMaterialBufferCache.ts";
import { WEBGPU_TEXTURE_SLOT_COUNT } from "../../../src/backends/webgpu/constants.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";
import { DRAW_PACKET_FLAG_SHADOW_RECEIVER } from "../../../src/pipeline/types.ts";

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
	return createTestDrawPacket({
		id,
		material,
		worldMatrix,
		previousWorldMatrix,
		normalMatrix: Matrix4.identity(),
		meshInstance: { id: `instance:${id}`, skeleton: null, renderLayers: 1 },
		primitive: {
			id: `primitive:${id}`,
			geometry: geometryData,
			material,
			receiveShadows: true,
		},
		passFlags: 1 << 4,
	});
}

const packets = [packet("a", 1), packet("b", 2)];
const materialBuffers = new WebGPUMaterialBufferCache(backend);
const batcher = new WebGPUStaticMeshBatcher(
	backend,
	{
		modelBindGroupLayouts: {
			pbr: { id: "model-layout:pbr" },
			phong: { id: "model-layout:phong" },
			flat: { id: "model-layout:flat" },
			unlit: { id: "model-layout:unlit" },
		},
	},
	animations,
	materialBuffers,
);
batcher.beginFrame();
batcher.preparePackets(packets);
const materialData = createWebGPUMaterialUniformData(material, false);
const snapshot = {
	revision: material.revision,
	data: materialData,
	textures: Array.from(
		{ length: WEBGPU_TEXTURE_SLOT_COUNT },
		(_, index) => ({ id: `texture:${index}` })
	),
	samplers: Array.from(
		{ length: WEBGPU_TEXTURE_SLOT_COUNT },
		(_, index) => ({ id: `sampler:${index}` })
	),
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

const arenaWrites = () => writes.filter(write => write.buffer.label === "WebGPUStaticInstanceArena");
writes.length = 0;
batcher.beginFrame();
batcher.preparePackets(packets);
assert.equal(arenaWrites().length, 0, "unchanged records need no arena upload");
batcher.commitFrame();

packets[1].submission.instance.worldMatrix.elements[0][3] = 3;
packets[1].submission.instance.previousWorldMatrix.elements[0][3] = 2;
writes.length = 0;
batcher.beginFrame();
batcher.preparePackets(packets);
assert.equal(arenaWrites().length, 1);
assert.equal(arenaWrites()[0].offset, 52 * 4, "upload starts at the changed instance");
assert.equal(arenaWrites()[0].data.byteLength, 52 * 4);
const changed = new Float32Array(arenaWrites()[0].data.buffer, arenaWrites()[0].data.byteOffset, 52);
assert.equal(changed[12], 3, "direct current-matrix edits are observed");
assert.equal(changed[28], 2, "direct previous-matrix edits are observed");
batcher.commitFrame();

packets[0].submission.instance.normalMatrix.elements[0][0] = 2;
writes.length = 0;
batcher.beginFrame();
batcher.preparePackets(packets);
assert.equal(arenaWrites()[0].offset, 0);
assert.equal(arenaWrites()[0].data.byteLength, 52 * 4);
batcher.commitFrame();

packets[0].submission.instance.renderLayers = 7;
packets[0].submission.passFlags ^= DRAW_PACKET_FLAG_SHADOW_RECEIVER;
writes.length = 0;
batcher.beginFrame();
batcher.preparePackets(packets);
const flags = new Float32Array(arenaWrites()[0].data.buffer, arenaWrites()[0].data.byteOffset, 52);
assert.equal(flags[48], 7, "layer edits update the instance record");
assert.equal(flags[49], (packets[0].submission.passFlags & DRAW_PACKET_FLAG_SHADOW_RECEIVER) ? 1 : 0);
batcher.commitFrame();

packets[0].submission.instance.worldMatrix.elements[1][3] = -0;
writes.length = 0;
batcher.beginFrame();
batcher.preparePackets(packets);
const signed = new Float32Array(arenaWrites()[0].data.buffer, arenaWrites()[0].data.byteOffset, 52);
assert.ok(Object.is(signed[13], -0), "record reuse preserves float32 signed-zero edits");
batcher.commitFrame();

packets[1].submission.instance.previousWorldMatrix = undefined;
packets[1].submission.instance.worldMatrix.elements[0][3] = 5;
batcher.beginFrame();
batcher.preparePackets(packets);
batcher.abortFrame();
writes.length = 0;
batcher.beginFrame();
batcher.preparePackets(packets);
assert.equal(arenaWrites().length, 0, "aborted pose must not advance history");
batcher.commitFrame();
writes.length = 0;
batcher.beginFrame();
batcher.preparePackets(packets);
assert.equal(arenaWrites().length, 1, "previous pose settles after a committed move");
assert.equal(arenaWrites()[0].data.byteLength, 52 * 4);
batcher.commitFrame();
writes.length = 0;
batcher.beginFrame();
batcher.preparePackets(packets);
assert.equal(arenaWrites().length, 0, "settled poses reuse arena data");
batcher.commitFrame();

writes.length = 0;
batcher.beginFrame();
batcher.preparePackets([...packets].reverse());
assert.equal(arenaWrites()[0].data.byteLength, 2 * 52 * 4, "reordering updates both slots");
assert.equal(batcher.getDrawState(packets[1], pipeline, geometry, snapshot, "default").firstInstance, 0);
batcher.commitFrame();

writes.length = 0;
batcher.beginFrame();
batcher.preparePackets(Array.from({ length: 300 }, (_, i) => packet(`growth:${i}`, i)));
assert.equal(arenaWrites()[0].data.byteLength, 300 * 52 * 4, "replacement arenas upload all records");
batcher.commitFrame();
batcher.beginFrame();
batcher.preparePackets([packets[0]]);
writes.length = 0;
for (let i = 0; i < 512; i++) {
	batcher.getDrawState(packet(`lazy-growth:${i}`, i), pipeline, geometry, snapshot, "default");
}
const grown = arenaWrites().find(write => write.offset === 0 && write.data.byteLength === 513 * 52 * 4);
assert.ok(grown, "lazy arena replacement also uploads existing records");
const grownFloats = new Float32Array(grown.data.buffer, grown.data.byteOffset, 52);
assert.equal(grownFloats[12], 1, "lazy growth preserves an earlier prepared instance");
batcher.destroy();
materialBuffers.destroy();
console.log("WebGPU static mesh batcher tests passed");
