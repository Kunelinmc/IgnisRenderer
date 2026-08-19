import assert from "node:assert/strict";

import {
	WebGPUAnimationPayloadPool,
} from "../../../src/backends/webgpu/WebGPUAnimationPayloadPool.ts";

function createBackend() {
	const buffers = [];
	const writes = [];
	return {
		buffers,
		writes,
		createBuffer(desc) {
			const buffer = {
				label: desc.label,
				size: desc.size,
				destroyed: false,
				destroy() {
					this.destroyed = true;
				},
			};
			buffers.push(buffer);
			return buffer;
		},
		writeBuffer(buffer, data, offset = 0) {
			const copy = new data.constructor(data);
			writes.push({ buffer, data: copy, offset });
		},
	};
}

function createPacket(id = "packet") {
	const primitive = { id: `primitive:${id}` };
	const mesh = { primitives: [primitive] };
	const meshInstance = {
		id: `instance:${id}`,
		mesh,
		skeleton: null,
		morphWeights: [],
		worldMatrix: {},
	};
	return {
		id,
		primitive,
		mesh,
		meshInstance,
		deformationRevision: 0,
	};
}

function createGeometry(options = {}) {
	return {
		vertexCount: options.vertexCount ?? 3,
		morphTargetCount: options.morphTargetCount ?? 0,
		morphSemanticMask: options.morphSemanticMask ?? 0,
	};
}

function testStaticPacketsUseOnlyFallbackBuffers() {
	const backend = createBackend();
	const pool = new WebGPUAnimationPayloadPool(backend);
	const geometry = createGeometry();
	pool.beginFrame();
	for (let i = 0; i < 10_000; i++) {
		const packet = createPacket(`static:${i}`);
		const shadow = pool.getShadowPayload(packet, geometry, null, null);
		const scene = pool.getScenePayload(packet, geometry, null, null);
		assert.equal(shadow.generation, 0);
		assert.equal(scene.generation, 0);
	}

	const stats = pool.getDebugStats();
	assert.equal(stats.entryCount, 10_000);
	assert.equal(stats.staticEntryCount, 10_000);
	assert.equal(stats.liveBufferCount, 3);
	assert.equal(stats.totalUploadCalls, 0);
	assert.equal(backend.buffers.length, 3);
	assert.equal(backend.writes.length, 0);
	pool.destroy();
}

function testSceneAndShadowShareOneTemporalStorageUpload() {
	const backend = createBackend();
	const pool = new WebGPUAnimationPayloadPool(backend);
	const packet = createPacket("animated");
	const geometry = createGeometry();
	const matrices = new Float32Array(16).fill(1);
	const jointMap = new Map([
		[packet.meshInstance.id, { skeleton: {}, matrices }],
	]);

	pool.beginFrame();
	packet.deformationRevision = 1;
	const shadow = pool.getShadowPayload(packet, geometry, jointMap, null);
	const scene = pool.getScenePayload(packet, geometry, jointMap, null);
	assert.strictEqual(shadow.jointMatricesBuffer, scene.jointMatricesBuffer);
	assert.equal(pool.getDebugStats().totalUploadCalls, 3);
	pool.getShadowPayload(packet, geometry, jointMap, null);
	pool.getScenePayload(packet, geometry, jointMap, null);
	assert.equal(pool.getDebugStats().totalUploadCalls, 3);

	pool.beginFrame();
	packet.deformationRevision = 2;
	matrices.fill(2);
	const changed = pool.getScenePayload(packet, geometry, jointMap, null);
	pool.getShadowPayload(packet, geometry, jointMap, null);
	assert.equal(pool.getDebugStats().totalUploadCalls, 4);
	const changedWrite = backend.writes.at(-1);
	assert.strictEqual(changedWrite.buffer, changed.jointMatricesBuffer);
	assert.deepEqual(Array.from(changedWrite.data.subarray(0, 16)), new Array(16).fill(2));
	assert.deepEqual(Array.from(changedWrite.data.subarray(16, 32)), new Array(16).fill(1));

	pool.beginFrame();
	pool.getScenePayload(packet, geometry, jointMap, null);
	assert.equal(pool.getDebugStats().totalUploadCalls, 5);
	const settledWrite = backend.writes.at(-1);
	assert.deepEqual(Array.from(settledWrite.data.subarray(16, 32)), new Array(16).fill(2));

	pool.beginFrame();
	pool.getScenePayload(packet, geometry, jointMap, null);
	assert.equal(pool.getDebugStats().totalUploadCalls, 5);
	assert.ok(pool.getDebugStats().totalSkippedUploads >= 1);
	pool.destroy();
}

function testManualSkeletonMutationUsesArrayComparison() {
	const backend = createBackend();
	const pool = new WebGPUAnimationPayloadPool(backend);
	const packet = createPacket("manual");
	const geometry = createGeometry();
	let value = 3;
	packet.meshInstance.skeleton = {
		jointCount: 1,
		updateJointMatrices() {},
		toFloat32Array(out) {
			out.fill(value);
			return out;
		},
	};

	pool.beginFrame();
	pool.getScenePayload(packet, geometry, null, null);
	const initialUploads = pool.getDebugStats().totalUploadCalls;
	pool.beginFrame();
	pool.getScenePayload(packet, geometry, null, null);
	assert.equal(pool.getDebugStats().totalUploadCalls, initialUploads);

	value = 4;
	pool.beginFrame();
	pool.getScenePayload(packet, geometry, null, null);
	assert.equal(pool.getDebugStats().totalUploadCalls, initialUploads + 1);
	pool.destroy();
}

function testMorphStorageIsSharedWithoutJointAllocation() {
	const backend = createBackend();
	const pool = new WebGPUAnimationPayloadPool(backend);
	const packet = createPacket("morph");
	const geometry = createGeometry({
		morphTargetCount: 2,
		morphSemanticMask: 1,
	});
	const morphMap = new Map([
		[packet.id, {
			packetId: packet.id,
			weights: new Float32Array([0.25, 0.5]),
			targetCount: 2,
		}],
	]);

	pool.beginFrame();
	packet.deformationRevision = 1;
	const scene = pool.getScenePayload(packet, geometry, null, morphMap);
	const shadow = pool.getShadowPayload(packet, geometry, null, morphMap);
	const stats = pool.getDebugStats();

	assert.strictEqual(scene.morphWeightsBuffer, shadow.morphWeightsBuffer);
	assert.strictEqual(scene.jointMatricesBuffer, pool.getFallbackStorageBuffer());
	assert.equal(stats.jointBufferCount, 0);
	assert.equal(stats.morphBufferCount, 1);
	assert.equal(stats.liveBufferCount, 6);
	pool.destroy();
}

function testCapacityGrowthAdvancesGeneration() {
	const backend = createBackend();
	const pool = new WebGPUAnimationPayloadPool(backend);
	const packet = createPacket("growth");
	const geometry = createGeometry();
	const state = { skeleton: {}, matrices: new Float32Array(16).fill(1) };
	const jointMap = new Map([[packet.meshInstance.id, state]]);

	pool.beginFrame();
	packet.deformationRevision = 1;
	const first = pool.getScenePayload(packet, geometry, jointMap, null);
	pool.beginFrame();
	packet.deformationRevision = 2;
	state.matrices = new Float32Array(32).fill(2);
	const grown = pool.getScenePayload(packet, geometry, jointMap, null);

	assert.notStrictEqual(grown.jointMatricesBuffer, first.jointMatricesBuffer);
	assert.notEqual(grown.generation, first.generation);
	assert.equal(first.jointMatricesBuffer.destroyed, true);
	assert.equal(pool.getDebugStats().capacityRebuilds, 2);
	pool.destroy();
}

function testInactiveResourcesReleaseAfterSixtyFrames() {
	const backend = createBackend();
	const pool = new WebGPUAnimationPayloadPool(backend);
	const packet = createPacket("release");
	const geometry = createGeometry();
	let jointMap = new Map([
		[packet.meshInstance.id, { skeleton: {}, matrices: new Float32Array(16) }],
	]);

	pool.beginFrame();
	packet.deformationRevision = 1;
	pool.getScenePayload(packet, geometry, jointMap, null);
	assert.equal(pool.getDebugStats().liveBufferCount, 5);

	jointMap = null;
	for (let inactiveFrame = 1; inactiveFrame <= 59; inactiveFrame++) {
		pool.beginFrame();
		const payload = pool.getScenePayload(packet, geometry, jointMap, null);
		assert.equal(payload.generation, 0);
	}
	assert.equal(pool.getDebugStats().liveBufferCount, 5);

	pool.beginFrame();
	pool.getScenePayload(packet, geometry, jointMap, null);
	const released = pool.getDebugStats();
	assert.equal(released.liveBufferCount, 3);
	assert.equal(released.staticEntryCount, 1);
	assert.equal(released.graceReleases, 2);
	pool.destroy();
}

function testGracePeriodResumeReusesCapacityWithoutFalseMotion() {
	const backend = createBackend();
	const pool = new WebGPUAnimationPayloadPool(backend);
	const packet = createPacket("resume");
	const geometry = createGeometry();
	const matrices = new Float32Array(16).fill(1);
	let jointMap = new Map([
		[packet.meshInstance.id, { skeleton: {}, matrices }],
	]);

	pool.beginFrame();
	packet.deformationRevision = 1;
	const first = pool.getScenePayload(packet, geometry, jointMap, null);
	pool.beginFrame();
	jointMap = null;
	pool.getScenePayload(packet, geometry, jointMap, null);

	pool.beginFrame();
	packet.deformationRevision = 2;
	matrices.fill(5);
	jointMap = new Map([
		[packet.meshInstance.id, { skeleton: {}, matrices }],
	]);
	const resumed = pool.getScenePayload(packet, geometry, jointMap, null);
	const resumeWrite = backend.writes.at(-1);

	assert.strictEqual(resumed.jointMatricesBuffer, first.jointMatricesBuffer);
	assert.deepEqual(Array.from(resumeWrite.data.subarray(0, 16)), new Array(16).fill(5));
	assert.deepEqual(Array.from(resumeWrite.data.subarray(16, 32)), new Array(16).fill(5));
	pool.destroy();
}

testStaticPacketsUseOnlyFallbackBuffers();
testSceneAndShadowShareOneTemporalStorageUpload();
testManualSkeletonMutationUsesArrayComparison();
testMorphStorageIsSharedWithoutJointAllocation();
testCapacityGrowthAdvancesGeneration();
testInactiveResourcesReleaseAfterSixtyFrames();
testGracePeriodResumeReusesCapacityWithoutFalseMotion();
console.log("WebGPU animation payload pool tests passed");
