import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { createWebGPUResourceManager } from "../../helpers/fakes.mjs";
import { WebGPUOcclusionCullingRuntime } from "../../../src/backends/webgpu/WebGPUOcclusionCullingRuntime.ts";
import { WebGPUCommandEncoder } from "../../../src/backends/webgpu/WebGPUCommandEncoder.ts";
import { normalizeOcclusionCullingOptions } from "../../../src/pipeline/OcclusionCulling.ts";
import { FakeWebGPUBackend } from "../../helpers/fakes.mjs";

function createContext() {
	return {
		incremental: {
			temporalHistoryReset: false,
		},
	};
}

function createCandidate(overrides = {}) {
	return {
		packetId: "packet-A",
		eligible: true,
		signatureA: 11,
		signatureB: 17,
		...overrides,
	};
}

function createPending(frameIndex, options) {
	return {
		visibilityGeneration: 0,
		frameIndex,
		packetIds: ["packet-A"],
		signaturesA: [11],
		signaturesB: [17],
		options,
		buffer: {
			destroy() {},
		},
		byteLength: 4,
		queued: true,
		done: false,
	};
}

function testVisibilityHysteresisAndRecovery() {
	const runtime = new WebGPUOcclusionCullingRuntime({});
	const options = normalizeOcclusionCullingOptions({
		hysteresisFrames: 2,
		maxReadbackLatencyFrames: 3,
	});
	const context = createContext();
	const candidate = createCandidate();

	runtime.beginFrame(context);
	assert.equal(runtime.getVisibilityProvider(options).isPacketVisible(candidate), true);

	runtime._applyReadback(createPending(1, options), new Uint32Array([0]));
	assert.equal(runtime.getVisibilityProvider(options).isPacketVisible(candidate), true);

	runtime._applyReadback(createPending(2, options), new Uint32Array([0]));
	assert.equal(runtime.getVisibilityProvider(options).isPacketVisible(candidate), false);

	runtime._applyReadback(createPending(3, options), new Uint32Array([1]));
	assert.equal(runtime.getVisibilityProvider(options).isPacketVisible(candidate), true);
}

function testStaleAndSignatureFallbacksAreVisible() {
	const runtime = new WebGPUOcclusionCullingRuntime({});
	const options = normalizeOcclusionCullingOptions({
		hysteresisFrames: 1,
		maxReadbackLatencyFrames: 1,
	});
	const context = createContext();
	const candidate = createCandidate();

	runtime.beginFrame(context);
	runtime._applyReadback(createPending(1, options), new Uint32Array([0]));
	assert.equal(runtime.getVisibilityProvider(options).isPacketVisible(candidate), false);
	assert.equal(
		runtime.getVisibilityProvider(options).isPacketVisible(
			createCandidate({ signatureA: 99 })
		),
		true
	);

	runtime.beginFrame(context);
	runtime.beginFrame(context);
	assert.equal(runtime.getVisibilityProvider(options).isPacketVisible(candidate), true);
}

function testTemporalResetClearsVisibility() {
	const runtime = new WebGPUOcclusionCullingRuntime({});
	const options = normalizeOcclusionCullingOptions({
		hysteresisFrames: 1,
	});
	const candidate = createCandidate();

	runtime.beginFrame(createContext());
	runtime._applyReadback(createPending(1, options), new Uint32Array([0]));
	assert.equal(runtime.getVisibilityProvider(options).isPacketVisible(candidate), false);

	runtime.beginFrame({
		incremental: {
			temporalHistoryReset: true,
		},
	});
	assert.equal(runtime.getVisibilityProvider(options).isPacketVisible(candidate), true);
}

function testResetIgnoresLateReadback() {
	const runtime = new WebGPUOcclusionCullingRuntime({});
	const options = normalizeOcclusionCullingOptions({
		hysteresisFrames: 1,
	});
	const candidate = createCandidate();
	const stalePending = createPending(1, options);

	runtime.beginFrame(createContext());
	runtime.resetVisibility();
	runtime._applyReadback(stalePending, new Uint32Array([0]));

	assert.equal(runtime.getVisibilityProvider(options).isPacketVisible(candidate), true);
}

async function testCompactDepthBiasUploadPreservesCandidateAlignment() {
	globalThis.GPUBufferUsage ??= { COPY_DST: 8, MAP_READ: 1 };
	const backend = new FakeWebGPUBackend();
	const runtime = new WebGPUOcclusionCullingRuntime(backend);
	const nativeEncoder = backend.createCommandEncoder();
	const encoder = new WebGPUCommandEncoder(nativeEncoder, {
		resourceManager: backend.resourceManager,
		createPassTimestampWrites: () => undefined,
	}, {});
	const context = {
		...createContext(),
		scene: { occlusion: { candidates: [createCandidate({
			screenRect: { x: 1, y: 2, width: 3, height: 4 },
			nearDepth: 0.2, farDepth: 0.8, screenAreaPx: 12,
		})] } },
	};
	runtime.beginFrame(context);
	const hiZ = backend.createTexture({ width: 4, height: 4, format: "r32float", usage: 0 });
	await runtime.recordVisibilityPass({
		context, encoder, hiZ, options: normalizeOcclusionCullingOptions({}),
	});
	assert.equal(backend.bindingGroupDestroyCalls, 1,
		"occlusion recording releases its transient binding acquisition");
	const params = backend.buffers.find((buffer) => buffer.label === "WebGPUOcclusionCullingParams");
	assert.equal(params.size, 4);
	assert.equal(params.lastWrite.data.byteLength, 4);
	assert.deepEqual(Array.from(params.lastWrite.data), [Math.fround(0.05)]);
	const candidates = backend.buffers.find((buffer) => buffer.label === "WebGPUOcclusionCandidates");
	assert.equal(candidates.size, 32);
	assert.deepEqual(Array.from(candidates.lastWrite.data),
		[1, 2, 3, 4, Math.fround(0.2), Math.fround(0.8), 12, 0]);
	assert.deepEqual(backend.dispatches, [[1, 1, 1]]);
	const shader = readFileSync(new URL(
		"../../../src/shaders/webgpu/utility/occlusionCulling.wgsl", import.meta.url,
	), "utf8");
	assert.equal(shader.match(/struct OcclusionParams\s*\{([^}]+)\}/)[1]
		.match(/:\s*f32\s*,/g).length * 4, 4);
	runtime.destroy();
}

async function run() {
	testVisibilityHysteresisAndRecovery();
	testStaleAndSignatureFallbacksAreVisible();
	testTemporalResetClearsVisibility();
	testResetIgnoresLateReadback();
	await testCompactDepthBiasUploadPreservesCandidateAlignment();
	console.log("test_webgpu_occlusion_culling_runtime: ok");
}

await run();
