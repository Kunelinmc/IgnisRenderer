import assert from "node:assert/strict";

import { WebGPUOcclusionCullingRuntime } from "../../../src/renderers/webgpu/WebGPUOcclusionCullingRuntime.ts";
import { normalizeOcclusionCullingOptions } from "../../../src/pipeline/OcclusionCulling.ts";

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

function run() {
	testVisibilityHysteresisAndRecovery();
	testStaleAndSignatureFallbacksAreVisible();
	testTemporalResetClearsVisibility();
	testResetIgnoresLateReadback();
	console.log("test_webgpu_occlusion_culling_runtime: ok");
}

run();
