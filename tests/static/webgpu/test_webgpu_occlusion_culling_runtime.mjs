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
	assert.equal(backend.bindingGroupDestroyCalls, 0,
		"occlusion retains its binding for reuse");
	const params = backend.buffers.find((buffer) => buffer.label === "WebGPUOcclusionCullingParams");
	assert.equal(params.size, 4);
	assert.equal(params.lastWrite.data.byteLength, 4);
	assert.deepEqual(Array.from(params.lastWrite.data), [Math.fround(0.05)]);
	const candidates = backend.buffers.find((buffer) => buffer.label === "WebGPUOcclusionCandidates");
	assert.equal(candidates.size, 32);
	assert.deepEqual(Array.from(candidates.lastWrite.data),
		[1, 2, 3, 4, Math.fround(0.2), Math.fround(0.8), 12, 0]);
	assert.deepEqual(backend.dispatches, [[1, 1, 1]]);
	assert.equal(nativeEncoder.calls.filter(([name]) => name === "copyBufferToBuffer").length, 1,
		"live occlusion results schedule a GPU readback");
	const shader = readFileSync(new URL(
		"../../../src/shaders/webgpu/utility/occlusionCulling.wgsl", import.meta.url,
	), "utf8");
	assert.equal(shader.match(/struct OcclusionParams\s*\{([^}]+)\}/)[1]
		.match(/:\s*f32\s*,/g).length * 4, 4);
	runtime.destroy();
}

async function testOcclusionBindingReuseAndInvalidation() {
	globalThis.GPUBufferUsage ??= { COPY_DST: 8, MAP_READ: 1 };
	const backend = new FakeWebGPUBackend();
	const runtime = new WebGPUOcclusionCullingRuntime(backend);
	const nativeEncoder = backend.createCommandEncoder();
	const encoder = new WebGPUCommandEncoder(nativeEncoder, {
		resourceManager: backend.resourceManager,
		createPassTimestampWrites: () => undefined,
	}, {});
	const candidate = createCandidate({
		screenRect: { x: 1, y: 2, width: 3, height: 4 },
		nearDepth: 0.2, farDepth: 0.8, screenAreaPx: 12,
	});
	const context = { ...createContext(), scene: { occlusion: { candidates: [candidate] } } };
	const options = normalizeOcclusionCullingOptions({});
	const firstHiZ = backend.createTexture({ width: 4, height: 4, format: "r32float", usage: 0 });
	const record = (hiZ) => runtime.recordVisibilityPass({ context, encoder, hiZ, options });

	await record(firstHiZ);
	const first = backend.bindingGroups.at(-1);
	runtime.beginFrame(context);
	await record(firstHiZ);
	assert.equal(backend.bindingGroups.length, 1, "unchanged resources reuse the binding");
	assert.equal(first.destroyed, false);
	assert.equal(backend.dispatches.length, 2);

	const secondHiZ = backend.createTexture({ width: 8, height: 8, format: "r32float", usage: 0 });
	await record(secondHiZ);
	const second = backend.bindingGroups.at(-1);
	assert.equal(backend.bindingGroups.length, 2);
	assert.equal(first.destroyed, true, "a replacement texture releases the old binding");
	assert.equal(second.desc.entries[0].resource, secondHiZ);

	context.scene.occlusion.candidates.push(createCandidate({ ...candidate, packetId: "packet-B" }));
	await record(secondHiZ);
	const grown = backend.bindingGroups.at(-1);
	assert.equal(backend.bindingGroups.length, 3);
	assert.equal(second.destroyed, true, "buffer growth releases the previous binding");
	assert.equal(grown.desc.entries[1].resource.size, 64);
	assert.equal(grown.desc.entries[3].resource.size, 8);

	runtime.onShaderRuntimeChanged();
	assert.equal(grown.destroyed, true, "shader changes release the old binding");
	await record(secondHiZ);
	const rebuilt = backend.bindingGroups.at(-1);
	assert.equal(backend.bindingGroups.length, 4);
	assert.notEqual(rebuilt.desc.pipeline, grown.desc.pipeline);

	secondHiZ.destroy();
	const dispatchesBeforeInvalidTexture = backend.dispatches.length;
	await record(secondHiZ);
	assert.equal(backend.dispatches.length, dispatchesBeforeInvalidTexture,
		"a destroyed texture must not record another pass");
	assert.equal(rebuilt.destroyed, true, "invalid resources release the retained binding");
	const thirdHiZ = backend.createTexture({ width: 8, height: 8, format: "r32float", usage: 0 });
	await record(thirdHiZ);
	const recovered = backend.bindingGroups.at(-1);
	assert.equal(backend.bindingGroups.length, 5, "a live texture can recover after failure");

	runtime.destroy();
	assert.equal(recovered.destroyed, true, "runtime destruction releases the retained binding");
}

async function testRetiredCandidateReleasesBindingBeforeRecording() {
	const backend = new FakeWebGPUBackend();
	const runtime = new WebGPUOcclusionCullingRuntime(backend);
	const encoder = new WebGPUCommandEncoder(backend.createCommandEncoder(), {
		resourceManager: backend.resourceManager,
		createPassTimestampWrites: () => undefined,
	}, {});
	const context = { ...createContext(), scene: { occlusion: { candidates: [createCandidate({
		screenRect: { x: 1, y: 2, width: 3, height: 4 },
		nearDepth: 0.2, farDepth: 0.8, screenAreaPx: 12,
	})] } } };
	const hiZ = backend.createTexture({ width: 4, height: 4, format: "r32float", usage: 0 });
	const request = { context, encoder, hiZ, options: normalizeOcclusionCullingOptions({}) };
	await runtime.recordVisibilityPass(request);
	const binding = backend.bindingGroups.at(-1);
	backend.resourceManager.invalidate(binding.desc.entries[1].resource);
	await runtime.recordVisibilityPass(request);
	assert.equal(backend.dispatches.length, 1, "retired candidates cannot be dispatched");
	assert.equal(binding.destroyed, true, "upload failure releases the retained binding");
	runtime.destroy();
}

function testUnavailableResultBuffersSkipReadback() {
	for (const retirement of ["destroy", "invalidate", "reset"]) {
		const backend = new FakeWebGPUBackend();
		const runtime = new WebGPUOcclusionCullingRuntime(backend);
		const nativeEncoder = backend.createCommandEncoder();
		const encoder = new WebGPUCommandEncoder(nativeEncoder, {
			resourceManager: backend.resourceManager,
			createPassTimestampWrites: () => undefined,
		}, {});
		const result = backend.createBuffer({ size: 4, usage: 0 });
		runtime._resultBuffer = result;
		if (retirement === "invalidate") backend.resourceManager.invalidate(result);
		else if (retirement === "reset") backend.resourceManager.reset();
		else result.destroy();
		const bufferCount = backend.buffers.length;
		const candidate = createCandidate();
		const options = normalizeOcclusionCullingOptions({});
		runtime._queueReadback(encoder, [candidate], options);
		assert.equal(backend.buffers.length, bufferCount,
			`${retirement} of results must skip staging allocation`);
		assert.equal(nativeEncoder.calls.some(([name]) => name === "copyBufferToBuffer"), false);
		assert.equal(runtime.getVisibilityProvider(options).isPacketVisible(candidate), true);
		runtime.destroy();
		backend.resourceManager.reset();
	}
}

async function run() {
	testVisibilityHysteresisAndRecovery();
	testStaleAndSignatureFallbacksAreVisible();
	testTemporalResetClearsVisibility();
	testResetIgnoresLateReadback();
	await testCompactDepthBiasUploadPreservesCandidateAlignment();
	await testOcclusionBindingReuseAndInvalidation();
	await testRetiredCandidateReleasesBindingBeforeRecording();
	testUnavailableResultBuffersSkipReadback();
	console.log("test_webgpu_occlusion_culling_runtime: ok");
}

await run();
