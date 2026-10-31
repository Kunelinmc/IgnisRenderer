import assert from "node:assert/strict";

import { createWebGPUResourceManager } from "../../helpers/fakes.mjs";
import { WebGPUHiZBuilder } from "../../../src/backends/webgpu/WebGPUHiZBuilder.ts";
import { createWebGPUComputeFacade } from "../../../src/backends/webgpu/ComputeFacade.ts";

import {
	DEFAULT_WEBGPU_DENOISE_OPTIONS,
	WebGPUDenoiser,
	resolveWebGPUDenoiseOptions,
	writeWebGPUDenoiseParams,
} from "../../../src/backends/webgpu/WebGPUDenoiser.ts";
import {
	embeddedShaderSources,
} from "../../../src/shaders/generated/embeddedShaderSources.ts";
import {
	WebGPUPostProcessRuntime,
} from "../../../src/backends/webgpu/WebGPUPostProcessRuntime.ts";
import {
	FakeBackend,
	FakeEncoder,
	createTexture,
} from "../../helpers/webgpu_postprocess_runtime_test_helpers.mjs";

function createRequest(encoder, overrides = {}) {
	const source = overrides.source ?? createTexture(16, 8, "source");
	return {
		scope: overrides.scope ?? "test-fast",
		encoder,
		source,
		scratch: overrides.scratch ?? createTexture(16, 8, "scratch"),
		output: overrides.output ?? createTexture(16, 8, "output"),
		depth: overrides.depth ?? createTexture(32, 16, "depth"),
		normal: overrides.normal ?? createTexture(32, 16, "normal"),
		moments: overrides.moments,
		sampler: overrides.sampler ?? { label: "sampler" },
		options: overrides.options,
	};
}

function testOptionsAndParameterPacking() {
	assert.equal(resolveWebGPUDenoiseOptions({ varianceGuided: true }).varianceGuided, true);
	assert.equal(writeWebGPUDenoiseParams(new Float32Array(12), 8, 8, 1,
		resolveWebGPUDenoiseOptions({ varianceGuided: true }))[11], 1);
	assert.equal(resolveWebGPUDenoiseOptions({ roughnessAware: true }).roughnessAware, true);
	assert.equal(writeWebGPUDenoiseParams(new Float32Array(12), 8, 8, 1,
		resolveWebGPUDenoiseOptions({ roughnessAware: true }))[10], 1);
	assert.deepEqual(resolveWebGPUDenoiseOptions(), DEFAULT_WEBGPU_DENOISE_OPTIONS);
	assert.deepEqual(
		resolveWebGPUDenoiseOptions({
			mode: "quality",
			signal: "scalar",
			radius: 99,
			depthPhi: -1,
			normalPhi: Number.NaN,
			valuePhi: Number.POSITIVE_INFINITY,
			confidenceFloor: 4,
		}),
		{
			mode: "quality",
			signal: "scalar",
			radius: 2,
			depthPhi: 0,
			normalPhi: 16,
			valuePhi: 2,
			confidenceFloor: 1,
			roughnessAware: false,
			varianceGuided: false,
		}
	);
	const params = writeWebGPUDenoiseParams(
		new Float32Array(12).fill(Number.NaN),
		16,
		8,
		4,
		resolveWebGPUDenoiseOptions({
			mode: "quality",
			signal: "scalar",
		})
	);
	assert.deepEqual(Array.from(params.slice(0, 7)), [
		1 / 16,
		1 / 8,
		2,
		4,
		24,
		16,
		2,
	]);
	assert.ok(Math.abs(params[7] - 0.05) <= 1e-6);
	assert.equal(params.byteLength, 48);
	assert.deepEqual(Array.from(params.slice(8)), [1, 1, 0, 0]);
	const oversizedFootprint = writeWebGPUDenoiseParams(
		new Float32Array(12),
		Number.NaN,
		Number.POSITIVE_INFINITY,
		99,
		{
			...DEFAULT_WEBGPU_DENOISE_OPTIONS,
			radius: 4,
			depthPhi: -1,
			normalPhi: Number.NaN,
			valuePhi: Number.POSITIVE_INFINITY,
			confidenceFloor: -2,
		}
	);
	assert.deepEqual(Array.from(oversizedFootprint.slice(0, 4)), [
		1,
		1,
		4,
		2,
	]);
	assert.deepEqual(Array.from(oversizedFootprint.slice(4, 8)), [
		0,
		DEFAULT_WEBGPU_DENOISE_OPTIONS.normalPhi,
		DEFAULT_WEBGPU_DENOISE_OPTIONS.valuePhi,
		0,
	]);
	for (const length of [4, 16]) {
		assert.throws(
			() => writeWebGPUDenoiseParams(
				new Float32Array(length),
				1,
				1,
				1,
				resolveWebGPUDenoiseOptions()
			),
			/12 floats/
		);
	}
}

function testShaderDefensiveFiltering() {
	const shader =
		embeddedShaderSources["./webgpu/postprocess/denoise.wgsl"];
	assert.ok(shader);
	assert.match(shader, /max\(TILE_HALO \/ radius, 1\)/);
	assert.match(shader, /centerIndex >= TILE_COUNT/);
	assert.match(shader, /sampleIndexSigned < 0/);
	assert.match(shader, /sampleIndexSigned >= i32\(TILE_COUNT\)/);
	assert.match(shader, /signalValueDistance/);
	assert.match(shader, /log\(1\.0 \+ sampleValue\)/);
	assert.match(shader, /mix\(centerSignal\.a, neighborhoodConfidence/);
	assert.match(shader, /mix\(1\.0, rawValueWeight, valueTrust\)/);
}

async function testFastAndQualityDispatches() {
	const backend = new FakeBackend();
	const denoiser = new WebGPUDenoiser(createWebGPUComputeFacade(backend));
	const fastEncoder = new FakeEncoder();
	const fastRequest = createRequest(fastEncoder);
	const fast = await denoiser.encode(fastRequest);

	assert.equal(fast.texture, fastRequest.output);
	assert.equal(fast.mode, "fast");
	assert.equal(fast.dispatchCount, 2);
	const fastParams = backend.buffers.find(
		(buffer) => buffer.label === "WebGPUDenoiseParams_test-fast_0"
	);
	assert.equal(fastParams.size, 48);
	assert.equal(fastParams.lastWrite.data.byteLength, 48);
	assert.deepEqual(fastParams.lastWrite.slice(0, 7), [1 / 16, 1 / 8, 2, 1, 24, 16, 2]);
	assert.deepEqual(fastParams.lastWrite.slice(8), [0, 0, 0, 0]);
	assert.equal(
		backend.shaderModules.filter(
			(module) => module.label === "WebGPUDenoiseShader"
		).length,
		1
	);
	assert.deepEqual(
		backend.computePipelines
			.filter((pipeline) => pipeline.label.startsWith("WebGPUDenoise"))
			.map((pipeline) => pipeline.desc.compute.entryPoint),
		["csDenoiseHorizontal", "csDenoiseVertical"]
	);
	assert.deepEqual(
		fastEncoder.calls.filter((call) => call[0] === "beginComputePass"),
		[
			["beginComputePass", "WebGPUDenoise_test-fast_fast_H_1"],
			["beginComputePass", "WebGPUDenoise_test-fast_fast_V_1"],
		]
	);
	await denoiser.encode(fastRequest);
	assert.equal(
		backend.buffers.some(
			(buffer) => buffer.label === "WebGPUDenoiseParams_test-fast_3"
		),
		true
	);

	const bindingCount = backend.bindingGroups.length;
	await denoiser.encode({
		...fastRequest,
		encoder: new FakeEncoder(),
	});
	assert.equal(backend.bindingGroups.length, bindingCount);

	const qualityEncoder = new FakeEncoder();
	const qualityRequest = createRequest(qualityEncoder, {
		scope: "test-quality",
		options: { mode: "quality" },
	});
	const quality = await denoiser.encode(qualityRequest);
	assert.equal(quality.mode, "quality");
	assert.equal(quality.dispatchCount, 6);
	const qualityParams = backend.buffers.filter(
		(buffer) => buffer.label.startsWith("WebGPUDenoiseParams_test-quality_")
	);
	assert.deepEqual(qualityParams.map((buffer) => buffer.size), [48, 48, 48]);
	assert.deepEqual(qualityParams.map((buffer) => buffer.lastWrite.data.byteLength), [48, 48, 48]);
	assert.deepEqual(qualityParams.map((buffer) => buffer.lastWrite[3]), [1, 2, 4]);
	assert.ok(qualityParams.every((buffer) => buffer.lastWrite[9] === 1));
	assert.deepEqual(
		qualityEncoder.calls
			.filter((call) => call[0] === "beginComputePass")
			.map((call) => call[1]),
		[
			"WebGPUDenoise_test-quality_quality_H_1",
			"WebGPUDenoise_test-quality_quality_V_1",
			"WebGPUDenoise_test-quality_quality_H_2",
			"WebGPUDenoise_test-quality_quality_V_2",
			"WebGPUDenoise_test-quality_quality_H_4",
			"WebGPUDenoise_test-quality_quality_V_4",
		]
	);
	assert.deepEqual(
		backend.buffers
			.filter((buffer) => buffer.label.startsWith("WebGPUDenoiseParams"))
			.map((buffer) => buffer.label),
		[
			"WebGPUDenoiseParams_test-fast_0",
			"WebGPUDenoiseParams_test-fast_3",
			"WebGPUDenoiseParams_test-quality_0",
			"WebGPUDenoiseParams_test-quality_1",
			"WebGPUDenoiseParams_test-quality_2",
		]
	);
	denoiser.destroy();
}

async function testSourceOverwriteAndValidation() {
	const backend = new FakeBackend();
	const denoiser = new WebGPUDenoiser(createWebGPUComputeFacade(backend));
	const encoder = new FakeEncoder();
	const source = createTexture(16, 8, "source-output");
	const overwrite = await denoiser.encode(createRequest(encoder, {
		scope: "test-overwrite",
		source,
		output: source,
		options: { signal: "scalar" },
	}));
	assert.equal(overwrite.texture, source);
	for (const group of backend.bindingGroups) {
		const entries = group.desc.entries;
		assert.equal(entries.find((entry) => entry.binding === 6).resource,
			entries.find((entry) => entry.binding === 0).resource);
		assert.notEqual(entries.find((entry) => entry.binding === 6).resource,
			entries.find((entry) => entry.binding === 5).resource);
	}
	for (const target of ["scratch", "output"]) {
		const writable = createTexture(16, 8, `aliased-${target}`);
		await assert.rejects(
			denoiser.encode(createRequest(new FakeEncoder(), {
				scope: `bad-moments-${target}`,
				[target]: writable,
				moments: writable,
				options: { varianceGuided: true },
			})),
			/moments distinct from scratch and output/
		);
	}

	await assert.rejects(
		denoiser.encode(createRequest(new FakeEncoder(), {
			scope: "bad-alias",
			source,
			scratch: source,
		})),
		/distinct from source and output/
	);
	await assert.rejects(
		denoiser.encode(createRequest(new FakeEncoder(), {
			scope: "bad-size",
			output: createTexture(8, 8, "wrong-size"),
		})),
		/does not match/
	);
	await assert.rejects(
		denoiser.encode(createRequest(new FakeEncoder(), {
			scope: "bad-format",
			output: {
				...createTexture(16, 8, "wrong-format"),
				format: "rgba8unorm",
			},
		})),
		/must use rgba16float/
	);

	const groupsBeforeInvalidate = backend.bindingGroups.length;
	denoiser.invalidateBindings();
	assert.equal(backend.bindingGroupDestroyCalls, groupsBeforeInvalidate);
	denoiser.invalidateShaderResources();
	assert.equal(backend.shaderModuleDestroyCalls, 1);
	assert.equal(backend.computePipelineDestroyCalls, 2);
	assert.equal(backend.bufferDestroyCalls, 1);
	denoiser.destroy();
}

async function testRuntimeOwnershipAndInvalidation() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(createWebGPUComputeFacade(backend), () => {});
	const denoiser = runtime.getDenoiser();
	assert.equal(runtime.getDenoiser(), denoiser);

	await denoiser.encode(createRequest(new FakeEncoder(), {
		scope: "runtime-owned",
	}));
	const bindingCount = backend.bindingGroups.length;
	runtime.invalidateBindings();
	assert.equal(backend.bindingGroupDestroyCalls, bindingCount);

	runtime.onShaderRuntimeChanged();
	assert.equal(backend.shaderModuleDestroyCalls, 1);
	assert.equal(backend.computePipelineDestroyCalls, 2);
	assert.equal(backend.bufferDestroyCalls, 1);
	assert.equal(runtime.getDenoiser(), denoiser);

	await denoiser.ensureResources();
	runtime.destroy();
	assert.equal(backend.shaderModuleDestroyCalls, backend.shaderModules.length);
	assert.equal(
		backend.computePipelineDestroyCalls,
		backend.computePipelines.length
	);
	assert.equal(backend.bufferDestroyCalls, backend.buffers.length);
}

function testDeviceLossCleanupContinuesAfterReleaseFailures() {
	const backend = new FakeBackend();
	const facade = createWebGPUComputeFacade(backend);
	for (const owner of [new WebGPUDenoiser(facade), new WebGPUHiZBuilder(facade)]) {
		const released = [];
		const first = backend.resourceManager.registerBindingGroup({}, "first", () => {
			released.push("first");
			throw new Error("device lost during binding release");
		});
		const second = backend.resourceManager.registerBindingGroup({}, "second", () => {
			released.push("second");
		});
		owner._bindings.set("first", { group: first, resources: [] });
		owner._bindings.set("second", { group: second, resources: [] });
		assert.doesNotThrow(() => owner.invalidateBindings());
		assert.deepEqual(released, ["first", "second"]);
		assert.equal(owner._bindings.size, 0);
		assert.doesNotThrow(() => owner.invalidateBindings());
		assert.deepEqual(released, ["first", "second"]);
		owner.destroy();
	}
}

testDeviceLossCleanupContinuesAfterReleaseFailures();
testOptionsAndParameterPacking();
testShaderDefensiveFiltering();
await testFastAndQualityDispatches();
await testSourceOverwriteAndValidation();
await testRuntimeOwnershipAndInvalidation();
console.log("WebGPU denoiser tests passed");
