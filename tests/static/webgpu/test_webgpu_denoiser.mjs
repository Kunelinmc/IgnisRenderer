import assert from "node:assert/strict";

import {
	DEFAULT_WEBGPU_DENOISE_OPTIONS,
	WebGPUDenoiser,
	resolveWebGPUDenoiseOptions,
	writeWebGPUDenoiseParams,
} from "../../../src/backends/webgpu/WebGPUDenoiser.ts";
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
		sampler: overrides.sampler ?? { label: "sampler" },
		options: overrides.options,
	};
}

function testOptionsAndParameterPacking() {
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
		}
	);
	const params = writeWebGPUDenoiseParams(
		new Float32Array(16),
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
	assert.deepEqual(Array.from(params.slice(8, 10)), [1, 1]);
	assert.throws(
		() => writeWebGPUDenoiseParams(
			new Float32Array(4),
			1,
			1,
			1,
			resolveWebGPUDenoiseOptions()
		),
		/16 floats/
	);
}

async function testFastAndQualityDispatches() {
	const backend = new FakeBackend();
	const denoiser = new WebGPUDenoiser(backend);
	const fastEncoder = new FakeEncoder();
	const fastRequest = createRequest(fastEncoder);
	const fast = await denoiser.encode(fastRequest);

	assert.equal(fast.texture, fastRequest.output);
	assert.equal(fast.mode, "fast");
	assert.equal(fast.dispatchCount, 2);
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
	const denoiser = new WebGPUDenoiser(backend);
	const encoder = new FakeEncoder();
	const source = createTexture(16, 8, "source-output");
	const overwrite = await denoiser.encode(createRequest(encoder, {
		scope: "test-overwrite",
		source,
		output: source,
		options: { signal: "scalar" },
	}));
	assert.equal(overwrite.texture, source);

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
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
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

testOptionsAndParameterPacking();
await testFastAndQualityDispatches();
await testSourceOverwriteAndValidation();
await testRuntimeOwnershipAndInvalidation();
console.log("WebGPU denoiser tests passed");
