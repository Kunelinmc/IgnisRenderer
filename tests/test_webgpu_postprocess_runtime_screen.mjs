import assert from "node:assert/strict";
import { WebGPUPostProcessRuntime } from "../src/renderers/webgpu/WebGPUPostProcessRuntime.ts";
import {
	FakeBackend,
	FakeEncoder,
	assertClose,
	createTexture,
} from "./helpers/webgpu_postprocess_runtime_test_helpers.mjs";
import { createResolvedPostProcess } from "./helpers/postprocess.mjs";

function createFrameContext(postProcessRequest = {}) {
	return {
		features: {},
		postProcess: createResolvedPostProcess(postProcessRequest),
		transient: new Map(),
	};
}

async function testFXAARuntimeUsesDedicatedPipeline() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(17, 9, "scene");
	const postPing = createTexture(17, 9, "ping");
	const postPong = createTexture(17, 9, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};

	await runtime.executePass({
		passId: "fxaa",
		encoder,
		targets,
		frameContext: createFrameContext(),
	});

	assert.equal(backend.samplers.length, 1);
	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.shaderModules[0].label, "WebGPUFXAAShader");
	assert.ok(backend.shaderModules[0].desc.code.includes("fn perceptualLuma"));
	assert.ok(backend.shaderModules[0].desc.code.includes("FXAA_QUALITY"));
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.computePipelines[0].label, "WebGPUFXAAPipeline");
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.buffers[0].desc.label, "WebGPUFXAAParams");
	assert.equal(backend.buffers[0].desc.size, 24);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroups[0].desc.entries.length, 4);
	assert.equal(
		backend.bindingGroups[0].desc.entries[0].resource,
		sceneColorMain
	);
	assert.equal(backend.bindingGroups[0].desc.entries[3].resource, postPong);

	const params = backend.buffers[0].lastWrite;
	assert.equal(params.length, 6);
	assertClose(params[0], 1 / 17);
	assertClose(params[1], 1 / 9);
	assertClose(params[2], 0.03125);
	assertClose(params[3], 0.166);
	assertClose(params[4], 0.75);
	assertClose(params[5], 0);

	assert.deepEqual(encoder.calls, [
		["beginComputePass", "WebGPUFXAA"],
		["setComputePipeline", "WebGPUFXAAPipeline"],
		["setBindingGroup", 0, "WebGPUFXAA_Binding"],
		["dispatchWorkgroups", 3, 2, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPong);
}

async function testToneMappingRuntimeUsesDedicatedPipeline() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(18, 10, "scene");
	const postPing = createTexture(18, 10, "ping");
	const postPong = createTexture(18, 10, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};

	await runtime.executePass({
		passId: "tonemap",
		encoder,
		targets,
		frameContext: createFrameContext(),
	});

	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.shaderModules[0].label, "WebGPUToneMappingShader");
	assert.ok(backend.shaderModules[0].desc.code.includes("acesFitted"));
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.computePipelines[0].label, "WebGPUToneMappingPipeline");
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroups[0].desc.entries.length, 2);
	assert.equal(
		backend.bindingGroups[0].desc.entries[0].resource,
		sceneColorMain
	);
	assert.equal(backend.bindingGroups[0].desc.entries[1].resource, postPong);
	assert.deepEqual(encoder.calls, [
		["beginComputePass", "WebGPUToneMapping"],
		["setComputePipeline", "WebGPUToneMappingPipeline"],
		["setBindingGroup", 0, "WebGPUToneMapping_Binding"],
		["dispatchWorkgroups", 3, 2, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPong);
}

async function testBloomRuntimeUsesDedicatedPipeline() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(40, 20, "scene");
	const postPing = createTexture(40, 20, "ping");
	const postPong = createTexture(40, 20, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};

	await runtime.executePass({
		passId: "bloom",
		encoder,
		targets,
		frameContext: createFrameContext({
			bloom: {
				enabled: true,
				options: {
					threshold: 1.2,
					softKnee: 0.35,
					intensity: 1.5,
					radius: 2,
				},
			},
		}),
	});

	assert.equal(backend.samplers.length, 1);
	assert.deepEqual(
		backend.shaderModules.map((module) => module.label),
		[
			"WebGPUBloomDownsampleShader",
			"WebGPUBloomBlurHShader",
			"WebGPUBloomBlurVShader",
			"WebGPUBloomUpsampleShader",
			"WebGPUBloomCompositeShader",
		]
	);
	assert.deepEqual(
		backend.computePipelines.map((pipeline) => pipeline.label),
		[
			"WebGPUBloomDownsamplePipeline",
			"WebGPUBloomBlurHPipeline",
			"WebGPUBloomBlurVPipeline",
			"WebGPUBloomUpsamplePipeline",
			"WebGPUBloomCompositePipeline",
		]
	);
	assert.equal(backend.buffers.length, 4);
	assert.deepEqual(
		backend.buffers.map((buffer) => buffer.desc.label),
		[
			"WebGPUBloomDownsampleParams",
			"WebGPUBloomBlurParams",
			"WebGPUBloomUpsampleParams",
			"WebGPUBloomCompositeParams",
		]
	);
	assert.ok(backend.buffers.every((buffer) => buffer.desc.size === 16));
	assert.equal(backend.textures.length, 10);
	assert.equal(backend.bindingGroups.length, 20);
	assert.equal(backend.writeBufferCalls, 20);

	const compositeParams = backend.buffers.find(
		(buffer) => buffer.desc.label === "WebGPUBloomCompositeParams"
	)?.lastWrite;
	assert.ok(compositeParams);
	assert.equal(compositeParams.length, 4);
	assertClose(compositeParams[0], 1 / 40);
	assertClose(compositeParams[1], 1 / 20);
	assertClose(compositeParams[2], 1.5);
	assertClose(compositeParams[3], 0);

	assert.deepEqual(encoder.calls.slice(0, 5), [
		["beginComputePass", "WebGPUBloom_Downsample0"],
		["setComputePipeline", "WebGPUBloomDownsamplePipeline"],
		["setBindingGroup", 0, "WebGPUBloom_Downsample0"],
		["dispatchWorkgroups", 3, 2, 1],
		["endComputePass"],
	]);
	assert.deepEqual(encoder.calls.slice(-5), [
		["beginComputePass", "WebGPUBloom_Composite"],
		["setComputePipeline", "WebGPUBloomCompositePipeline"],
		["setBindingGroup", 0, "WebGPUBloom_Composite"],
		["dispatchWorkgroups", 5, 3, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPong);
}

async function testMotionBlurRuntimeUsesDedicatedPipeline() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(48, 24, "scene");
	const postPing = createTexture(48, 24, "ping");
	const postPong = createTexture(48, 24, "pong");
	const gMotionDepth = createTexture(48, 24, "motion-depth");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
		gMotionDepth,
	};

	await runtime.executePass({
		passId: "motion-blur",
		encoder,
		targets,
		frameContext: createFrameContext({
			"motion-blur": {
				enabled: true,
				options: {
					shutterScale: 1.1,
					maxSamples: 18,
					velocityClamp: 0.08,
					depthReject: 0.03,
					centerWeight: 1.25,
				},
			},
		}),
	});

	assert.equal(backend.samplers.length, 1);
	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.shaderModules[0].label, "WebGPUMotionBlurShader");
	assert.ok(backend.shaderModules[0].desc.code.includes("depthConfidence"));
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.computePipelines[0].label, "WebGPUMotionBlurPipeline");
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.buffers[0].desc.label, "WebGPUMotionBlurParams");
	assert.equal(backend.buffers[0].desc.size, 32);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroups[0].desc.entries.length, 5);
	assert.equal(
		backend.bindingGroups[0].desc.entries[0].resource,
		sceneColorMain
	);
	assert.equal(backend.bindingGroups[0].desc.entries[1].resource, gMotionDepth);
	assert.equal(backend.bindingGroups[0].desc.entries[4].resource, postPong);

	const params = backend.buffers[0].lastWrite;
	assert.equal(params.length, 8);
	assertClose(params[0], 1 / 48);
	assertClose(params[1], 1 / 24);
	assertClose(params[2], 1.1);
	assertClose(params[3], 18);
	assertClose(params[4], 0.08);
	assertClose(params[5], 0.03);
	assertClose(params[6], 1.25);
	assertClose(params[7], 0);

	assert.deepEqual(encoder.calls, [
		["beginComputePass", "WebGPUMotionBlur"],
		["setComputePipeline", "WebGPUMotionBlurPipeline"],
		["setBindingGroup", 0, "WebGPUMotionBlur_Binding"],
		["dispatchWorkgroups", 6, 3, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPong);
}

async function testFogRuntimeUsesDedicatedPipeline() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(24, 12, "scene");
	const postPing = createTexture(24, 12, "ping");
	const postPong = createTexture(24, 12, "pong");
	const gMotionDepth = createTexture(24, 12, "motion-depth");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
		gMotionDepth,
	};

	await runtime.executePass({
		passId: "fog",
		encoder,
		targets,
		frameContext: createFrameContext({
			fog: {
				enabled: true,
				options: {
					mode: "exp2",
					color: [0.1, 0.2, 0.3],
					start: 10,
					end: 120,
					density: 0.02,
					strength: 0.75,
				},
			},
		}),
	});

	assert.equal(backend.samplers.length, 1);
	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.shaderModules[0].label, "WebGPUFogShader");
	assert.ok(backend.shaderModules[0].desc.code.includes("ignisComputeFogFactor"));
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.computePipelines[0].label, "WebGPUFogPipeline");
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.buffers[0].desc.label, "WebGPUFogParams");
	assert.equal(backend.buffers[0].desc.size, 32);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroups[0].desc.entries.length, 5);
	assert.equal(backend.bindingGroups[0].desc.entries[0].resource, sceneColorMain);
	assert.equal(backend.bindingGroups[0].desc.entries[1].resource, gMotionDepth);
	assert.equal(backend.bindingGroups[0].desc.entries[4].resource, postPong);

	const params = backend.buffers[0].lastWrite;
	assert.equal(params.length, 8);
	assertClose(params[0], 2);
	assertClose(params[1], 10);
	assertClose(params[2], 120);
	assertClose(params[3], 0.02);
	assertClose(params[4], 0.1);
	assertClose(params[5], 0.2);
	assertClose(params[6], 0.3);
	assertClose(params[7], 0.75);

	assert.deepEqual(encoder.calls, [
		["beginComputePass", "WebGPUFog"],
		["setComputePipeline", "WebGPUFogPipeline"],
		["setBindingGroup", 0, "WebGPUFog_Binding"],
		["dispatchWorkgroups", 3, 2, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPong);
}

async function testDOFRuntimeUsesDedicatedPipeline() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(36, 18, "scene");
	const postPing = createTexture(36, 18, "ping");
	const postPong = createTexture(36, 18, "pong");
	const gMotionDepth = createTexture(36, 18, "motion-depth");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
		gMotionDepth,
	};

	await runtime.executePass({
		passId: "dof",
		encoder,
		targets,
		frameContext: createFrameContext({
			dof: {
				enabled: true,
				options: {
					focusDistance: 6,
					focusRange: 2,
					nearStrength: 0.7,
					farStrength: 1.2,
					maxBlurRadius: 10,
					depthCurve: 1.5,
					highlightThreshold: 1.1,
					highlightGain: 0.4,
					chromaticAberration: 0.3,
				},
			},
		}),
	});

	assert.equal(backend.samplers.length, 1);
	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.shaderModules[0].label, "WebGPUDOFShader");
	assert.ok(backend.shaderModules[0].desc.code.includes("computeCoC"));
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.computePipelines[0].label, "WebGPUDOFPipeline");
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.buffers[0].desc.label, "WebGPUDOFParams");
	assert.equal(backend.buffers[0].desc.size, 48);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroups[0].desc.entries.length, 5);
	assert.equal(
		backend.bindingGroups[0].desc.entries[0].resource,
		sceneColorMain
	);
	assert.equal(backend.bindingGroups[0].desc.entries[1].resource, gMotionDepth);
	assert.equal(backend.bindingGroups[0].desc.entries[4].resource, postPong);

	const params = backend.buffers[0].lastWrite;
	assert.equal(params.length, 12);
	assertClose(params[0], 1 / 36);
	assertClose(params[1], 1 / 18);
	assertClose(params[2], 6);
	assertClose(params[3], 2);
	assertClose(params[4], 0.7);
	assertClose(params[5], 1.2);
	assertClose(params[6], 10);
	assertClose(params[7], 1.5);
	assertClose(params[8], 1.1);
	assertClose(params[9], 0.4);
	assertClose(params[10], 0.3);
	assertClose(params[11], 0);

	assert.deepEqual(encoder.calls, [
		["beginComputePass", "WebGPUDOF"],
		["setComputePipeline", "WebGPUDOFPipeline"],
		["setBindingGroup", 0, "WebGPUDOF_Binding"],
		["dispatchWorkgroups", 5, 3, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPong);
}

async function testMotionBlurSkipsRedundantParamUploads() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const sceneColorMain = createTexture(48, 24, "scene");
	const postPing = createTexture(48, 24, "ping");
	const postPong = createTexture(48, 24, "pong");
	const gMotionDepth = createTexture(48, 24, "motion-depth");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
		gMotionDepth,
	};
	const frameContext = createFrameContext({
		"motion-blur": {
			enabled: true,
			options: {
				shutterScale: 1.1,
				maxSamples: 18,
				velocityClamp: 0.08,
				depthReject: 0.03,
				centerWeight: 1.25,
			},
		},
	});

	await runtime.executePass({
		passId: "motion-blur",
		encoder: new FakeEncoder(),
		targets,
		frameContext,
	});
	await runtime.executePass({
		passId: "motion-blur",
		encoder: new FakeEncoder(),
		targets,
		frameContext,
	});

	assert.equal(backend.writeBufferCalls, 1);
}

async function testFXAARuntimePingPongsAndCachesResources() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const firstEncoder = new FakeEncoder();
	const secondEncoder = new FakeEncoder();
	const sceneColorMain = createTexture(32, 16, "scene");
	const postPing = createTexture(32, 16, "ping");
	const postPong = createTexture(32, 16, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};
	const frameContext = createFrameContext();

	await runtime.executePass({
		passId: "fxaa",
		encoder: firstEncoder,
		targets,
		frameContext,
	});
	await runtime.executePass({
		passId: "fxaa",
		encoder: secondEncoder,
		targets,
		frameContext,
	});

	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.bindingGroups.length, 2);
	assert.equal(backend.bindingGroups[1].desc.entries[0].resource, postPong);
	assert.equal(backend.bindingGroups[1].desc.entries[3].resource, postPing);
	assert.equal(targets.sceneColor, postPing);
	assert.deepEqual(secondEncoder.calls, [
		["beginComputePass", "WebGPUFXAA"],
		["setComputePipeline", "WebGPUFXAAPipeline"],
		["setBindingGroup", 0, "WebGPUFXAA_Binding"],
		["dispatchWorkgroups", 4, 2, 1],
		["endComputePass"],
	]);
}

async function testInvalidateBindingsDestroysCachedBindingGroups() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const sceneColorMain = createTexture(20, 12, "scene");
	const postPing = createTexture(20, 12, "ping");
	const postPong = createTexture(20, 12, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};

	await runtime.executePass({
		passId: "fxaa",
		encoder: new FakeEncoder(),
		targets,
		frameContext: createFrameContext(),
	});
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroupDestroyCalls, 0);

	runtime.invalidateBindings();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
	runtime.invalidateBindings();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
}

async function testBindingReplacementDestroysStaleBindingGroup() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const sceneColorMain = createTexture(28, 14, "scene");
	const postPing = createTexture(28, 14, "ping");
	const postPong = createTexture(28, 14, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};
	const frameContext = createFrameContext();

	await runtime.executePass({
		passId: "fxaa",
		encoder: new FakeEncoder(),
		targets,
		frameContext,
	});
	await runtime.executePass({
		passId: "fxaa",
		encoder: new FakeEncoder(),
		targets,
		frameContext,
	});
	await runtime.executePass({
		passId: "fxaa",
		encoder: new FakeEncoder(),
		targets,
		frameContext,
	});

	assert.equal(backend.bindingGroups.length, 3);
	assert.equal(backend.bindingGroupDestroyCalls, 1);
}

export async function run() {
	await testFXAARuntimeUsesDedicatedPipeline();
	await testToneMappingRuntimeUsesDedicatedPipeline();
	await testBloomRuntimeUsesDedicatedPipeline();
	await testFogRuntimeUsesDedicatedPipeline();
	await testMotionBlurRuntimeUsesDedicatedPipeline();
	await testMotionBlurSkipsRedundantParamUploads();
	await testDOFRuntimeUsesDedicatedPipeline();
	await testFXAARuntimePingPongsAndCachesResources();
	await testInvalidateBindingsDestroysCachedBindingGroups();
	await testBindingReplacementDestroysStaleBindingGroup();
	console.log("WebGPU postprocess screen runtime tests passed");
}

if (import.meta.url === `file://${process.argv[1]}`) {
	await run();
}
