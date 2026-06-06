import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import {
	BloomPass,
	ColorFilterPass,
	DepthOfFieldPass,
	FastApproximateAntiAliasingPass,
	FogPass,
	InteractionOutlinePass,
	MotionBlurPass,
	ToneMappingPass,
} from "../../../src/postprocess/index.ts";
import { INTERACTION_TRANSIENT_STATE_KEY } from "../../../src/pipeline/types.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { WebGPUPostProcessRuntime } from "../../../src/renderers/webgpu/WebGPUPostProcessRuntime.ts";
import {
	FakeBackend,
	FakeEncoder,
	assertClose,
	createTexture,
} from "../../helpers/webgpu_postprocess_runtime_test_helpers.mjs";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createFrameContext(postProcessRequest = {}) {
	return {
		features: {},
		postProcess: createResolvedPostProcess(postProcessRequest),
		transient: new Map(),
	};
}

function destroySnapshotPasses(snapshot) {
	for (const resolved of snapshot.getEnabledPasses()) {
		resolved.pass.destroy();
	}
}

function createFXAAPassRequest(
	frameContext,
	pass = new FastApproximateAntiAliasingPass({ enabled: true })
) {
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "fxaa",
		options: frameContext.postProcess.getOptions("fxaa"),
		startPassId: null,
	};
}

function createBloomPassRequest(frameContext) {
	const pass = new BloomPass({ enabled: true });
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "bloom",
		options: frameContext.postProcess.getOptions("bloom"),
		startPassId: null,
	};
}

function createFogPassRequest(frameContext) {
	const pass = new FogPass({ enabled: true });
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "fog",
		options: frameContext.postProcess.getOptions("fog"),
		startPassId: null,
	};
}

function createMotionBlurPassRequest(frameContext, pass) {
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "motion-blur",
		options: frameContext.postProcess.getOptions("motion-blur"),
		startPassId: null,
	};
}

function createDepthOfFieldPassRequest(frameContext, pass) {
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "dof",
		options: frameContext.postProcess.getOptions("dof"),
		startPassId: null,
	};
}

function createToneMappingPassRequest(frameContext, pass) {
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "tonemap",
		options: frameContext.postProcess.getOptions("tonemap"),
		startPassId: null,
	};
}

function createColorFilterPassRequest(frameContext, pass) {
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "color-filter",
		options: frameContext.postProcess.getOptions("color-filter"),
		startPassId: null,
	};
}

function createInteractionOutlinePassRequest(frameContext, pass) {
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "interaction-outline",
		options: frameContext.postProcess.getOptions("interaction-outline"),
		startPassId: null,
	};
}

function createWebGPUImplementationContext(runtime, encoder, targets) {
	return {
		encoder,
		targets,
		shared: runtime.sharedContext,
		publishColorTarget: (texture) => {
			targets.sceneColor = texture;
		},
	};
}

async function executeFXAAPass(runtime, encoder, targets, frameContext, pass) {
	const request = createFXAAPassRequest(frameContext, pass);
	return request.pass.getImplementation("webgpu").execute(request, {
		encoder,
		targets,
		shared: runtime.sharedContext,
		publishColorTarget: (texture) => {
			targets.sceneColor = texture;
		},
	});
}

async function executeBloomPass(runtime, encoder, targets, frameContext) {
	const request = createBloomPassRequest(frameContext);
	const result = await request.pass.getImplementation("webgpu").execute(request, {
		encoder,
		targets,
		shared: runtime.sharedContext,
		publishColorTarget: (texture) => {
			targets.sceneColor = texture;
		},
	});
	return { request, result };
}

async function executeFogPass(runtime, encoder, targets, frameContext) {
	const request = createFogPassRequest(frameContext);
	const result = await request.pass.getImplementation("webgpu").execute(request, {
		encoder,
		targets,
		shared: runtime.sharedContext,
		publishColorTarget: (texture) => {
			targets.sceneColor = texture;
		},
	});
	return { request, result };
}

async function executeMotionBlurPass(
	runtime,
	encoder,
	targets,
	frameContext,
	pass = new MotionBlurPass({ enabled: true })
) {
	const request = createMotionBlurPassRequest(frameContext, pass);
	const result = await request.pass
		.getImplementation("webgpu")
		.execute(
			request,
			createWebGPUImplementationContext(runtime, encoder, targets)
		);
	return { request, result };
}

async function executeDepthOfFieldPass(
	runtime,
	encoder,
	targets,
	frameContext,
	pass = new DepthOfFieldPass({ enabled: true })
) {
	const request = createDepthOfFieldPassRequest(frameContext, pass);
	const result = await request.pass
		.getImplementation("webgpu")
		.execute(
			request,
			createWebGPUImplementationContext(runtime, encoder, targets)
		);
	return { request, result };
}

async function executeToneMappingPass(
	runtime,
	encoder,
	targets,
	frameContext,
	pass = new ToneMappingPass({ enabled: true })
) {
	const request = createToneMappingPassRequest(frameContext, pass);
	const result = await request.pass
		.getImplementation("webgpu")
		.execute(
			request,
			createWebGPUImplementationContext(runtime, encoder, targets)
		);
	return { request, result };
}

async function executeColorFilterPass(
	runtime,
	encoder,
	targets,
	frameContext,
	pass = new ColorFilterPass({ enabled: true })
) {
	const request = createColorFilterPassRequest(frameContext, pass);
	const result = await request.pass
		.getImplementation("webgpu")
		.execute(
			request,
			createWebGPUImplementationContext(runtime, encoder, targets)
		);
	return { request, result };
}

async function executeInteractionOutlinePass(
	runtime,
	encoder,
	targets,
	frameContext,
	pass = new InteractionOutlinePass({ enabled: true })
) {
	const request = createInteractionOutlinePassRequest(frameContext, pass);
	const result = await request.pass
		.getImplementation("webgpu")
		.execute(
			request,
			createWebGPUImplementationContext(runtime, encoder, targets)
		);
	return { request, result };
}

async function testFXAAPassImplementationUsesDedicatedPipeline() {
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
	const frameContext = createFrameContext();
	const pass = new FastApproximateAntiAliasingPass({ enabled: true });

	const result = await executeFXAAPass(
		runtime,
		encoder,
		targets,
		frameContext,
		pass
	);
	assert.deepEqual(result, { ran: true });

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

	pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testToneMappingPassImplementationUsesDedicatedPipeline() {
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
	const frameContext = createFrameContext();
	const pass = new ToneMappingPass({ enabled: true });

	const { request, result } = await executeToneMappingPass(
		runtime,
		encoder,
		targets,
		frameContext,
		pass
	);
	assert.deepEqual(result, { ran: true });

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

	request.pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testBloomPassImplementationUsesDedicatedPipeline() {
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
	const frameContext = createFrameContext({
		bloom: {
			enabled: true,
			options: {
				threshold: 1.2,
				softKnee: 0.35,
				intensity: 1.5,
				radius: 2,
			},
		},
	});

	const { request, result } = await executeBloomPass(
		runtime,
		encoder,
		targets,
		frameContext
	);
	assert.deepEqual(result, { ran: true });

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

	request.pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testMotionBlurPassImplementationUsesDedicatedPipeline() {
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

	const { request, result } = await executeMotionBlurPass(
		runtime,
		encoder,
		targets,
		frameContext
	);
	assert.deepEqual(result, { ran: true });

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

	request.pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testFogPassImplementationUsesDedicatedPipeline() {
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
	const frameContext = createFrameContext({
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
	});

	const { request, result } = await executeFogPass(
		runtime,
		encoder,
		targets,
		frameContext
	);
	assert.deepEqual(result, { ran: true });

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

	request.pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testDOFPassImplementationUsesDedicatedPipeline() {
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
	const frameContext = createFrameContext({
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
	});

	const { request, result } = await executeDepthOfFieldPass(
		runtime,
		encoder,
		targets,
		frameContext
	);
	assert.deepEqual(result, { ran: true });

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

	request.pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testColorFilterPassImplementationUsesDedicatedPipeline() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(30, 14, "scene");
	const postPing = createTexture(30, 14, "ping");
	const postPong = createTexture(30, 14, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};
	const frameContext = createFrameContext({
		"color-filter": {
			enabled: true,
			options: {
				brightness: 0.2,
				saturation: 1.4,
				contrast: 0.75,
				temperature: -0.25,
				tint: 0.35,
			},
		},
	});

	const { request, result } = await executeColorFilterPass(
		runtime,
		encoder,
		targets,
		frameContext
	);
	assert.deepEqual(result, { ran: true });

	assert.equal(backend.samplers.length, 1);
	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.shaderModules[0].label, "WebGPUColorFilterShader");
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.computePipelines[0].label, "WebGPUColorFilterPipeline");
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.buffers[0].desc.label, "WebGPUColorFilterParams");
	assert.equal(backend.buffers[0].desc.size, 32);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroups[0].desc.entries.length, 4);
	assert.equal(backend.bindingGroups[0].desc.entries[0].resource, sceneColorMain);
	assert.equal(backend.bindingGroups[0].desc.entries[3].resource, postPong);

	const params = backend.buffers[0].lastWrite;
	assert.equal(params.length, 8);
	assertClose(params[0], 0.2);
	assertClose(params[1], 1.4);
	assertClose(params[2], 0.75);
	assertClose(params[3], -0.25);
	assertClose(params[4], 0.35);

	assert.deepEqual(encoder.calls, [
		["beginComputePass", "WebGPUColorFilter"],
		["setComputePipeline", "WebGPUColorFilterPipeline"],
		["setBindingGroup", 0, "WebGPUColorFilter_Binding"],
		["dispatchWorkgroups", 4, 2, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPong);

	request.pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testInteractionOutlinePassImplementationUsesDedicatedPipeline() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(64, 32, "scene");
	const postPing = createTexture(64, 32, "ping");
	const postPong = createTexture(64, 32, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};
	const frameContext = createFrameContext({
		"interaction-outline": { enabled: true },
	});
	frameContext.attachments = { width: 64, height: 32 };
	frameContext.camera = {
		type: "perspective",
		fov: 60,
		viewMatrix: Matrix4.identity(),
		viewProjectionMatrix: Matrix4.identity(),
	};
	frameContext.scene = {
		meshInstances: [
			{
				entityId: 7,
				visible: true,
				getWorldBoundingSphere() {
					return {
						center: { x: 0, y: 0, z: -5 },
						radius: 1,
					};
				},
			},
		],
	};
	frameContext.transient.set(INTERACTION_TRANSIENT_STATE_KEY, {
		selectedEntityIds: [7],
		outline: {
			color: { r: 128, g: 64, b: 255, a: 0.5 },
			opacity: 0.8,
			thickness: 3,
			shape: "diamond",
		},
	});

	const { request, result } = await executeInteractionOutlinePass(
		runtime,
		encoder,
		targets,
		frameContext
	);
	assert.deepEqual(result, { ran: true });

	assert.equal(backend.samplers.length, 1);
	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.shaderModules[0].label, "WebGPUInteractionOutlineShader");
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(
		backend.computePipelines[0].label,
		"WebGPUInteractionOutlinePipeline"
	);
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.buffers[0].desc.label, "WebGPUInteractionOutlineParams");
	assert.equal(backend.buffers[0].desc.size, 1088);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroups[0].desc.entries.length, 3);
	assert.equal(backend.bindingGroups[0].desc.entries[0].resource, sceneColorMain);
	assert.equal(backend.bindingGroups[0].desc.entries[2].resource, postPong);

	const params = backend.buffers[0].lastWrite;
	assert.equal(params.length, 272);
	assertClose(params[0], 1 / 64);
	assertClose(params[1], 1 / 32);
	assertClose(params[2], 0.4);
	assertClose(params[3], 3);
	assertClose(params[8], 1);
	assertClose(params[9], 2);

	assert.deepEqual(encoder.calls, [
		["beginComputePass", "WebGPUInteractionOutline"],
		["setComputePipeline", "WebGPUInteractionOutlinePipeline"],
		["setBindingGroup", 0, "WebGPUInteractionOutline_Binding"],
		["dispatchWorkgroups", 8, 4, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPong);

	request.pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
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
	const pass = new MotionBlurPass({ enabled: true });

	await executeMotionBlurPass(
		runtime,
		new FakeEncoder(),
		targets,
		frameContext,
		pass
	);
	await executeMotionBlurPass(
		runtime,
		new FakeEncoder(),
		targets,
		frameContext,
		pass
	);

	assert.equal(backend.writeBufferCalls, 1);

	pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testFXAAPassImplementationPingPongsAndCachesResources() {
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
	const pass = new FastApproximateAntiAliasingPass({ enabled: true });

	await executeFXAAPass(runtime, firstEncoder, targets, frameContext, pass);
	await executeFXAAPass(runtime, secondEncoder, targets, frameContext, pass);

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

	pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
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
	const frameContext = createFrameContext();
	const pass = new FastApproximateAntiAliasingPass({ enabled: true });

	await executeFXAAPass(
		runtime,
		new FakeEncoder(),
		targets,
		frameContext,
		pass
	);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroupDestroyCalls, 0);

	runtime.invalidateBindings();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
	runtime.invalidateBindings();
	assert.equal(backend.bindingGroupDestroyCalls, 1);

	pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
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
	const pass = new FastApproximateAntiAliasingPass({ enabled: true });

	await executeFXAAPass(runtime, new FakeEncoder(), targets, frameContext, pass);
	await executeFXAAPass(runtime, new FakeEncoder(), targets, frameContext, pass);
	await executeFXAAPass(runtime, new FakeEncoder(), targets, frameContext, pass);

	assert.equal(backend.bindingGroups.length, 3);
	assert.equal(backend.bindingGroupDestroyCalls, 1);

	pass.destroy();
	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testDestroyReleasesToneMappingImplementationResources() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const sceneColorMain = createTexture(32, 18, "scene");
	const postPing = createTexture(32, 18, "ping");
	const postPong = createTexture(32, 18, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};
	const frameContext = createFrameContext();
	const pass = new ToneMappingPass({ enabled: true });
	const implementation = pass.getImplementation("webgpu");

	await executeToneMappingPass(
		runtime,
		new FakeEncoder(),
		targets,
		frameContext,
		pass
	);
	assert.equal(backend.shaderModuleDestroyCalls, 0);
	assert.equal(backend.computePipelineDestroyCalls, 0);
	assert.equal(backend.samplerDestroyCalls, 0);

	implementation.destroy();
	assert.equal(backend.shaderModuleDestroyCalls, 1);
	assert.equal(backend.computePipelineDestroyCalls, 1);
	assert.equal(backend.samplerDestroyCalls, 0);
	assert.equal(backend.bufferDestroyCalls, 0);
	assert.equal(backend.bindingGroupDestroyCalls, 1);
	assert.deepEqual(
		backend.shaderModules
			.filter((module) => module.destroyed)
			.map((module) => module.label)
			.sort(),
		["WebGPUToneMappingShader"]
	);
	assert.deepEqual(
		backend.computePipelines
			.filter((pipeline) => pipeline.destroyed)
			.map((pipeline) => pipeline.label)
			.sort(),
		["WebGPUToneMappingPipeline"]
	);

	implementation.destroy();
	assert.equal(backend.shaderModuleDestroyCalls, 1);
	assert.equal(backend.computePipelineDestroyCalls, 1);
	assert.equal(backend.samplerDestroyCalls, 0);
	assert.equal(backend.bufferDestroyCalls, 0);
	assert.equal(backend.bindingGroupDestroyCalls, 1);

	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testBloomImplementationLifecycleReleasesOwnedResources() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const targets = {
		sceneColor: createTexture(40, 20, "scene"),
		postPing: createTexture(40, 20, "ping"),
		postPong: createTexture(40, 20, "pong"),
	};
	const request = createBloomPassRequest(
		createFrameContext({ bloom: { enabled: true } })
	);
	const implementation = request.pass.getImplementation("webgpu");

	await implementation.execute(request, {
		encoder,
		targets,
		shared: runtime.sharedContext,
		publishColorTarget: (texture) => {
			targets.sceneColor = texture;
		},
	});
	assert.equal(backend.textureDestroyCalls, 0);

	implementation.invalidate();
	assert.equal(backend.textureDestroyCalls, 10);
	implementation.invalidate();
	assert.equal(backend.textureDestroyCalls, 10);

	implementation.destroy();
	assert.equal(backend.shaderModuleDestroyCalls, 5);
	assert.equal(backend.computePipelineDestroyCalls, 5);
	assert.equal(backend.bufferDestroyCalls, 4);
	implementation.destroy();
	assert.equal(backend.shaderModuleDestroyCalls, 5);
	assert.equal(backend.computePipelineDestroyCalls, 5);
	assert.equal(backend.bufferDestroyCalls, 4);

	destroySnapshotPasses(request.frameContext.postProcess);
	runtime.destroy();
}

export async function run() {
	await testFXAAPassImplementationUsesDedicatedPipeline();
	await testToneMappingPassImplementationUsesDedicatedPipeline();
	await testBloomPassImplementationUsesDedicatedPipeline();
	await testFogPassImplementationUsesDedicatedPipeline();
	await testMotionBlurPassImplementationUsesDedicatedPipeline();
	await testMotionBlurSkipsRedundantParamUploads();
	await testDOFPassImplementationUsesDedicatedPipeline();
	await testColorFilterPassImplementationUsesDedicatedPipeline();
	await testInteractionOutlinePassImplementationUsesDedicatedPipeline();
	await testFXAAPassImplementationPingPongsAndCachesResources();
	await testInvalidateBindingsDestroysCachedBindingGroups();
	await testBindingReplacementDestroysStaleBindingGroup();
	await testDestroyReleasesToneMappingImplementationResources();
	await testBloomImplementationLifecycleReleasesOwnedResources();
	console.log("WebGPU postprocess screen runtime tests passed");
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
	await run();
}
