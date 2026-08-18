import assert from "node:assert/strict";

import * as frameExecutorFixture from "../../helpers/webgpu_frame_executor_resilience.mjs";
import { WebGPUColorDirtyClearPass } from "../../../src/backends/webgpu/rendergraph/WebGPUColorDirtyClearPass.ts";

const {
	FakeBackend,
	Logger,
	PBRMaterial,
	WebGPUFrameExecutor,
	createDeferredLightingResourcesStub,
	createFrameContext,
	createModeTrackingResourcesStub,
	createOITBackend,
	createOITSequencingResourcesStub,
	findEncoderCallIndex,
	getFrameGraphDebugState,
	getFrameTargets,
	initializeIsolatedWebGPUTestState,
} = frameExecutorFixture;

const restoreTestState = initializeIsolatedWebGPUTestState();

async function testOITTransparentAndParticleExecutionOrder() {
	const backend = createOITBackend();
	const resources = createOITSequencingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.features.enableOIT = true;
	context.scene.transparentPackets = [
		{
			id: "transparent-oit",
			material: { transmissionFactor: 0 },
		},
		{
			id: "transparent-transmission",
			material: { transmissionFactor: 1 },
		},
	];
	context.scene.particleSystems = [{ id: "ps-0" }];

	await executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-transparent", executor: "backend", enabled: true },
		context
	);
	await executor.executePass(
		{ stage: "particles", executor: "backend", enabled: true },
		context
	);

	const labels = backend.recordedRenderPasses.map((pass) => pass.label);
	assert.deepEqual(labels, [
		"WebGPUOITClear",
		"WebGPUOITMeshAccumulate",
		"WebGPUParticlesOIT",
		"WebGPUOITResolvePass",
		"WebGPUTransmissionMRT",
		"WebGPUParticlesMRT_Additive",
	]);
	assert.ok(
		resources._state.events.includes("draw:transparent-oit:oit:default")
	);
	assert.ok(
		resources._state.events.includes(
			"draw:transparent-transmission:transmission:default"
		)
	);
	assert.ok(resources._state.events.includes("prepare:mrt"));
	assert.deepEqual(
		resources._state.drawOptions.filter((entry) =>
			entry.packetId.startsWith("transparent-")
		),
		[
			{
				packetId: "transparent-oit",
				sceneTargetMode: "mrt",
				transparentPipelineMode: "oit",
				drawMode: "default",
			},
			{
				packetId: "transparent-transmission",
				sceneTargetMode: "mrt",
				transparentPipelineMode: "transmission",
				drawMode: "default",
			},
		]
	);
	assert.ok(
		resources._state.events.some((event) =>
			event.startsWith("particles:WebGPUParticlesOIT:oit:")
		)
	);
	assert.ok(
		resources._state.events.some((event) =>
			event.startsWith("particles:WebGPUParticlesMRT_Additive:legacy:")
		)
	);
	const oitDrawIndex = findEncoderCallIndex(
		backend,
		(call) => call[0] === "beginRenderPass" && call[1]?.label === "WebGPUOITMeshAccumulate"
	);
	const copyIndex = findEncoderCallIndex(
		backend,
		(call) => call[0] === "copyTextureToTexture"
	);
	const resolveIndex = findEncoderCallIndex(
		backend,
		(call) =>
			call[0] === "beginRenderPass" &&
			call[1]?.label === "WebGPUOITResolvePass"
	);
	assert.ok(oitDrawIndex >= 0);
	assert.ok(copyIndex >= 0);
	assert.ok(copyIndex < oitDrawIndex);
	assert.ok(resolveIndex > oitDrawIndex);
	assert.equal(backend.encoderCopyCalls.length >= 1, true);
}

async function testOITTransparentResolvesImmediatelyWithoutParticles() {
	const backend = createOITBackend();
	const resources = createOITSequencingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.features.enableOIT = true;
	context.scene.transparentPackets = [
		{
			id: "transparent-oit-only",
			material: { transmissionFactor: 0 },
		},
		{
			id: "transparent-transmission-only",
			material: { transmissionFactor: 1 },
		},
	];
	context.scene.particleSystems = [];

	await executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-transparent", executor: "backend", enabled: true },
		context
	);

	const labels = backend.recordedRenderPasses.map((pass) => pass.label);
	assert.deepEqual(labels, [
		"WebGPUOITClear",
		"WebGPUOITMeshAccumulate",
		"WebGPUOITResolvePass",
		"WebGPUTransmissionMRT",
	]);
	assert.ok(
		resources._state.events.includes(
			"draw:transparent-transmission-only:transmission:default"
		)
	);
	assert.ok(resources._state.events.includes("prepare:mrt"));
	assert.deepEqual(
		resources._state.drawOptions.filter((entry) =>
			entry.packetId.startsWith("transparent-")
		),
		[
			{
				packetId: "transparent-oit-only",
				sceneTargetMode: "mrt",
				transparentPipelineMode: "oit",
				drawMode: "default",
			},
			{
				packetId: "transparent-transmission-only",
				sceneTargetMode: "mrt",
				transparentPipelineMode: "transmission",
				drawMode: "default",
			},
		]
	);
	assert.equal(
		resources._state.events.some((event) =>
			event.startsWith("particles:WebGPUParticlesOIT")
		),
		false
	);
	const oitDrawIndex = findEncoderCallIndex(
		backend,
		(call) => call[0] === "beginRenderPass" && call[1]?.label === "WebGPUOITMeshAccumulate"
	);
	const copyIndex = findEncoderCallIndex(
		backend,
		(call) => call[0] === "copyTextureToTexture"
	);
	const resolveIndex = findEncoderCallIndex(
		backend,
		(call) =>
			call[0] === "beginRenderPass" &&
			call[1]?.label === "WebGPUOITResolvePass"
	);
	assert.ok(oitDrawIndex >= 0);
	assert.ok(copyIndex >= 0);
	assert.ok(copyIndex < oitDrawIndex);
	assert.ok(resolveIndex > oitDrawIndex);
	assert.equal(backend.encoderCopyCalls.length >= 1, true);
}

async function testDeferredLightingBindsUnusedGroupOnePlaceholder() {
	const backend = new FakeBackend();
	const resources = createDeferredLightingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [
		{
			id: "deferred-anisotropic",
			material: new PBRMaterial({ anisotropyStrength: 0.8 }),
		},
	];

	await executor.beginFrame(context);
	const bridge = executor.frameRuntime.postProcess.createGBufferBridge(context);
	assert.equal(bridge.channels.specular.format, "rgba16float");
	assert.equal(bridge.channels.specular.encoding, "specular-color-factor.rgba");
	assert.equal(
		bridge.channels.specular.handle.texture,
		getFrameTargets(executor).gSpecular
	);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	const frameEncoder = backend.commandEncoders[0];
	assert.ok(frameEncoder);
	const deferredPassIndex = frameEncoder.calls.findIndex(
		(call) =>
			call[0] === "beginRenderPass" &&
			call[1].label === "WebGPUDeferredLighting"
	);
	assert.notEqual(deferredPassIndex, -1);
	const drawIndex = frameEncoder.calls.findIndex(
		(call, index) => index > deferredPassIndex && call[0] === "draw"
	);
	assert.notEqual(drawIndex, -1);
	const groupOneBinding = frameEncoder.calls.find(
		(call, index) =>
			index > deferredPassIndex &&
			index < drawIndex &&
			call[0] === "setBindGroup" &&
			call[1] === 1
	);
	assert.ok(groupOneBinding);
	assert.equal(groupOneBinding[2], resources._state.deferredUnusedBinding);
	assert.ok(
		resources._state.events.includes(
			"draw:deferred-anisotropic:gbuffer:early-z-prepass"
		)
	);
	assert.ok(
		resources._state.events.includes(
			"draw:deferred-anisotropic:gbuffer:early-z-color"
		)
	);
	assert.deepEqual(
		resources._state.events.filter((event) =>
			event.startsWith("draw:deferred-anisotropic:")
		),
		[
			"draw:deferred-anisotropic:gbuffer:early-z-prepass",
			"draw:deferred-anisotropic:gbuffer:early-z-color",
		]
	);
}

async function testDeferredIncrementalClearsDirtyColorAndGBufferAttachments() {
	const backend = new FakeBackend();
	const resources = createDeferredLightingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [
		{
			id: "deferred-animated",
			material: new PBRMaterial({ anisotropyStrength: 0.8 }),
		},
	];
	context.incremental = {
		enabled: true,
		forceFullFrame: false,
		dirtyRects: [{ x: 8, y: 12, width: 20, height: 24 }],
		dirtyTileSize: 16,
		dirtyTileColumns: 4,
		dirtyTileRows: 4,
		dirtyTiles: [0],
		dirtyAreaRatio: 0.125,
		firstPass: "main-opaque",
		reasonMask: 0,
		temporalHistoryReset: false,
	};

	await executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context,
	);

	const colorClearIndex = backend.recordedRenderPasses.findIndex(
		(pass) => pass.label === "WebGPUColorDirtyClear",
	);
	const gbufferIndex = backend.recordedRenderPasses.findIndex(
		(pass) => pass.label === "WebGPUGBuffer_Load",
	);
	const lightingIndex = backend.recordedRenderPasses.findIndex(
		(pass) => pass.label === "WebGPUDeferredLighting",
	);
	assert.ok(colorClearIndex >= 0);
	assert.ok(colorClearIndex < gbufferIndex);
	assert.ok(gbufferIndex < lightingIndex);
	const colorClearPass = backend.recordedRenderPasses[colorClearIndex];
	assert.equal(colorClearPass.colorAttachments.length, 8);
	assert.equal(
		colorClearPass.colorAttachments.every(
			(attachment) => attachment.loadOp === "load",
		),
		true,
	);
	const frameEncoder = backend.commandEncoders[0];
	assert.ok(
		frameEncoder.calls.some(
			(call) =>
				call[0] === "setScissorRect" &&
				call[1] === 8 &&
				call[2] === 12 &&
				call[3] === 20 &&
				call[4] === 24,
		),
	);
}

async function testDeferredFallbackPreservesIncrementalState() {
	const backend = new FakeBackend();
	const resources = createDeferredLightingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [
		{
			id: "deferred-extended",
			material: new PBRMaterial({ anisotropyStrength: 0.8 }),
		},
		{
			id: "forward-fallback",
			material: new PBRMaterial({ wireframe: true }),
		},
	];
	context.incremental = {
		enabled: true,
		forceFullFrame: false,
		dirtyRects: [{ x: 8, y: 12, width: 20, height: 24 }],
		dirtyTileSize: 16,
		dirtyTileColumns: 4,
		dirtyTileRows: 4,
		dirtyTiles: [0],
		dirtyAreaRatio: 0.125,
		firstPass: "main-opaque",
		reasonMask: 0,
		temporalHistoryReset: false,
	};

	await executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context,
	);

	const colorClearPasses = backend.recordedRenderPasses.filter(
		(pass) => pass.label === "WebGPUColorDirtyClear",
	);
	assert.equal(colorClearPasses.length, 1);
	const fallbackPass = backend.recordedRenderPasses.find(
		(pass) => pass.label === "WebGPUMainMRT_Load",
	);
	assert.ok(fallbackPass);
	assert.equal(fallbackPass.depthStencilAttachment.depthLoadOp, "load");
}

async function testDeferredBaseClearSelectsCanonicalPipeline() {
	const backend = new FakeBackend();
	const clearPass = new WebGPUColorDirtyClearPass(backend);
	const view = backend.getCanvasColorTexture();
	await clearPass.record(
		backend.createCommandEncoder(),
		"deferred",
		[
			{ view, format: "rgba16float" },
			{ view, format: "rgba8unorm" },
			{ view, format: "rgba8unorm" },
			{ view, format: "rgba16float" },
			{ view, format: "rgba16float" },
		],
		1,
		[{ x: 0, y: 0, width: 8, height: 8 }],
	);

	const pipeline = backend.renderPipelines.find((candidate) =>
		candidate.label.startsWith("WebGPUColorDirtyClearPipeline_deferred|"),
	);
	assert.ok(pipeline);
	assert.equal(pipeline.desc.fragment.entryPoint, "fsDeferred");
	clearPass.destroy();
}

async function testDeferredLightingKeepsTransmissionOutOfGBuffer() {
	const backend = new FakeBackend();
	const resources = createDeferredLightingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [
		{
			id: "opaque-transmission",
			material: new PBRMaterial({ transmissionFactor: 1 }),
		},
	];

	await executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	const frameEncoder = backend.commandEncoders[0];
	assert.ok(frameEncoder);
	assert.equal(
		frameEncoder.calls.some(
			(call) =>
				call[0] === "beginRenderPass" &&
				call[1].label === "WebGPUDeferredLighting"
		),
		false
	);
	assert.ok(
		resources._state.events.includes(
			"draw:opaque-transmission:mrt:early-z-prepass"
		)
	);
	assert.ok(
		resources._state.events.includes(
			"draw:opaque-transmission:mrt:early-z-color"
		)
	);
}

async function testDeferredLightingCanBeExplicitlyDisabled() {
	const backend = new FakeBackend();
	backend.enableDeferredLighting = false;
	const resources = createDeferredLightingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [
		{
			id: "deferred-disabled",
			material: new PBRMaterial({ anisotropyStrength: 0.8 }),
		},
	];

	await executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	const frameEncoder = backend.commandEncoders[0];
	assert.ok(frameEncoder);
	assert.equal(getFrameGraphDebugState(executor).targetManager.sceneTargetMode, "mrt");
	assert.equal(
		frameEncoder.calls.some(
			(call) =>
				call[0] === "beginRenderPass" &&
				call[1].label === "WebGPUDeferredLighting"
		),
		false
	);
	assert.ok(
		resources._state.events.includes(
			"draw:deferred-disabled:mrt:early-z-prepass"
		)
	);
	assert.ok(
		resources._state.events.includes(
			"draw:deferred-disabled:mrt:early-z-color"
		)
	);
}

async function testDeferredPipelineFailureFallsBackBeforeGBufferCommands() {
	const backend = new FakeBackend();
	const resources = createDeferredLightingResourcesStub();
	resources.getDeferredLightingPipeline = async () => {
		throw new Error("simulated deferred pipeline failure");
	};
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [
		{
			id: "deferred-pipeline-fallback",
			material: new PBRMaterial({ anisotropyStrength: 0.8 }),
		},
	];
	const warnings = [];
	Logger.configure({
		level: "warn",
		resetOnceKeys: true,
		sink: {
			warn: (...args) =>
				warnings.push(args.map((arg) => String(arg)).join(" ")),
		},
	});
	try {
		await executor.beginFrame(context);
		await executor.executePass(
			{ stage: "main-opaque", executor: "backend", enabled: true },
			context
		);
		const labels = backend.recordedRenderPasses.map((pass) => pass.label);
		assert.equal(labels.some((label) => label.startsWith("WebGPUGBuffer")), false);
		assert.ok(labels.some((label) => label.startsWith("WebGPUMainMRT")));
		assert.equal(
			warnings.filter((warning) =>
				warning.includes("[webgpu-deferred-runtime-fallback]")
			).length,
			1
		);
		assert.ok(
			resources._state.events.includes(
				"draw:deferred-pipeline-fallback:mrt:early-z-color"
			)
		);
	} finally {
		Logger.reset();
		executor.destroy();
	}
}

async function testDeferredLightingWarnsWhenRequestedButMRTUnavailable() {
	const backend = new FakeBackend();
	backend.device.limits.maxColorAttachments = 1;
	backend.device.limits.maxColorAttachmentBytesPerSample = 16;
	const resources = createModeTrackingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	const warnings = [];

	Logger.configure({
		level: "warn",
		resetOnceKeys: true,
		sink: {
			warn: (...args) =>
				warnings.push(args.map((arg) => String(arg)).join(" ")),
		},
	});
	try {
		await executor.beginFrame(context);
		assert.equal(getFrameGraphDebugState(executor).targetManager.sceneTargetMode, "single");
		assert.equal(
			warnings.some((warning) =>
				warning.includes("[webgpu-deferred-disabled-attachments]")
			),
			true
		);
	} finally {
		Logger.reset();
	}
}

async function testOITMSAAFallsBackToLegacyAndWarns() {
	const backend = createOITBackend({ sampleCount: 4 });
	const resources = createModeTrackingResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources, backend.msaaContext);
	const context = createFrameContext(64, 64);
	context.features.enableOIT = true;
	context.scene.transparentPackets = [{ id: "packet", material: {} }];
	const warnings = [];

	Logger.configure({
		level: "warn",
		resetOnceKeys: true,
		sink: {
			warn: (...args) =>
				warnings.push(args.map((arg) => String(arg)).join(" ")),
		},
	});
	try {
		await executor.beginFrame(context);
		assert.equal(getFrameGraphDebugState(executor).oitActive, false);
		await executor.executePass(
			{ stage: "main-transparent", executor: "backend", enabled: true },
			context
		);
		const oitMSAAWarnings = warnings.filter((warning) =>
			warning.includes("[webgpu-oit-disabled-msaa]")
		);
		assert.equal(oitMSAAWarnings.length, 1);
		assert.notEqual(resources._state.drawModeAtRequest, null);
	} finally {
		Logger.reset();
	}
}

async function testOITRuntimeFallbackWarnsWithoutEncoderCopy() {
	const backend = new FakeBackend();
	const originalCreateCommandEncoder =
		backend.createCommandEncoder.bind(backend);
	backend.createCommandEncoder = () => {
		const encoder = originalCreateCommandEncoder();
		encoder.copyTextureToTexture = undefined;
		return encoder;
	};
	const resources = createModeTrackingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.features.enableOIT = true;
	context.scene.transparentPackets = [{ id: "transparent", material: {} }];
	const warnings = [];

	Logger.configure({
		level: "warn",
		resetOnceKeys: true,
		sink: {
			warn: (...args) =>
				warnings.push(args.map((arg) => String(arg)).join(" ")),
		},
	});
	try {
		await executor.beginFrame(context);
		assert.equal(getFrameGraphDebugState(executor).oitActive, false);
		const runtimeWarnings = warnings.filter((warning) =>
			warning.includes("[webgpu-oit-disabled-runtime]")
		);
		assert.equal(runtimeWarnings.length, 1);
	} finally {
		Logger.reset();
	}
}

async function run() {
	try {
		await testOITTransparentAndParticleExecutionOrder();
		await testOITTransparentResolvesImmediatelyWithoutParticles();
		await testDeferredLightingBindsUnusedGroupOnePlaceholder();
		await testDeferredIncrementalClearsDirtyColorAndGBufferAttachments();
		await testDeferredFallbackPreservesIncrementalState();
		await testDeferredBaseClearSelectsCanonicalPipeline();
		await testDeferredLightingKeepsTransmissionOutOfGBuffer();
		await testDeferredLightingCanBeExplicitlyDisabled();
		await testDeferredPipelineFailureFallsBackBeforeGBufferCommands();
		await testDeferredLightingWarnsWhenRequestedButMRTUnavailable();
		await testOITMSAAFallsBackToLegacyAndWarns();
		await testOITRuntimeFallbackWarnsWithoutEncoderCopy();
		console.log("WebGPU frame-executor transparency/deferred tests passed");
	} finally {
		restoreTestState();
	}
}

await run();
