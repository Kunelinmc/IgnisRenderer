import assert from "node:assert/strict";

import * as frameExecutorFixture from "../../helpers/webgpu_frame_executor_resilience.mjs";

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

	executor.beginFrame(context);
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

	executor.beginFrame(context);
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

	executor.beginFrame(context);
	const bridge = executor.createGBufferBridge(context);
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

	executor.beginFrame(context);
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

	executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	const frameEncoder = backend.commandEncoders[0];
	assert.ok(frameEncoder);
	assert.equal(executor.getSceneTargetModeForFrame(), "mrt");
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
		executor.beginFrame(context);
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

function testDeferredLightingWarnsWhenRequestedButMRTUnavailable() {
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
		executor.beginFrame(context);
		assert.equal(executor.getSceneTargetModeForFrame(), "single");
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
		executor.beginFrame(context);
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

function testOITRuntimeFallbackWarnsWithoutEncoderCopy() {
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
		executor.beginFrame(context);
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
