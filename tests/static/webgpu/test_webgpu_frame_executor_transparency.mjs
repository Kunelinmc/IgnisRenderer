import assert from "node:assert/strict";

import * as frameExecutorFixture from "../../helpers/webgpu_frame_executor_resilience.mjs";

const {
	FakeBackend,
	Logger,
	PBRMaterial,
	WebGPUFrameExecutor,
	createFrameContext,
	createResolvedPostProcess,
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

async function testForwardMaterialsUseRequestedSceneTargets() {
	for (const mode of ["single", "color", "mrt"]) {
		const backend = new FakeBackend();
		const resources = createOITSequencingResourcesStub();
		const executor = new WebGPUFrameExecutor(backend, resources);
		const context = createFrameContext(64, 64);
		context.postProcess = createResolvedPostProcess(mode === "single" ? {}
			: mode === "color" ? { bloom: { enabled: true } }
			: { ssao: { enabled: true } }, "webgpu");
		context.scene.opaquePackets = [
			{ id: "anisotropic", material: new PBRMaterial({ anisotropyStrength: 0.8 }) },
			{ id: "transmissive", material: new PBRMaterial({ transmissionFactor: 1 }) },
		];
		try {
			await executor.beginFrame(context);
			assert.equal(getFrameGraphDebugState(executor).targetManager.sceneTargetMode, mode);
			await executor.executePass(
				{ stage: "main-opaque", executor: "backend", enabled: true }, context,
			);
			for (const id of ["anisotropic", "transmissive"]) {
				assert.deepEqual(resources._state.drawOptions.filter((entry) =>
					entry.packetId === id), ["early-z-prepass", "early-z-color"].map((drawMode) => ({
					packetId: id,
					sceneTargetMode: mode,
					transparentPipelineMode: "default",
					drawMode,
				})));
			}
			if (mode === "single") assert.equal(getFrameTargets(executor), null);
			else assert.ok(getFrameTargets(executor).sceneColorMain);
		} finally {
			executor.destroy();
		}
	}
}

async function testForwardMRTIncrementalClearsAllRasterAttachments() {
	const backend = new FakeBackend();
	const resources = createOITSequencingResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [{ id: "animated", material: new PBRMaterial() }];
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
	try {
		await executor.beginFrame(context);
		await executor.executePass(
			{ stage: "main-opaque", executor: "backend", enabled: true }, context,
		);
		const passes = backend.recordedRenderPasses;
		const clearIndex = passes.findIndex((pass) => pass.label === "WebGPUColorDirtyClear");
		const depthIndex = passes.findIndex((pass) => pass.label === "WebGPUDepthDirtyClear");
		const earlyIndex = passes.findIndex((pass) => pass.label === "WebGPUEarlyZPrepassMRT");
		const sceneIndex = passes.findIndex((pass) => pass.label === "WebGPUMainMRT_Load");
		assert.ok(clearIndex >= 0 && clearIndex < depthIndex);
		assert.ok(depthIndex < earlyIndex && earlyIndex < sceneIndex);
		const targets = getFrameTargets(executor);
		assert.deepEqual(passes[clearIndex].colorAttachments.map((attachment) => attachment.view), [
			targets.sceneColorMain, targets.gAlbedoAlpha, targets.gNormalRoughMetal,
			targets.gEmissiveOcclusion, targets.gMotionDepth,
		]);
		assert.ok(passes[clearIndex].colorAttachments.every((attachment) =>
			attachment.loadOp === "load"));
		assert.equal(passes[sceneIndex].depthStencilAttachment.depthLoadOp, "load");
		assert.ok(backend.commandEncoders[0].calls.some((call) =>
			call[0] === "setScissorRect" && call.slice(1).join(",") === "8,12,20,24"));
	} finally {
		executor.destroy();
	}
}

async function run() {
	try {
		await testForwardMaterialsUseRequestedSceneTargets();
		await testForwardMRTIncrementalClearsAllRasterAttachments();
		await testOITTransparentAndParticleExecutionOrder();
		await testOITTransparentResolvesImmediatelyWithoutParticles();
		await testOITMSAAFallsBackToLegacyAndWarns();
		await testOITRuntimeFallbackWarnsWithoutEncoderCopy();
		console.log("WebGPU frame-executor transparency tests passed");
	} finally {
		restoreTestState();
	}
}

await run();
