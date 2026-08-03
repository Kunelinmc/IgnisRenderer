import assert from "node:assert/strict";

import * as frameExecutorFixture from "../../helpers/webgpu_frame_executor_resilience.mjs";

const {
	FakeBackend,
	Logger,
	WebGPUFrameExecutor,
	createFrameContext,
	createMSAAContext,
	createModeTrackingResourcesStub,
	createResolvedPostProcess,
	createResourcesStub,
	getFrameGraphDebugState,
	getFrameTargets,
	getMSAATargets,
	initializeIsolatedWebGPUTestState,
} = frameExecutorFixture;

const restoreTestState = initializeIsolatedWebGPUTestState();

function testFrameTargetAllocationFailureReleasesPartialResources() {
	const backend = new FakeBackend();
	backend.failTextureAtCall = 4;
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	assert.throws(
		() => executor.beginFrame(context),
		/simulated allocation failure/
	);
	assert.equal(getFrameGraphDebugState(executor).texturePoolOwnerCount, 0);
	assert.equal(getFrameTargets(executor), null);
	assert.equal(getMSAATargets(executor), null);
}

async function testMSAAAllocationFallbackPersistsForDeviceRuntime() {
	const backend = new FakeBackend();
	const msaa = createMSAAContext(4);
	const createTexture = backend.createTexture.bind(backend);
	let multisampleAllocationAttempts = 0;
	backend.createTexture = (desc) => {
		if ((desc.sampleCount ?? 1) > 1) {
			multisampleAllocationAttempts++;
			throw new Error("simulated MSAA allocation failure");
		}
		return createTexture(desc);
	};
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub(), msaa);
	const context = createFrameContext(64, 64);
	const warnings = [];

	Logger.configure({
		level: "warn",
		resetOnceKeys: true,
		sink: {
			warn: (...args) => warnings.push(args.map((arg) => String(arg)).join(" ")),
		},
	});
	try {
		executor.beginFrame(context);
		assert.equal(msaa.sampleCount, 1);
		assert.equal(getFrameGraphDebugState(executor).msaaTargets, null);
		assert.equal(
			getFrameGraphDebugState(executor).targetManager.msaaSampleCount,
			1
		);
		assert.equal(multisampleAllocationAttempts, 1);
		assert.equal(
			warnings.filter((warning) => warning.includes("[webgpu-msaa-runtime-fallback-1x]")).length,
			1
		);

		await executor.endFrame();
		executor.beginFrame(context);
		assert.equal(multisampleAllocationAttempts, 1);
		assert.equal(msaa.sampleCount, 1);
	} finally {
		Logger.reset();
		executor.destroy();
	}
}

async function testLegacyMainPassForcesSingleSceneTargetMode() {
	const backend = new FakeBackend();
	const resources = createModeTrackingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [{ id: "packet" }];
	context.postProcess = createResolvedPostProcess({});

	executor.beginFrame(context);

	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	assert.equal(resources._state.environmentModeAtRequest, "single");
	assert.equal(resources._state.drawModeAtRequest, "single");
}

async function testIncrementalMainPassUsesDepthPartialReuse() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [{ id: "packet" }];
	context.incremental = {
		enabled: true,
		forceFullFrame: false,
		dirtyRects: [{
			x: 8,
			y: 8,
			width: 16,
			height: 16,
		}],
		dirtyTileSize: 16,
		dirtyTileColumns: 4,
		dirtyTileRows: 4,
		dirtyTiles: [0],
		dirtyAreaRatio: 0.0625,
		firstPass: "main-opaque",
		reasonMask: 0,
		temporalHistoryReset: false,
	};

	executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	assert.equal(backend.recordedRenderPasses.length >= 3, true);
	const depthClearPass = backend.recordedRenderPasses[0];
	assert.equal(depthClearPass.depthStencilAttachment.depthLoadOp, "load");
	assert.equal(depthClearPass.colorAttachments.length, 0);
	const earlyZPass = backend.recordedRenderPasses[1];
	assert.equal(earlyZPass.label, "WebGPUEarlyZPrepassMRT");
	assert.equal(earlyZPass.colorAttachments.length, 0);
	assert.equal(earlyZPass.depthStencilAttachment.depthLoadOp, "load");
	const mainPass = backend.recordedRenderPasses[2];
	assert.equal(mainPass.depthStencilAttachment.depthLoadOp, "load");
}

async function testMainOpaqueDisablesEarlyZWhenConfiguredOff() {
	const backend = new FakeBackend();
	backend.enableEarlyZPrepass = false;
	const resources = createModeTrackingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [{ id: "packet" }];

	executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	const labels = backend.recordedRenderPasses.map((pass) => pass.label);
	assert.equal(labels.includes("WebGPUEarlyZPrepassMRT"), false);
	assert.equal(resources._state.drawPipelineModeAtRequest, "default");
}

async function testLegacyMainPassScalesDirtyRectsToCanvasTarget() {
	const backend = new FakeBackend();
	backend.canvasColorTexture.width = 1223;
	backend.canvasColorTexture.height = 869;
	backend.canvasDepthTexture.width = 1223;
	backend.canvasDepthTexture.height = 869;
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(1920, 869);
	context.scene.opaquePackets = [{ id: "packet" }];
	context.postProcess = createResolvedPostProcess({});
	context.incremental = {
		enabled: true,
		forceFullFrame: false,
		dirtyRects: [{
			x: 960,
			y: 0,
			width: 960,
			height: 869,
		}],
		dirtyTileSize: 32,
		dirtyTileColumns: 60,
		dirtyTileRows: 28,
		dirtyTiles: [0],
		dirtyAreaRatio: 0.5,
		firstPass: "main-opaque",
		reasonMask: 0,
		temporalHistoryReset: false,
	};

	executor.beginFrame(context);

	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	const frameEncoder = backend.commandEncoders[0];
	assert.ok(frameEncoder);
	const scissorCalls = frameEncoder.calls.filter(
		(call) => call[0] === "setScissorRect"
	);
	assert.deepEqual(scissorCalls, [
		["setScissorRect", 611, 0, 612, 869],
		["setScissorRect", 611, 0, 612, 869],
		["setScissorRect", 611, 0, 612, 869],
	]);
}

function testFrameTargetsSkipOptionalTargetsWhenUnused() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	assert.ok(targets);
	assert.equal(executor.getSceneTargetModeForFrame(), "mrt");
	assert.ok(targets.postPing);
	assert.ok(targets.postPong);
	assert.ok(targets.gMotionDepth);
	assert.equal(targets.oitAccum, null);
	assert.equal(targets.oitReveal, null);
	assert.equal(targets.oitSceneColorCopy, null);
	assert.equal(targets.planarReflectionMask, null);

	executor.invalidateFrameTargets();
	assert.equal(
		getFrameGraphDebugState(executor).pendingFrameTargetInvalidation,
		true
	);
	executor.abortFrame();
	assert.equal(getFrameTargets(executor), null);
}

function testGBufferBridgeReportsAllocatedWebGPUFormats() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	executor.beginFrame(context);
	const bridge = executor.createGBufferBridge(context);

	assert.equal(bridge.depthEncoding, "linear-view-z");
	assert.equal(bridge.motionEncoding, "ndc-delta");
	assert.equal(bridge.channels.depth.format, "rgba16float");
	assert.equal(bridge.channels.depth.encoding, "motion-depth.z");
	assert.equal(bridge.channels.motion.format, "rgba16float");
	assert.equal(bridge.channels.motion.encoding, "motion-depth.xy");
	assert.equal(bridge.channels.normal.format, "rgba16float");
	assert.equal(bridge.channels.normal.encoding, "encoded-world-normal");
	assert.equal(bridge.channels.roughness.format, "rgba16float");
	assert.equal(bridge.channels.roughness.encoding, "normal-roughness-metallic.z");
	assert.equal(bridge.channels.metallic.format, "rgba16float");
	assert.equal(bridge.channels.metallic.encoding, "normal-roughness-metallic.w");
	assert.equal(bridge.channels.specular, undefined);
	assert.equal(bridge.channels.albedo.format, "rgba8unorm");
	assert.equal(bridge.channels.albedo.encoding, "linear-rgb-alpha");
	assert.equal(
		bridge.channels.normal.handle.texture,
		getFrameTargets(executor).gNormalRoughMetal
	);
	assert.equal(
		bridge.channels.roughness.handle.texture,
		getFrameTargets(executor).gNormalRoughMetal
	);
	assert.equal(
		bridge.channels.metallic.handle.texture,
		getFrameTargets(executor).gNormalRoughMetal
	);
	assert.equal(
		bridge.channels.albedo.handle.texture,
		getFrameTargets(executor).gAlbedoAlpha
	);
	assert.equal(
		bridge.channels.depth.handle.texture,
		getFrameTargets(executor).gMotionDepth
	);

	executor.abortFrame();
}

function testHiZIsPlannableBeforeItsBuildExecutes() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess({
		ssgi: { enabled: true },
	}, "webgpu");

	executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	const debug = getFrameGraphDebugState(executor);
	const port = executor.createPostProcessSessionPort();

	assert.ok(targets.hiZ);
	assert.equal(debug.hiZ.status, "pending");
	assert.equal(
		port.isGraphResourceAvailable("backend:frame-hiz"),
		true,
	);

	executor.abortFrame();
}

function testFrameTargetsAllocateAndReleaseOptionalTargetsWhenNeeded() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess({});
	context.features.enableOIT = true;
	context.features.enableReflection = true;
	context.scene.transparentPackets = [{ id: "transparent", material: {} }];
	context.scene.reflectivePackets = [{ id: "mirror", material: {} }];

	executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	assert.ok(targets);
	assert.equal(executor.getSceneTargetModeForFrame(), "color");
	assert.equal(targets.postPing, null);
	assert.equal(targets.postPong, null);
	assert.equal(targets.gMotionDepth, null);
	const {
		oitAccum,
		oitReveal,
		oitSceneColorCopy,
		planarReflectionMask,
	} = targets;
	assert.ok(oitAccum);
	assert.ok(oitReveal);
	assert.ok(oitSceneColorCopy);
	assert.ok(planarReflectionMask);
	assert.equal(getFrameGraphDebugState(executor).texturePoolOwnerCount >= 4, true);

	executor.invalidateFrameTargets();
	executor.abortFrame();
	assert.equal(getFrameTargets(executor), null);
	assert.equal(getFrameGraphDebugState(executor).texturePoolOwnerCount, 0);
}

function testFrameTargetsSkippedWhenFrameHasNoOffscreenWork() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess({});

	executor.beginFrame(context);
	assert.equal(getFrameTargets(executor), null);
	assert.equal(executor.getSceneTargetModeForFrame(), "single");
	assert.equal(getFrameGraphDebugState(executor).texturePoolOwnerCount, 0);
}

function testTransmissionTargetsAllocateOnlyWhenRefractionHasWork() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess(
		{ ssrefraction: { enabled: true } },
		"webgpu"
	);
	context.scene.transparentPackets = [
		{ id: "glass", material: { transmissionFactor: 1 } },
	];

	executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	assert.ok(targets.transmissionSceneColorCopy);
	assert.ok(targets.transmissionLighting);
	assert.ok(targets.gTransmissionSurface0);
	assert.ok(targets.gTransmissionSurface1);
	assert.ok(targets.gTransmissionSurface2);
	assert.ok(targets.transmissionDepth);
	const bridge = executor.createGBufferBridge(context);
	assert.equal(
		bridge.channels.transmission.handle.texture,
		targets.gTransmissionSurface0
	);
	executor.abortFrame();

	const noWorkContext = createFrameContext(64, 64);
	noWorkContext.postProcess = createResolvedPostProcess(
		{ ssrefraction: { enabled: true } },
		"webgpu"
	);
	noWorkContext.scene.transparentPackets = [
		{ id: "alpha", material: { transmissionFactor: 0 } },
	];
	executor.beginFrame(noWorkContext);
	assert.equal(getFrameTargets(executor), null);
	executor.abortFrame();
}

async function testFrameTargetReuseIgnoresPostProcessDownsampleOptions() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const firstContext = createFrameContext(64, 64);
	firstContext.postProcess = createResolvedPostProcess({
		ssao: { enabled: true, options: { downsample: 2 } },
		ssr: { enabled: true, options: { downsample: 2 } },
	});
	executor.beginFrame(firstContext);
	const firstTargets = getFrameTargets(executor);
	const firstTextureCount = backend.createTextureCalls.length;
	await executor.endFrame();

	const secondContext = createFrameContext(64, 64);
	secondContext.postProcess = createResolvedPostProcess({
		ssao: { enabled: true, options: { downsample: 4 } },
		ssr: { enabled: true, options: { downsample: 4 } },
	});
	executor.beginFrame(secondContext);
	assert.strictEqual(getFrameTargets(executor), firstTargets);
	assert.equal(backend.createTextureCalls.length, firstTextureCount);
	await executor.endFrame();
	executor.destroy();
}

async function run() {
	try {
		await testFrameTargetAllocationFailureReleasesPartialResources();
		await testMSAAAllocationFallbackPersistsForDeviceRuntime();
		await testLegacyMainPassForcesSingleSceneTargetMode();
		await testIncrementalMainPassUsesDepthPartialReuse();
		await testMainOpaqueDisablesEarlyZWhenConfiguredOff();
		await testLegacyMainPassScalesDirtyRectsToCanvasTarget();
		await testFrameTargetsSkipOptionalTargetsWhenUnused();
		await testGBufferBridgeReportsAllocatedWebGPUFormats();
		await testHiZIsPlannableBeforeItsBuildExecutes();
		await testFrameTargetsAllocateAndReleaseOptionalTargetsWhenNeeded();
		await testFrameTargetsSkippedWhenFrameHasNoOffscreenWork();
		await testTransmissionTargetsAllocateOnlyWhenRefractionHasWork();
		await testFrameTargetReuseIgnoresPostProcessDownsampleOptions();
		console.log("WebGPU frame-executor targets/MSAA tests passed");
	} finally {
		restoreTestState();
	}
}

await run();
