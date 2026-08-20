import assert from "node:assert/strict";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";

import * as frameExecutorFixture from "../../helpers/webgpu_frame_executor_resilience.mjs";

const {
	BackendPostProcessRuntime,
	FakeBackend,
	Logger,
	WebGPUPostProcessExecutor,
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

async function testFrameTargetAllocationFailureReleasesPartialResources() {
	const backend = new FakeBackend();
	backend.failTextureAtCall = 4;
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	await assert.rejects(
		executor.beginFrame(context),
		/simulated allocation failure/,
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
		await executor.beginFrame(context);
		assert.equal(msaa.sampleCount, 1);
		assert.equal(getFrameGraphDebugState(executor).msaaTargets, null);
		assert.equal(
			getFrameGraphDebugState(executor).targetManager.sampleCount,
			1
		);
		assert.equal(multisampleAllocationAttempts, 1);
		assert.equal(
			warnings.filter((warning) =>
				warning.includes("[webgpu-scene-sample-count-runtime-fallback-1x]")
			).length,
			1
		);
		assert.ok(warnings.some((warning) => warning.includes("4x scene sample-count")));

		await executor.endFrame();
		await executor.beginFrame(context);
		assert.equal(multisampleAllocationAttempts, 1);
		assert.equal(msaa.sampleCount, 1);
	} finally {
		Logger.reset();
		executor.destroy();
	}
}

async function testMSAATargetFormatsMatchLegacyMRTPipeline() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(
		backend,
		createResourcesStub(),
		createMSAAContext(4)
	);
	try {
		await executor.beginFrame(createFrameContext(64, 64));
		const targets = getMSAATargets(executor);
		assert.ok(targets);
		assert.deepEqual(
			[
				targets.sceneColorMain,
				targets.gAlbedoAlpha,
				targets.gNormalRoughMetal,
				targets.gEmissiveOcclusion,
				targets.gMotionDepth,
			].map((target) => target?.format),
			[
				"rgba16float",
				"rgba8unorm",
				"rgba8unorm",
				"rgba16float",
				"rgba16float",
			]
		);
	} finally {
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

	await executor.beginFrame(context);

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

	await executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	assert.equal(backend.recordedRenderPasses.length >= 4, true);
	const colorClearPass = backend.recordedRenderPasses[0];
	assert.equal(colorClearPass.label, "WebGPUColorDirtyClear");
	assert.equal(colorClearPass.colorAttachments.length, 5);
	assert.equal(
		colorClearPass.colorAttachments.every((attachment) => attachment.loadOp === "load"),
		true
	);
	const depthClearPass = backend.recordedRenderPasses[1];
	assert.equal(depthClearPass.depthStencilAttachment.depthLoadOp, "load");
	assert.equal(depthClearPass.colorAttachments.length, 0);
	const earlyZPass = backend.recordedRenderPasses[2];
	assert.equal(earlyZPass.label, "WebGPUEarlyZPrepassMRT");
	assert.equal(earlyZPass.colorAttachments.length, 0);
	assert.equal(earlyZPass.depthStencilAttachment.depthLoadOp, "load");
	const mainPass = backend.recordedRenderPasses[3];
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

	await executor.beginFrame(context);
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

	await executor.beginFrame(context);

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
		["setScissorRect", 611, 0, 612, 869],
	]);
}

async function testFrameTargetsSkipOptionalTargetsWhenUnused() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	await executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	assert.ok(targets);
	assert.equal(getFrameGraphDebugState(executor).targetManager.sceneTargetMode, "mrt");
	assert.ok(targets.postPing);
	assert.ok(targets.postPong);
	assert.ok(targets.gMotionDepth);
	assert.equal(targets.oitAccum, null);
	assert.equal(targets.oitReveal, null);
	assert.equal(targets.oitSceneColorCopy, null);
	assert.equal(targets.planarReflectionMask, null);

	executor.invalidateFrameTargets();
	executor.abortFrame();
	assert.equal(getFrameTargets(executor), null);
}

async function testGBufferBridgeReportsAllocatedWebGPUFormats() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	await executor.beginFrame(context);
	const bridge = executor.frameRuntime.postProcess.createGBufferBridge(context);

	assert.equal(bridge.normalSpace, "view");
	assert.equal(bridge.depthEncoding, "linear-view-z");
	assert.equal(bridge.motionEncoding, "ndc-delta");
	assert.equal(bridge.channels.depth.format, "rgba16float");
	assert.equal(bridge.channels.depth.encoding, "motion-depth.z");
	assert.equal(bridge.channels.motion.format, "rgba16float");
	assert.equal(bridge.channels.motion.encoding, "motion-depth.xy");
	assert.equal(bridge.channels.normal.format, "rgba8unorm");
	assert.equal(bridge.channels.normal.encoding, "encoded-view-normal");
	assert.equal(bridge.channels.roughness.format, "rgba8unorm");
	assert.equal(bridge.channels.roughness.encoding, "normal-roughness-metallic.z");
	assert.equal(bridge.channels.metallic.format, "rgba8unorm");
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

async function testHiZIsPlannableBeforeItsBuildExecutes() {
	const backend = new FakeBackend();
	const postProcessExecutor = new WebGPUPostProcessExecutor(backend);
	backend.postProcessRuntime = new BackendPostProcessRuntime({
		executor: postProcessExecutor,
		backend,
	});
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess({
		ssgi: { enabled: true },
	}, "webgpu");
	const port = executor.frameRuntime.postProcess.createSessionPort();
	postProcessExecutor.bindSession(port);

	try {
		await executor.beginFrame(context);
		const targets = getFrameTargets(executor);

		assert.ok(targets.hiZ);
		assert.equal(
			port.isGraphResourceAvailable("backend:frame-hiz"),
			true,
		);
		assert.deepEqual(
			executor.frameRuntime.postProcess.graphFrame.graph.passes.map((pass) => pass.id),
			["ssgi"],
			"SSGI finalization must see the allocated MRT G-buffer channels",
		);
	} finally {
		executor.abortFrame();
		postProcessExecutor.unbindSession(port);
	}
}

async function testFrameTargetsAllocateAndReleaseOptionalTargetsWhenNeeded() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess({});
	context.features.enableOIT = true;
	context.features.enableReflection = true;
	context.scene.transparentPackets = [{ id: "transparent", material: {} }];
	context.scene.reflectivePackets = [{ id: "mirror", material: {} }];

	await executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	assert.ok(targets);
	assert.equal(getFrameGraphDebugState(executor).targetManager.sceneTargetMode, "color");
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

async function testFrameTargetsSkippedWhenFrameHasNoOffscreenWork() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess({});

	await executor.beginFrame(context);
	assert.equal(getFrameTargets(executor), null);
	assert.equal(getFrameGraphDebugState(executor).targetManager.sceneTargetMode, "single");
	assert.equal(getFrameGraphDebugState(executor).texturePoolOwnerCount, 0);
}

async function testTransmissionTargetsAllocateOnlyWhenRefractionHasWork() {
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

	await executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	assert.ok(targets.transmissionSceneColorCopy);
	assert.ok(targets.transmissionLighting);
	assert.ok(targets.gTransmissionSurface0);
	assert.ok(targets.gTransmissionSurface1);
	assert.ok(targets.gTransmissionSurface2);
	assert.ok(targets.transmissionDepth);
	const bridge = executor.frameRuntime.postProcess.createGBufferBridge(context);
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
	await executor.beginFrame(noWorkContext);
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
	await executor.beginFrame(firstContext);
	const firstTargets = getFrameTargets(executor);
	const firstTextureCount = backend.createTextureCalls.length;
	await executor.endFrame();

	const secondContext = createFrameContext(64, 64);
	secondContext.postProcess = createResolvedPostProcess({
		ssao: { enabled: true, options: { downsample: 4 } },
		ssr: { enabled: true, options: { downsample: 4 } },
	});
	await executor.beginFrame(secondContext);
	assert.strictEqual(getFrameTargets(executor), firstTargets);
	assert.equal(backend.createTextureCalls.length, firstTextureCount);
	await executor.endFrame();
	executor.destroy();
}

async function testDeferredBaseSkipsFrameSizedExtensionTargets() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess({});
	context.scene.opaquePackets = [{ material: new PBRMaterial() }];
	await executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	assert.equal(getFrameGraphDebugState(executor).targetManager.sceneTargetMode, "gbuffer");
	assert.equal(
		getFrameGraphDebugState(executor).targetManager.deferredGBufferLayout,
		"base"
	);
	assert.equal(targets.gSpecular, null);
	assert.equal(targets.gCoatSheen, null);
	assert.equal(targets.gSheenReflectance, null);
	assert.equal(targets.gMaterialExt0, null);
	assert.equal(targets.gMaterialExt3, null);
	await executor.endFrame();
	executor.destroy();
}

async function testDeferredAllocationFallbackReleasesCompactTargetsAtomically() {
	const backend = new FakeBackend();
	const createTexture = backend.createTexture.bind(backend);
	let failedExtendedAllocation = false;
	backend.createTexture = (desc) => {
		if (!failedExtendedAllocation && desc.format === "rgba16uint") {
			failedExtendedAllocation = true;
			throw new Error("simulated deferred extension allocation failure");
		}
		return createTexture(desc);
	};
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess({});
	context.scene.opaquePackets = [
		{ material: new PBRMaterial({ anisotropyStrength: 1 }) },
	];
	const warnings = [];
	Logger.configure({
		level: "warn",
		resetOnceKeys: true,
		sink: {
			warn: (...args) => warnings.push(args.map((arg) => String(arg)).join(" ")),
		},
	});
	try {
		await executor.beginFrame(context);
		assert.equal(failedExtendedAllocation, true);
		assert.equal(getFrameGraphDebugState(executor).targetManager.sceneTargetMode, "mrt");
		assert.equal(
			getFrameGraphDebugState(executor).targetManager.sceneTargetMode,
			"mrt"
		);
		assert.equal(
			warnings.filter((warning) =>
				warning.includes("[webgpu-deferred-runtime-fallback]")
			).length,
			1
		);
		const liveCompactTargets = backend.textures.filter(
			(texture) =>
				texture.label?.startsWith("WebGPUGBuffer") && !texture.destroyed
		);
		assert.equal(
			liveCompactTargets.some((texture) => texture.format === "rgba16uint"),
			false
		);
		await executor.endFrame();
	} finally {
		Logger.reset();
		executor.destroy();
	}
}

async function run() {
	try {
		await testFrameTargetAllocationFailureReleasesPartialResources();
		await testMSAAAllocationFallbackPersistsForDeviceRuntime();
		await testMSAATargetFormatsMatchLegacyMRTPipeline();
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
		await testDeferredBaseSkipsFrameSizedExtensionTargets();
		await testDeferredAllocationFallbackReleasesCompactTargetsAtomically();
		console.log("WebGPU frame-executor targets/MSAA tests passed");
	} finally {
		restoreTestState();
	}
}

await run();
