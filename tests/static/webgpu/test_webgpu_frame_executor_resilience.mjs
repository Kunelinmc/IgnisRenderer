import assert from "node:assert/strict";
import { WebGPUFrameOrchestrator as WebGPUFrameExecutor } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameOrchestrator.ts";
import { Logger } from "../../../src/foundation/Logger.ts";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Material } from "../../../src/materials/Material.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";

import { FakeWebGPUBackend as FakeBackend } from "../../helpers/fakes.mjs";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createPreparedFrameResources(options = {}) {
	return {
		scopeKey: options.scopeKey ?? "main",
		sceneTargetMode: options.sceneTargetMode ?? "mrt",
		frameBinding: { id: "frame-binding" },
		decalFrameBinding: { id: "decal-frame-binding" },
		environmentBinding: { id: "environment-binding" },
		clusteredSceneBinding: { id: "clustered-binding" },
		lightingState: {},
		featureData: { get: () => undefined },
		featureState: {},
		environmentState: {},
		jointMatrixMap: null,
		morphWeightMap: null,
	};
}

function createFrameScopeAdapter(resources) {
	return {
		prepare: (context, options) => resources.prepareFrame(context, options),
		updateParticleShadowVolumes() {},
		destroy() {},
	};
}

function createResourcesStub() {
	return {
		createFrameScope() { return createFrameScopeAdapter(this); },
		sceneFrameLayout: {},
		prepareFrame(_context, options = {}) {
			return createPreparedFrameResources(options);
		},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources() {
			return null;
		},
		async renderParticles() {},
		buildParticleMeshDrawPackets() {
			return [];
		},
	};
}

function createModeTrackingResourcesStub() {
	const state = {
		mode: "single",
		modeTransitions: [],
		environmentModeAtRequest: null,
		drawModeAtRequest: null,
		drawPipelineModeAtRequest: null,
	};
	return {
		createFrameScope() { return createFrameScopeAdapter(this); },
		sceneFrameLayout: {},
		prepareFrame(_context, options = {}) {
			state.mode = options.sceneTargetMode ?? state.mode;
			state.modeTransitions.push(state.mode);
			return createPreparedFrameResources(options);
		},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getEnvironmentResources(_frameResources, sceneTargetMode) {
			state.environmentModeAtRequest = sceneTargetMode ?? state.mode;
			return {
				pipeline: {},
				frameBinding: {},
			};
		},
		async getDrawResources(_packet, _frameResources, options = {}) {
			state.drawModeAtRequest = options.sceneTargetMode ?? state.mode;
			state.drawPipelineModeAtRequest = options.drawMode ?? "default";
			return null;
		},
		async renderParticles() {},
		buildParticleMeshDrawPackets() {
			return [];
		},
		_state: state,
	};
}

function createFrameContext(width, height) {
	return {
		createFrameScope() { return createFrameScopeAdapter(this); },
		viewCamera: {},
		attachments: { width, height },
		features: {
			enableLighting: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableEnvironment: false,
			enableOIT: false,
			enableClusteredLighting: false,
			warnings: [],
			clusteredLightingOptions: {},
		},
		postProcess: createResolvedPostProcess({
			ssao: { enabled: true },
			taa: { enabled: true },
		}, "webgpu"),
		shadowMaps: new Map(),
		scene: {
			particleSystems: [],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
			decalPackets: [],
		},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: {},
		transient: new Map(),
	};
}

function createOITBackend({ sampleCount = 1 } = {}) {
	const backend = new FakeBackend();
	backend.msaaContext = createMSAAContext(sampleCount);
	return backend;
}

function createMSAAContext(initialSampleCount = 1) {
	let sampleCount = initialSampleCount;
	return {
		createFrameScope() { return createFrameScopeAdapter(this); },
		get sampleCount() {
			return sampleCount;
		},
		resolveSupportedSampleCount(requested) {
			return Math.max(1, Math.floor(requested));
		},
		fallbackToSingleSample() {
			if (sampleCount === 1) {
				return false;
			}
			sampleCount = 1;
			return true;
		},
	};
}

function findEncoderCallIndex(backend, predicate) {
	const encoder = backend.commandEncoders[0];
	if (!encoder) {
		return -1;
	}
	return encoder.calls.findIndex(predicate);
}

function getFrameGraphDebugState(executor) {
	return executor.getDebugState();
}

function getFrameTargets(executor) {
	return getFrameGraphDebugState(executor).frameTargets;
}

function getMSAATargets(executor) {
	return getFrameGraphDebugState(executor).msaaTargets;
}

function createOITSequencingResourcesStub() {
	const state = {
		events: [],
		drawOptions: [],
	};
	const drawResource = {
		pipeline: {},
		frameBinding: {},
		modelBinding: {},
		clusteredBinding: {},
		vertexBuffer: {},
		indexBuffer: {},
		indexCount: 3,
	};
	return {
		sceneFrameLayout: {},
		createFrameScope() { return createFrameScopeAdapter(this); },
		prepareFrame(_context, options = {}) {
			state.events.push(`prepare:${options.sceneTargetMode ?? "default"}`);
			return createPreparedFrameResources(options);
		},
		async buildClusteredLighting() {
			state.events.push("clustered:build");
		},
		renderShadows() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources(packet, _frameResources, options = {}) {
			state.drawOptions.push({
				packetId: packet.id,
				sceneTargetMode: options.sceneTargetMode ?? null,
				transparentPipelineMode: options.transparentPipelineMode ?? "default",
				drawMode: options.drawMode ?? "default",
			});
			state.events.push(
				`draw:${packet.id}:${options.transparentPipelineMode ?? "default"}:${options.drawMode ?? "default"}`
			);
			return [drawResource];
		},
		async renderParticles(
			encoder,
			_context,
			targets,
			_frameResources,
			_mode,
			options = {}
		) {
			const blendModes = options.includeBlendModes ?? [];
			state.events.push(
				`particles:${targets.label}:${options.pipelineMode ?? "legacy"}:${blendModes.join(",")}`
			);
			encoder.beginRenderPass({
				label: targets.label,
				colorAttachments: targets.colorAttachments,
				depthStencilAttachment: {
					view: targets.depth,
					depthLoadOp: "load",
					depthStoreOp: "store",
				},
			});
			encoder.endRenderPass();
			return options.pipelineMode === "oit" ? 1 : 1;
		},
		buildParticleMeshDrawPackets() {
			return [];
		},
		_state: state,
	};
}

function createDeferredLightingResourcesStub() {
	const state = {
		deferredUnusedBinding: { id: "deferred-unused-binding" },
		events: [],
	};
	const drawResource = {
		pipeline: { id: "gbuffer-pipeline" },
		frameBinding: { id: "frame-binding" },
		modelBinding: { id: "model-binding" },
		clusteredBinding: { id: "clustered-binding" },
		vertexBuffer: { id: "vertex-buffer" },
		indexBuffer: { id: "index-buffer" },
		indexCount: 3,
	};
	return {
		sceneFrameLayout: {},
		createFrameScope() { return createFrameScopeAdapter(this); },
		prepareFrame(_context, options = {}) {
			return createPreparedFrameResources(options);
		},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources(packet, _frameResources, options = {}) {
			state.events.push(
				`draw:${packet.id}:${options.sceneTargetMode ?? "none"}:${options.drawMode ?? "default"}`
			);
			return [drawResource];
		},
		async renderParticles() {},
		buildParticleMeshDrawPackets() {
			return [];
		},
		getGBufferWriteLayout() {
			return { id: "gbuffer-write-layout" };
		},
		getGBufferReadLayout() {
			return { id: "gbuffer-read-layout" };
		},
		async getDeferredLightingPipeline() {
			return { id: "deferred-lighting-pipeline" };
		},
		getFrameBinding() {
			return { id: "frame-binding" };
		},
		getDeferredUnusedBinding() {
			return state.deferredUnusedBinding;
		},
		getClusteredSceneBinding() {
			return { id: "clustered-binding" };
		},
		_state: state,
	};
}

function createPlanarReflectionResourcesStub() {
	const state = {
		events: [],
		drawOptions: [],
		environmentOptions: [],
		prepareContexts: [],
		throwOnClusteredBuild: false,
	};
	const drawResource = {
		pipeline: { id: "draw-pipeline" },
		frameBinding: { id: "frame-binding" },
		modelBinding: { id: "model-binding" },
		clusteredBinding: { id: "clustered-binding" },
		vertexBuffer: { id: "vertex-buffer" },
		indexBuffer: { id: "index-buffer" },
		indexCount: 3,
	};
	return {
		sceneFrameLayout: {},
		createFrameScope() {
			return {
				prepare: (context, options) => this.prepareFrame(context, options),
				updateParticleShadowVolumes() {},
				destroy() { state.events.push("scope:destroy"); },
			};
		},
		prepareFrame(context, options = {}) {
			state.prepareContexts.push(context);
			state.events.push(
				`prepare:reflection:${context.features.enableReflection}:ssr:${context.postProcess.isEnabled("ssr")}:opaque:${context.scene.opaquePackets.map((packet) => packet.id).join(",")}`
			);
			return createPreparedFrameResources(options);
		},
		async buildClusteredLighting(_encoder, frameResources) {
			state.events.push("clustered:build");
			if (frameResources) {
				state.events.push(`clustered-scope:${frameResources.scopeKey}`);
			}
			if (state.throwOnClusteredBuild) {
				throw new Error("simulated planar capture failure");
			}
		},
		renderShadows() {},
		async getEnvironmentResources(_frameResources, sceneTargetMode, options = {}) {
			state.environmentOptions.push({
				sceneTargetMode: sceneTargetMode ?? null,
				sampleCountOverride: options.sampleCountOverride ?? null,
			});
			return null;
		},
		async getDrawResources(packet, _frameResources, options = {}) {
			state.drawOptions.push({
				packetId: packet.id,
				sceneTargetMode: options.sceneTargetMode ?? null,
				drawMode: options.drawMode ?? "default",
				sampleCountOverride: options.sampleCountOverride ?? null,
			});
			state.events.push(
				`draw:${packet.id}:${options.sceneTargetMode ?? "default"}:${options.drawMode ?? "default"}`
			);
			return [drawResource];
		},
		async renderParticles() {},
		buildParticleMeshDrawPackets() {
			return [];
		},
		getPlanarReflectionLayout() {
			return { id: "planar-reflection-layout" };
		},
		releaseScope(scopeKey) {
			state.events.push(`release:${scopeKey}`);
		},
		_state: state,
	};
}

async function testZeroSizedFrameSkipsEncoderAndLegacyDepthPath() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(0, 0);

	executor.beginFrame(context);
	assert.equal(backend.createCommandEncoderCalls, 0);
	assert.equal(getFrameGraphDebugState(executor).active, true);

	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);
	await executor.endFrame();
	assert.equal(getFrameGraphDebugState(executor).active, false);
	assert.equal(getFrameGraphDebugState(executor).texturePoolOwnerCount, 0);
}

async function testDebugStateRetainsNodesFromEveryCompiledStage() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);
	await executor.endFrame();

	const debugState = getFrameGraphDebugState(executor);
	assert.ok(debugState.lastPlannedNodeIds.includes("main-opaque:opaque-scene"));
	assert.ok(debugState.lastPlannedNodeIds.includes("postprocess:presentation"));
	assert.deepEqual(
		debugState.compiledStages.map((stage) => stage.pass.stage),
		["main-opaque", "postprocess"]
	);
}

async function testFrameSessionRejectsInvalidLifecycleCalls() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	await assert.rejects(
		executor.executePass(
			{ stage: "main-opaque", executor: "backend", enabled: true },
			context
		),
		/no active frame session/
	);
	await assert.rejects(executor.endFrame(), /no active frame session/);
	assert.doesNotThrow(() => executor.abortFrame());

	executor.beginFrame(context);
	assert.throws(
		() => executor.beginFrame(context),
		/already has an active frame session/
	);
	assert.equal(getFrameGraphDebugState(executor).active, true);
	executor.abortFrame();
}

async function testFrameSessionRequiresOriginalContext() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	const mismatchedContext = createFrameContext(64, 64);

	executor.beginFrame(context);
	await assert.rejects(
		executor.executePass(
			{ stage: "main-opaque", executor: "backend", enabled: true },
			mismatchedContext
		),
		/must match the context passed to beginFrame/
	);
	assert.equal(getFrameGraphDebugState(executor).active, true);
	executor.abortFrame();
}

function testAbortFrameClearsActiveStateWithoutSubmit() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	executor.beginFrame(context);

	executor.abortFrame();

	assert.equal(backend.submits, 0);
	assert.equal(getFrameGraphDebugState(executor).active, false);
	assert.equal(getFrameGraphDebugState(executor).motionHistoryWriteTarget, null);
	assert.equal(getFrameGraphDebugState(executor).oitActive, false);
}

async function testEndFrameFailureClosesActiveState() {
	const backend = new FakeBackend();
	const error = new Error("submit failed");
	backend.submit = () => {
		throw error;
	};
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	executor.beginFrame(context);
	let caught = null;
	try {
		await executor.endFrame();
	} catch (caughtError) {
		caught = caughtError;
	}

	assert.strictEqual(caught, error);
	assert.equal(getFrameGraphDebugState(executor).active, false);
	assert.equal(getFrameGraphDebugState(executor).motionHistoryWriteTarget, null);
}

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

function testInvalidateFrameTargetsDestroysPresentBinding() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());

	executor.invalidateFrameTargets();

	assert.equal(getFrameTargets(executor), null);
}

function testInvalidateFrameTargetsDefersDuringActiveFrame() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	executor.beginFrame(context);
	const activeTargets = getFrameTargets(executor);
	assert.ok(activeTargets);

	executor.invalidateFrameTargets();
	assert.strictEqual(getFrameTargets(executor), activeTargets);
	assert.equal(
		getFrameGraphDebugState(executor).pendingFrameTargetInvalidation,
		true
	);

	executor.abortFrame();
	assert.equal(getFrameTargets(executor), null);
	assert.equal(
		getFrameGraphDebugState(executor).pendingFrameTargetInvalidation,
		false
	);
}

function testShaderRuntimeInvalidationDefersDuringActiveFrame() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	executor.beginFrame(context);
	executor.onShaderRuntimeChanged();
	assert.equal(
		getFrameGraphDebugState(executor).pendingShaderRuntimeInvalidation,
		true
	);

	executor.abortFrame();
	assert.equal(
		getFrameGraphDebugState(executor).pendingShaderRuntimeInvalidation,
		false
	);
}

async function testLegacyMainPassForcesSingleSceneTargetMode() {
	const backend = new FakeBackend();
	const resources = createModeTrackingResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
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
	const executor = new WebGPUFrameExecutor(backend, resources);
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

async function testPlanarReflectionCaptureAndCompositeSequencing() {
	const backend = new FakeBackend();
	backend.device.limits.maxStorageTexturesPerShaderStage = 0;
	const resources = createPlanarReflectionResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
	const context = createFrameContext(64, 64);
	const camera = new Camera();
	camera.position.set(0, 2, 5);
	camera.updateMatrices();
	context.viewCamera = camera;
	context.features.enableReflection = true;
	context.postProcess = createResolvedPostProcess({
		ssr: { enabled: true },
	});
	context.incremental = {
		enabled: false,
		forceFullFrame: true,
		dirtyRects: [{ x: 0, y: 0, width: 64, height: 64 }],
		dirtyTileSize: 64,
		dirtyTileColumns: 1,
		dirtyTileRows: 1,
		dirtyTiles: [0],
		dirtyAreaRatio: 1,
		firstPass: null,
		reasonMask: 0,
		temporalHistoryReset: false,
	};
	const mirrorMaterial = new Material({
		name: "mirror",
		reflectivity: 0.75,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	});
	const objectMaterial = new Material({ name: "object" });
	const mirrorPacket = createPlanarPacket("mirror", mirrorMaterial, 0);
	const objectPacket = createPlanarPacket("object", objectMaterial, 1);
	context.scene.opaquePackets = [mirrorPacket, objectPacket];
	context.scene.reflectivePackets = [mirrorPacket];
	context.scene.transparentPackets = [];
	context.scene.meshInstances = [];
	context.scene.lights = [];
	context.scene.shadowMaps = new Map();
	context.scene.environment = {
		backgroundEnabled: false,
		lightingEnabled: false,
		backgroundTexture: null,
		iblTexture: null,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};

	executor.beginFrame(context);
	await executor.executePass(
		{ stage: "reflection", executor: "backend", enabled: true },
		context
	);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	const colorTarget = backend.createTextureCalls.find((desc) =>
		String(desc.label).startsWith("WebGPUPlanarReflectionColor_")
	);
	assert.equal(colorTarget.width, 32);
	assert.equal(colorTarget.height, 32);
	assert.ok(
		resources._state.events.includes(
			"prepare:reflection:false:ssr:false:opaque:object"
		)
	);
	assert.ok(
		resources._state.events.includes("draw:object:color:reflection-capture")
	);
	assert.ok(
		resources._state.environmentOptions.some(
			(entry) => entry.sceneTargetMode === "color"
		)
	);
	assert.ok(
		resources._state.events.includes(
			"draw:mirror:mrt:planar-reflection-composite"
		)
	);
	const labels = backend.recordedRenderPasses.map((pass) => pass.label);
	const capturePass = backend.recordedRenderPasses.find(
		(pass) => pass.label === "WebGPUPlanarReflectionCaptureMain"
	);
	assert.ok(capturePass);
	assert.equal(capturePass.colorAttachments.length, 1);
	assert.ok(
		labels.indexOf("WebGPUPlanarReflectionCaptureMain") <
			labels.indexOf("WebGPUMainMRT_Clear")
	);
	assert.ok(
		labels.indexOf("WebGPUPlanarReflectionComposite") >
			labels.indexOf("WebGPUMainMRT_Clear")
	);
	const compositePass = backend.recordedRenderPasses.find(
		(pass) => pass.label === "WebGPUPlanarReflectionComposite"
	);
	assert.equal(compositePass.colorAttachments.length, 2);
	assert.equal(
		String(compositePass.colorAttachments[1].view.label).startsWith(
			"WebGPUPlanarReflectionMask"
		),
		true
	);
	assert.equal(backend.submits, 0);
	await executor.endFrame();
	assert.deepEqual(getFrameGraphDebugState(executor).commit.submittedLabels, [
		"main:before-reflection",
		"planar-reflection:0.000000,1.000000,0.000000,0.000000",
		"main:final",
	]);
}

async function testPlanarReflectionUsesColorTargetsWithoutPostProcess() {
	const backend = new FakeBackend();
	backend.device.limits.maxStorageTexturesPerShaderStage = 0;
	const resources = createPlanarReflectionResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
	const context = createFrameContext(64, 64);
	const camera = new Camera();
	camera.position.set(0, 2, 5);
	camera.updateMatrices();
	context.viewCamera = camera;
	context.features.enableReflection = true;
	context.postProcess = createResolvedPostProcess({});
	const mirrorMaterial = new Material({
		name: "mirror",
		reflectivity: 0.75,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	});
	const objectMaterial = new Material({ name: "object" });
	const mirrorPacket = createPlanarPacket("mirror", mirrorMaterial, 0);
	const objectPacket = createPlanarPacket("object", objectMaterial, 1);
	context.scene.opaquePackets = [mirrorPacket, objectPacket];
	context.scene.reflectivePackets = [mirrorPacket];
	context.scene.transparentPackets = [];
	context.scene.meshInstances = [];
	context.scene.lights = [];
	context.scene.shadowMaps = new Map();
	context.scene.environment = {
		backgroundEnabled: false,
		lightingEnabled: false,
		backgroundTexture: null,
		iblTexture: null,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};

	executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	assert.ok(targets);
	assert.equal(executor.getSceneTargetModeForFrame(), "color");
	assert.equal(targets.postPing, null);
	assert.equal(targets.postPong, null);
	assert.equal(targets.gAlbedoAlpha, null);
	assert.equal(targets.gMotionDepth, null);
	assert.ok(targets.planarReflectionMask);

	await executor.executePass(
		{ stage: "reflection", executor: "backend", enabled: true },
		context
	);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	assert.ok(
		resources._state.drawOptions.some(
			(entry) =>
				entry.packetId === "object" &&
				entry.sceneTargetMode === "color"
		)
	);
	const capturePass = backend.recordedRenderPasses.find(
		(pass) => pass.label === "WebGPUPlanarReflectionCaptureMain"
	);
	assert.ok(capturePass);
	assert.equal(capturePass.colorAttachments.length, 1);
	assert.ok(
		resources._state.events.includes(
			"draw:mirror:mrt:planar-reflection-composite"
		)
	);
	const labels = backend.recordedRenderPasses.map((pass) => pass.label);
	assert.ok(labels.includes("WebGPUMainColor_Clear"));
	assert.ok(labels.includes("WebGPUPlanarReflectionComposite"));
	const compositePass = backend.recordedRenderPasses.find(
		(pass) => pass.label === "WebGPUPlanarReflectionComposite"
	);
	assert.equal(compositePass.colorAttachments.length, 2);
	assert.strictEqual(
		compositePass.colorAttachments[1].view,
		targets.planarReflectionMask
	);
	executor.destroy();
}

async function testPlanarReflectionCaptureKeepsMSAAFrameTargetsAlive() {
	const backend = new FakeBackend();
	backend.device.limits.maxStorageTexturesPerShaderStage = 0;
	const msaa = createMSAAContext(4);
	let executor = null;
	const resources = createPlanarReflectionResourcesStub();
	executor = new WebGPUFrameExecutor(backend, resources, msaa);
	const context = createFrameContext(64, 64);
	const camera = new Camera();
	camera.position.set(0, 2, 5);
	camera.updateMatrices();
	context.viewCamera = camera;
	context.features.enableReflection = true;
	context.postProcess = createResolvedPostProcess({
		ssr: { enabled: true },
	});
	context.incremental = {
		enabled: false,
		forceFullFrame: true,
		dirtyRects: [{ x: 0, y: 0, width: 64, height: 64 }],
		dirtyTileSize: 64,
		dirtyTileColumns: 1,
		dirtyTileRows: 1,
		dirtyTiles: [0],
		dirtyAreaRatio: 1,
		firstPass: null,
		reasonMask: 0,
		temporalHistoryReset: false,
	};
	const mirrorMaterial = new Material({
		name: "mirror",
		reflectivity: 0.75,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	});
	const objectMaterial = new Material({ name: "object" });
	const mirrorPacket = createPlanarPacket("mirror", mirrorMaterial, 0);
	const objectPacket = createPlanarPacket("object", objectMaterial, 1);
	context.scene.opaquePackets = [mirrorPacket, objectPacket];
	context.scene.reflectivePackets = [mirrorPacket];
	context.scene.transparentPackets = [];
	context.scene.meshInstances = [];
	context.scene.lights = [];
	context.scene.shadowMaps = new Map();
	context.scene.environment = {
		backgroundEnabled: false,
		lightingEnabled: false,
		backgroundTexture: null,
		iblTexture: null,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};

	executor.beginFrame(context);
	const frameTargets = getFrameTargets(executor);
	const msaaTargets = getMSAATargets(executor);
	assert.ok(frameTargets);
	assert.ok(msaaTargets);

	await executor.executePass(
		{ stage: "reflection", executor: "backend", enabled: true },
		context
	);
	assert.equal(msaa.sampleCount, 4);
	assert.strictEqual(getFrameTargets(executor), frameTargets);
	assert.strictEqual(getMSAATargets(executor), msaaTargets);

	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);
	assert.strictEqual(getFrameTargets(executor), frameTargets);
	assert.strictEqual(getMSAATargets(executor), msaaTargets);
	const captureDrawOptions = resources._state.drawOptions.find(
		(options) => options.drawMode === "reflection-capture"
	);
	assert.ok(captureDrawOptions);
	assert.equal(captureDrawOptions.sceneTargetMode, "color");
	assert.equal(captureDrawOptions.sampleCountOverride, 1);
	const compositePass = backend.recordedRenderPasses.find(
		(pass) => pass.label === "WebGPUPlanarReflectionComposite"
	);
	assert.ok(compositePass);
	assert.equal(
		compositePass.colorAttachments[0].resolveTarget,
		frameTargets.sceneColorMain
	);
	assert.equal(
		compositePass.colorAttachments[1].resolveTarget,
		frameTargets.planarReflectionMask
	);
	executor.destroy();
}

async function testPlanarReflectionCaptureFailureKeepsMainFrameResources() {
	const backend = new FakeBackend();
	backend.device.limits.maxStorageTexturesPerShaderStage = 0;
	const resources = createPlanarReflectionResourcesStub();
	resources._state.throwOnClusteredBuild = true;
	const executor = new WebGPUFrameExecutor(backend, resources, backend.msaaContext);
	const context = createFrameContext(64, 64);
	const camera = new Camera();
	camera.position.set(0, 2, 5);
	camera.updateMatrices();
	context.viewCamera = camera;
	context.features.enableReflection = true;
	context.postProcess = createResolvedPostProcess({
		ssr: { enabled: true },
	});
	context.incremental = {
		enabled: false,
		forceFullFrame: true,
		dirtyRects: [{ x: 0, y: 0, width: 64, height: 64 }],
		dirtyTileSize: 64,
		dirtyTileColumns: 1,
		dirtyTileRows: 1,
		dirtyTiles: [0],
		dirtyAreaRatio: 1,
		firstPass: null,
		reasonMask: 0,
		temporalHistoryReset: false,
	};
	const mirrorMaterial = new Material({
		name: "mirror",
		reflectivity: 0.75,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	});
	const objectMaterial = new Material({ name: "object" });
	const mirrorPacket = createPlanarPacket("mirror", mirrorMaterial, 0);
	const objectPacket = createPlanarPacket("object", objectMaterial, 1);
	context.scene.opaquePackets = [mirrorPacket, objectPacket];
	context.scene.reflectivePackets = [mirrorPacket];
	context.scene.transparentPackets = [];
	context.scene.meshInstances = [];
	context.scene.lights = [];
	context.scene.shadowMaps = new Map();
	context.scene.environment = {
		backgroundEnabled: false,
		lightingEnabled: false,
		backgroundTexture: null,
		iblTexture: null,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};

	executor.beginFrame(context);
	const mainFrameResources = executor.getPreparedFrameResources();

	await assert.rejects(
		executor.executePass(
			{ stage: "reflection", executor: "backend", enabled: true },
			context
		),
		/simulated planar capture failure/
	);

	assert.strictEqual(executor.getPreparedFrameResources(), mainFrameResources);
	assert.equal(mainFrameResources.sceneTargetMode, "mrt");
	assert.ok(
		resources._state.events.some((event) =>
			event === "scope:destroy"
		)
	);
	executor.abortFrame();
	assert.equal(getFrameGraphDebugState(executor).active, false);
}

async function testPlanarReflectionCaptureUsesMirroredCameraAndCenterSide() {
	const backend = new FakeBackend();
	backend.device.limits.maxStorageTexturesPerShaderStage = 0;
	const resources = createPlanarReflectionResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
	const context = createFrameContext(64, 64);
	const camera = new Camera();
	camera.position.set(0, 2, 5);
	camera.updateMatrices();
	context.viewCamera = camera;
	context.features.enableReflection = true;
	context.postProcess = createResolvedPostProcess({
		ssr: { enabled: true },
	});

	const mirrorMaterial = new Material({
		name: "mirror",
		reflectivity: 0.75,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	});
	const objectMaterial = new Material({ name: "object" });
	const mirrorPacket = createPlanarPacket("mirror", mirrorMaterial, 0);
	const abovePacket = createPlanarPacket("above", objectMaterial, 1);
	const crossingPacket = createPlanarPacket("crossing", objectMaterial, -0.5);
	const culledPacket = createPlanarPacket("culled", objectMaterial, -2);
	culledPacket.worldBounds.radius = 0.25;
	context.scene.opaquePackets = [
		mirrorPacket,
		abovePacket,
		crossingPacket,
		culledPacket,
	];
	context.scene.reflectivePackets = [mirrorPacket];
	context.scene.transparentPackets = [];
	context.scene.meshInstances = [];
	context.scene.lights = [];
	context.scene.shadowMaps = new Map();
	context.scene.environment = {
		backgroundEnabled: false,
		lightingEnabled: false,
		backgroundTexture: null,
		iblTexture: null,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};

	executor.beginFrame(context);
	await executor.executePass(
		{ stage: "reflection", executor: "backend", enabled: true },
		context
	);

	const captureContext = resources._state.prepareContexts.find(
		(candidate) =>
			candidate.features.enableReflection === false &&
			candidate.postProcess.isEnabled("ssr") === false
	);
	assert.ok(captureContext);
	assert.deepEqual(captureContext.viewCamera.getWorldPosition(), {
		x: 0,
		y: -2,
		z: 5,
	});
	assert.deepEqual(
		captureContext.scene.opaquePackets.map((packet) => packet.id),
		["above"]
	);
	assert.ok(
		resources._state.events.includes(
			"prepare:reflection:false:ssr:false:opaque:above"
		)
	);
}

async function testOITTransparentAndParticleExecutionOrder() {
	const backend = createOITBackend();
	const resources = createOITSequencingResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
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
	const executor = new WebGPUFrameExecutor(backend, resources);
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

async function testScreenSpaceRefractionCapturesTransmissionPackets() {
	const backend = createOITBackend();
	const resources = createOITSequencingResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess(
		{ ssrefraction: { enabled: true } },
		"webgpu"
	);
	context.scene.transparentPackets = [
		{
			id: "transparent-transmission-capture",
			material: { transmissionFactor: 1 },
		},
	];

	executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	assert.ok(targets);
	await executor.executePass(
		{ stage: "main-transparent", executor: "backend", enabled: true },
		context
	);

	const labels = backend.recordedRenderPasses.map((pass) => pass.label);
	assert.deepEqual(labels, ["WebGPUTransmissionCapture"]);
	assert.equal(backend.encoderCopyCalls.length, 2);
	assert.equal(backend.encoderCopyCalls[0][0].texture, targets.sceneColorMain);
	assert.equal(
		backend.encoderCopyCalls[0][1].texture,
		targets.transmissionSceneColorCopy
	);
	assert.equal(backend.encoderCopyCalls[1][0].texture, targets.depth);
	assert.equal(backend.encoderCopyCalls[1][0].aspect, "depth-only");
	assert.equal(
		backend.encoderCopyCalls[1][1].texture,
		targets.transmissionDepth
	);
	assert.equal(backend.encoderCopyCalls[1][1].aspect, "depth-only");
	assert.equal(
		backend.recordedRenderPasses[0].depthStencilAttachment.depthLoadOp,
		"load"
	);
	assert.ok(
		resources._state.events.includes(
			"draw:transparent-transmission-capture:transmission-capture:default"
		)
	);
	assert.equal(
		resources._state.events.includes(
			"draw:transparent-transmission-capture:transmission:default"
		),
		false
	);
	assert.deepEqual(resources._state.drawOptions, [
		{
			packetId: "transparent-transmission-capture",
			sceneTargetMode: "mrt",
			transparentPipelineMode: "transmission-capture",
			drawMode: "default",
		},
	]);
}

async function testDeferredLightingBindsUnusedGroupOnePlaceholder() {
	const backend = new FakeBackend();
	const resources = createDeferredLightingResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
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
	const executor = new WebGPUFrameExecutor(backend, resources);
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
	const executor = new WebGPUFrameExecutor(backend, resources);
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

function testDeferredLightingWarnsWhenRequestedButMRTUnavailable() {
	const backend = new FakeBackend();
	backend.device.limits.maxColorAttachments = 1;
	backend.device.limits.maxColorAttachmentBytesPerSample = 16;
	const resources = createModeTrackingResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
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
				warning.includes("[webgpu-deferred-disabled-mrt]")
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
	const executor = new WebGPUFrameExecutor(backend, resources);
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
	await testZeroSizedFrameSkipsEncoderAndLegacyDepthPath();
	await testDebugStateRetainsNodesFromEveryCompiledStage();
	await testFrameSessionRejectsInvalidLifecycleCalls();
	await testFrameSessionRequiresOriginalContext();
	testAbortFrameClearsActiveStateWithoutSubmit();
	await testEndFrameFailureClosesActiveState();
	testFrameTargetAllocationFailureReleasesPartialResources();
	await testMSAAAllocationFallbackPersistsForDeviceRuntime();
	testInvalidateFrameTargetsDestroysPresentBinding();
	testInvalidateFrameTargetsDefersDuringActiveFrame();
	testShaderRuntimeInvalidationDefersDuringActiveFrame();
	await testLegacyMainPassForcesSingleSceneTargetMode();
	await testIncrementalMainPassUsesDepthPartialReuse();
	await testMainOpaqueDisablesEarlyZWhenConfiguredOff();
	await testLegacyMainPassScalesDirtyRectsToCanvasTarget();
	testFrameTargetsSkipOptionalTargetsWhenUnused();
	testGBufferBridgeReportsAllocatedWebGPUFormats();
	testFrameTargetsAllocateAndReleaseOptionalTargetsWhenNeeded();
	testFrameTargetsSkippedWhenFrameHasNoOffscreenWork();
	testTransmissionTargetsAllocateOnlyWhenRefractionHasWork();
	await testFrameTargetReuseIgnoresPostProcessDownsampleOptions();
	await testPlanarReflectionCaptureAndCompositeSequencing();
	await testPlanarReflectionUsesColorTargetsWithoutPostProcess();
	await testPlanarReflectionCaptureKeepsMSAAFrameTargetsAlive();
	await testPlanarReflectionCaptureFailureKeepsMainFrameResources();
	await testPlanarReflectionCaptureUsesMirroredCameraAndCenterSide();
	await testOITTransparentAndParticleExecutionOrder();
	await testOITTransparentResolvesImmediatelyWithoutParticles();
	await testScreenSpaceRefractionCapturesTransmissionPackets();
	await testDeferredLightingBindsUnusedGroupOnePlaceholder();
	await testDeferredLightingKeepsTransmissionOutOfGBuffer();
	await testDeferredLightingCanBeExplicitlyDisabled();
	testDeferredLightingWarnsWhenRequestedButMRTUnavailable();
	await testOITMSAAFallsBackToLegacyAndWarns();
	testOITRuntimeFallbackWarnsWithoutEncoderCopy();
	console.log("WebGPU frame executor resilience tests passed");
}

await run();

function createPlanarPacket(id, material, y) {
	const worldMatrix = Matrix4.identity();
	return {
		id,
		meshInstance: { id: `${id}-mesh`, worldMatrix, mesh: { primitives: [] } },
		mesh: { primitives: [], boundingSphere: { center: { x: 0, y, z: 0 }, radius: 1 } },
		primitive: {
			id: `${id}-primitive`,
			material,
			geometry: {},
			boundingSphere: { center: { x: 0, y, z: 0 }, radius: 1 },
		},
		material,
		geometry: {},
		worldMatrix,
		normalMatrix: worldMatrix,
		worldBounds: { center: { x: 0, y, z: 0 }, radius: 1 },
		sortDepth: 1,
		pipelineKey: id,
		passFlags: 0,
	};
}
