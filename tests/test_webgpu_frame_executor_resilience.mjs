import assert from "node:assert/strict";
import { WebGPUFrameExecutor } from "../src/renderers/webgpu/WebGPUFrameExecutor.ts";
import { Logger } from "../src/foundation/Logger.ts";

import { FakeWebGPUBackend as FakeBackend } from "./helpers/test_fakes.mjs";

function createResourcesStub() {
	return {
		sceneFrameLayout: {},
		setSceneTargetMode() {},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getSkyboxResources() {
			return null;
		},
		async getDrawResources() {
			return null;
		},
		async renderParticles() {},
	};
}

function createModeTrackingResourcesStub() {
	const state = {
		mode: "single",
		modeTransitions: [],
		skyboxModeAtRequest: null,
		drawModeAtRequest: null,
	};
	return {
		sceneFrameLayout: {},
		setSceneTargetMode(mode) {
			state.mode = mode;
			state.modeTransitions.push(mode);
		},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getSkyboxResources() {
			state.skyboxModeAtRequest = state.mode;
			return {
				pipeline: {},
				frameBinding: {},
			};
		},
		async getDrawResources() {
			state.drawModeAtRequest = state.mode;
			return null;
		},
		async renderParticles() {},
		_state: state,
	};
}

function createFrameContext(width, height) {
	return {
		camera: {},
		attachments: { width, height },
		features: {
			enableLighting: true,
			enableGamma: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableSkybox: false,
			enableOIT: false,
			enableSSAO: true,
			enableSSGI: false,
			enableTAA: true,
			enableSSR: false,
			enableVolumetric: false,
			enableFog: false,
			enableMotionBlur: false,
			enableDOF: false,
			enableBloom: false,
			enableClusteredLighting: false,
			enableFXAA: false,
			warnings: [],
			ssrOptions: {},
			ssaoOptions: {},
			ssgiOptions: {},
			taaOptions: {},
			volumetricOptions: {},
			fogOptions: {},
			motionBlurOptions: {},
			dofOptions: {},
			bloomOptions: {},
			clusteredLightingOptions: {},
		},
		shadowMaps: new Map(),
		scene: {
			particleSystems: [],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
		},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: {},
		transient: new Map(),
	};
}

function createOITBackend({ sampleCount = 1 } = {}) {
	const backend = new FakeBackend();
	backend.getMSAASampleCount = () => sampleCount;
	const originalCreateCommandEncoder =
		backend.createCommandEncoder.bind(backend);
	backend.nativeCopyCalls = [];
	backend.createCommandEncoder = () => {
		const encoder = originalCreateCommandEncoder();
		encoder.getNativeWebGPUCommandEncoder = () => ({
			copyTextureToTexture: (...args) => {
				backend.nativeCopyCalls.push(args);
			},
		});
		return encoder;
	};
	return backend;
}

function createOITSequencingResourcesStub() {
	const state = {
		events: [],
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
		setSceneTargetMode(mode) {
			state.events.push(`mode:${mode}`);
		},
		async buildClusteredLighting() {
			state.events.push("clustered:build");
		},
		renderShadows() {},
		async getSkyboxResources() {
			return null;
		},
		async getDrawResources(packet, options = {}) {
			state.events.push(
				`draw:${packet.id}:${options.transparentPipelineMode ?? "default"}`
			);
			return [drawResource];
		},
		async renderParticles(encoder, _context, targets, _mode, options = {}) {
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
		_state: state,
	};
}

async function testZeroSizedFrameSkipsEncoderAndLegacyDepthPath() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(0, 0);

	executor.beginFrame(context);
	assert.equal(backend.createCommandEncoderCalls, 0);

	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);
	await executor.endFrame();
	assert.equal(executor._texturePoolOwners.size, 0);
}

function testFrameTargetAllocationFailureReleasesPartialResources() {
	const backend = new FakeBackend();
	backend.failTextureAtCall = 4;
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());

	assert.throws(
		() => executor._ensureFrameTargets(64, 64, 2, 2),
		/simulated allocation failure/
	);
	assert.equal(executor._texturePoolOwners.size, 0);
	assert.equal(executor._frameTargets, null);
	assert.equal(executor._msaaTargets, null);
}

function testInvalidateFrameTargetsDestroysPresentBinding() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	executor._presentBinding = {
		destroy() {
			backend.bindingGroupDestroyCalls++;
		},
	};

	executor.invalidateFrameTargets();

	assert.equal(backend.bindingGroupDestroyCalls, 1);
	assert.equal(executor._presentBinding, null);
}

async function testLegacyMainPassForcesSingleSceneTargetMode() {
	const backend = new FakeBackend();
	const resources = createModeTrackingResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [{ id: "packet" }];

	executor.beginFrame(context);
	executor._frameTargets = null;

	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	assert.equal(resources._state.skyboxModeAtRequest, "single");
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

	assert.equal(backend.recordedRenderPasses.length >= 2, true);
	const depthClearPass = backend.recordedRenderPasses[0];
	assert.equal(depthClearPass.depthStencilAttachment.depthLoadOp, "load");
	assert.equal(depthClearPass.colorAttachments.length, 0);
	const mainPass = backend.recordedRenderPasses[1];
	assert.equal(mainPass.depthStencilAttachment.depthLoadOp, "load");
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
	executor._mrtEnabled = false;
	executor._frameTargets = null;

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
	]);
}

function testFrameTargetsIncludeAndReleaseOITResources() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	executor.beginFrame(context);
	assert.ok(executor._frameTargets);
	const { oitAccum, oitReveal, oitSceneColorCopy } = executor._frameTargets;
	assert.ok(oitAccum);
	assert.ok(oitReveal);
	assert.ok(oitSceneColorCopy);
	assert.equal(executor._texturePoolOwners.has(oitAccum), true);
	assert.equal(executor._texturePoolOwners.has(oitReveal), true);
	assert.equal(executor._texturePoolOwners.has(oitSceneColorCopy), true);

	executor.invalidateFrameTargets();
	assert.equal(executor._frameTargets, null);
	assert.equal(executor._texturePoolOwners.has(oitAccum), false);
	assert.equal(executor._texturePoolOwners.has(oitReveal), false);
	assert.equal(executor._texturePoolOwners.has(oitSceneColorCopy), false);
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
		"WebGPUOITDraw",
		"WebGPUParticlesOIT",
		"WebGPUOITResolvePass",
		"WebGPUTransmissionMRT",
		"WebGPUParticlesMRT_Additive",
	]);
	assert.ok(
		resources._state.events.includes("draw:transparent-oit:oit")
	);
	assert.ok(
		resources._state.events.includes(
			"draw:transparent-transmission:transmission"
		)
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
	assert.equal(backend.nativeCopyCalls.length >= 1, true);
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
		"WebGPUOITDraw",
		"WebGPUOITResolvePass",
		"WebGPUTransmissionMRT",
	]);
	assert.ok(
		resources._state.events.includes(
			"draw:transparent-transmission-only:transmission"
		)
	);
	assert.equal(
		resources._state.events.some((event) =>
			event.startsWith("particles:WebGPUParticlesOIT")
		),
		false
	);
	assert.equal(backend.nativeCopyCalls.length >= 1, true);
}

async function testOITMSAAFallsBackToLegacyAndWarns() {
	const backend = createOITBackend({ sampleCount: 4 });
	const resources = createModeTrackingResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
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
		assert.equal(executor._oitActive, false);
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

function testOITRuntimeFallbackWarnsWithoutNativeEncoder() {
	const backend = new FakeBackend();
	const resources = createModeTrackingResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
	const context = createFrameContext(64, 64);
	context.features.enableOIT = true;
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
		assert.equal(executor._oitActive, false);
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
	testFrameTargetAllocationFailureReleasesPartialResources();
	testInvalidateFrameTargetsDestroysPresentBinding();
	await testLegacyMainPassForcesSingleSceneTargetMode();
	await testIncrementalMainPassUsesDepthPartialReuse();
	await testLegacyMainPassScalesDirtyRectsToCanvasTarget();
	testFrameTargetsIncludeAndReleaseOITResources();
	await testOITTransparentAndParticleExecutionOrder();
	await testOITTransparentResolvesImmediatelyWithoutParticles();
	await testOITMSAAFallsBackToLegacyAndWarns();
	testOITRuntimeFallbackWarnsWithoutNativeEncoder();
	console.log("WebGPU frame executor resilience tests passed");
}

await run();
