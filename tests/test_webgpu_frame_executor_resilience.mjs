import assert from "node:assert/strict";
import { WebGPUFrameExecutor } from "../src/renderers/webgpu/WebGPUFrameExecutor.ts";
import { Logger } from "../src/foundation/Logger.ts";
import { Camera } from "../src/cameras/Camera.ts";
import { Material } from "../src/materials/Material.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";

import { FakeWebGPUBackend as FakeBackend } from "./helpers/test_fakes.mjs";
import { createResolvedPostProcess } from "./helpers/postprocess.mjs";

function createResourcesStub() {
	return {
		sceneFrameLayout: {},
		setSceneTargetMode() {},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getEnvironmentResources() {
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
		environmentModeAtRequest: null,
		drawModeAtRequest: null,
		drawPipelineModeAtRequest: null,
	};
	return {
		sceneFrameLayout: {},
		setSceneTargetMode(mode) {
			state.mode = mode;
			state.modeTransitions.push(mode);
		},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getEnvironmentResources() {
			state.environmentModeAtRequest = state.mode;
			return {
				pipeline: {},
				frameBinding: {},
			};
		},
		async getDrawResources(_packet, options = {}) {
			state.drawModeAtRequest = state.mode;
			state.drawPipelineModeAtRequest = options.drawMode ?? "default";
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
		}),
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
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources(packet, options = {}) {
			state.events.push(
				`draw:${packet.id}:${options.transparentPipelineMode ?? "default"}:${options.drawMode ?? "default"}`
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
		setSceneTargetMode() {},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources(packet, options = {}) {
			state.events.push(
				`draw:${packet.id}:${options.sceneTargetMode ?? "none"}:${options.drawMode ?? "default"}`
			);
			return [drawResource];
		},
		async renderParticles() {},
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
		prepareContexts: [],
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
		setSceneTargetMode(mode) {
			state.events.push(`mode:${mode}`);
		},
		prepareFrame(context) {
			state.prepareContexts.push(context);
			state.events.push(
				`prepare:reflection:${context.features.enableReflection}:ssr:${context.postProcess.isEnabled("ssr")}:opaque:${context.scene.opaquePackets.map((packet) => packet.id).join(",")}`
			);
		},
		async buildClusteredLighting() {
			state.events.push("clustered:build");
		},
		renderShadows() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources(packet, options = {}) {
			state.events.push(
				`draw:${packet.id}:${options.sceneTargetMode ?? "default"}:${options.drawMode ?? "default"}`
			);
			return [drawResource];
		},
		async renderParticles() {},
		getPlanarReflectionLayout() {
			return { id: "planar-reflection-layout" };
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
		() => executor._ensureFrameTargets(64, 64, false),
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
		["setScissorRect", 611, 0, 612, 869],
	]);
}

function testFrameTargetsIncludeAndReleaseOITResources() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	executor.beginFrame(context);
	assert.ok(executor._frameTargets);
	const {
		oitAccum,
		oitReveal,
		oitSceneColorCopy,
		planarReflectionMask,
	} = executor._frameTargets;
	assert.ok(oitAccum);
	assert.ok(oitReveal);
	assert.ok(oitSceneColorCopy);
	assert.ok(planarReflectionMask);
	assert.equal(executor._texturePoolOwners.has(oitAccum), true);
	assert.equal(executor._texturePoolOwners.has(oitReveal), true);
	assert.equal(executor._texturePoolOwners.has(oitSceneColorCopy), true);
	assert.equal(executor._texturePoolOwners.has(planarReflectionMask), true);

	executor.invalidateFrameTargets();
	assert.equal(executor._frameTargets, null);
	assert.equal(executor._texturePoolOwners.has(oitAccum), false);
	assert.equal(executor._texturePoolOwners.has(oitReveal), false);
	assert.equal(executor._texturePoolOwners.has(oitSceneColorCopy), false);
	assert.equal(executor._texturePoolOwners.has(planarReflectionMask), false);
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
	const firstTargets = executor._frameTargets;
	const firstTextureCount = backend.createTextureCalls.length;
	await executor.endFrame();

	const secondContext = createFrameContext(64, 64);
	secondContext.postProcess = createResolvedPostProcess({
		ssao: { enabled: true, options: { downsample: 4 } },
		ssr: { enabled: true, options: { downsample: 4 } },
	});
	executor.beginFrame(secondContext);
	assert.strictEqual(executor._frameTargets, firstTargets);
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
	context.camera = camera;
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
		resources._state.events.includes("draw:object:mrt:reflection-capture")
	);
	assert.ok(
		resources._state.events.includes(
			"draw:mirror:mrt:planar-reflection-composite"
		)
	);
	const labels = backend.recordedRenderPasses.map((pass) => pass.label);
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
		resources._state.events.includes("draw:transparent-oit:oit:default")
	);
	assert.ok(
		resources._state.events.includes(
			"draw:transparent-transmission:transmission:default"
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
			"draw:transparent-transmission-only:transmission:default"
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
	backend.isDeferredLightingEnabled = () => false;
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
	await testMainOpaqueDisablesEarlyZWhenConfiguredOff();
	await testLegacyMainPassScalesDirtyRectsToCanvasTarget();
	testFrameTargetsIncludeAndReleaseOITResources();
	await testFrameTargetReuseIgnoresPostProcessDownsampleOptions();
	await testPlanarReflectionCaptureAndCompositeSequencing();
	await testOITTransparentAndParticleExecutionOrder();
	await testOITTransparentResolvesImmediatelyWithoutParticles();
	await testDeferredLightingBindsUnusedGroupOnePlaceholder();
	await testDeferredLightingKeepsTransmissionOutOfGBuffer();
	await testDeferredLightingCanBeExplicitlyDisabled();
	testDeferredLightingWarnsWhenRequestedButMRTUnavailable();
	await testOITMSAAFallsBackToLegacyAndWarns();
	testOITRuntimeFallbackWarnsWithoutNativeEncoder();
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
