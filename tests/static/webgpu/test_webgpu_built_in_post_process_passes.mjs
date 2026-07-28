import assert from "node:assert/strict";
import {
	GammaPass,
	ScreenSpaceGlobalIlluminationPass,
	ScreenSpaceRefractionsPass,
	ScreenSpaceReflectionsPass,
	TemporalAntiAliasingPass,
	VolumetricLightingPass,
} from "../../../src/postprocess/index.ts";
import { WebGPUBackend } from "../../../src/backends/webgpu/WebGPUBackend.ts";
import {
	PROBE_CAPTURE_EXTENSION,
	RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
	RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT,
	WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT,
} from "../../../src/backends/BackendExtensions.ts";
import { WebGPUFrameOrchestrator as WebGPUFrameExecutor } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameOrchestrator.ts";
import { WebGPUFrameFeatureDataStore } from "../../../src/backends/webgpu/FrameFeatures.ts";
import { WEBGPU_VOLUMETRIC_LIGHTING_DATA } from "../../../src/backends/webgpu/WebGPUFrameFeatureModules.ts";
import { FakeWebGPUBackend as FakeBackend } from "../../helpers/fakes.mjs";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

const BUILTIN_PASS_BY_ID = new Map([
	["gamma", new GammaPass({ enabled: true })],
	["taa", new TemporalAntiAliasingPass({ enabled: true })],
	["ssgi", new ScreenSpaceGlobalIlluminationPass({ enabled: true })],
	["ssr", new ScreenSpaceReflectionsPass({ enabled: true })],
	["ssrefraction", new ScreenSpaceRefractionsPass({ enabled: true })],
	["volumetric", new VolumetricLightingPass({ enabled: true })],
]);

function createTemporalRequest(overrides = {}) {
	const frameContext = overrides.frameContext ?? {
		postProcess: {
			enabled: {
				gamma: true,
			},
			options: {},
		},
		transient: new Map(),
	};
	const histories = {
		taa: {
			valid: true,
			read: { resource: { id: "taa-read" } },
			write: { resource: { id: "taa-write" } },
		},
		ssr: {
			valid: true,
			read: { resource: { id: "ssr-read" } },
			write: { resource: { id: "ssr-write" } },
		},
		ssgi: {
			valid: true,
			read: { resource: { id: "ssgi-read" } },
			write: { resource: { id: "ssgi-write" } },
		},
		volumetric: {
			valid: true,
			read: { resource: { id: "vol-read" } },
			write: { resource: { id: "vol-write" } },
		},
		"volumetric-reservoir": {
			valid: true,
			read: { resource: { id: "res-read" } },
			write: { resource: { id: "res-write" } },
		},
		motion: {
			valid: true,
			read: { resource: { id: "motion-read" } },
			write: { resource: { id: "motion-write" } },
		},
		...(overrides.histories ?? {}),
	};

	return {
		...overrides,
		frameContext,
		histories,
		gBuffer: overrides.gBuffer ?? { channels: {} },
		transients: overrides.transients ?? {},
		options: overrides.options ?? {},
		startPassId: null,
	};
}

function createExecutionContextRequest(passId, request, overrides = {}) {
	const pass =
		overrides.pass ??
		BUILTIN_PASS_BY_ID.get(passId) ?? {
			id: passId,
			builtIn: true,
		};
	const implementation =
		overrides.implementation ??
		pass.getImplementation?.("webgpu") ?? {
			id: `${passId}:test`,
		};
	const options = request.options ?? pass.normalizeOptions?.({}) ?? {};
	const declaration = overrides.declaration ?? implementation.describeExecution?.({
		frameContext: request.frameContext,
		postProcess: request.postProcess,
		backend: "webgpu",
		gBuffer: request.gBuffer,
		width: request.frameContext?.attachments?.width ?? 64,
		height: request.frameContext?.attachments?.height ?? 64,
		options,
	}) ?? {
		color: { access: "read", output: "new-version" },
		histories: [],
		transients: [],
		gBuffer: [],
		shared: [],
	};
	return {
		...request,
		passId,
		pass,
		implementation,
		options,
		declaration,
	};
}

function createExecutorHarness(postProcessRequest = {
	taa: { enabled: true },
	ssr: { enabled: true },
	volumetric: { enabled: true },
}) {
	const backend = new FakeBackend();
	const featureData = new WebGPUFrameFeatureDataStore();
	featureData.set(WEBGPU_VOLUMETRIC_LIGHTING_DATA, {
		id: "volumetric-lighting-data",
	});
	const frameResources = {
		scopeKey: "main",
		sceneTargetMode: "mrt",
		frameBinding: { id: "frame-binding" },
		decalFrameBinding: { id: "decal-frame-binding" },
		environmentBinding: null,
		clusteredSceneBinding: null,
		lightingState: { id: "lighting-state" },
		featureData,
		featureState: {},
		environmentState: {},
		jointMatrixMap: new Map(),
		morphWeightMap: new Map(),
	};
	const resources = {
		sceneFrameLayout: {},
		createFrameScope() {
			return {
				prepare() { return frameResources; },
				updateParticleShadowVolumes() {},
				destroy() {},
			};
		},
		prepareFrame() {
			return frameResources;
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
	};
	const executor = new WebGPUFrameExecutor(backend, resources);
	const frameContext = {
		camera: {},
		attachments: { width: 64, height: 64 },
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
		postProcess: createResolvedPostProcess(postProcessRequest, "webgpu"),
		shadowMaps: new Map(),
		scene: {
			particleSystems: [],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
			decalPackets: [],
			spatialIndex: null,
		},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: {},
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [],
		},
		transient: new Map(),
	};
	executor.beginFrame(frameContext);
	return { executor, backend, frameContext };
}

async function testGammaOwnsWebGPUKernelBeforeRawPresent() {
	const { executor, backend, frameContext } = createExecutorHarness({
		gamma: { enabled: true },
	});
	const gammaPass = BUILTIN_PASS_BY_ID.get("gamma");
	const request = createTemporalRequest({ frameContext });
	const passRequest = createExecutionContextRequest("gamma", request, {
		pass: gammaPass,
		implementation: gammaPass.getImplementation("webgpu"),
	});
	const context = executor.createPassExecutionContext(passRequest);
	const targets = executor.getDebugState().frameTargets;

	assert.equal(
		executor.getDebugState().targetManager.needsPostProcessTargets,
		true
	);
	assert.ok(context.encoder);
	assert.ok(context.shared);
	assert.ok(context.resources.color.input);
	assert.ok(context.resources.color.output);

	const result = await passRequest.implementation.execute(passRequest, context);
	executor.completePostProcessPass(passRequest, result);
	assert.deepEqual(result, { ran: true });
	assert.equal(
		executor.getDebugState().frameTargets.sceneColor,
		targets.postPong
	);
	assert.equal(
		backend.computePipelines.some(
			(pipeline) => pipeline.label === "WebGPUGammaPipeline"
		),
		true
	);

	await executor.endFrame();
	assert.equal(
		backend.buffers.some((buffer) => buffer.label === "WebGPUPresentParams"),
		false
	);
	const presentBinding = backend.bindingGroups.find(
		(group) => group.label === "WebGPUPresentBinding"
	);
	assert.ok(presentBinding);
	assert.equal(presentBinding.entries.length, 2);
}

async function testTemporalExecutePassUsesPipelineHistories() {
	const { executor } = createExecutorHarness();

	const taaRequest = createTemporalRequest({
		histories: {
			motion: {
				valid: false,
				read: { resource: { id: "motion-read" } },
				write: { resource: { id: "motion-write" } },
			},
		},
	});
	const taaContext = executor.createPassExecutionContext(
		createExecutionContextRequest("taa", taaRequest)
	);
	assert.equal("historyRead" in executor.getDebugState().frameTargets, false);
	assert.equal(taaContext.resources.getHistory("taa").read.id, "taa-read");
	assert.equal(taaContext.resources.getHistory("taa").write.id, "taa-write");
	assert.equal(taaContext.resources.getHistory("motion").read.id, "motion-read");
	assert.equal(taaContext.resources.getHistory("motion").write.id, "motion-write");
	taaContext.resources.copyGBufferToHistory("motion", "motion");
	assert.equal(
		executor.getDebugState().motionHistoryWriteTarget.id,
		"motion-write"
	);

	const ssrRequest = createTemporalRequest({
		transients: {
			"ssr:denoise-a": {
				handle: { resource: { id: "ssr-denoise-a" } },
			},
			"ssr:denoise-b": {
				handle: { resource: { id: "ssr-denoise-b" } },
			},
		},
	});
	const ssrContext = executor.createPassExecutionContext(
		createExecutionContextRequest("ssr", ssrRequest)
	);
	assert.deepEqual(ssrContext.frameBinding, { id: "frame-binding" });
	assert.equal(ssrContext.resources.getHistory("ssr").read.id, "ssr-read");
	assert.equal(ssrContext.resources.getHistory("ssr").write.id, "ssr-write");
	assert.equal(ssrContext.resources.getHistory("motion").read.id, "motion-read");
	assert.equal(ssrContext.resources.getHistory("motion").write.id, "motion-write");
	assert.equal(
		ssrContext.resources.getTransient("ssr:denoise-a").id,
		"ssr-denoise-a"
	);
	assert.equal(
		ssrContext.resources.getTransient("ssr:denoise-b").id,
		"ssr-denoise-b"
	);

	const ssgiRequest = createTemporalRequest({
		transients: {
			"ssgi:denoise-a": {
				handle: { resource: { id: "ssgi-denoise-a" } },
			},
			"ssgi:denoise-b": {
				handle: { resource: { id: "ssgi-denoise-b" } },
			},
		},
	});
	const ssgiContext = executor.createPassExecutionContext(
		createExecutionContextRequest("ssgi", ssgiRequest)
	);
	assert.deepEqual(ssgiContext.frameBinding, { id: "frame-binding" });
	assert.equal(ssgiContext.resources.getHistory("ssgi").read.id, "ssgi-read");
	assert.equal(ssgiContext.resources.getHistory("ssgi").write.id, "ssgi-write");
	assert.equal(ssgiContext.resources.getHistory("motion").read.id, "motion-read");
	assert.equal(ssgiContext.resources.getHistory("motion").write.id, "motion-write");
	assert.equal(
		ssgiContext.resources.getTransient("ssgi:denoise-a").id,
		"ssgi-denoise-a"
	);
	assert.equal(
		ssgiContext.resources.getTransient("ssgi:denoise-b").id,
		"ssgi-denoise-b"
	);

	const ssrefractionRequest = createTemporalRequest({
		transients: {
			"ssrefraction:raw": {
				handle: { resource: { id: "ssrefraction-raw" } },
			},
			"ssrefraction:denoise-scratch": {
				handle: { resource: { id: "ssrefraction-denoise-scratch" } },
			},
		},
	});
	const ssrefractionContext = executor.createPassExecutionContext(
		createExecutionContextRequest("ssrefraction", ssrefractionRequest)
	);
	assert.deepEqual(ssrefractionContext.frameBinding, { id: "frame-binding" });
	assert.equal(
		ssrefractionContext.resources.getTransient("ssrefraction:raw").id,
		"ssrefraction-raw"
	);
	assert.equal(
		ssrefractionContext.resources
			.getTransient("ssrefraction:denoise-scratch").id,
		"ssrefraction-denoise-scratch"
	);
	assert.throws(
		() => ssrefractionContext.resources.getShared("backend:frame-hiz"),
		/missing required shared resource/
	);

	const volumetricContext = executor.createPassExecutionContext(
		createExecutionContextRequest("volumetric", ssrRequest)
	);
	assert.deepEqual(volumetricContext.frameBinding, { id: "frame-binding" });
	assert.deepEqual(volumetricContext.getFrameData(WEBGPU_VOLUMETRIC_LIGHTING_DATA), {
		id: "volumetric-lighting-data",
	});
	assert.ok(volumetricContext.shared);
	assert.equal(volumetricContext.resources.getHistory("volumetric").read.id, "vol-read");
	assert.equal(volumetricContext.resources.getHistory("volumetric").write.id, "vol-write");
	assert.equal(
		volumetricContext.resources.getHistory("volumetric-reservoir").read.id,
		"res-read"
	);
	assert.equal(
		volumetricContext.resources.getHistory("volumetric-reservoir").write.id,
		"res-write"
	);
	assert.equal(volumetricContext.resources.getHistory("motion").read.id, "motion-read");
	assert.equal(volumetricContext.resources.getHistory("motion").write.id, "motion-write");
	executor.abortFrame();
}

function testCustomImplementationUsesFixedContext() {
	const { executor } = createExecutorHarness();
	const request = createTemporalRequest();
	const passRequest = createExecutionContextRequest("custom-webgpu", request, {
			pass: {
				id: "custom-webgpu",
				builtIn: false,
			},
			implementation: {
				id: "custom-webgpu:test",
				describeExecution: () => ({
					color: { access: "read", output: "new-version" },
					histories: [{
						descriptor: { id: "taa", format: "rgba16float" },
						write: [{ access: "write", usage: "storage" }],
					}],
				}),
				execute: () => ({ ran: true }),
			},
		});
	const context = executor.createPassExecutionContext(passRequest);

	assert.ok(context.encoder);
	assert.equal(Object.isFrozen(context.targets), true);
	assert.ok(context.shared);
	assert.deepEqual(context.frameBinding, { id: "frame-binding" });
	assert.equal(context.resources.getHistory("taa").write.id, "taa-write");
	assert.equal("customHistoryWrite" in context, false);
	assert.equal("publishColorTarget" in context, false);
	const targets = executor.getDebugState().frameTargets;
	assert.equal(targets.sceneColor, targets.sceneColorMain);
	executor.completePostProcessPass(passRequest, { ran: true });
	assert.equal(
		executor.getDebugState().frameTargets.sceneColor,
		context.resources.color.output
	);
	executor.abortFrame();
}

async function testWarmupHintsFollowPlanPostProcessPasses() {
	const backend = new FakeBackend();
	const resources = {
		sceneFrameLayout: {},
		createFrameScope() {
			return {
				prepare() { throw new Error("not used by warmup test"); },
				updateParticleShadowVolumes() {},
				destroy() {},
			};
		},
	};
	const executor = new WebGPUFrameExecutor(backend, resources);

	const emptyWarmup = await executor.warmup(
		{
			transient: new Map(),
			postProcess: {
				getEnabledPasses: () => [],
				getOptions: () => undefined,
			},
		},
		{
			materials: [],
			shaderMaterials: [],
			enableEnvironment: false,
			enableShadows: false,
			enableParticles: false,
			includePostProcess: true,
			postProcessPasses: ["ssr", "volumetric", "ssr", "gamma", "custom-pass"],
			sceneTargetMode: "mrt",
		}
	);

	assert.equal(emptyWarmup.failed, 0);

	const taaPass = new TemporalAntiAliasingPass({ enabled: true });
	const taaWarmup = await executor.warmup(
		{
			transient: new Map(),
			postProcess: {
				getEnabledPasses: () => [{ pass: taaPass }],
				getOptions: () => undefined,
			},
		},
		{
			materials: [],
			shaderMaterials: [],
			enableEnvironment: false,
			enableShadows: false,
			enableParticles: false,
			includePostProcess: true,
			postProcessPasses: ["taa", "custom-pass"],
			sceneTargetMode: "mrt",
		}
	);

	assert.equal(taaWarmup.failed, 0);
	assert.equal(taaWarmup.compiled > 0, true);
}

function testBackendPostProcessSurfaceKeepsOnlyExecutorBridge() {
	const backend = new WebGPUBackend();
	backend.attach({
		surface: { canvas: {} },
		events: { emit: () => {} },
	});
	assert.equal(backend.profile.capabilities.postProcess, true);
	assert.equal(
		backend.extensions
			.listExtensions()
			.some((extension) => extension.id === "renderer.postprocess"),
		false
	);
	assert.equal("postProcessAdapter" in backend, false);
	assert.equal("postProcessExecutor" in backend, false);
	assert.equal("createPostProcessGBufferBridge" in backend, false);
	assert.equal("postProcess" in backend, false);
}

function testWebGPUOcclusionExtensionDescriptor() {
	const backend = new WebGPUBackend();
	backend.attach({
		surface: { canvas: {} },
		events: { emit: () => {} },
	});
	const extension = backend.extensions.getExtension(
		RENDERER_OCCLUSION_CULLING_EXTENSION_ID
	);
	assert.ok(extension);
	assert.deepEqual(extension.insertionPoints, [
		RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT,
		WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT,
	]);
	assert.equal(typeof extension.api.getVisibilityProvider, "function");
	assert.equal(typeof extension.api.resetOcclusionCulling, "function");
}

async function testProbeCaptureRemainsExtensionOnly() {
	const backend = new WebGPUBackend();
	backend.attach({
		surface: { canvas: {} },
		events: { emit: () => {} },
	});
	const extension = backend.extensions.requireBackendExtension(
		PROBE_CAPTURE_EXTENSION
	);

	assert.equal(await extension.captureProbeFace({}), null);
	assert.equal("captureProbeFace" in backend, false);
}

function testSSRRequirementsExposeMaterialChannels() {
	const pass = new ScreenSpaceReflectionsPass({ enabled: true });
	const options = pass.normalizeOptions({});
	const declaration = pass.getImplementation("webgpu").describeExecution({
		backend: "webgpu",
		options,
	});
	assert.deepEqual(declaration.gBuffer.map((entry) => entry.semantic), [
		"depth",
		"normal",
		"roughness",
		"metallic",
		"motion",
	]);
	pass.destroy();
}

async function run() {
	await testGammaOwnsWebGPUKernelBeforeRawPresent();
	await testTemporalExecutePassUsesPipelineHistories();
	testCustomImplementationUsesFixedContext();
	await testWarmupHintsFollowPlanPostProcessPasses();
	testBackendPostProcessSurfaceKeepsOnlyExecutorBridge();
	testWebGPUOcclusionExtensionDescriptor();
	await testProbeCaptureRemainsExtensionOnly();
	testSSRRequirementsExposeMaterialChannels();
	console.log("WebGPU post-process executor tests passed");
}

await run();
