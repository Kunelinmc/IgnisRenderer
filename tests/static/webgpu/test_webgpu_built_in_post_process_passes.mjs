import assert from "node:assert/strict";
import { createWebGPUComputeFacade } from "../../../src/backends/webgpu/ComputeFacade.ts";
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
	RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
	RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT,
	WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT,
} from "../../../src/backends/BackendExtensions.ts";
import { FakeWebGPUBackend as FakeBackend } from "../../helpers/fakes.mjs";
import {
	WebGPUFrameExecutor,
	createFrameContext,
	getFrameGraphDebugState,
} from "../../helpers/webgpu_frame_executor_resilience.mjs";
import { toShaderCompileError } from "../../../src/pipeline/WarmupPlanner.ts";

const particleRenderer = {
	async renderParticles() {
		return 0;
	},
};

const singleSampleResolver = {
	sampleCount: 1,
	resolveDomainSampleCount(domain, requestedSampleCount, formats) {
		return {
			domain,
			requestedSampleCount,
			sampleCount: 1,
			signature: `${domain}|${formats.join(",")}`,
			runtimeFallbackActive: false,
		};
	},
	fallbackToSingleSample() {
		return false;
	},
};

function createFrameExecutor(backend, resources) {
	backend.computeFacade = createWebGPUComputeFacade(backend);
	return new WebGPUFrameExecutor(
		backend,
		resources,
		singleSampleResolver,
		undefined,
		particleRenderer,
	);
}
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

async function createExecutorHarness(postProcessRequest = {
	taa: { enabled: true },
	ssr: { enabled: true },
	volumetric: { enabled: true },
}) {
	const backend = new FakeBackend();
	const frameResources = {
		scopeKey: "main",
		sceneTargetMode: "mrt",
		frameBinding: { id: "frame-binding" },
		decalFrameBinding: { id: "decal-frame-binding" },
		environmentBinding: null,
		clusteredSceneBinding: null,
		lightingState: { id: "lighting-state" },
		volumetricLightingData: { lights: [], warnings: [] },
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
	};
	const executor = createFrameExecutor(backend, resources);
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
			shadowCasterSubmissions: [],
			shadowTransmitterSubmissions: [],
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
	await executor.beginFrame(frameContext);
	return { executor, backend, frameContext };
}

async function testGammaOwnsWebGPUKernelBeforeRawPresent() {
	const { executor, backend, frameContext } = await createExecutorHarness({
		gamma: { enabled: true },
	});
	const gammaPass = BUILTIN_PASS_BY_ID.get("gamma");
	const request = createTemporalRequest({ frameContext });
	const passRequest = createExecutionContextRequest("gamma", request, {
		pass: gammaPass,
		implementation: gammaPass.getImplementation("webgpu"),
	});
	const context = executor.frameRuntime.postProcess
		.createPassExecutionContext(passRequest);
	const targets = getFrameGraphDebugState(executor).frameTargets;

	assert.equal(
		getFrameGraphDebugState(executor).targetManager.needsPostProcessTargets,
		true
	);
	assert.ok(context.encoder);
	assert.ok(context.shared);
	assert.ok(context.resources.color.input);
	assert.ok(context.resources.color.output);

	const result = await passRequest.implementation.execute(passRequest, context);
	executor.frameRuntime.postProcess.completePass(passRequest, result);
	assert.deepEqual(result, { ran: true });
	const gammaParams = backend.buffers.find((buffer) => buffer.label === "WebGPUGammaParams");
	assert.equal(gammaParams.size, 8);
	assert.deepEqual(Array.from(gammaParams.lastWrite), [0, 4]);
	assert.equal(backend.writeCalls.findLast((call) => call[1] === gammaParams)[2].byteLength, 8);
	assert.equal(
		getFrameGraphDebugState(executor).frameTargets.sceneColor,
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
		true
	);
	const presentBinding = backend.bindingGroups.find(
		(group) => group.label === "WebGPUPresentBinding"
	);
	assert.ok(presentBinding);
	assert.equal(presentBinding.entries.length, 3);
}

async function testTemporalExecutePassUsesPipelineHistories() {
	const { executor } = await createExecutorHarness();

	const taaRequest = createTemporalRequest({
		histories: {
			motion: {
				valid: false,
				read: { resource: { id: "motion-read" } },
				write: { resource: { id: "motion-write" } },
			},
		},
	});
	const taaContext = executor.frameRuntime.postProcess.createPassExecutionContext(
		createExecutionContextRequest("taa", taaRequest)
	);
	assert.equal("historyRead" in getFrameGraphDebugState(executor).frameTargets, false);
	assert.equal(taaContext.resources.getHistory("taa").read.id, "taa-read");
	assert.equal(taaContext.resources.getHistory("taa").write.id, "taa-write");
	assert.equal(taaContext.resources.getHistory("motion").read.id, "motion-read");
	assert.equal(taaContext.resources.getHistory("motion").write.id, "motion-write");
	taaContext.resources.copyGBufferToHistory("motion", "motion");

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
	const ssrContext = executor.frameRuntime.postProcess.createPassExecutionContext(
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
	const ssgiContext = executor.frameRuntime.postProcess.createPassExecutionContext(
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
	const ssrefractionContext = executor.frameRuntime.postProcess.createPassExecutionContext(
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

	const volumetricContext = executor.frameRuntime.postProcess.createPassExecutionContext(
		createExecutionContextRequest("volumetric", ssrRequest)
	);
	assert.deepEqual(volumetricContext.frameBinding, { id: "frame-binding" });
	assert.deepEqual(volumetricContext.volumetricLightingData, {
		lights: [], warnings: [],
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

async function testCustomImplementationUsesFixedContext() {
	const { executor } = await createExecutorHarness();
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
	const context = executor.frameRuntime.postProcess
		.createPassExecutionContext(passRequest);

	assert.ok(context.encoder);
	assert.equal(Object.isFrozen(context.targets), true);
	assert.ok(context.shared);
	assert.deepEqual(context.frameBinding, { id: "frame-binding" });
	assert.equal(context.resources.getHistory("taa").write.id, "taa-write");
	const targets = getFrameGraphDebugState(executor).frameTargets;
	assert.equal(targets.sceneColor, targets.sceneColorMain);
	executor.frameRuntime.postProcess.completePass(passRequest, { ran: true });
	assert.equal(
		getFrameGraphDebugState(executor).frameTargets.sceneColor,
		context.resources.color.output
	);
	executor.abortFrame();
}

function createWarmupHarness(resourcePhases = []) {
	const backend = new FakeBackend();
	let resourceRequest;
	const resources = {
		sceneFrameLayout: {},
		createFrameScope() {
			return {
				prepare() { throw new Error("warmup must not prepare a graph frame"); },
				destroy() {},
			};
		},
		async warmup(context, plan, options, framePackets, runtimeOptions) {
			resourceRequest = { context, plan, options, framePackets, runtimeOptions };
			return resourcePhases;
		},
	};
	const executor = createFrameExecutor(backend, resources);
	const context = createFrameContext(4, 4);
	context.postProcess = createResolvedPostProcess({
		tonemap: { enabled: true },
		gamma: { enabled: true },
	}, "webgpu");
	return {
		backend,
		resources,
		runtime: executor.frameRuntime,
		context,
		getResourceRequest: () => resourceRequest,
	};
}

async function testWarmupAggregatesFeaturePhasesAndProgress() {
	const compileError = toShaderCompileError(new Error("shadow compile failure"),
		"webgpu", "WebGPUShadowWarmup");
	const { backend, resources, runtime, context, getResourceRequest } = createWarmupHarness([
		{ phase: "scene", total: 3, compiled: 2, skipped: 1, failed: 0, errors: [] },
		{ phase: "shadows", total: 1, compiled: 0, skipped: 0, failed: 1,
			errors: [compileError] },
	]);
	const progress = [];
	const events = [];
	const warmupResources = resources.warmup.bind(resources);
	resources.warmup = async (...args) => {
		events.push("resources");
		return warmupResources(...args);
	};
	const options = {
		includeParticles: false,
		scheduling: "idle",
		yieldIntervalMs: 0,
		onProgress(update) {
			progress.push(update);
			events.push(update.phase);
		},
	};
	const report = await runtime.warmup(context, options, 4);
	assert.equal(report.backend, "webgpu");
	assert.equal(report.total, 7);
	assert.equal(report.compiled, 5);
	assert.equal(report.skipped, 1);
	assert.equal(report.failed, 1);
	assert.deepEqual(report.errors, [compileError]);
	assert.deepEqual(report.phases.map((phase) => phase.phase), [
		"webgpu-frame", "scene", "shadows", "webgpu-reflection",
	]);
	assert.deepEqual(progress, [
		{ phase: "webgpu-frame", completed: 3, total: 3 },
		{ phase: "scene", completed: 3, total: 3 },
		{ phase: "shadows", completed: 1, total: 1 },
		{ phase: "webgpu-reflection", completed: 0, total: 0 },
	]);
	assert.deepEqual(events, [
		"webgpu-frame", "resources", "scene", "shadows", "webgpu-reflection",
	]);
	assert.ok(report.finishedAt >= report.startedAt);
	assert.equal(report.durationMs, report.finishedAt - report.startedAt);
	const request = getResourceRequest();
	assert.equal(request.context, context);
	assert.equal(request.options, options);
	assert.deepEqual(request.plan.postProcessPasses, ["tonemap", "gamma"]);
	assert.equal(request.plan.enableParticles, false);
	assert.deepEqual(request.framePackets.all, []);
	assert.deepEqual(request.runtimeOptions, {
		enableEarlyZPrepass: true,
		enableDeferredLighting: true,
		sampleCount: 4,
	});
	assert.equal(backend.textures.length, 0);
	assert.equal(backend.submits, 0);
}

async function testWarmupReportsPostProcessCompilationFailures() {
	const { backend, runtime, context } = createWarmupHarness();
	const failure = new Error("gamma compile failure");
	const createComputePipeline = backend.createComputePipeline.bind(backend);
	backend.createComputePipeline = async (desc) => {
		if (desc.label === "WebGPUGammaPipeline") throw failure;
		return createComputePipeline(desc);
	};
	const report = await runtime.warmup(context, {}, 1);
	assert.equal(report.total, 3);
	assert.equal(report.compiled, 2);
	assert.equal(report.failed, 1);
	assert.equal(report.errors[0].cause, failure);
	assert.deepEqual(report.phases.map((phase) => phase.phase), [
		"webgpu-frame", "webgpu-reflection",
	]);
}

async function testWarmupRejectsSetupFailures() {
	const { backend, runtime, context } = createWarmupHarness();
	const failure = new Error("warmup planning failed");
	backend.postProcessRuntime.planWarmup = () => { throw failure; };
	await assert.rejects(runtime.warmup(context, {}, 1), (error) => error === failure);
	assert.equal(backend.pipelines.length, 0);
}

async function testWarmupRetainsCompletedPhasesAfterUnexpectedFailure() {
	const { resources, runtime, context } = createWarmupHarness();
	const failure = new Error("resource warmup failed");
	resources.warmup = async () => { throw failure; };
	const progress = [];
	const report = await runtime.warmup(context, {
		onProgress: (update) => progress.push(update),
	}, 1);
	assert.equal(report.total, 4);
	assert.equal(report.compiled, 3);
	assert.equal(report.failed, 1);
	assert.equal(report.errors[0].cause, failure);
	assert.deepEqual(report.phases.map((phase) => phase.phase), [
		"webgpu-frame", "webgpu-warmup",
	]);
	assert.deepEqual(progress, [
		{ phase: "webgpu-frame", completed: 3, total: 3 },
		{ phase: "webgpu-warmup", completed: 1, total: 1 },
	]);
}

async function testWarmupHintsFollowPlanPostProcessPasses() {
	const backend = new FakeBackend();
	const resources = {
		sceneFrameLayout: {},
		createFrameScope() {
			return {
				prepare() { throw new Error("not used by warmup test"); },
				destroy() {},
			};
		},
	};
	const executor = createFrameExecutor(backend, resources);
	const postProcess = executor.frameRuntime.modules.modules.find(
		(module) => module.id === "post-process",
	);

	const emptyWarmup = await postProcess.warmup(
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
	const taaWarmup = await postProcess.warmup(
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

async function testBackendPostProcessSurfaceKeepsOnlyExecutorBridge() {
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
}

async function testWebGPUOcclusionExtensionDescriptor() {
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

async function testProbeCaptureExtensionIsRemoved() {
	const backend = new WebGPUBackend();
	backend.attach({
		surface: { canvas: {} },
		events: { emit: () => {} },
	});
	const extension = backend.extensions.getBackendExtension({
		id: "renderer.probe-capture",
	});
	assert.equal(extension, null);
}

async function testSSRRequirementsExposeMaterialChannels() {
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
	await testCustomImplementationUsesFixedContext();
	await testWarmupAggregatesFeaturePhasesAndProgress();
	await testWarmupReportsPostProcessCompilationFailures();
	await testWarmupRejectsSetupFailures();
	await testWarmupRetainsCompletedPhasesAfterUnexpectedFailure();
	await testWarmupHintsFollowPlanPostProcessPasses();
	await testBackendPostProcessSurfaceKeepsOnlyExecutorBridge();
	await testWebGPUOcclusionExtensionDescriptor();
	await testProbeCaptureExtensionIsRemoved();
	await testSSRRequirementsExposeMaterialChannels();
	console.log("WebGPU post-process executor tests passed");
}

await run();
