import assert from "node:assert/strict";
import {
	GammaPass,
	ScreenSpaceRefractionsPass,
	ScreenSpaceReflectionsPass,
	TemporalAntiAliasingPass,
	VolumetricLightingPass,
} from "../../../src/postprocess/index.ts";
import { WebGPUBackend } from "../../../src/renderers/WebGPUBackend.ts";
import {
	RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
	RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT,
	WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT,
} from "../../../src/renderers/BackendExtensions.ts";
import { WebGPUFrameExecutor } from "../../../src/renderers/webgpu/WebGPUFrameExecutor.ts";
import { WebGPUFrameFeatureDataStore } from "../../../src/renderers/webgpu/FrameFeatures.ts";
import { WEBGPU_VOLUMETRIC_LIGHTING_DATA } from "../../../src/renderers/webgpu/WebGPUFrameFeatureModules.ts";
import { FakeWebGPUBackend as FakeBackend } from "../../helpers/fakes.mjs";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

const BUILTIN_PASS_BY_ID = new Map([
	["gamma", new GammaPass({ enabled: true })],
	["taa", new TemporalAntiAliasingPass({ enabled: true })],
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
	return {
		...request,
		passId,
		pass,
		implementation,
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
	const context = executor.getPassExecutionContext(passRequest);
	const targets = executor.getFrameGraphDebugState().frameTargets;

	assert.equal(
		executor.getFrameGraphDebugState().targetManager.needsPostProcessTargets,
		true
	);
	assert.ok(context.encoder);
	assert.ok(context.shared);
	assert.equal(typeof context.publishColorTarget, "function");

	const result = await passRequest.implementation.execute(passRequest, context);
	executor.completePostProcessPass(passRequest, result);
	assert.deepEqual(result, { ran: true });
	assert.equal(
		executor.getFrameGraphDebugState().frameTargets.sceneColor,
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
	const taaContext = executor.getPassExecutionContext(
		createExecutionContextRequest("taa", taaRequest)
	);
	assert.equal("historyRead" in executor.getFrameGraphDebugState().frameTargets, false);
	assert.equal(taaContext.historyRead.id, "taa-read");
	assert.equal(taaContext.historyWrite.id, "taa-write");
	assert.equal(taaContext.motionHistoryRead.id, "motion-read");
	assert.equal(taaContext.motionHistoryWrite.id, "motion-write");
	taaContext.writeMotionHistoryFromCurrent();
	assert.equal(
		executor.getFrameGraphDebugState().motionHistoryWriteTarget.id,
		"motion-write"
	);

	const ssrRequest = createTemporalRequest();
	const ssrContext = executor.getPassExecutionContext(
		createExecutionContextRequest("ssr", ssrRequest)
	);
	assert.deepEqual(ssrContext.frameBinding, { id: "frame-binding" });
	assert.equal(ssrContext.historyRead.id, "ssr-read");
	assert.equal(ssrContext.historyWrite.id, "ssr-write");
	assert.equal(ssrContext.motionHistoryRead.id, "motion-read");
	assert.equal(ssrContext.motionHistoryWrite.id, "motion-write");

	const ssrefractionRequest = createTemporalRequest({
		transients: {
			"ssrefraction:raw": {
				handle: { resource: { id: "ssrefraction-raw" } },
			},
			hiz: {
				handle: { resource: { id: "hiz" } },
			},
		},
	});
	const ssrefractionContext = executor.getPassExecutionContext(
		createExecutionContextRequest("ssrefraction", ssrefractionRequest)
	);
	assert.deepEqual(ssrefractionContext.frameBinding, { id: "frame-binding" });
	assert.equal(ssrefractionContext.refractionRaw.id, "ssrefraction-raw");
	assert.equal(ssrefractionContext.hiZ.id, "hiz");

	const volumetricContext = executor.getPassExecutionContext(
		createExecutionContextRequest("volumetric", ssrRequest)
	);
	assert.deepEqual(volumetricContext.frameBinding, { id: "frame-binding" });
	assert.deepEqual(volumetricContext.volumetricLighting, {
		id: "volumetric-lighting-data",
	});
	assert.ok(volumetricContext.shared);
	assert.equal(volumetricContext.historyRead.id, "vol-read");
	assert.equal(volumetricContext.historyWrite.id, "vol-write");
	assert.equal(volumetricContext.reservoirHistoryRead.id, "res-read");
	assert.equal(volumetricContext.reservoirHistoryWrite.id, "res-write");
	assert.equal(volumetricContext.motionHistoryRead.id, "motion-read");
	assert.equal(volumetricContext.motionHistoryWrite.id, "motion-write");
	executor.abortFrame();
}

function testCustomImplementationMetadataPacksContext() {
	const { executor } = createExecutorHarness();
	const request = createTemporalRequest();
	const context = executor.getPassExecutionContext(
		createExecutionContextRequest("custom-webgpu", request, {
			pass: {
				id: "custom-webgpu",
				builtIn: false,
			},
			implementation: {
				id: "custom-webgpu:test",
				metadata: {
					context: {
						backend: "webgpu",
						kind: "screen",
						publishColorTarget: true,
						frameBinding: true,
						histories: [
							{
								property: "customHistoryWrite",
								historyId: "taa",
								side: "write",
							},
						],
					},
				},
			},
		})
	);

	assert.ok(context.encoder);
	assert.equal(Object.isFrozen(context.targets), true);
	assert.ok(context.shared);
	assert.deepEqual(context.frameBinding, { id: "frame-binding" });
	assert.equal(context.customHistoryWrite.id, "taa-write");
	const targets = executor.getFrameGraphDebugState().frameTargets;
	context.publishColorTarget(targets.postPing);
	assert.equal(targets.sceneColor, targets.sceneColorMain);
	executor.completePostProcessPass(
		createExecutionContextRequest("custom-webgpu", request),
		{ ran: true }
	);
	assert.equal(executor.getFrameGraphDebugState().frameTargets.sceneColor, targets.postPing);
	executor.abortFrame();
}

async function testWarmupHintsFollowPlanPostProcessPasses() {
	const backend = new FakeBackend();
	const resources = {
		sceneFrameLayout: {},
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

function testSSRRequirementsExposeMaterialChannels() {
	const pass = new ScreenSpaceReflectionsPass({ enabled: true });
	assert.deepEqual(pass.getRequirements({}).gBuffer, [
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
	testCustomImplementationMetadataPacksContext();
	await testWarmupHintsFollowPlanPostProcessPasses();
	testBackendPostProcessSurfaceKeepsOnlyExecutorBridge();
	testWebGPUOcclusionExtensionDescriptor();
	testSSRRequirementsExposeMaterialChannels();
	console.log("WebGPU post-process executor tests passed");
}

await run();
