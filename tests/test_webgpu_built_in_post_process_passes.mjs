import assert from "node:assert/strict";
import {
	resolvePostProcessBackendAdapter,
	ScreenSpaceReflectionsPass,
	TemporalAntiAliasingPass,
	VolumetricLightingPass,
} from "../src/postprocess/index.ts";
import { WebGPUBackend } from "../src/renderers/WebGPUBackend.ts";
import { WebGPUFrameExecutor } from "../src/renderers/webgpu/WebGPUFrameExecutor.ts";
import { FakeWebGPUBackend as FakeBackend } from "./helpers/test_fakes.mjs";
import { createResolvedPostProcess } from "./helpers/postprocess.mjs";

const BUILTIN_PASS_BY_ID = new Map([
	["taa", new TemporalAntiAliasingPass({ enabled: true })],
	["ssr", new ScreenSpaceReflectionsPass({ enabled: true })],
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

function createExecutorHarness() {
	const backend = new FakeBackend();
	const frameResources = {
		scopeKey: "main",
		sceneTargetMode: "mrt",
		frameBinding: { id: "frame-binding" },
		decalFrameBinding: { id: "decal-frame-binding" },
		environmentBinding: null,
		clusteredSceneBinding: null,
		lightingState: { id: "lighting-state" },
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
		postProcess: createResolvedPostProcess({
			taa: { enabled: true },
			ssr: { enabled: true },
			volumetric: { enabled: true },
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
	return { executor, backend };
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

	const volumetricContext = executor.getPassExecutionContext(
		createExecutionContextRequest("volumetric", ssrRequest)
	);
	assert.deepEqual(volumetricContext.frameBinding, { id: "frame-binding" });
	assert.deepEqual(volumetricContext.lightingState, { id: "lighting-state" });
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
	const adapter = resolvePostProcessBackendAdapter(backend);
	assert.ok(adapter);
	assert.equal(adapter.backend, "webgpu");
	assert.equal(typeof adapter.executePass, "function");
	assert.equal(typeof adapter.createGBufferBridge, "function");
	assert.equal("executor" in adapter, false);
	assert.equal("postProcessExecutor" in backend, false);
	assert.equal("createPostProcessGBufferBridge" in backend, false);
	assert.equal("postProcess" in backend, false);
}

async function run() {
	await testTemporalExecutePassUsesPipelineHistories();
	testCustomImplementationMetadataPacksContext();
	await testWarmupHintsFollowPlanPostProcessPasses();
	testBackendPostProcessSurfaceKeepsOnlyExecutorBridge();
	console.log("WebGPU post-process executor tests passed");
}

await run();
