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
	const runtimeCalls = [];
	const executor = Object.create(WebGPUFrameExecutor.prototype);
	executor._encoder = { id: "encoder" };
	executor._frameTargets = {
		sceneColor: { id: "scene-color" },
		sceneColorMain: { id: "scene-color-main" },
	};
	executor._frameResources = {
		scopeKey: "main",
		sceneTargetMode: "mrt",
		frameBinding: { id: "frame-binding" },
		environmentBinding: null,
		clusteredSceneBinding: null,
		lightingState: { id: "lighting-state" },
		featureState: {},
		environmentState: {},
		jointMatrixMap: new Map(),
		morphWeightMap: new Map(),
	};
	executor._postRuntime = {
		sharedContext: { id: "shared-context" },
		executePass: async (request) => {
			runtimeCalls.push(request);
			return {
				ran: true,
				historyUpdated: true,
			};
		},
	};
	executor._resources = {
	};
	return { executor, runtimeCalls };
}

async function testTemporalExecutePassUsesPipelineHistories() {
	const { executor, runtimeCalls } = createExecutorHarness();

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
	assert.equal("historyRead" in executor._frameTargets, false);
	assert.equal(taaContext.historyRead.id, "taa-read");
	assert.equal(taaContext.historyWrite.id, "taa-write");
	assert.equal(taaContext.motionHistoryRead.id, "motion-read");
	assert.equal(taaContext.motionHistoryWrite.id, "motion-write");
	taaContext.writeMotionHistoryFromCurrent();
	assert.equal(executor._motionHistoryWriteTarget.id, "motion-write");
	assert.equal(runtimeCalls.length, 0);

	const ssrRequest = createTemporalRequest();
	const ssrContext = executor.getPassExecutionContext(
		createExecutionContextRequest("ssr", ssrRequest)
	);
	assert.equal(runtimeCalls.length, 0);
	assert.deepEqual(ssrContext.frameBinding, { id: "frame-binding" });
	assert.equal(ssrContext.historyRead.id, "ssr-read");
	assert.equal(ssrContext.historyWrite.id, "ssr-write");
	assert.equal(ssrContext.motionHistoryRead.id, "motion-read");
	assert.equal(ssrContext.motionHistoryWrite.id, "motion-write");

	const volumetricContext = executor.getPassExecutionContext(
		createExecutionContextRequest("volumetric", ssrRequest)
	);
	assert.equal(runtimeCalls.length, 0);
	assert.deepEqual(volumetricContext.frameBinding, { id: "frame-binding" });
	assert.deepEqual(volumetricContext.lightingState, { id: "lighting-state" });
	assert.deepEqual(volumetricContext.shared, { id: "shared-context" });
	assert.equal(volumetricContext.historyRead.id, "vol-read");
	assert.equal(volumetricContext.historyWrite.id, "vol-write");
	assert.equal(volumetricContext.reservoirHistoryRead.id, "res-read");
	assert.equal(volumetricContext.reservoirHistoryWrite.id, "res-write");
	assert.equal(volumetricContext.motionHistoryRead.id, "motion-read");
	assert.equal(volumetricContext.motionHistoryWrite.id, "motion-write");
	assert.equal(runtimeCalls.length, 0);
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

	assert.equal(context.encoder.id, "encoder");
	assert.deepEqual(context.shared, { id: "shared-context" });
	assert.deepEqual(context.frameBinding, { id: "frame-binding" });
	assert.equal(context.customHistoryWrite.id, "taa-write");
	context.publishColorTarget({ id: "custom-color" });
	assert.equal(executor._frameTargets.sceneColor.id, "custom-color");
}

async function testWarmupHintsFollowPlanPostProcessPasses() {
	const backend = new FakeBackend();
	const resources = {
		sceneFrameLayout: {},
		setSceneTargetMode() {},
	};
	const executor = new WebGPUFrameExecutor(backend, resources);
	const warmupHints = [];

	executor._ensurePresentResources = async () => {};
	executor._postRuntime.warmupHints = async (hints) => {
		warmupHints.push(...hints);
		return { compiled: hints.length, failed: 0, errors: [] };
	};

	await executor.warmup(
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

	assert.deepEqual(warmupHints, []);

	const taaPass = new TemporalAntiAliasingPass({ enabled: true });
	await executor.warmup(
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

	assert.deepEqual(warmupHints, ["postprocess:taa"]);
}

function testBackendPostProcessSurfaceKeepsOnlyExecutorBridge() {
	const backend = new WebGPUBackend();
	const adapter = resolvePostProcessBackendAdapter(backend);
	assert.ok(adapter);
	assert.equal(adapter.backend, "webgpu");
	assert.equal(typeof adapter.executor.executePass, "function");
	assert.equal(typeof adapter.createGBufferBridge, "function");
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
