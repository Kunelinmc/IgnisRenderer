import assert from "node:assert/strict";
import { WebGPUBackend } from "../src/renderers/WebGPUBackend.ts";
import { WebGPUFrameExecutor } from "../src/renderers/webgpu/WebGPUFrameExecutor.ts";
import { FakeWebGPUBackend as FakeBackend } from "./helpers/test_fakes.mjs";

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

function createExecutorHarness() {
	const runtimeCalls = [];
	const executor = Object.create(WebGPUFrameExecutor.prototype);
	executor._encoder = { id: "encoder" };
	executor._frameTargets = {
		sceneColor: { id: "scene-color" },
		sceneColorMain: { id: "scene-color-main" },
		historyRead: { id: "initial-taa-read" },
		historyWrite: { id: "initial-taa-write" },
		ssrHistoryRead: { id: "initial-ssr-read" },
		ssrHistoryWrite: { id: "initial-ssr-write" },
		volumetricHistoryRead: { id: "initial-vol-read" },
		volumetricHistoryWrite: { id: "initial-vol-write" },
		volumetricReservoirHistoryRead: { id: "initial-res-read" },
		volumetricReservoirHistoryWrite: { id: "initial-res-write" },
		motionHistoryRead: { id: "initial-motion-read" },
		motionHistoryWrite: { id: "initial-motion-write" },
	};
	executor._postRuntime = {
		executePass: async (request) => {
			runtimeCalls.push(request);
			return {
				ran: true,
				historyUpdated: true,
			};
		},
	};
	executor._resources = {
		getFrameBinding() {
			return { id: "frame-binding" };
		},
		getLightingState() {
			return { id: "lighting-state" };
		},
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
	const taaResult = await executor.executePostProcessPass("taa", taaRequest);
	assert.deepEqual(taaResult, {
		ran: true,
		updatedHistoryIds: ["taa"],
	});
	assert.equal(runtimeCalls[0].passId, "taa");
	assert.equal(runtimeCalls[0].historyValid, false);
	assert.equal(executor._frameTargets.historyRead.id, "taa-read");
	assert.equal(executor._frameTargets.historyWrite.id, "taa-write");

	const ssrRequest = createTemporalRequest();
	const ssrResult = await executor.executePostProcessPass("ssr", ssrRequest);
	assert.deepEqual(ssrResult, {
		ran: true,
		updatedHistoryIds: ["ssr"],
	});
	assert.equal(runtimeCalls[1].passId, "ssr");
	assert.equal(runtimeCalls[1].historyValid, true);
	assert.deepEqual(runtimeCalls[1].frameBinding, { id: "frame-binding" });
	assert.equal(executor._frameTargets.ssrHistoryRead.id, "ssr-read");
	assert.equal(executor._frameTargets.ssrHistoryWrite.id, "ssr-write");

	const volumetricResult = await executor.executePostProcessPass(
		"volumetric",
		ssrRequest
	);
	assert.deepEqual(volumetricResult, {
		ran: true,
		updatedHistoryIds: ["volumetric", "volumetric-reservoir"],
	});
	assert.equal(runtimeCalls[2].passId, "volumetric");
	assert.equal(runtimeCalls[2].historyValid, true);
	assert.deepEqual(runtimeCalls[2].frameBinding, { id: "frame-binding" });
	assert.deepEqual(runtimeCalls[2].lightingState, { id: "lighting-state" });
	assert.equal(executor._frameTargets.volumetricHistoryRead.id, "vol-read");
	assert.equal(executor._frameTargets.volumetricHistoryWrite.id, "vol-write");
	assert.equal(
		executor._frameTargets.volumetricReservoirHistoryRead.id,
		"res-read"
	);
	assert.equal(
		executor._frameTargets.volumetricReservoirHistoryWrite.id,
		"res-write"
	);
	assert.equal(executor._frameTargets.motionHistoryRead.id, "motion-read");
	assert.equal(executor._frameTargets.motionHistoryWrite.id, "motion-write");
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
		{},
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

	assert.deepEqual(warmupHints, [
		"postprocess:ssr",
		"postprocess:hiz",
		"postprocess:copy",
		"postprocess:volumetric",
	]);
}

function testBackendPostProcessSurfaceKeepsOnlyExecutorBridge() {
	const backend = new WebGPUBackend();
	assert.equal(backend.postProcessExecutor.backend, "webgpu");
	assert.equal(typeof backend.postProcessExecutor.executePass, "function");
	assert.equal(typeof backend.createPostProcessGBufferBridge, "function");
	assert.equal("postProcess" in backend, false);
}

async function run() {
	await testTemporalExecutePassUsesPipelineHistories();
	await testWarmupHintsFollowPlanPostProcessPasses();
	testBackendPostProcessSurfaceKeepsOnlyExecutorBridge();
	console.log("WebGPU post-process executor tests passed");
}

await run();
