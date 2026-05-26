import assert from "node:assert/strict";
import { Logger } from "../src/foundation/Logger.ts";
import {
	ScreenSpaceReflectionsPass,
	VolumetricLightingPass,
} from "../src/postprocess/index.ts";
import { WebGPUPostProcessRuntime } from "../src/renderers/webgpu/WebGPUPostProcessRuntime.ts";
import {
	FakeBackend,
	FakeEncoder,
	createTexture,
} from "./helpers/webgpu_postprocess_runtime_test_helpers.mjs";
import { createResolvedPostProcess } from "./helpers/postprocess.mjs";

const SSR_PASS = new ScreenSpaceReflectionsPass({ enabled: true });
const VOLUMETRIC_PASS = new VolumetricLightingPass({ enabled: true });

function createTemporalTargets(width = 32, height = 16) {
	return {
		sceneColor: createTexture(width, height, "scene"),
		postPing: createTexture(width, height, "ping"),
		postPong: createTexture(width, height, "pong"),
		gNormalRoughMetal: createTexture(width, height, "g-normal"),
		gMotionDepth: createTexture(width, height, "g-motion-depth"),
		planarReflectionMask: createTexture(width, height, "planar-reflection-mask"),
		ssrRaw: createTexture(width, height, "ssr-raw"),
		hiZ: createTexture(width, height, "hiz"),
	};
}

function createTemporalHistories(width = 32, height = 16) {
	return {
		historyRead: createTexture(width, height, "history-read"),
		historyWrite: createTexture(width, height, "history-write"),
		motionHistoryRead: createTexture(width, height, "motion-history-read"),
		motionHistoryWrite: createTexture(width, height, "motion-history-write"),
		ssrHistoryRead: createTexture(width, height, "ssr-history-read"),
		ssrHistoryWrite: createTexture(width, height, "ssr-history-write"),
		volumetricHistoryRead: createTexture(width, height, "vol-history-read"),
		volumetricHistoryWrite: createTexture(width, height, "vol-history-write"),
		volumetricReservoirHistoryRead: createTexture(
			width,
			height,
			"vol-reservoir-read"
		),
		volumetricReservoirHistoryWrite: createTexture(
			width,
			height,
			"vol-reservoir-write"
		),
	};
}

function createSSRPassRequest(frameContext, histories, historyValid = true) {
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {
			ssr: {
				valid: historyValid,
				read: { resource: histories.ssrHistoryRead },
				write: { resource: histories.ssrHistoryWrite },
			},
			motion: {
				valid: historyValid,
				read: { resource: histories.motionHistoryRead },
				write: { resource: histories.motionHistoryWrite },
			},
		},
		pass: SSR_PASS,
		passId: "ssr",
		options: frameContext.postProcess.getOptions("ssr"),
		startPassId: null,
	};
}

function createVolumetricPassRequest(frameContext, histories, historyValid = true) {
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {
			volumetric: {
				valid: historyValid,
				read: { resource: histories.volumetricHistoryRead },
				write: { resource: histories.volumetricHistoryWrite },
			},
			"volumetric-reservoir": {
				valid: historyValid,
				read: { resource: histories.volumetricReservoirHistoryRead },
				write: { resource: histories.volumetricReservoirHistoryWrite },
			},
			motion: {
				valid: historyValid,
				read: { resource: histories.motionHistoryRead },
				write: { resource: histories.motionHistoryWrite },
			},
		},
		pass: VOLUMETRIC_PASS,
		passId: "volumetric",
		options: frameContext.postProcess.getOptions("volumetric"),
		startPassId: null,
	};
}

async function executeSSRImplementation(
	backend,
	runtime,
	options = {}
) {
	const {
		targets = createTemporalTargets(),
		frameContext = createPerspectiveFrameContext(),
		historyValid = true,
		histories = createTemporalHistories(
			targets.sceneColor.width,
			targets.sceneColor.height
		),
		encoder = new FakeEncoder(backend),
	} = options;
	const frameBinding =
		Object.prototype.hasOwnProperty.call(options, "frameBinding") ?
			options.frameBinding
		:	{ label: "frame-binding" };
	let published = null;
	let motionWrites = 0;
	const request = createSSRPassRequest(frameContext, histories, historyValid);
	const context = {
		encoder,
		targets,
		shared: runtime.sharedContext,
		frameBinding,
		historyRead: histories.ssrHistoryRead,
		historyWrite: histories.ssrHistoryWrite,
		motionHistoryRead: histories.motionHistoryRead,
		motionHistoryWrite: histories.motionHistoryWrite,
		publishColorTarget: (texture) => {
			published = texture;
			targets.sceneColor = texture;
		},
		writeMotionHistoryFromCurrent: () => {
			motionWrites++;
		},
	};
	const result = await request.pass.getImplementation("webgpu").execute(
		request,
		context
	);
	return { result, published, motionWrites };
}

async function executeVolumetricImplementation(
	backend,
	runtime,
	options = {}
) {
	const {
		targets = createTemporalTargets(),
		frameContext = createPerspectiveFrameContext(),
		historyValid = true,
		histories = createTemporalHistories(
			targets.sceneColor.width,
			targets.sceneColor.height
		),
		lightingState = null,
		encoder = new FakeEncoder(backend),
	} = options;
	const frameBinding =
		Object.prototype.hasOwnProperty.call(options, "frameBinding") ?
			options.frameBinding
		:	{ label: "frame-binding" };
	let published = null;
	let motionWrites = 0;
	const request = createVolumetricPassRequest(
		frameContext,
		histories,
		historyValid
	);
	const context = {
		encoder,
		targets,
		shared: runtime.sharedContext,
		frameBinding,
		lightingState,
		historyRead: histories.volumetricHistoryRead,
		historyWrite: histories.volumetricHistoryWrite,
		reservoirHistoryRead: histories.volumetricReservoirHistoryRead,
		reservoirHistoryWrite: histories.volumetricReservoirHistoryWrite,
		motionHistoryRead: histories.motionHistoryRead,
		motionHistoryWrite: histories.motionHistoryWrite,
		publishColorTarget: (texture) => {
			published = texture;
			targets.sceneColor = texture;
		},
		writeMotionHistoryFromCurrent: () => {
			motionWrites++;
		},
	};
	const result = await request.pass.getImplementation("webgpu").execute(
		request,
		context
	);
	return { result, published, motionWrites };
}

function createPerspectiveFrameContext(postProcessRequest = {}) {
	return {
		camera: {
			type: "perspective",
			fov: 60,
			aspectRatio: 2,
		},
		features: {},
		postProcess: createResolvedPostProcess(postProcessRequest),
		transient: new Map(),
	};
}

async function captureWarnMessagesAsync(run) {
	const warnings = [];
	Logger.configure({
		level: "warn",
		sink: {
			warn: (...args) => {
				warnings.push(args.map((arg) => String(arg)).join(" "));
			},
		},
		resetOnceKeys: true,
	});
	try {
		await run();
	} finally {
		Logger.reset();
	}
	return warnings;
}

async function testTAAIsOwnedByLogicalPassImplementation() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const targets = createTemporalTargets();
	const result = await runtime.executePass({
		passId: "taa",
		encoder: new FakeEncoder(),
		targets,
		frameContext: createPerspectiveFrameContext({
			taa: {
				enabled: true,
				options: {
					historyWeight: 0.8,
				},
			},
		}),
		historyValid: true,
	});

	assert.equal(result.ran, false);
	assert.equal(result.historyUpdated, undefined);
	assert.equal(targets.sceneColor.label, "scene");
}

async function testSSRAndVolumetricReportHistoryUpdates() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const frameBinding = { label: "frame-binding" };

	const ssrTargets = createTemporalTargets(32, 16);
	const ssrRun = await executeSSRImplementation(backend, runtime, {
		targets: ssrTargets,
		frameContext: createPerspectiveFrameContext({
			ssr: {
				enabled: true,
				options: {
					maxSteps: 24,
				},
			},
		}),
		historyValid: true,
		frameBinding,
	});
	assert.deepEqual(ssrRun.result, {
		ran: true,
		updatedHistoryIds: ["ssr", "motion"],
	});
	assert.strictEqual(ssrRun.published, ssrTargets.postPing);
	assert.strictEqual(ssrTargets.sceneColor, ssrTargets.postPing);
	assert.equal(ssrRun.motionWrites, 1);

	const volumetricTargets = createTemporalTargets(32, 16);
	const volumetricRun = await executeVolumetricImplementation(backend, runtime, {
		targets: volumetricTargets,
		frameContext: createPerspectiveFrameContext({
			volumetric: {
				enabled: true,
				options: {
					samples: 12,
				},
			},
		}),
		historyValid: true,
		frameBinding,
		lightingState: null,
	});
	assert.deepEqual(volumetricRun.result, {
		ran: true,
		updatedHistoryIds: ["volumetric", "volumetric-reservoir", "motion"],
	});
	assert.strictEqual(volumetricRun.published, volumetricTargets.postPong);
	assert.strictEqual(volumetricTargets.sceneColor, volumetricTargets.postPong);
	assert.equal(volumetricRun.motionWrites, 1);
}

async function testOrthographicTemporalPassesSkipAndReturnFalse() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const frameContext = {
		camera: {
			type: "orthographic",
		},
		features: {},
		postProcess: createResolvedPostProcess(),
		transient: new Map(),
	};
	const frameBinding = { label: "frame-binding" };

	const warnings = await captureWarnMessagesAsync(async () => {
		const ssrRun = await executeSSRImplementation(backend, runtime, {
			targets: createTemporalTargets(),
			frameContext,
			historyValid: true,
			frameBinding,
		});
		assert.deepEqual(ssrRun.result, { ran: false });
		assert.equal(ssrRun.motionWrites, 0);

		const volumetricRun = await executeVolumetricImplementation(backend, runtime, {
			targets: createTemporalTargets(),
			frameContext,
			historyValid: true,
			frameBinding,
			lightingState: null,
		});
		assert.deepEqual(volumetricRun.result, { ran: false });
		assert.equal(volumetricRun.motionWrites, 0);
	});

	assert.equal(warnings.length, 2);
	assert.equal(
		warnings.some((warning) =>
			warning.includes("[webgpu-ssr-orthographic-disabled]")
		),
		true
	);
	assert.equal(
		warnings.some((warning) =>
			warning.includes("[webgpu-volumetric-orthographic-disabled]")
		),
		true
	);
}

async function testHiZResourcesAreSharedAcrossTemporalPasses() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const frameBinding = { label: "frame-binding" };
	const targets = createTemporalTargets(16, 8);
	const frameContext = createPerspectiveFrameContext({});
	const encoder = new FakeEncoder(backend);

	await executeSSRImplementation(backend, runtime, {
		targets,
		frameContext,
		historyValid: false,
		frameBinding,
		encoder,
	});
	const firstViewCount = backend.textureViews.length;
	assert.equal(firstViewCount, 5);
	const firstHiZPassCount = encoder.calls.filter(
		(call) =>
			call[0] === "beginComputePass" &&
			typeof call[1] === "string" &&
			call[1].startsWith("WebGPUHiZ")
	).length;
	assert.equal(firstHiZPassCount, 5);

	await executeSSRImplementation(backend, runtime, {
		targets,
		frameContext,
		historyValid: false,
		frameBinding,
		encoder,
	});
	assert.equal(backend.textureViews.length, firstViewCount);
	assert.equal(
		encoder.calls.filter(
			(call) =>
				call[0] === "beginComputePass" &&
				typeof call[1] === "string" &&
				call[1].startsWith("WebGPUHiZ")
		).length,
		firstHiZPassCount
	);

	await executeVolumetricImplementation(backend, runtime, {
		targets,
		frameContext,
		historyValid: false,
		frameBinding,
		lightingState: null,
		encoder,
	});
	assert.equal(backend.textureViews.length, firstViewCount);
	assert.equal(
		encoder.calls.filter(
			(call) =>
				call[0] === "beginComputePass" &&
				typeof call[1] === "string" &&
				call[1].startsWith("WebGPUHiZ")
		).length,
		firstHiZPassCount
	);
	assert.equal(
		backend.shaderModules.filter((module) => module.label === "WebGPUHiZShader")
			.length,
		1
	);
	assert.equal(
		backend.computePipelines.filter(
			(pipeline) =>
				pipeline.label === "WebGPUHiZInitPipeline" ||
				pipeline.label === "WebGPUHiZReducePipeline"
		).length,
		2
	);
}

async function testUnknownPassReturnsRanFalse() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const result = await runtime.executePass({
		passId: "gamma",
		encoder: new FakeEncoder(),
		targets: createTemporalTargets(),
		frameContext: createPerspectiveFrameContext(),
	});
	assert.deepEqual(result, { ran: false });
}

async function testMissingSSRFrameBindingSkipsImplementation() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const targets = createTemporalTargets(32, 16);
	const ssrRun = await executeSSRImplementation(backend, runtime, {
		targets,
		frameContext: createPerspectiveFrameContext(),
		historyValid: true,
		frameBinding: undefined,
	});
	assert.deepEqual(ssrRun.result, { ran: false });
	assert.equal(ssrRun.motionWrites, 0);
	assert.equal(backend.textureViews.length, 0);
}

async function testMigratedScreenWarmupHintsDoNotAllocateRuntimeResources() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});

	await runtime.warmupHints([
		"postprocess:motion-blur",
		"postprocess:dof",
	]);
	assert.equal(backend.buffers.length, 0);
	assert.equal(backend.shaderModules.length, 0);
	assert.equal(backend.computePipelines.length, 0);
	assert.equal(backend.bufferDestroyCalls, 0);
	assert.equal(backend.bindingGroupDestroyCalls, 0);
	assert.equal(backend.shaderModuleDestroyCalls, 0);
	assert.equal(backend.computePipelineDestroyCalls, 0);

	runtime.onShaderRuntimeChanged();
	assert.equal(backend.bindingGroupDestroyCalls, 0);
	assert.equal(backend.bufferDestroyCalls, 0);
	assert.equal(backend.shaderModuleDestroyCalls, 0);
	assert.equal(backend.computePipelineDestroyCalls, 0);

	runtime.onShaderRuntimeChanged();
	assert.equal(backend.bindingGroupDestroyCalls, 0);
	assert.equal(backend.bufferDestroyCalls, 0);
	assert.equal(backend.shaderModuleDestroyCalls, 0);
	assert.equal(backend.computePipelineDestroyCalls, 0);
}

async function run() {
	await testTAAIsOwnedByLogicalPassImplementation();
	await testSSRAndVolumetricReportHistoryUpdates();
	await testOrthographicTemporalPassesSkipAndReturnFalse();
	await testHiZResourcesAreSharedAcrossTemporalPasses();
	await testUnknownPassReturnsRanFalse();
	await testMissingSSRFrameBindingSkipsImplementation();
	await testMigratedScreenWarmupHintsDoNotAllocateRuntimeResources();
	console.log("WebGPU postprocess temporal runtime tests passed");
}

await run();
