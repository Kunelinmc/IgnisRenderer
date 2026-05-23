import assert from "node:assert/strict";
import { Logger } from "../src/foundation/Logger.ts";
import { SCREEN_SPACE_REFLECTIONS_PASS } from "../src/postprocess/index.ts";
import { WebGPUPostProcessRuntime } from "../src/renderers/webgpu/WebGPUPostProcessRuntime.ts";
import {
	FakeBackend,
	FakeEncoder,
	createTexture,
} from "./helpers/webgpu_postprocess_runtime_test_helpers.mjs";
import { createResolvedPostProcess } from "./helpers/postprocess.mjs";

function createTemporalTargets(width = 32, height = 16) {
	return {
		sceneColor: createTexture(width, height, "scene"),
		postPing: createTexture(width, height, "ping"),
		postPong: createTexture(width, height, "pong"),
		gNormalRoughMetal: createTexture(width, height, "g-normal"),
		gMotionDepth: createTexture(width, height, "g-motion-depth"),
		planarReflectionMask: createTexture(width, height, "planar-reflection-mask"),
		historyRead: createTexture(width, height, "history-read"),
		historyWrite: createTexture(width, height, "history-write"),
		motionHistoryRead: createTexture(width, height, "motion-history-read"),
		ssrRaw: createTexture(width, height, "ssr-raw"),
		ssrHistoryRead: createTexture(width, height, "ssr-history-read"),
		ssrHistoryWrite: createTexture(width, height, "ssr-history-write"),
		hiZ: createTexture(width, height, "hiz"),
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

function createSSRPassRequest(frameContext, targets, historyValid = true) {
	const implementation = SCREEN_SPACE_REFLECTIONS_PASS.implementations.webgpu;
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {
			ssr: {
				valid: historyValid,
				read: { resource: targets.ssrHistoryRead },
				write: { resource: targets.ssrHistoryWrite },
			},
			motion: {
				valid: historyValid,
				read: { resource: targets.motionHistoryRead },
				write: { resource: targets.motionHistoryWrite },
			},
		},
		pass: SCREEN_SPACE_REFLECTIONS_PASS,
		passId: "ssr",
		implementation,
		options: frameContext.postProcess.options.ssr,
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
	} = options;
	const frameBinding =
		Object.prototype.hasOwnProperty.call(options, "frameBinding") ?
			options.frameBinding
		:	{ label: "frame-binding" };
	let published = null;
	let motionWrites = 0;
	const implementation = SCREEN_SPACE_REFLECTIONS_PASS.implementations.webgpu;
	const request = createSSRPassRequest(frameContext, targets, historyValid);
	const context = {
		encoder: new FakeEncoder(backend),
		targets,
		shared: runtime.sharedContext,
		frameBinding,
		publishColorTarget: (texture) => {
			published = texture;
			targets.sceneColor = texture;
		},
		writeMotionHistoryFromCurrent: () => {
			motionWrites++;
		},
	};
	const result = await implementation.execute(request, context);
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
	const volumetricResult = await runtime.executePass({
		passId: "volumetric",
		encoder: new FakeEncoder(),
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
	assert.equal(volumetricResult.ran, true);
	assert.equal(volumetricResult.historyUpdated, true);
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

		const volumetricResult = await runtime.executePass({
			passId: "volumetric",
			encoder: new FakeEncoder(),
			targets: createTemporalTargets(),
			frameContext,
			historyValid: true,
			frameBinding,
			lightingState: null,
		});
		assert.equal(volumetricResult.ran, false);
		assert.equal(volumetricResult.historyUpdated, false);
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

async function testHiZMipViewsAreCachedAcrossSSRExecutions() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const frameBinding = { label: "frame-binding" };
	const targets = createTemporalTargets(16, 8);
	const frameContext = createPerspectiveFrameContext({});

	await executeSSRImplementation(backend, runtime, {
		targets,
		frameContext,
		historyValid: false,
		frameBinding,
	});
	const firstViewCount = backend.textureViews.length;
	assert.equal(firstViewCount, 5);

	await executeSSRImplementation(backend, runtime, {
		targets,
		frameContext,
		historyValid: false,
		frameBinding,
	});
	assert.equal(backend.textureViews.length, firstViewCount);
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

async function testOnShaderRuntimeChangedDestroysParameterBuffers() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const sceneColorMain = createTexture(24, 12, "scene");
	const postPing = createTexture(24, 12, "ping");
	const postPong = createTexture(24, 12, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};

	void targets;
	await runtime.warmupHints([
		"postprocess:volumetric",
		"postprocess:motion-blur",
		"postprocess:dof",
	]);
	assert.equal(backend.bufferDestroyCalls, 0);
	assert.equal(backend.bindingGroupDestroyCalls, 0);
	assert.equal(backend.shaderModuleDestroyCalls, 0);
	assert.equal(backend.computePipelineDestroyCalls, 0);

	runtime.onShaderRuntimeChanged();
	assert.equal(backend.bindingGroupDestroyCalls, 0);
	const destroyedLabels = new Set(
		backend.buffers
			.filter((buffer) => buffer.destroyed)
			.map((buffer) => buffer.desc.label)
	);
	assert.ok(destroyedLabels.has("WebGPUVolumetricParams"));
	assert.ok(destroyedLabels.has("WebGPUMotionBlurParams"));
	assert.ok(destroyedLabels.has("WebGPUDOFParams"));
	assert.equal(destroyedLabels.has("WebGPUSSRTraceParams"), false);
	assert.equal(destroyedLabels.has("WebGPUSSRComposeParams"), false);
	assert.equal(destroyedLabels.has("WebGPUFXAAParams"), false);
	const destroyedShaderLabels = new Set(
		backend.shaderModules
			.filter((module) => module.destroyed)
			.map((module) => module.label)
	);
	assert.ok(destroyedShaderLabels.has("WebGPUHiZShader"));
	assert.ok(destroyedShaderLabels.has("WebGPUVolumetricShader"));
	assert.ok(destroyedShaderLabels.has("WebGPUMotionBlurShader"));
	assert.ok(destroyedShaderLabels.has("WebGPUDOFShader"));
	assert.equal(destroyedShaderLabels.has("WebGPUSSRShader"), false);
	assert.equal(destroyedShaderLabels.has("WebGPUFXAAShader"), false);
	const bufferDestroyCalls = backend.bufferDestroyCalls;
	const shaderModuleDestroyCalls = backend.shaderModuleDestroyCalls;
	const computePipelineDestroyCalls = backend.computePipelineDestroyCalls;

	runtime.onShaderRuntimeChanged();
	assert.equal(backend.bindingGroupDestroyCalls, 0);
	assert.equal(backend.bufferDestroyCalls, bufferDestroyCalls);
	assert.equal(backend.shaderModuleDestroyCalls, shaderModuleDestroyCalls);
	assert.equal(backend.computePipelineDestroyCalls, computePipelineDestroyCalls);
}

async function run() {
	await testTAAIsOwnedByLogicalPassImplementation();
	await testSSRAndVolumetricReportHistoryUpdates();
	await testOrthographicTemporalPassesSkipAndReturnFalse();
	await testHiZMipViewsAreCachedAcrossSSRExecutions();
	await testUnknownPassReturnsRanFalse();
	await testMissingSSRFrameBindingSkipsImplementation();
	await testOnShaderRuntimeChangedDestroysParameterBuffers();
	console.log("WebGPU postprocess temporal runtime tests passed");
}

await run();
