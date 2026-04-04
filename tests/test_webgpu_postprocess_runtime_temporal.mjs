import assert from "node:assert/strict";
import { WebGPUPostProcessRuntime } from "../src/renderers/webgpu/WebGPUPostProcessRuntime.ts";
import {
	FakeBackend,
	FakeEncoder,
	createTexture,
} from "./helpers/webgpu_postprocess_runtime_test_helpers.mjs";

function createTemporalTargets(width = 32, height = 16) {
	return {
		sceneColor: createTexture(width, height, "scene"),
		postPing: createTexture(width, height, "ping"),
		postPong: createTexture(width, height, "pong"),
		gNormalRoughMetal: createTexture(width, height, "g-normal"),
		gMotionDepth: createTexture(width, height, "g-motion-depth"),
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

function createPerspectiveFrameContext(features = {}) {
	return {
		camera: {
			type: "perspective",
			fov: 60,
			aspectRatio: 2,
		},
		features,
		transient: new Map(),
	};
}

async function testTAAExecutePassReportsHistoryUpdate() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const targets = createTemporalTargets();
	const result = await runtime.executePass({
		passId: "taa",
		encoder: new FakeEncoder(),
		targets,
		frameContext: createPerspectiveFrameContext({
			taaOptions: {
				historyWeight: 0.8,
			},
		}),
		historyValid: true,
	});

	assert.equal(result.ran, true);
	assert.equal(result.historyUpdated, true);
	assert.equal(targets.sceneColor, targets.postPong);
}

async function testSSRAndVolumetricReportHistoryUpdates() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const frameBinding = { label: "frame-binding" };

	const ssrTargets = createTemporalTargets(32, 16);
	const ssrResult = await runtime.executePass({
		passId: "ssr",
		encoder: new FakeEncoder(),
		targets: ssrTargets,
		frameContext: createPerspectiveFrameContext({
			ssrOptions: {
				maxSteps: 24,
			},
		}),
		historyValid: true,
		frameBinding,
	});
	assert.equal(ssrResult.ran, true);
	assert.equal(ssrResult.historyUpdated, true);

	const volumetricTargets = createTemporalTargets(32, 16);
	const volumetricResult = await runtime.executePass({
		passId: "volumetric",
		encoder: new FakeEncoder(),
		targets: volumetricTargets,
		frameContext: createPerspectiveFrameContext({
			volumetricOptions: {
				samples: 12,
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
	const warnings = [];
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(
		backend,
		(key, message) => warnings.push([key, message])
	);
	const frameContext = {
		camera: {
			type: "orthographic",
		},
		features: {},
		transient: new Map(),
	};
	const frameBinding = { label: "frame-binding" };

	const ssrResult = await runtime.executePass({
		passId: "ssr",
		encoder: new FakeEncoder(),
		targets: createTemporalTargets(),
		frameContext,
		historyValid: true,
		frameBinding,
	});
	assert.equal(ssrResult.ran, false);
	assert.equal(ssrResult.historyUpdated, false);

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

	assert.equal(warnings.length, 2);
	assert.equal(warnings[0][0], "webgpu-ssr-orthographic-disabled");
	assert.equal(warnings[1][0], "webgpu-volumetric-orthographic-disabled");
}

async function testHiZMipViewsAreCachedAcrossSSRExecutions() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const frameBinding = { label: "frame-binding" };
	const targets = createTemporalTargets(16, 8);
	const frameContext = createPerspectiveFrameContext({});

	await runtime.executePass({
		passId: "ssr",
		encoder: new FakeEncoder(),
		targets,
		frameContext,
		historyValid: false,
		frameBinding,
	});
	const firstViewCount = backend.textureViews.length;
	assert.equal(firstViewCount, 5);

	await runtime.executePass({
		passId: "ssr",
		encoder: new FakeEncoder(),
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

async function testMissingSSRFrameBindingThrowsAtRuntime() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const targets = createTemporalTargets(32, 16);
	await assert.rejects(async () => {
		await runtime.executePass({
			passId: "ssr",
			encoder: new FakeEncoder(),
			targets,
			frameContext: createPerspectiveFrameContext(),
			historyValid: true,
		});
	});
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

	await runtime.executePass({
		passId: "fxaa",
		encoder: new FakeEncoder(),
		targets,
		frameContext: { features: {}, transient: new Map() },
	});
	await runtime.warmupHints([
		"postprocess:ssao",
		"postprocess:ssgi",
		"postprocess:taa",
		"postprocess:ssr",
		"postprocess:volumetric",
		"postprocess:motion-blur",
		"postprocess:dof",
	]);
	assert.equal(backend.bufferDestroyCalls, 0);
	assert.equal(backend.bindingGroupDestroyCalls, 0);

	runtime.onShaderRuntimeChanged();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
	assert.equal(backend.bufferDestroyCalls, 9);
	const destroyedLabels = new Set(
		backend.buffers
			.filter((buffer) => buffer.destroyed)
			.map((buffer) => buffer.desc.label)
	);
	assert.ok(destroyedLabels.has("WebGPUSSAOParams"));
	assert.ok(destroyedLabels.has("WebGPUSSGIParams"));
	assert.ok(destroyedLabels.has("WebGPUTAAParams"));
	assert.ok(destroyedLabels.has("WebGPUSSRTraceParams"));
	assert.ok(destroyedLabels.has("WebGPUSSRComposeParams"));
	assert.ok(destroyedLabels.has("WebGPUVolumetricParams"));
	assert.ok(destroyedLabels.has("WebGPUMotionBlurParams"));
	assert.ok(destroyedLabels.has("WebGPUDOFParams"));
	assert.ok(destroyedLabels.has("WebGPUFXAAParams"));

	runtime.onShaderRuntimeChanged();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
	assert.equal(backend.bufferDestroyCalls, 9);
}

async function run() {
	await testTAAExecutePassReportsHistoryUpdate();
	await testSSRAndVolumetricReportHistoryUpdates();
	await testOrthographicTemporalPassesSkipAndReturnFalse();
	await testHiZMipViewsAreCachedAcrossSSRExecutions();
	await testUnknownPassReturnsRanFalse();
	await testMissingSSRFrameBindingThrowsAtRuntime();
	await testOnShaderRuntimeChangedDestroysParameterBuffers();
	console.log("WebGPU postprocess temporal runtime tests passed");
}

await run();
