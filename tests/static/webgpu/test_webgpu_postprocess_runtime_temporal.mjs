import assert from "node:assert/strict";

import { createWebGPUResourceManager } from "../../helpers/fakes.mjs";
import { createWebGPUComputeFacade } from "../../../src/backends/webgpu/ComputeFacade.ts";
import {
	ScreenSpaceReflectionsPass,
	VolumetricLightingPass,
} from "../../../src/postprocess/index.ts";
import { WebGPUPostProcessRuntime } from "../../../src/backends/webgpu/WebGPUPostProcessRuntime.ts";
import {
	FakeBackend,
	FakeEncoder,
} from "../../helpers/webgpu_postprocess_runtime_test_helpers.mjs";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

const SSR_PASS = new ScreenSpaceReflectionsPass({ enabled: true });
const VOLUMETRIC_PASS = new VolumetricLightingPass({ enabled: true });

function ownedTexture(backend, width, height, label) {
	return backend.createTexture({
		width, height, label, format: label === "hiz" ? "r32float" : "rgba16float", usage: 0,
	});
}

function createTemporalTargets(backend, width = 32, height = 16) {
	return {
		sceneColor: ownedTexture(backend, width, height, "scene"),
		postPing: ownedTexture(backend, width, height, "ping"),
		postPong: ownedTexture(backend, width, height, "pong"),
		gNormalRoughMetal: ownedTexture(backend, width, height, "g-normal"),
		gMotionDepth: ownedTexture(backend, width, height, "g-motion-depth"),
		planarReflectionMask: ownedTexture(backend, width, height, "planar-reflection-mask"),
	};
}

function createTemporalTransients(backend, width = 32, height = 16) {
	return {
		ssrRaw: ownedTexture(backend, width, height, "ssr-raw"),
		ssrDenoiseA: ownedTexture(backend, width, height, "ssr-denoise-a"),
		ssrDenoiseB: ownedTexture(backend, width, height, "ssr-denoise-b"),
		hiZ: ownedTexture(backend, width, height, "hiz"),
	};
}

function createTemporalHistories(backend, width = 32, height = 16) {
	return {
		historyRead: ownedTexture(backend, width, height, "history-read"),
		historyWrite: ownedTexture(backend, width, height, "history-write"),
		motionHistoryRead: ownedTexture(backend, width, height, "motion-history-read"),
		motionHistoryWrite: ownedTexture(backend, width, height, "motion-history-write"),
		ssrHistoryRead: ownedTexture(backend, width, height, "ssr-history-read"),
		ssrHistoryWrite: ownedTexture(backend, width, height, "ssr-history-write"),
		ssrMomentsRead: ownedTexture(backend, width, height, "ssr-moments-read"),
		ssrMomentsWrite: ownedTexture(backend, width, height, "ssr-moments-write"),
		volumetricHistoryRead: ownedTexture(backend, width, height, "vol-history-read"),
		volumetricHistoryWrite: ownedTexture(backend, width, height, "vol-history-write"),
		volumetricReservoirHistoryRead: ownedTexture(backend,
			width,
			height,
			"vol-reservoir-read"
		),
		volumetricReservoirHistoryWrite: ownedTexture(backend,
			width,
			height,
			"vol-reservoir-write"
		),
	};
}

function createSSRPassRequest(frameContext, histories, historyValid = true) {
	const resolvedPass = frameContext.postProcess.getPass("ssr");
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
			"ssr:moments": {
				valid: historyValid,
				read: { resource: histories.ssrMomentsRead },
				write: { resource: histories.ssrMomentsWrite },
			},
			motion: {
				valid: historyValid,
				read: { resource: histories.motionHistoryRead },
				write: { resource: histories.motionHistoryWrite },
			},
		},
		transients: {},
		pass: resolvedPass?.pass ?? SSR_PASS,
		passId: "ssr",
		options: resolvedPass?.options ?? frameContext.postProcess.getOptions("ssr"),
		startPassId: null,
	};
}

function createVolumetricPassRequest(frameContext, histories, historyValid = true) {
	const resolvedPass = frameContext.postProcess.getPass("volumetric");
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
		transients: {},
		pass: resolvedPass?.pass ?? VOLUMETRIC_PASS,
		passId: "volumetric",
		options:
			resolvedPass?.options ?? frameContext.postProcess.getOptions("volumetric"),
		startPassId: null,
	};
}

async function executeSSRImplementation(
	backend,
	runtime,
	options = {}
) {
	const {
		targets = createTemporalTargets(backend),
		frameContext = createPerspectiveFrameContext(),
		historyValid = true,
		histories = createTemporalHistories(backend,
			targets.sceneColor.width,
			targets.sceneColor.height
		),
		transients = createTemporalTransients(backend,
			histories.ssrHistoryWrite.width,
			histories.ssrHistoryWrite.height
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
	const output = targets.postPing;
	const context = {
		encoder,
		targets,
		shared: runtime,
		frameBinding,
		resources: {
			color: { input: targets.sceneColor, output },
			getGBuffer: (semantic) => semantic === "normal" ?
				targets.gNormalRoughMetal : targets.gMotionDepth,
			getHistory: (id) => id === "ssr" ? {
				read: histories.ssrHistoryRead,
				write: histories.ssrHistoryWrite,
				valid: historyValid,
			} : id === "ssr:moments" ? {
				read: histories.ssrMomentsRead,
				write: histories.ssrMomentsWrite,
				valid: historyValid,
			} : {
				read: histories.motionHistoryRead,
				write: histories.motionHistoryWrite,
				valid: historyValid,
			},
			getTransient: (id) => ({
				"ssr:denoise-a": transients.ssrDenoiseA,
				"ssr:denoise-b": transients.ssrDenoiseB,
			})[id] ?? null,
			getShared: (id) => id === "backend:frame-hiz" ? transients.hiZ :
				id === "backend:planar-reflection-mask" ?
					targets.planarReflectionMask : null,
			copyGBufferToHistory: () => {
				motionWrites++;
			},
		},
	};
	const result = await request.pass.getImplementation("webgpu").execute(
		request,
		context
	);
	if (result.ran !== false) {
		published = output;
		targets.sceneColor = output;
	}
	return { result, published, motionWrites, transients, encoder };
}

async function executeVolumetricImplementation(
	backend,
	runtime,
	options = {}
) {
	const {
		targets = createTemporalTargets(backend),
		frameContext = createPerspectiveFrameContext(),
		historyValid = true,
		histories = createTemporalHistories(backend,
			targets.sceneColor.width,
			targets.sceneColor.height
		),
		transients = createTemporalTransients(backend,
			targets.sceneColor.width,
			targets.sceneColor.height
		),
		lightingState = null,
		volumetricLightingData = null,
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
		shared: runtime,
		frameBinding,
		lightingState,
		volumetricLightingData,
		resources: {
			color: { input: targets.sceneColor, output: targets.postPong },
			getGBuffer: () => targets.gMotionDepth,
			getHistory: (id) => {
				if (id === "volumetric") return {
					read: histories.volumetricHistoryRead,
					write: histories.volumetricHistoryWrite,
					valid: historyValid,
				};
				if (id === "volumetric-reservoir") return {
					read: histories.volumetricReservoirHistoryRead,
					write: histories.volumetricReservoirHistoryWrite,
					valid: historyValid,
				};
				return {
					read: histories.motionHistoryRead,
					write: histories.motionHistoryWrite,
					valid: historyValid,
				};
			},
			getTransient: () => null,
			getShared: (id) => id === "backend:frame-hiz" ? transients.hiZ : null,
			copyGBufferToHistory: () => {
				motionWrites++;
			},
		},
	};
	const result = await request.pass.getImplementation("webgpu").execute(
		request,
		context
	);
	if (result.ran !== false) {
		published = targets.postPong;
		targets.sceneColor = targets.postPong;
	}
	return { result, published, motionWrites };
}

function createPerspectiveFrameContext(postProcessRequest = {}) {
	return {
		viewCamera: {
			type: "perspective",
			fov: 60,
			aspectRatio: 2,
		},
		features: {},
		postProcess: createResolvedPostProcess(postProcessRequest),
		transient: new Map(),
	};
}

function destroySnapshotPasses(snapshot) {
	for (const resolved of snapshot.getEnabledPasses()) {
		resolved.pass.destroy();
	}
}

async function testSSRAndVolumetricReportHistoryUpdates() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(createWebGPUComputeFacade(backend), () => {});
	const frameBinding = { label: "frame-binding" };

	const ssrTargets = createTemporalTargets(backend, 32, 16);
	const ssrHistories = createTemporalHistories(backend, 16, 8);
	const ssrFrameContext = createPerspectiveFrameContext({
		ssr: {
			enabled: true,
			options: {
				maxSteps: 24,
			},
		},
	});
	const ssrRun = await executeSSRImplementation(backend, runtime, {
		targets: ssrTargets,
		histories: ssrHistories,
		frameContext: ssrFrameContext,
		historyValid: true,
		frameBinding,
	});
	assert.deepEqual(ssrRun.result, {
		ran: true,
		updatedHistoryIds: ["ssr", "ssr:moments", "motion"],
	});
	assert.strictEqual(ssrRun.published, ssrTargets.postPing);
	assert.strictEqual(ssrTargets.sceneColor, ssrTargets.postPing);
	assert.equal(ssrRun.motionWrites, 1);
	const ssrImplementation = ssrFrameContext.postProcess
		.getPass("ssr")
		.pass.getImplementation("webgpu");
	const ssrDeclaration = ssrImplementation.describeExecution({
		options: ssrFrameContext.postProcess.getOptions("ssr"),
	});
	assert.deepEqual(
		ssrDeclaration.transients.map((entry) => entry.descriptor.id),
		["ssr:denoise-a", "ssr:denoise-b"]
	);
	assert.deepEqual(
		ssrDeclaration.histories
			.find((entry) => entry.descriptor.id === "ssr")
			.write,
		[
			{ access: "write", usage: "storage" },
			{ access: "read", usage: "sampled" },
		]
	);
	const traceBinding = backend.bindingGroups.find(
		(group) => group.label === "WebGPUSSR_TraceBinding"
	);
	const composeBinding = backend.bindingGroups.find(
		(group) => group.label === "WebGPUSSR_ComposeBinding"
	);
	const composeParams = composeBinding.entries[4].resource;
	assert.equal(composeParams.size, 8);
	assert.deepEqual(Array.from(composeParams.lastWrite), [1 / 32, 1 / 16]);
	assert.equal(backend.writeCalls.findLast((call) => call[1] === composeParams)[2].byteLength, 8);
	assert.strictEqual(
		traceBinding.entries[8].resource,
		ssrHistories.ssrHistoryWrite
	);
	assert.notStrictEqual(composeBinding.entries[1].resource, ssrHistories.ssrHistoryWrite,
		"Spatial reconstruction must not overwrite the next frame's temporal radiance.");
	assert.strictEqual(traceBinding.entries[9].resource, ssrHistories.ssrMomentsRead);
	assert.strictEqual(traceBinding.entries[10].resource, ssrHistories.ssrMomentsWrite);
	assert.strictEqual(
		composeBinding.entries[1].resource,
		ssrRun.transients.ssrDenoiseB
	);

	const volumetricTargets = createTemporalTargets(backend, 32, 16);
	const volumetricFrameContext = createPerspectiveFrameContext({
		volumetric: {
			enabled: true,
			options: {
				samples: 12,
			},
		},
	});
	const volumetricRun = await executeVolumetricImplementation(backend, runtime, {
		targets: volumetricTargets,
		frameContext: volumetricFrameContext,
		historyValid: true,
		frameBinding,
		lightingState: null,
		volumetricLightingData: {
			lights: [{
				type: 1,
				position: [2, 3, 4],
				range: 10,
				direction: [0, 0, 0],
				outerCos: -2,
				innerCos: -2,
				color: [0.5, 0.25, 0.125],
			}],
			warnings: [],
		},
	});
	assert.deepEqual(volumetricRun.result, {
		ran: true,
		updatedHistoryIds: ["volumetric", "volumetric-reservoir", "motion"],
	});
	assert.strictEqual(volumetricRun.published, volumetricTargets.postPong);
	assert.strictEqual(volumetricTargets.sceneColor, volumetricTargets.postPong);
	assert.equal(volumetricRun.motionWrites, 1);
	const lightBuffer = backend.buffers.find((buffer) => buffer.label === "WebGPUVolumetricLights");
	assert.deepEqual(Array.from(lightBuffer?.lastWrite?.data ?? []), [
		2, 3, 4, 10,
		0, 0, 0, -2,
		0.5, 0.25, 0.125, -2,
	]);

	destroySnapshotPasses(ssrFrameContext.postProcess);
	destroySnapshotPasses(volumetricFrameContext.postProcess);
	runtime.destroy();
}

async function testOrthographicTemporalPassesSkipAndReturnFalse() {
	const backend = new FakeBackend();
	const warnings = [];
	const runtime = new WebGPUPostProcessRuntime(
		createWebGPUComputeFacade(backend),
		(key, message) => warnings.push(`[${key}] ${message}`)
	);
	assert.equal(backend.shaderModules.length, 0);
	assert.equal(backend.computePipelines.length, 0);
	assert.equal(backend.buffers.length, 0);
	const frameContext = {
		viewCamera: {
			type: "orthographic",
		},
		features: {},
		postProcess: createResolvedPostProcess(),
		transient: new Map(),
	};
	const frameBinding = { label: "frame-binding" };

	const ssrRun = await executeSSRImplementation(backend, runtime, {
		targets: createTemporalTargets(backend),
		frameContext,
		historyValid: true,
		frameBinding,
	});
	assert.deepEqual(ssrRun.result, { ran: false });
	assert.equal(ssrRun.motionWrites, 0);

	const volumetricRun = await executeVolumetricImplementation(backend, runtime, {
		targets: createTemporalTargets(backend),
		frameContext,
		historyValid: true,
		frameBinding,
		lightingState: null,
	});
	assert.deepEqual(volumetricRun.result, { ran: false });
	assert.equal(volumetricRun.motionWrites, 0);

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

	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testHiZResourcesAreSharedAcrossTemporalPasses() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(createWebGPUComputeFacade(backend), () => {});
	const frameBinding = { label: "frame-binding" };
	const targets = createTemporalTargets(backend, 16, 8);
	const transients = createTemporalTransients(backend, 16, 8);
	const frameContext = createPerspectiveFrameContext({
		ssr: { enabled: true },
		volumetric: { enabled: true },
	});
	const encoder = new FakeEncoder(backend);
	await runtime.getHiZBuilder().build({
		encoder,
		depth: targets.gMotionDepth,
		hiZ: transients.hiZ,
	});

	await executeSSRImplementation(backend, runtime, {
		targets,
		frameContext,
		historyValid: false,
		frameBinding,
		transients,
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
		transients,
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
		transients,
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

	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

async function testSSRDestroyReleasesCachedBindings() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(createWebGPUComputeFacade(backend), () => {});
	const frameContext = createPerspectiveFrameContext({
		ssr: { enabled: true },
	});

	await executeSSRImplementation(backend, runtime, {
		frameContext,
		historyValid: true,
	});
	assert.ok(backend.bindingGroups.length > 2);

	const ssrPass = frameContext.postProcess.getEnabledPasses()
		.find((resolved) => resolved.id === "ssr")?.pass;
	assert.ok(ssrPass);
	ssrPass.destroy();

	assert.equal(backend.bindingGroupDestroyCalls, 2);
	assert.equal(backend.shaderModuleDestroyCalls, 1);
	assert.equal(backend.computePipelineDestroyCalls, 2);
	assert.equal(backend.bufferDestroyCalls, 2);

	runtime.destroy();
	assert.equal(
		backend.bindingGroupDestroyCalls,
		backend.bindingGroups.length
	);
	assert.equal(
		backend.shaderModuleDestroyCalls,
		backend.shaderModules.length
	);
	assert.equal(
		backend.computePipelineDestroyCalls,
		backend.computePipelines.length
	);
	assert.equal(backend.bufferDestroyCalls, backend.buffers.length);
}

async function testMissingSSRFrameBindingSkipsImplementation() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(createWebGPUComputeFacade(backend), () => {});
	const targets = createTemporalTargets(backend, 32, 16);
	const frameContext = createPerspectiveFrameContext();
	const ssrRun = await executeSSRImplementation(backend, runtime, {
		targets,
		frameContext,
		historyValid: true,
		frameBinding: undefined,
	});
	assert.deepEqual(ssrRun.result, { ran: false });
	assert.equal(ssrRun.motionWrites, 0);
	assert.equal(backend.textureViews.length, 0);

	destroySnapshotPasses(frameContext.postProcess);
	runtime.destroy();
}

function testCleanupPreservesErrorContext() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(createWebGPUComputeFacade(backend), () => {});
	let releasedBuffers = 0;
	const buffer = backend.resourceManager.registerBuffer({}, { size: 16 }, () => {
		releasedBuffers++;
		throw new Error("buffer release failed");
	});
	assert.throws(() => runtime.destroyManagedResource(buffer, "SSR params"),
		{ message: "Failed to destroy SSR params: buffer release failed" });
	assert.equal(releasedBuffers, 1);
	assert.doesNotThrow(() => runtime.destroyManagedResource(buffer, "SSR params"));

	let releasedGroups = 0;
	const group = backend.resourceManager.registerBindingGroup({}, "failed binding", () => {
		releasedGroups++;
		throw "binding release failed";
	});
	assert.throws(() => runtime.destroyBindingGroup(group),
		{ message: "Failed to destroy post-process binding group: binding release failed" });
	assert.equal(releasedGroups, 1);
	assert.doesNotThrow(() => runtime.destroyBindingGroup(group));
	assert.doesNotThrow(() => runtime.destroyManagedResource({}));
	assert.doesNotThrow(() => runtime.destroyManagedResource(null));
	assert.doesNotThrow(() => runtime.destroyBindingGroup(null));
	runtime.destroy();
}

async function run() {
	testCleanupPreservesErrorContext();
	await testSSRAndVolumetricReportHistoryUpdates();
	await testOrthographicTemporalPassesSkipAndReturnFalse();
	await testHiZResourcesAreSharedAcrossTemporalPasses();
	await testSSRDestroyReleasesCachedBindings();
	await testMissingSSRFrameBindingSkipsImplementation();
	console.log("WebGPU postprocess temporal runtime tests passed");
}

await run();
