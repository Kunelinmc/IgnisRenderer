import assert from "node:assert/strict";
import {
	PostProcessPlanner,
	PostProcessPass,
	PostProcessPassRegistry,
	resolvePostProcessExecutionOrder,
} from "../../../src/postprocess/index.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createPostProcess(overrides = {}) {
	return createResolvedPostProcess(overrides, "webgpu");
}

function createFrameContext(postProcess, incremental = {}) {
	return {
		viewCamera: {
			type: "perspective",
			fov: 60,
			aspectRatio: 1,
			near: 0.1,
			far: 100,
		},
		attachments: {
			width: 64,
			height: 32,
			pixels: new Uint8ClampedArray(64 * 32 * 4),
			depthBuffer: new Float32Array(64 * 32),
			normalBuffer: new Float32Array(64 * 32 * 3),
			motionBuffer: new Float32Array(64 * 32 * 2),
		},
		features: {},
		scene: {},
		shadowMaps: [],
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: null,
		postProcess,
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width: 64, height: 32 }],
			dirtyTileSize: 64,
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			postProcessStartPass: null,
			reasonMask: 0,
			temporalHistoryReset: false,
			...incremental,
		},
		transient: new Map(),
	};
}

function createGBufferBridge() {
	return {
		width: 64,
		height: 32,
		normalSpace: "view",
		depthEncoding: "linear-view-z",
		motionEncoding: "ndc-delta",
		channels: {
			color: { semantic: "color", handle: { backend: "test" }, width: 64, height: 32 },
			depth: { semantic: "depth", handle: { backend: "test" }, width: 64, height: 32 },
			normal: { semantic: "normal", handle: { backend: "test" }, width: 64, height: 32 },
			roughness: { semantic: "roughness", handle: { backend: "test" }, width: 64, height: 32 },
			metallic: { semantic: "metallic", handle: { backend: "test" }, width: 64, height: 32 },
			albedo: { semantic: "albedo", handle: { backend: "test" }, width: 64, height: 32 },
			motion: { semantic: "motion", handle: { backend: "test" }, width: 64, height: 32 },
		},
		worldPosition: {
			source: "derived",
			available: true,
		},
	};
}

function testBuiltInOrderUsesPipelineAuthority() {
	const order = resolvePostProcessExecutionOrder(
		createPostProcess({
			ssao: { enabled: true },
			ssgi: { enabled: true },
			taa: { enabled: true },
			ssr: { enabled: true },
			volumetric: { enabled: true },
			fog: { enabled: true, options: { application: "postprocess" } },
			"motion-blur": { enabled: true },
			dof: { enabled: true },
			bloom: { enabled: true },
			tonemap: { enabled: true },
			"color-filter": { enabled: true },
			fxaa: { enabled: true },
			gamma: { enabled: true },
		}),
		{ backend: "webgpu" }
	);
	assert.deepEqual(
		order.map((pass) => pass.id),
		[
			"ssao",
			"ssgi",
			"taa",
			"ssr",
			"volumetric",
			"fog",
			"motion-blur",
			"dof",
			"bloom",
			"tonemap",
			"color-filter",
			"fxaa",
			"gamma",
		]
	);
}

function testFogSceneModeSkipsFogInPipelineOrder() {
	const order = resolvePostProcessExecutionOrder(
		createPostProcess({
			volumetric: { enabled: true },
			fog: { enabled: true, options: { application: "scene" } },
			"motion-blur": { enabled: true },
		}),
		{ backend: "webgpu" }
	);
	assert.equal(order.some((pass) => pass.id === "fog"), false);
}

function testIncrementalStartPassIsResolvedByPlanner() {
	const postProcess = createPostProcess({
		bloom: { enabled: true },
		tonemap: { enabled: true },
		"color-filter": { enabled: true },
		fxaa: { enabled: true },
		gamma: { enabled: true },
	});
	const frameContext = createFrameContext(postProcess, {
		enabled: true,
		forceFullFrame: false,
		firstPass: "postprocess",
		postProcessStartPass: "color-filter",
	});
	const graph = new PostProcessPlanner().plan({
		frameContext,
		backend: "webgpu",
		postProcess,
		gBuffer: createGBufferBridge(),
		resolveImplementation: (pass) => pass.getImplementation("webgpu"),
		isSharedResourceAvailable: () => true,
	});

	assert.equal(graph.startPassId, "color-filter");
	assert.deepEqual(
		graph.passes.map((pass) => pass.id),
		["color-filter", "fxaa", "gamma"]
	);
	assert.equal(graph.frameContext.incremental.postProcessStartPass, "color-filter");
}

function testSSGIRequiresHiZAndSharesMotionHistory() {
	const planner = new PostProcessPlanner();
	const unavailableWarnings = [];
	const ssgiOnly = createPostProcess({ ssgi: { enabled: true } });
	const unavailable = planner.plan({
		frameContext: createFrameContext(ssgiOnly),
		backend: "webgpu",
		postProcess: ssgiOnly,
		gBuffer: createGBufferBridge(),
		resolveImplementation: (pass) => pass.getImplementation("webgpu"),
		isSharedResourceAvailable: () => false,
		warn: (key, message) => unavailableWarnings.push([key, message]),
	});
	assert.deepEqual(unavailable.passes, []);
	assert.deepEqual(unavailableWarnings, [[
		"postprocess-backend-shared-unavailable-ssgi",
		"Post-process pass \"ssgi\" requires unavailable shared resource " +
			"\"backend:frame-hiz\"; skipping it",
	]]);

	const temporal = createPostProcess({
		ssgi: { enabled: true },
		taa: { enabled: true },
		ssr: { enabled: true },
	});
	const graph = planner.plan({
		frameContext: createFrameContext(temporal),
		backend: "webgpu",
		postProcess: temporal,
		gBuffer: createGBufferBridge(),
		resolveImplementation: (pass) => pass.getImplementation("webgpu"),
		isSharedResourceAvailable: () => true,
	});
	const motionDescriptors = graph.historyDescriptors.filter(
		(descriptor) => descriptor.id === "motion"
	);
	assert.deepEqual(motionDescriptors, [{
		id: "motion",
		usage: ["sampled", "copy-dst", "render-target"],
	}]);
}

function planColorDomains(overrides, warnings = []) {
	const postProcess = createPostProcess(overrides);
	const frameContext = createFrameContext(postProcess);
	return new PostProcessPlanner().plan({
		frameContext,
		backend: "webgpu",
		postProcess,
		gBuffer: createGBufferBridge(),
		resolveImplementation: (pass) => pass.getImplementation("webgpu"),
		isSharedResourceAvailable: () => true,
		warn: (key, message) => warnings.push([key, message]),
		displayOutput: {
			requested: { mode: "hdr", exposure: 1, hdrHeadroom: 4 },
			activeDynamicRange: "hdr",
			colorSpace: "display-p3",
		},
	});
}

function testColorDomainPlanning() {
	let warnings = [];
	let graph = planColorDomains({
		tonemap: { enabled: true },
		gamma: { enabled: true },
	}, warnings);
	assert.equal(graph.initialColorDomain, "scene-linear-hdr");
	assert.equal(graph.outputColorDomain, "display-encoded");
	assert.deepEqual(warnings, []);

	warnings = [];
	graph = planColorDomains({
		tonemap: { enabled: false },
		gamma: { enabled: true },
	}, warnings);
	assert.deepEqual(graph.passes.map((pass) => pass.id), []);
	assert.equal(graph.outputColorDomain, "scene-linear-hdr");
	assert.equal(
		warnings.some(([key]) =>
			key === "postprocess-color-domain-mismatch-gamma"
		),
		true,
	);

	graph = planColorDomains({
		tonemap: { enabled: true },
		gamma: { enabled: false },
	});
	assert.equal(graph.outputColorDomain, "display-linear");
}

class LegacyHDRPass extends PostProcessPass {
	constructor() {
		super({
			id: "legacy-hdr",
			enabled: true,
			implementations: {
				webgpu: () => ({
					describeExecution: () => ({
						color: { access: "read", output: "new-version" },
					}),
					execute: () => ({ ran: true }),
				}),
			},
		});
	}
}

function testUndeclaredHDRCustomPassWarnsButRuns() {
	const registry = new PostProcessPassRegistry();
	registry.registerPass(new LegacyHDRPass());
	const postProcess = registry.createSnapshot("webgpu");
	const frameContext = createFrameContext(postProcess);
	const warnings = [];
	const graph = new PostProcessPlanner().plan({
		frameContext,
		backend: "webgpu",
		postProcess,
		gBuffer: createGBufferBridge(),
		resolveImplementation: (pass) => pass.getImplementation("webgpu"),
		isSharedResourceAvailable: () => true,
		warn: (key, message) => warnings.push([key, message]),
		displayOutput: {
			requested: { mode: "hdr", exposure: 1, hdrHeadroom: 4 },
			activeDynamicRange: "hdr",
			colorSpace: "display-p3",
		},
	});
	assert.deepEqual(graph.passes.map((pass) => pass.id), ["legacy-hdr"]);
	assert.equal(graph.outputColorDomain, "scene-linear-hdr");
	assert.equal(
		warnings.some(([key]) =>
			key === "postprocess-color-domain-undeclared-legacy-hdr"
		),
		true,
	);
}

async function run() {
	testBuiltInOrderUsesPipelineAuthority();
	testFogSceneModeSkipsFogInPipelineOrder();
	testIncrementalStartPassIsResolvedByPlanner();
	testSSGIRequiresHiZAndSharesMotionHistory();
	testColorDomainPlanning();
	testUndeclaredHDRCustomPassWarnsButRuns();
	console.log("WebGPU post-process pipeline-order tests passed");
}

await run();
