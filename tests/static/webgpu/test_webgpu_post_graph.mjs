import assert from "node:assert/strict";
import {
	PostProcessGraphCompiler,
	resolvePostProcessExecutionOrder,
} from "../../../src/postprocess/index.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createPostProcess(overrides = {}) {
	return createResolvedPostProcess(overrides, "webgpu");
}

function createFrameContext(postProcess, incremental = {}) {
	return {
		camera: {
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
		normalSpace: "world",
		depthEncoding: "linear-view-z",
		motionEncoding: "ndc-delta",
		channels: {
			color: { semantic: "color", handle: { backend: "test" }, width: 64, height: 32 },
			depth: { semantic: "depth", handle: { backend: "test" }, width: 64, height: 32 },
			normal: { semantic: "normal", handle: { backend: "test" }, width: 64, height: 32 },
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

function testIncrementalStartPassIsResolvedByGraphCompiler() {
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
	const graph = new PostProcessGraphCompiler().compile({
		frameContext,
		backend: "webgpu",
		postProcess,
		gBuffer: createGBufferBridge(),
	});

	assert.equal(graph.startPassId, "color-filter");
	assert.deepEqual(
		graph.passes.map((pass) => pass.id),
		["color-filter", "fxaa", "gamma"]
	);
	assert.equal(graph.frameContext.incremental.postProcessStartPass, "color-filter");
}

async function run() {
	testBuiltInOrderUsesPipelineAuthority();
	testFogSceneModeSkipsFogInPipelineOrder();
	testIncrementalStartPassIsResolvedByGraphCompiler();
	console.log("WebGPU post-process pipeline-order tests passed");
}

await run();
