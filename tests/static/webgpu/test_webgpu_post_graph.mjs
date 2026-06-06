import assert from "node:assert/strict";
import { PostProcessPipeline } from "../../../src/postprocess/index.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createPostProcess(overrides = {}) {
	return createResolvedPostProcess(overrides, "webgpu");
}

function createExecutor() {
	const shared = {
		sampler: null,
		compute: {
			createShaderModule: async () => ({ label: "fxaa-shader" }),
			createComputePipeline: () => ({ label: "fxaa-pipeline" }),
			createBuffer: () => ({ label: "fxaa-params" }),
			writeBuffer() {},
		},
		async ensureCommonResources() {
			this.sampler = this.sampler ?? { label: "fxaa-sampler" };
		},
		getCachedBindGroup() {
			return { label: "fxaa-binding" };
		},
	};
	return {
		backend: "webgpu",
		createResource() {
			throw new Error("Unexpected history allocation in this test");
		},
		destroyResource() {},
		executePass() {
			return { ran: true };
		},
		getPassExecutionContext(request) {
			const passId = request.passId;
			const targets = {
				sceneColor: { width: 64, height: 32, label: "scene" },
				postPing: { width: 64, height: 32, label: "ping" },
				postPong: { width: 64, height: 32, label: "pong" },
				gMotionDepth: { width: 64, height: 32, label: "motion-depth" },
			};
			if (
				[
					"motion-blur",
					"dof",
					"tonemap",
					"color-filter",
					"interaction-outline",
				].includes(passId)
			) {
				return {
					encoder: {
						beginComputePass() {},
						setComputePipeline() {},
						setBindingGroup() {},
						dispatchWorkgroups() {},
						endComputePass() {},
					},
					targets,
					shared,
					publishColorTarget(texture) {
						targets.sceneColor = texture;
					},
				};
			}
			if (passId === "gamma") {
				return {
					targets,
					presentToCanvas() {},
				};
			}
			if (passId !== "fxaa") {
				return undefined;
			}
			return {
				encoder: {
					beginComputePass() {},
					setComputePipeline() {},
					setBindingGroup() {},
					dispatchWorkgroups() {},
					endComputePass() {},
				},
				targets,
				shared,
				publishColorTarget(texture) {
					targets.sceneColor = texture;
				},
			};
		},
	};
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
	const pipeline = new PostProcessPipeline();
	const order = pipeline.getExecutionOrder(
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
			"interaction-outline": { enabled: true },
			gamma: { enabled: true },
		}),
		createExecutor()
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
			"interaction-outline",
			"gamma",
		]
	);
}

function testFogSceneModeSkipsFogInPipelineOrder() {
	const pipeline = new PostProcessPipeline();
	const order = pipeline.getExecutionOrder(
		createPostProcess({
			volumetric: { enabled: true },
			fog: { enabled: true, options: { application: "scene" } },
			"motion-blur": { enabled: true },
		}),
		createExecutor()
	);
	assert.equal(order.some((pass) => pass.id === "fog"), false);
}

async function testIncrementalStartPassIsResolvedByPipeline() {
	const pipeline = new PostProcessPipeline();
	const executed = [];
	const executor = {
		...createExecutor(),
		executePass(passId, request) {
			executed.push({ passId, startPassId: request.startPassId });
			return { ran: true };
		},
	};
	const postProcess = createPostProcess({
		bloom: { enabled: true },
		tonemap: { enabled: true },
		"color-filter": { enabled: true },
		fxaa: { enabled: true },
		gamma: { enabled: true },
	});
	const result = await pipeline.execute({
		frameContext: createFrameContext(postProcess, {
			enabled: true,
			forceFullFrame: false,
			firstPass: "postprocess",
			postProcessStartPass: "color-filter",
		}),
		executor,
		gBuffer: createGBufferBridge(),
	});

	assert.equal(result.startPassId, "color-filter");
	assert.deepEqual(
		result.executedPassIds,
		["color-filter", "fxaa", "gamma"]
	);
	assert.deepEqual(
		executed.map((entry) => entry.passId),
		[]
	);
	assert.ok(executed.every((entry) => entry.startPassId === "color-filter"));
}

async function run() {
	testBuiltInOrderUsesPipelineAuthority();
	testFogSceneModeSkipsFogInPipelineOrder();
	await testIncrementalStartPassIsResolvedByPipeline();
	console.log("WebGPU post-process pipeline-order tests passed");
}

await run();
