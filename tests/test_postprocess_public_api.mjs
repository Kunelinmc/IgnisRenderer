import assert from "node:assert/strict";
import {
	FastApproximateAntiAliasingPass,
	GammaPass,
	PostProcessHistoryManager,
	PostProcessPass,
	PostProcessPassRegistry,
	PostProcessPipeline,
	ScreenSpaceAmbientOcclusionPass,
	ScreenSpaceReflectionsPass,
	ToneMappingPass,
} from "../src/index.ts";
import {
	ALL_POST_PROCESS_CAPABILITIES,
	createNoopPostProcessSupport,
	createResolvedPostProcess,
} from "./helpers/postprocess.mjs";

class FakeExecutor {
	constructor(backend = "webgpu", capabilities = ALL_POST_PROCESS_CAPABILITIES) {
		this.backend = backend;
		this.capabilities = capabilities;
		this.created = [];
		this.destroyed = [];
		this.executed = [];
	}

	createResource(desc) {
		const handle = {
			id: desc.id,
			backend: this.backend,
			width: desc.width,
			height: desc.height,
			format: desc.format,
			resource: { desc },
		};
		this.created.push(handle);
		return handle;
	}

	destroyResource(handle) {
		this.destroyed.push(handle);
	}

	executePass(passId, request) {
		this.executed.push({
			passId,
			options: request.options,
			startPassId: request.startPassId,
			histories: request.histories,
		});
		return { ran: true };
	}

	getPassExecutionContext(passId, request) {
		if (this.backend === "software" && passId === "fxaa") {
			return {
				attachments: request.frameContext.attachments,
				canvasContext: null,
			};
		}
		return undefined;
	}
}

class CustomPass extends PostProcessPass {
	constructor(id, config = {}) {
		super({
			id,
			placement: config.placement,
			order: config.order,
			enabled: config.enabled,
			options: config.options,
			implementations: config.implementations ?? { webgpu: {} },
		});
	}
}

class SkippingPass extends CustomPass {
	execute() {
		return { ran: false };
	}
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
			motionBuffer: new Float32Array(64 * 32 * 4),
			albedoBuffer: new Float32Array(64 * 32 * 4),
		},
		features: {},
		postProcess,
		shadowMaps: [],
		scene: {},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: null,
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
			color: { semantic: "color", width: 64, height: 32, handle: {} },
			depth: { semantic: "depth", width: 64, height: 32, handle: {} },
			normal: { semantic: "normal", width: 64, height: 32, handle: {} },
			albedo: { semantic: "albedo", width: 64, height: 32, handle: {} },
			motion: { semantic: "motion", width: 64, height: 32, handle: {} },
		},
		worldPosition: {
			source: "derived",
			available: true,
		},
	};
}

function testRegistryOnlySurfaceAndPassMutation() {
	const registry = new PostProcessPassRegistry();
	assert.equal(typeof registry.registerPass, "function");
	assert.equal(typeof registry.getPass, "function");
	assert.equal(registry.enable, undefined);
	assert.equal(registry.disable, undefined);
	assert.equal(registry.setOptions, undefined);
	assert.equal(registry.reset, undefined);
	assert.throws(
		() => registry.registerPass({ id: "plain-object" }),
		/requires a PostProcessPass/
	);

	let changes = 0;
	const ssao = new ScreenSpaceAmbientOcclusionPass({
		enabled: true,
		options: { samples: 12 },
	});
	registry.on("change", () => changes++);
	registry.registerPass(ssao);
	ssao.setOptions({ radius: 4 });
	ssao.disable();
	assert.equal(registry.getPass("ssao"), ssao);
	assert.equal(changes, 3);
}

function testSnapshotNormalizationAndWarnings() {
	const registry = new PostProcessPassRegistry();
	registry.registerPass(
		new ScreenSpaceAmbientOcclusionPass({
			enabled: true,
			options: { samples: 500, radius: 2 },
		})
	);
	const snapshot = registry.createSnapshot(
		{ ...ALL_POST_PROCESS_CAPABILITIES, ssao: false },
		"software"
	);
	assert.equal(snapshot.isEnabled("ssao"), false);
	assert.ok(
		snapshot
			.getWarnings()
			.some((warning) => warning.key === "software-postprocess-unsupported-ssao")
	);

	const supported = registry.createSnapshot(ALL_POST_PROCESS_CAPABILITIES, "software");
	assert.equal(supported.isEnabled("ssao"), true);
	assert.equal(supported.getOptions("ssao").samples, 48);
	assert.equal(supported.getOptions("ssao").radius, 2);
}

async function testPipelineOrderingAndIncrementalStartPass() {
	const registry = new PostProcessPassRegistry();
	registry.registerPass(new ToneMappingPass({ enabled: true }));
	registry.registerPass(new GammaPass({ enabled: true }));
	registry.registerPass(new CustomPass("custom-hdr", {
		enabled: true,
		placement: "hdr",
	}));
	registry.registerPass(new CustomPass("custom-overlay", {
		enabled: true,
		placement: "overlay",
		order: -1,
	}));
	const snapshot = registry.createSnapshot(ALL_POST_PROCESS_CAPABILITIES, "webgpu");
	const pipeline = new PostProcessPipeline();
	const executor = new FakeExecutor("webgpu");
	await pipeline.execute({
		frameContext: createFrameContext(snapshot),
		executor,
		gBuffer: createGBufferBridge(),
	});

	const order = executor.executed.map((entry) => entry.passId);
	assert.deepEqual(order, ["custom-hdr", "tonemap", "custom-overlay", "gamma"]);

	const incrementalExecutor = new FakeExecutor("webgpu");
	const result = await pipeline.execute({
		frameContext: createFrameContext(snapshot, {
			enabled: true,
			forceFullFrame: false,
			firstPass: "postprocess",
			postProcessStartPass: "custom-overlay",
		}),
		executor: incrementalExecutor,
		gBuffer: createGBufferBridge(),
	});
	assert.equal(result.startPassId, "custom-overlay");
	assert.deepEqual(
		incrementalExecutor.executed.map((entry) => entry.passId),
		["custom-overlay", "gamma"]
	);
}

async function testPassOwnedImplementationsAndFallback() {
	const pipeline = new PostProcessPipeline();
	const fxaaSnapshot = createResolvedPostProcess(
		{ fxaa: { enabled: true } },
		ALL_POST_PROCESS_CAPABILITIES,
		"software"
	);
	const executor = new FakeExecutor("software");
	executor.executePass = function executePass(passId) {
		this.executed.push({ passId });
		throw new Error(`Unexpected fallback execution for ${passId}`);
	};
	const result = await pipeline.execute({
		frameContext: createFrameContext(fxaaSnapshot),
		executor,
		gBuffer: createGBufferBridge(),
	});
	assert.deepEqual(result.executedPassIds, ["fxaa"]);
	assert.deepEqual(executor.executed, []);

	const fallbackSnapshot = createResolvedPostProcess(
		{ bloom: { enabled: true } },
		ALL_POST_PROCESS_CAPABILITIES,
		"webgpu"
	);
	const fallbackExecutor = new FakeExecutor("webgpu");
	await pipeline.execute({
		frameContext: createFrameContext(fallbackSnapshot),
		executor: fallbackExecutor,
		gBuffer: createGBufferBridge(),
	});
	assert.deepEqual(fallbackExecutor.executed.map((entry) => entry.passId), ["bloom"]);
}

async function testSSRHistorySignatureUsesOptions() {
	const pipeline = new PostProcessPipeline();
	const executor = new FakeExecutor("webgpu");
	const createSnapshot = (downsample) =>
		createResolvedPostProcess(
			{ ssr: { enabled: true, options: { downsample } } },
			ALL_POST_PROCESS_CAPABILITIES,
			"webgpu"
		);

	await pipeline.execute({
		frameContext: createFrameContext(createSnapshot(2)),
		executor,
		gBuffer: createGBufferBridge(),
	});
	const firstSSRRead = executor.created.find((handle) => handle.id === "ssr:read");
	assert.equal(firstSSRRead.width, 32);
	assert.equal(firstSSRRead.height, 16);

	await pipeline.execute({
		frameContext: createFrameContext(createSnapshot(4)),
		executor,
		gBuffer: createGBufferBridge(),
	});
	const recreatedSSRReads = executor.created.filter(
		(handle) => handle.id === "ssr:read"
	);
	assert.equal(recreatedSSRReads.at(-1).width, 16);
	assert.equal(recreatedSSRReads.at(-1).height, 8);
	assert.ok(executor.destroyed.some((handle) => handle.id === "ssr:read"));
}

async function testRanFalsePassIsExcludedFromExecutedIds() {
	const registry = new PostProcessPassRegistry();
	registry.registerPass(new SkippingPass("custom-skip", { enabled: true }));
	const snapshot = registry.createSnapshot(ALL_POST_PROCESS_CAPABILITIES, "webgpu");
	const pipeline = new PostProcessPipeline();
	const executor = new FakeExecutor("webgpu");
	const result = await pipeline.execute({
		frameContext: createFrameContext(snapshot),
		executor,
		gBuffer: createGBufferBridge(),
	});
	assert.deepEqual(result.executedPassIds, []);
}

function testHistoryManagerInvalidationAndResize() {
	const manager = new PostProcessHistoryManager();
	const executor = new FakeExecutor("webgpu");
	const descriptors = [{ id: "taa", format: "rgba16float" }];

	let slots = manager.prepare({
		executor,
		descriptors,
		width: 32,
		height: 16,
		reset: false,
		signature: "camera-a",
	});
	assert.equal(slots.taa.valid, false);
	assert.equal(executor.created.length, 2);

	const firstRead = slots.taa.read;
	const firstWrite = slots.taa.write;
	manager.markUpdated("taa");
	manager.endFrame();
	slots = manager.prepare({
		executor,
		descriptors,
		width: 32,
		height: 16,
		reset: false,
		signature: "camera-a",
	});
	assert.equal(slots.taa.valid, true);
	assert.strictEqual(slots.taa.read, firstWrite);
	assert.strictEqual(slots.taa.write, firstRead);

	slots = manager.prepare({
		executor,
		descriptors,
		width: 64,
		height: 16,
		reset: false,
		signature: "camera-a",
	});
	assert.equal(slots.taa.valid, false);
	assert.equal(executor.created.length, 4);
	assert.equal(executor.destroyed.length, 2);
}

function testLogicalGBufferBridgeHelperShape() {
	const support = createNoopPostProcessSupport(
		"software",
		ALL_POST_PROCESS_CAPABILITIES
	);
	const bridge = support.createGBufferBridge(
		createFrameContext(
			createResolvedPostProcess({}, ALL_POST_PROCESS_CAPABILITIES, "software")
		)
	);
	assert.equal(bridge.width, 64);
	assert.equal(bridge.height, 32);
	assert.equal(bridge.normalSpace, "world");
	assert.equal(bridge.depthEncoding, "linear-view-z");
	assert.equal(bridge.worldPosition.source, "derived");
	assert.equal(bridge.worldPosition.available, true);
	assert.equal(bridge.channels.depth.handle.backend, "software");
	assert.equal(bridge.channels.normal.handle.backend, "software");
}

async function run() {
	testRegistryOnlySurfaceAndPassMutation();
	testSnapshotNormalizationAndWarnings();
	await testPipelineOrderingAndIncrementalStartPass();
	await testPassOwnedImplementationsAndFallback();
	await testSSRHistorySignatureUsesOptions();
	await testRanFalsePassIsExcludedFromExecutedIds();
	testHistoryManagerInvalidationAndResize();
	testLogicalGBufferBridgeHelperShape();
	console.log("Postprocess public API tests passed");
}

await run();
