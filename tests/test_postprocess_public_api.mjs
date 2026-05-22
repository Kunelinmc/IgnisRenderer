import assert from "node:assert/strict";
import {
	DEFAULT_POST_PROCESS_CAPABILITIES,
	PostProcessController,
	resolvePostProcessState,
} from "../src/pipeline/PostProcessController.ts";
import {
	PostProcessHistoryManager,
	PostProcessPipeline,
} from "../src/postprocess/index.ts";
import {
	ALL_POST_PROCESS_CAPABILITIES,
	createNoopPostProcessSupport,
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
			startPassId: request.startPassId,
			histories: request.histories,
		});
		return {
			ran: true,
			updatedHistoryIds: request.pass.history?.map((history) => history.id) ?? [],
		};
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
			motionBuffer: new Float32Array(64 * 32 * 2),
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
			color: {
				semantic: "color",
				width: 64,
				height: 32,
				handle: { backend: "test", resource: "color" },
			},
			depth: {
				semantic: "depth",
				width: 64,
				height: 32,
				handle: { backend: "test", resource: "depth" },
			},
			normal: {
				semantic: "normal",
				width: 64,
				height: 32,
				handle: { backend: "test", resource: "normal" },
			},
			motion: {
				semantic: "motion",
				width: 64,
				height: 32,
				handle: { backend: "test", resource: "motion" },
			},
		},
		worldPosition: {
			source: "derived",
			available: true,
		},
	};
}

function testControllerDefaultsAndOptionsMerge() {
	const controller = new PostProcessController();
	let state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);

	assert.equal(state.enabled.tonemap, true);
	assert.equal(state.enabled.gamma, true);
	assert.equal(state.enabled["interaction-outline"], true);
	assert.equal(state.enabled.ssao, false);

	controller.setOptions("ssao", {
		radius: 4,
		intensity: 0.5,
	});
	state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);
	assert.equal(state.enabled.ssao, false);
	assert.equal(state.options.ssao.radius, 4);
	assert.equal(state.options.ssao.intensity, 0.5);

	controller.enable("ssao", {
		samples: 12,
		radius: 6,
	});
	state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);
	assert.equal(state.enabled.ssao, true);
	assert.equal(state.options.ssao.samples, 12);
	assert.equal(state.options.ssao.radius, 6);
	assert.equal(state.options.ssao.intensity, 0.5);

	controller.disable("tonemap");
	state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);
	assert.equal(state.enabled.tonemap, false);

	controller.reset("ssao");
	state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);
	assert.equal(state.enabled.ssao, false);
	assert.equal(state.options.ssao.radius, 8);
	assert.equal(state.options.ssao.intensity, 1);

	controller.reset();
	state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);
	assert.equal(state.enabled.tonemap, true);
	assert.equal(state.enabled.gamma, true);
	assert.equal(state.enabled["interaction-outline"], true);
	assert.equal(state.enabled.ssao, false);
}

function testUnsupportedExplicitEnableWarning() {
	const defaultState = resolvePostProcessState(
		{},
		DEFAULT_POST_PROCESS_CAPABILITIES,
		"software"
	);
	assert.equal(defaultState.enabled.tonemap, false);
	assert.equal(defaultState.enabled.gamma, false);
	assert.equal(defaultState.warnings.length, 0);

	const controller = new PostProcessController();
	controller.enable("ssr");

	const state = resolvePostProcessState(
		controller.getState(),
		{
			...ALL_POST_PROCESS_CAPABILITIES,
			ssr: false,
		},
		"software"
	);

	assert.equal(state.enabled.ssr, false);
	assert.ok(
		state.warnings.some(
			(warning) => warning.key === "software-postprocess-unsupported-ssr"
		)
	);
}

function testLogicalCustomPassRegistration() {
	const calls = {
		changed: 0,
		registered: [],
		unregistered: [],
	};
	const controller = new PostProcessController(undefined, {
		onRegisterPass(pass) {
			calls.registered.push(pass);
		},
		onUnregisterPass(id) {
			calls.unregistered.push(id);
		},
		onChange() {
			calls.changed++;
		},
	});
	const pass = {
		id: "custom-edge",
		implementations: {
			webgpu: {},
		},
	};

	assert.throws(
		() => controller.enable("custom-edge"),
		/Unknown post-process pass/
	);

	controller.registerPass(pass).enable("custom-edge", {
		strength: 0.75,
	});

	const state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"webgpu"
	);
	assert.strictEqual(calls.registered[0], pass);
	assert.equal(state.enabled["custom-edge"], true);
	assert.deepEqual(state.options["custom-edge"], { strength: 0.75 });

	controller.disable("custom-edge");
	const disabledState = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"webgpu"
	);
	assert.equal(disabledState.enabled["custom-edge"], false);

	controller.unregisterPass("custom-edge");
	assert.deepEqual(calls.unregistered, ["custom-edge"]);
	assert.throws(
		() => controller.enable("custom-edge"),
		/Unknown post-process pass/
	);
	assert.ok(calls.changed >= 4);
}

async function testPipelineBackendImplementationSelection() {
	const controller = new PostProcessController();
	const pass = {
		id: "custom-webgpu-only",
		dependsOn: ["tonemap"],
		isEnabled: (state) => state.enabled["custom-webgpu-only"],
		implementations: {
			webgpu: {},
		},
	};
	controller.registerPass(pass).enable("custom-webgpu-only");
	controller.disable("gamma");
	const state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"webgpu"
	);
	const pipeline = new PostProcessPipeline();
	pipeline.registerPass(pass);

	const webgpuExecutor = new FakeExecutor("webgpu");
	await pipeline.execute({
		frameContext: createFrameContext(state),
		executor: webgpuExecutor,
		gBuffer: createGBufferBridge(),
	});
	assert.ok(
		webgpuExecutor.executed.some(
			(entry) => entry.passId === "custom-webgpu-only"
		)
	);

	const softwareExecutor = new FakeExecutor("software");
	await pipeline.execute({
		frameContext: createFrameContext(state),
		executor: softwareExecutor,
		gBuffer: createGBufferBridge(),
	});
	assert.equal(
		softwareExecutor.executed.some(
			(entry) => entry.passId === "custom-webgpu-only"
		),
		false
	);
}

async function testPipelineDiagnosticsAndIncrementalStartPass() {
	const pipeline = new PostProcessPipeline();
	const warnings = [];
	pipeline.registerPass({
		id: "missing-dependency-pass",
		dependsOn: ["not-registered"],
		isEnabled: (state) => state.enabled["missing-dependency-pass"],
		implementations: { webgpu: {} },
	});
	pipeline.registerPass({
		id: "cycle-a",
		dependsOn: ["cycle-b"],
		isEnabled: (state) => state.enabled["cycle-a"],
		implementations: { webgpu: {} },
	});
	pipeline.registerPass({
		id: "cycle-b",
		dependsOn: ["cycle-a"],
		isEnabled: (state) => state.enabled["cycle-b"],
		implementations: { webgpu: {} },
	});

	const state = resolvePostProcessState(
		{
			fxaa: { enabled: true },
			"missing-dependency-pass": { enabled: true },
			"cycle-a": { enabled: true },
			"cycle-b": { enabled: true },
		},
		ALL_POST_PROCESS_CAPABILITIES,
		"webgpu"
	);
	const executor = new FakeExecutor("webgpu");
	const result = await pipeline.execute({
		frameContext: createFrameContext(state, {
			enabled: true,
			forceFullFrame: false,
			firstPass: "postprocess",
			postProcessStartPass: "fxaa",
		}),
		executor,
		gBuffer: createGBufferBridge(),
		warn: (key, message) => warnings.push({ key, message }),
	});

	assert.equal(result.firstStage, "postprocess");
	assert.equal(result.startPassId, "fxaa");
	assert.equal(executor.executed[0].passId, "fxaa");
	assert.equal(
		executor.executed.some((entry) => entry.passId === "tonemap"),
		false
	);
	assert.ok(
		warnings.some((warning) =>
			warning.key.startsWith("postprocess-dependency-missing-")
		)
	);
	assert.ok(
		warnings.some((warning) =>
			warning.key.startsWith("postprocess-cycle-")
		)
	);
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
		width: 32,
		height: 16,
		reset: false,
		signature: "camera-b",
	});
	assert.equal(slots.taa.valid, false);
	assert.equal(executor.created.length, 2);

	manager.markUpdated("taa");
	manager.endFrame();
	slots = manager.prepare({
		executor,
		descriptors,
		width: 64,
		height: 16,
		reset: false,
		signature: "camera-b",
	});
	assert.equal(slots.taa.valid, false);
	assert.equal(executor.created.length, 4);
	assert.equal(executor.destroyed.length, 2);

	manager.markUpdated("taa");
	manager.endFrame();
	slots = manager.prepare({
		executor,
		descriptors,
		width: 64,
		height: 16,
		reset: true,
		signature: "camera-b",
	});
	assert.equal(slots.taa.valid, false);
	assert.equal(executor.created.length, 4);
}

function testLogicalGBufferBridgeHelperShape() {
	const support = createNoopPostProcessSupport(
		"software",
		ALL_POST_PROCESS_CAPABILITIES
	);
	const bridge = support.createGBufferBridge(
		createFrameContext(
			resolvePostProcessState({}, ALL_POST_PROCESS_CAPABILITIES, "software")
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
	testControllerDefaultsAndOptionsMerge();
	testUnsupportedExplicitEnableWarning();
	testLogicalCustomPassRegistration();
	await testPipelineBackendImplementationSelection();
	await testPipelineDiagnosticsAndIncrementalStartPass();
	testHistoryManagerInvalidationAndResize();
	testLogicalGBufferBridgeHelperShape();
	console.log("Postprocess public API tests passed");
}

await run();
