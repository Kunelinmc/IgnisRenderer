import assert from "node:assert/strict";
import {
	BloomPass,
	FastApproximateAntiAliasingPass,
	FogPass,
	GammaPass,
	hasPostProcessExecutionPasses,
	PostProcessHistoryManager,
	PostProcessPass,
	PostProcessPassRegistry,
	PostProcessPipeline,
	resolvePostProcessExecutionOrder,
	ScreenSpaceAmbientOcclusionPass,
	ScreenSpaceGlobalIlluminationPass,
	ScreenSpaceReflectionsPass,
	ToneMappingPass,
	VolumetricLightingPass,
} from "../src/index.ts";
import {
	createNoopPostProcessSupport,
	createResolvedPostProcess,
} from "./helpers/postprocess.mjs";

class FakeExecutor {
	constructor(backend = "webgpu") {
		this.backend = backend;
		this.created = [];
		this.destroyed = [];
		this.executed = [];
		this.ownedExecuted = [];
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
			implementation: request.implementation,
			options: request.options,
			startPassId: request.startPassId,
			histories: request.histories,
		});
		return { ran: true };
	}

	getPassExecutionContext(request) {
		const passId = request.passId;
		if (
			this.backend === "webgpu" &&
			[
				"motion-blur",
				"dof",
				"tonemap",
				"color-filter",
				"interaction-outline",
			].includes(passId)
		) {
			const targets = {
				sceneColor: { id: "scene", width: 64, height: 32 },
				postPing: { id: "ping", width: 64, height: 32 },
				postPong: { id: "pong", width: 64, height: 32 },
				gMotionDepth: { id: "motion-depth", width: 64, height: 32 },
			};
			return {
				encoder: {
					beginComputePass() {},
					setComputePipeline() {},
					setBindingGroup() {},
					dispatchWorkgroups() {},
					endComputePass() {},
				},
				targets,
				shared: {
					sampler: { id: "sampler" },
					compute: {
						createShaderModule: async (desc) => ({ label: desc.label }),
						createComputePipeline: (desc) => ({ label: desc.label }),
						createBuffer: (desc) => ({ label: desc.label }),
						writeBuffer() {},
					},
					async ensureCommonResources() {},
					getCachedBindGroup(_key, _pipeline, _entries, label) {
						return { label };
					},
					invalidateBindingsByPrefix() {},
					destroyManagedResource() {},
				},
				publishColorTarget: (texture) => {
					targets.sceneColor = texture;
					this.ownedExecuted.push(passId);
				},
			};
		}
		if (this.backend === "webgpu" && passId === "gamma") {
			return {
				targets: {
					sceneColor: { id: "scene" },
				},
				presentToCanvas: () => {
					this.ownedExecuted.push(passId);
				},
			};
		}
		if (
			this.backend === "software" &&
			["tonemap", "color-filter", "interaction-outline", "gamma"].includes(passId)
		) {
			return {
				canvasContext: {},
			};
		}
		if (this.backend === "software" && passId === "fxaa") {
			return {
				attachments: request.frameContext.attachments,
				canvasContext: null,
			};
		}
		if (this.backend === "software" && passId === "volumetric") {
			return {
				canvasContext: {},
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

class ConditionalPass extends CustomPass {
	shouldExecute(request) {
		return request.frameContext?.transient.get("run-conditional-pass") === true;
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
	assert.equal(typeof registry.invalidatePasses, "function");
	assert.equal(typeof registry.destroyPasses, "function");
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
	assert.equal(ssao.builtIn, true);
	assert.equal("capabilityId" in ssao, false);
	assert.equal(ssao.warningLabel, "SSAO");
	ssao.setOptions({ radius: 4 });
	ssao.disable();
	assert.equal(registry.getPass("ssao"), ssao);
	assert.equal(changes, 3);
}

function testSnapshotNormalizationAndWarnings() {
	const unsupportedRegistry = new PostProcessPassRegistry();
	unsupportedRegistry.registerPass(
		new ScreenSpaceGlobalIlluminationPass({ enabled: true })
	);
	const snapshot = unsupportedRegistry.createSnapshot("software");
	const unsupportedWarning = snapshot
		.getWarnings()
		.find((warning) => warning.key === "software-postprocess-unsupported-ssgi");
	assert.equal(snapshot.isEnabled("ssgi"), false);
	assert.ok(unsupportedWarning);
	assert.ok(unsupportedWarning.message.includes("SSGI post-processing"));

	const registry = new PostProcessPassRegistry();
	registry.registerPass(
		new ScreenSpaceAmbientOcclusionPass({
			enabled: true,
			options: { samples: 500, radius: 2 },
		})
	);
	const supported = registry.createSnapshot("software");
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
	assert.equal(registry.getPass("custom-hdr").builtIn, false);
	assert.equal("capabilityId" in registry.getPass("custom-hdr"), false);
	const snapshot = registry.createSnapshot("webgpu");
	const pipeline = new PostProcessPipeline();
	const executor = new FakeExecutor("webgpu");
	const result = await pipeline.execute({
		frameContext: createFrameContext(snapshot),
		executor,
		gBuffer: createGBufferBridge(),
	});

	assert.deepEqual(result.executedPassIds, [
		"custom-hdr",
		"tonemap",
		"custom-overlay",
		"gamma",
	]);
	assert.deepEqual(
		executor.executed.map((entry) => entry.passId),
		["custom-hdr", "custom-overlay"]
	);
	assert.ok(executor.executed.every((entry) => entry.implementation));
	assert.deepEqual(executor.ownedExecuted, ["tonemap", "gamma"]);

	const incrementalExecutor = new FakeExecutor("webgpu");
	const incrementalResult = await pipeline.execute({
		frameContext: createFrameContext(snapshot, {
			enabled: true,
			forceFullFrame: false,
			firstPass: "postprocess",
			postProcessStartPass: "custom-overlay",
		}),
		executor: incrementalExecutor,
		gBuffer: createGBufferBridge(),
	});
	assert.equal(incrementalResult.startPassId, "custom-overlay");
	assert.deepEqual(incrementalResult.executedPassIds, ["custom-overlay", "gamma"]);
	assert.deepEqual(
		incrementalExecutor.executed.map((entry) => entry.passId),
		["custom-overlay"]
	);
	assert.deepEqual(incrementalExecutor.ownedExecuted, ["gamma"]);
}

function testExecutionPredicatesDrivePipelineWork() {
	const registry = new PostProcessPassRegistry();
	registry.registerPass(
		new ConditionalPass("custom-conditional", {
			enabled: true,
		})
	);
	const snapshot = registry.createSnapshot("webgpu");
	const frameContext = createFrameContext(snapshot);

	assert.equal(
		hasPostProcessExecutionPasses(snapshot, { frameContext }),
		false
	);
	assert.deepEqual(
		resolvePostProcessExecutionOrder(snapshot, { frameContext }).map(
			(pass) => pass.id
		),
		[]
	);

	frameContext.transient.set("run-conditional-pass", true);
	assert.equal(
		hasPostProcessExecutionPasses(snapshot, { frameContext }),
		true
	);
	assert.deepEqual(
		resolvePostProcessExecutionOrder(snapshot, { frameContext }).map(
			(pass) => pass.id
		),
		["custom-conditional"]
	);
}

async function testPassOwnedImplementationsAndFallback() {
	const pipeline = new PostProcessPipeline();
	const fxaaSnapshot = createResolvedPostProcess(
		{ fxaa: { enabled: true } },
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

	const webgpuSnapshot = createResolvedPostProcess(
		{
			volumetric: { enabled: true },
			fog: { enabled: true, options: { application: "postprocess" } },
			bloom: { enabled: true },
		},
		"webgpu"
	);
	const webgpuExecutor = new FakeExecutor("webgpu");
	webgpuExecutor.executePass = function executePass(passId) {
		this.executed.push({ passId });
		throw new Error(`Unexpected fallback execution for ${passId}`);
	};
	await pipeline.execute({
		frameContext: createFrameContext(webgpuSnapshot),
		executor: webgpuExecutor,
		gBuffer: createGBufferBridge(),
	});
	assert.deepEqual(webgpuExecutor.executed, []);

	const volumetricSnapshot = createResolvedPostProcess(
		{ volumetric: { enabled: true } },
		"software"
	);
	const volumetricExecutor = new FakeExecutor("software");
	volumetricExecutor.executePass = function executePass(passId) {
		this.executed.push({ passId });
		throw new Error(`Unexpected fallback execution for ${passId}`);
	};
	const volumetricResult = await pipeline.execute({
		frameContext: createFrameContext(volumetricSnapshot),
		executor: volumetricExecutor,
		gBuffer: createGBufferBridge(),
	});
	assert.deepEqual(volumetricResult.executedPassIds, ["volumetric"]);
	assert.deepEqual(volumetricExecutor.executed, []);

	const fallbackSnapshot = createResolvedPostProcess(
		{ tonemap: { enabled: true } },
		"software"
	);
	const fallbackExecutor = new FakeExecutor("software");
	const fallbackResult = await pipeline.execute({
		frameContext: createFrameContext(fallbackSnapshot),
		executor: fallbackExecutor,
		gBuffer: createGBufferBridge(),
	});
	assert.deepEqual(fallbackResult.executedPassIds, ["tonemap"]);
	assert.deepEqual(fallbackExecutor.executed.map((entry) => entry.passId), []);
}

function testRegistryLifecycleDelegatesToPassImplementations() {
	const registry = new PostProcessPassRegistry();
	const passes = [
		new FogPass({ enabled: true }),
		new BloomPass({ enabled: true }),
		new VolumetricLightingPass({ enabled: true }),
	];
	const calls = [];
	for (const pass of passes) {
		registry.registerPass(pass);
		const implementation = pass.getImplementation("webgpu");
		implementation.invalidate = () => {
			calls.push(`${pass.id}:invalidate`);
		};
		implementation.destroy = () => {
			calls.push(`${pass.id}:destroy`);
		};
	}

	assert.equal(registry.invalidatePasses("software"), registry);
	assert.deepEqual(calls, []);
	assert.equal(registry.invalidatePasses("webgpu"), registry);
	assert.deepEqual(calls, [
		"fog:invalidate",
		"bloom:invalidate",
		"volumetric:invalidate",
	]);
	assert.equal(registry.destroyPasses("webgpu"), registry);
	assert.deepEqual(calls.slice(3), [
		"fog:destroy",
		"bloom:destroy",
		"volumetric:destroy",
	]);

	calls.length = 0;
	registry.unregisterPass("bloom");
	assert.deepEqual(calls, ["bloom:destroy"]);
}

async function testSSRHistorySignatureUsesOptions() {
	const pipeline = new PostProcessPipeline();
	const executor = new FakeExecutor("webgpu");
	const createSnapshot = (downsample) =>
		createResolvedPostProcess(
			{ ssr: { enabled: true, options: { downsample } } },
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
	const snapshot = registry.createSnapshot("webgpu");
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
		"software"
	);
	const bridge = support.createGBufferBridge(
		createFrameContext(
			createResolvedPostProcess({}, "software")
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
	testExecutionPredicatesDrivePipelineWork();
	await testPassOwnedImplementationsAndFallback();
	testRegistryLifecycleDelegatesToPassImplementations();
	await testSSRHistorySignatureUsesOptions();
	await testRanFalsePassIsExcludedFromExecutedIds();
	testHistoryManagerInvalidationAndResize();
	testLogicalGBufferBridgeHelperShape();
	console.log("Postprocess public API tests passed");
}

await run();
