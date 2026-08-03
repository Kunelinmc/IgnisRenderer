import assert from "node:assert/strict";

import * as frameExecutorFixture from "../../helpers/webgpu_frame_executor_resilience.mjs";

const {
	BackendPostProcessRuntime,
	FakeBackend,
	Material,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	WebGPUFrameExecutor,
	WebGPUPostProcessExecutor,
	createFrameContext,
	createResolvedPostProcess,
	createResourcesStub,
	getFrameGraphDebugState,
	initializeIsolatedWebGPUTestState,
} = frameExecutorFixture;

const restoreTestState = initializeIsolatedWebGPUTestState();

async function testWholeFramePlanningOccursOnlyAtBeginFrame() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.incremental = {
		enabled: false,
		forceFullFrame: true,
		dirtyRects: [],
	};
	const passes = [
		{ stage: "main-opaque", executor: "backend", enabled: true, dependsOn: [] },
		{
			stage: "postprocess",
			executor: "backend",
			enabled: true,
			dependsOn: ["main-opaque"],
		},
	];
	context.framePlan = { stageOrder: [], backendPasses: passes };
	let plannerCalls = 0;
	const originalPlanStage = executor._graphPlanner.planStage.bind(executor._graphPlanner);
	executor._graphPlanner.planStage = (...args) => {
		plannerCalls++;
		return originalPlanStage(...args);
	};

	executor.beginFrame(context);
	assert.equal(plannerCalls, 1);
	await executor.executePass(passes[0], context);
	await executor.executePass(passes[1], context);
	assert.equal(plannerCalls, 1);
	await executor.endFrame();

	const debug = getFrameGraphDebugState(executor);
	assert.deepEqual(debug.compiledStages.map((stage) => stage.pass.stage), [
		"webgpu-setup",
		"main-opaque",
		"postprocess",
		"webgpu-present",
	]);
	assert.equal(debug.compiledGraph.resources.some(
		(resource) =>
			resource.id === "shadow-atlas" ||
			resource.id.startsWith("paged-shadow:"),
	), false);
	assert.equal(debug.compiledGraph.shadowDiagnostics.some((diagnostic) =>
		diagnostic.resourceId === "shadow-atlas" ||
		diagnostic.resourceId?.startsWith("paged-shadow:")), false);
	assert.equal(debug.compiledGraph.completeness, "coarse");
	const postStage = debug.compiledStages.find(
		(stage) => stage.pass.stage === "postprocess",
	);
	assert.deepEqual(postStage.nodes.map((node) => node.id), [
		"postprocess:pass:ssao",
		"postprocess:pass:taa",
	]);
	assert.ok(postStage.nodes.every((node) => node.kind === "post-process-pass"));
	assert.ok(debug.compiledGraph.dependencies.some((edge) =>
		edge.toNodeId === "postprocess:pass:ssao"));
	assert.ok(debug.compiledGraph.dependencies.some((edge) =>
		edge.fromNodeId === "postprocess:pass:ssao" &&
		edge.toNodeId === "postprocess:pass:taa"));
	const presentNodeId = debug.compiledStages
		.find((stage) => stage.pass.stage === "webgpu-present")
		.nodes.at(-1).id;
	assert.ok(debug.compiledGraph.dependencies.some((edge) =>
		edge.fromNodeId === "postprocess:pass:taa" &&
		edge.toNodeId === presentNodeId));
}

function testParticleSimulationDefersFrameSealing() {
	const backend = new FakeBackend();
	const resources = createResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess({}, "webgpu");
	const particlePass = {
		stage: "particle-sim",
		executor: "backend",
		enabled: true,
		dependsOn: [],
	};
	const opaquePass = {
		stage: "main-opaque",
		executor: "backend",
		enabled: true,
		dependsOn: ["particle-sim"],
	};
	context.framePlan = {
		stageOrder: [],
		backendPasses: [particlePass, opaquePass],
	};

	executor.beginFrame(context);
	assert.equal(backend.createCommandEncoderCalls, 0);
	assert.equal(getFrameGraphDebugState(executor).compiledGraph, null);

	const material = new Material({ name: "simulated-mesh-particle" });
	const primitive = {
		id: "simulated-mesh-particle-primitive",
		material,
		geometry: {},
		boundingSphere: {
			center: { x: 0, y: 0, z: 0 },
			radius: 1,
		},
	};
	const mesh = {
		primitives: [primitive],
		defaultMorphWeights: [],
	};
	context.transient.set(PARTICLE_MESH_TRANSIENT_BATCHES_KEY, [
		{
			kind: "mesh",
			systemId: "simulated-system",
			templateIndex: 0,
			mesh,
			primitive,
			material,
			receiveShadows: true,
			castShadows: true,
			shadowDensity: 1,
			shadowSoftness: 1,
			particles: [
				{
					templateIndex: 0,
					position: { x: 0, y: 0, z: -2 },
					previousPosition: { x: 0, y: 0, z: -2 },
					size: 1,
					color: { r: 255, g: 255, b: 255, a: 1 },
					rotation: 0,
					previousRotation: 0,
					depth: 2,
				},
			],
		},
	]);

	executor.sealParticleSimulation(context);
	assert.equal(backend.createCommandEncoderCalls, 1);
	assert.equal(executor._session.framePackets.opaque.length, 1);
	assert.ok(getFrameGraphDebugState(executor).compiledGraph);
	assert.equal(resources._state.particleShadowVolumeUpdates.length, 1);
	assert.strictEqual(resources._state.particleShadowVolumeUpdates[0], context);
	executor.abortFrame();
}

function testWholeFrameShadowCatalogFollowsFramePlan() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess({}, "webgpu");
	const shadowPass = {
		stage: "shadow",
		executor: "backend",
		enabled: true,
		dependsOn: [],
	};
	const opaquePass = {
		stage: "main-opaque",
		executor: "backend",
		enabled: true,
		dependsOn: ["shadow"],
	};
	context.framePlan = {
		stageOrder: [],
		backendPasses: [shadowPass, opaquePass],
	};

	executor.beginFrame(context);
	const graph = getFrameGraphDebugState(executor).compiledGraph;
	assert.ok(graph.resources.some(
		(resource) => resource.id === "shadow-atlas",
	));
	assert.equal(graph.shadowDiagnostics.some((diagnostic) =>
		diagnostic.code === "read-content-unknown" &&
		diagnostic.resourceId === "shadow-atlas"), false);
	executor.abortFrame();
}

function testSSGIWholeFrameGraphCompilation() {
	const backend = new FakeBackend();
	const postProcessExecutor = new WebGPUPostProcessExecutor(backend);
	backend.postProcessRuntime = new BackendPostProcessRuntime({
		executor: postProcessExecutor,
		backend,
	});
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const postProcessPort = executor.createPostProcessSessionPort();
	postProcessExecutor.bindSession(postProcessPort);
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess({
		ssgi: { enabled: true },
	}, "webgpu");
	context.scene.transparentPackets = [
		{ id: "glass", material: { transmissionFactor: 1 } },
	];
	context.incremental = {
		enabled: false,
		forceFullFrame: true,
		dirtyRects: [],
	};
	context.framePlan = {
		stageOrder: [],
		backendPasses: [
			{ stage: "main-opaque", executor: "backend", enabled: true, dependsOn: [] },
			{
				stage: "main-transparent",
				executor: "backend",
				enabled: true,
				dependsOn: ["main-opaque"],
			},
			{
				stage: "postprocess",
				executor: "backend",
				enabled: true,
				dependsOn: ["main-transparent"],
			},
		],
	};

	executor.beginFrame(context);
	const debug = getFrameGraphDebugState(executor);
	const ssgiNode = debug.compiledGraph.nodes.find(
		(node) => node.id === "postprocess:pass:ssgi",
	);
	assert.ok(ssgiNode);
	assert.equal(ssgiNode.internalAccesses, "ordered");
	assert.ok(!debug.compiledGraph.diagnostics.some(
		(entry) =>
			entry.code === "incompatible-subgraph-port" ||
			entry.code === "physical-feedback-loop",
	));
	executor.abortFrame();
	postProcessExecutor.unbindSession(postProcessPort);
}

async function run() {
	try {
		await testWholeFramePlanningOccursOnlyAtBeginFrame();
		await testParticleSimulationDefersFrameSealing();
		await testWholeFrameShadowCatalogFollowsFramePlan();
		await testSSGIWholeFrameGraphCompilation();
		console.log("WebGPU frame-executor whole-frame planning tests passed");
	} finally {
		restoreTestState();
	}
}

await run();
