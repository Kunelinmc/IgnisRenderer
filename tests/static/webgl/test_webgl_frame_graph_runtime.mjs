import assert from "node:assert/strict";
import {
	WebGLFrameGraphCompiler,
} from "../../../src/backends/webgl/rendergraph/WebGLFrameGraphCompiler.ts";
import {
	WebGLFrameGraphPlanner,
} from "../../../src/backends/webgl/rendergraph/WebGLFrameGraphPlanner.ts";
import {
	WebGLFrameGraphRuntime,
} from "../../../src/backends/webgl/rendergraph/WebGLFrameGraphRuntime.ts";
import {
	WebGLFrameNodeExecutorRegistry,
} from "../../../src/backends/webgl/rendergraph/WebGLFrameNodeExecutorRegistry.ts";
import {
	WebGLFrameTargetManager,
} from "../../../src/backends/webgl/WebGLFrameTargetManager.ts";
import {
	WebGLTemporalFrameState,
} from "../../../src/backends/webgl/WebGLTemporalFrameState.ts";
import { CameraType } from "../../../src/cameras/Camera.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";

function createContext(overrides = {}) {
	return {
		features: {
			enableEnvironment: false,
		},
		scene: {
			particleSystems: [],
			environment: {
				backgroundEnabled: false,
				backgroundTexture: null,
			},
		},
		incremental: {
			enabled: false,
			forceFullFrame: false,
			dirtyRects: [],
		},
		...overrides,
	};
}

function createPass(stage) {
	return {
		stage,
		executor: "backend",
		enabled: true,
		dependsOn: [],
	};
}

function testWebGLTemporalStateRollsBackAbortedFrames() {
	const temporal = new WebGLTemporalFrameState();
	const context = {
		viewCamera: {
			type: CameraType.Perspective,
			viewProjectionMatrix: new Matrix4(),
		},
		attachments: { width: 32, height: 16 },
		incremental: { temporalHistoryReset: false },
	};
	const requirements = {
		cameraJitter: { sequence: "halton-2-3", scale: 1 },
	};
	temporal.beginFrame(context, requirements);
	const first = Array.from(temporal.jitterCurrentPrev);
	temporal.abortFrame();
	temporal.beginFrame(context, requirements);
	assert.deepEqual(Array.from(temporal.jitterCurrentPrev), first);
	temporal.commitFrame();

	context.viewCamera.viewProjectionMatrix = new Matrix4([
		[1, 0, 0, 4],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	]);
	temporal.beginFrame(context, requirements);
	assert.ok(temporal.previousViewProjection);
	temporal.abortFrame();
}

function createExecutor(events, options = {}) {
	return {
		beginFrame() {
			events.push("begin");
			if (options.beginError) throw options.beginError;
		},
		beginTemporalFrame() {},
		clearFrameTargets() {
			events.push("clear");
		},
		renderEnvironmentNode() {
			events.push("environment");
		},
		isOITActive() {
			return options.oitActive === true;
		},
		hasPresentedInFrame() {
			return events.includes("present");
		},
		collectFrameGraphResources() {
			const resources = [
				"frame:scene-color",
				"frame:motion-depth",
				"frame:normal",
				"frame:depth",
				"frame:present-source",
				"post:color",
				"frame:transmission-background",
				"frame:transmission-linear-depth",
			];
			if (options.oitActive) {
				resources.push("oit:accum", "oit:reveal");
			}
			return resources;
		},
		collectFrameGraphResourceCatalog(includeShadowResources = true) {
			const ids = [
				"frame:scene-color",
				"frame:motion-depth",
				"frame:normal",
				"frame:depth",
				"frame:present-source",
				"post:color",
				"frame:transmission-background",
				"frame:transmission-linear-depth",
				"canvas:color",
			];
			if (options.oitActive) ids.push("oit:accum", "oit:reveal");
			if (includeShadowResources) {
				ids.push(
					"shadow:atlas",
					"shadow:transmittance",
					"shadow:particle-volume",
				);
			}
			return {
				resources: ids.map((id) => ({
					id,
					origin: "imported",
					kind: "texture",
					residency: id === "canvas:color" ? "external" : "frame",
					initialContent: "unknown",
					format: "rgba8unorm",
					width: 16,
					height: 16,
					depthOrArrayLayers: 1,
					dimension: "2d",
					sampleCount: 1,
					mipLevelCount: 1,
				})),
				bindings: ids.map((id) => ({
					resourceId: id,
					physicalId:
						id === "frame:scene-color" || id === "frame:present-source"
							? "webgl:scene-color"
							: `webgl:${id}`,
					kind: "texture",
				})),
			};
		},
		renderShadowNode() {
			events.push("shadow");
		},
		renderOpaqueDepthPrepass() {
			events.push("depth-prepass");
			return new Set(["opaque-0"]);
		},
		renderOpaqueScene(_context, earlyZPacketIds) {
			events.push(`opaque:${earlyZPacketIds.has("opaque-0")}`);
		},
		renderTransparentLegacy() {
			events.push("transparent");
		},
		prepareTransmissionDepth() {
			events.push("transmission-depth-copy");
		},
		renderLegacyTransparentSegment(_context, start, end) {
			events.push(`transparent:${start}-${end}`);
		},
		copyTransmissionBackground() {
			events.push("transmission-background-copy");
		},
		renderTransmissionPacket(_context, index) {
			events.push(`transmission-draw:${index}`);
		},
		prepareOITTransparent() {
			events.push("oit-clear:transparent");
		},
		renderOITTransparentAccum() {
			events.push("oit-accum:transparent");
		},
		renderOITTransparentReveal() {
			events.push("oit-reveal:transparent");
		},
		copySceneColorForOIT() {
			events.push("oit-copy-scene-color");
		},
		resolveOIT() {
			events.push("oit-resolve");
		},
		renderOITLegacyTransparent() {
			events.push("oit-legacy");
		},
		prepareOITParticles() {
			events.push("oit-clear:particles");
		},
		renderOITParticleAccum() {
			events.push("oit-accum:particles");
		},
		renderOITParticleReveal() {
			events.push("oit-reveal:particles");
		},
		renderParticlesLegacy() {
			events.push("particles");
		},
		renderOITAdditiveParticles() {
			events.push("particles:additive");
		},
		presentFrame() {
			if (options.presentError) throw options.presentError;
			events.push("present");
		},
		finishFrame() {
			events.push("finish");
		},
		abortFrame() {
			events.push("abort");
		},
	};
}

function createRuntime(events, options = {}) {
	const executor = createExecutor(events, options);
	const postProcessRuntime = {
		describeFrame(context) {
			return {
				backend: "webgl",
				postProcess: context.postProcess,
				frameContext: context,
				gBuffer: {},
				width: 16,
				height: 16,
				orderedPasses: [],
				passes: [],
				startPassId: null,
			};
		},
		buildRenderGraphFrame() {
			const payload = {
				passId: "test-pass",
				color: { access: "read", output: "new-version" },
				inputColor: "scene-color",
				plannedOutputColor: "color:0",
				compatibilityOpaque: false,
			};
			return {
				graph: {
					passes: options.emptyPostProcess ? [] : [{ id: "test-pass" }],
					frameRequirements: {},
				},
				subgraph: {
					resources: [
						{
							id: "scene-color",
							origin: "imported",
							kind: "texture",
							residency: "frame",
							initialContent: "valid",
							format: "rgba8unorm",
							width: 16,
							height: 16,
						},
						{
							id: "gbuffer:normal",
							origin: "imported",
							kind: "texture",
							residency: "frame",
							initialContent: "valid",
							format: "rgba8unorm",
							width: 16,
							height: 16,
						},
						{
							id: "gbuffer:roughness",
							origin: "imported",
							kind: "texture",
							residency: "frame",
							initialContent: "valid",
							format: "rgba8unorm",
							width: 16,
							height: 16,
						},
						{
							id: "color:0",
							origin: "graph",
							kind: "texture",
							residency: "transient",
							initialContent: "undefined",
							format: "rgba8unorm",
							width: 16,
							height: 16,
						},
					],
					nodes: [{
						id: "pass:test-pass",
						stage: "postprocess",
						kind: "postprocess-pass",
						dependsOn: [],
						creates: ["color:0"],
						resources: [
							{ resource: "scene-color", access: "read", usage: "sampled" },
							{ resource: "gbuffer:normal", access: "read", usage: "sampled" },
							{ resource: "gbuffer:roughness", access: "read", usage: "sampled" },
							{ resource: "color:0", access: "write", usage: "color-attachment" },
						],
						payload,
					}],
					imports: [
						{ name: "scene-color", resource: "scene-color" },
						{ name: "gbuffer:normal", resource: "gbuffer:normal" },
						{ name: "gbuffer:roughness", resource: "gbuffer:roughness" },
					],
					outputPorts: [{ name: "color", resource: "color:0" }],
					exports: [{ name: "color", resource: "color:0" }],
					outputColor: "color:0",
					resourceRoles: {
						"scene-color": "scene-color",
						"gbuffer:normal": "gbuffer",
						"gbuffer:roughness": "gbuffer",
						"color:0": "color-version",
					},
				},
			};
		},
		beginGraphFrame(plan) {
			events.push("postprocess:begin");
			return { graph: plan.graph, token: {}, compiled: plan };
		},
		executeGraphPass(_frame, passId) {
			events.push(`postprocess:${passId}`);
			return { ran: options.skipPostProcess !== true };
		},
		endGraphFrame() {
			events.push("postprocess:end");
		},
		resolveGraphColor(_frame, color) {
			return options.skipPostProcess ? "frame:scene-color" : color;
		},
		abortFrame() {
			events.push("postprocess:abort");
		},
	};
	return new WebGLFrameGraphRuntime(executor, postProcessRuntime);
}

function testRuntimeExecutesOpaqueNodesInOrder() {
	const events = [];
	const runtime = createRuntime(events);
	const context = createContext();

	runtime.beginFrame(context);
	runtime.executePass(createPass("main-opaque"), context);
	runtime.endFrame(context);
	runtime.commitGraphAnalysis();

	assert.deepEqual(events, [
		"begin",
		"clear",
		"depth-prepass",
		"opaque:true",
		"present",
		"finish",
	]);
	assert.deepEqual(runtime.getDebugState().lastExecutedNodeIds, [
		"webgl-begin-frame:scene-clear:frame",
		"main-opaque:opaque-depth-prepass:frame",
		"main-opaque:opaque-scene:frame",
		"webgl-present:present:frame",
	]);
	assert.equal(runtime.getDebugState().graphAnalysis.state, "committed");
	assert.equal(
		runtime.getDebugState().graphAnalysis.lastSuccessful.state,
		"committed",
	);
}

function testBeginFrameFailureAbortsPreparedServices() {
	const events = [];
	const failure = new Error("shadow allocation failed");
	const runtime = createRuntime(events, { beginError: failure });
	assert.throws(() => runtime.beginFrame(createContext()), failure);
	assert.deepEqual(events, ["begin", "abort"]);
}

async function testRuntimeDelegatesPostProcessNode() {
	const events = [];
	const runtime = createRuntime(events);
	const pass = createPass("postprocess");
	const context = createContext({
		framePlan: { stageOrder: [], backendPasses: [pass] },
	});

	runtime.beginFrame(context);
	await runtime.executePass(pass, context);

	assert.deepEqual(events.slice(2), [
		"postprocess:begin",
		"postprocess:test-pass",
		"postprocess:end",
	]);
	assert.ok(
		runtime
			.getDebugState()
			.lastPlannedNodeIds.includes("postprocess:pass:test-pass")
	);
}

async function testSkippedPostProcessRecordsExecutionOverlay() {
	const events = [];
	const runtime = createRuntime(events, { skipPostProcess: true });
	const pass = createPass("postprocess");
	const context = createContext({
		framePlan: { stageOrder: [], backendPasses: [pass] },
	});
	runtime.beginFrame(context);
	await runtime.executePass(pass, context);
	assert.deepEqual(
		runtime.getDebugState().graphAnalysis.current.executionOverlay,
		{
			skippedNodeIds: ["postprocess:pass:test-pass"],
			resourceAliases: [{
				resourceId: "postprocess:color:0",
				resolvedResourceId: "frame:scene-color",
			}],
		},
	);
}

function testRuntimePlansOITParticleFlow() {
	const events = [];
	const runtime = createRuntime(events, { oitActive: true });
	const context = createContext({
		scene: {
			particleSystems: [{}],
			environment: {
				backgroundEnabled: false,
				backgroundTexture: null,
			},
		},
	});

	runtime.beginFrame(context);
	runtime.executePass(createPass("main-transparent"), context);
	runtime.executePass(createPass("particles"), context);

	assert.deepEqual(events.slice(2), [
		"oit-clear:transparent",
		"oit-accum:transparent",
		"oit-reveal:transparent",
		"oit-clear:particles",
		"oit-accum:particles",
		"oit-reveal:particles",
		"oit-copy-scene-color",
		"oit-resolve",
		"oit-legacy",
		"particles:additive",
	]);
}

function testRuntimeExecutesTransmissionPacketsInSortedOrder() {
	const events = [];
	const runtime = createRuntime(events);
	const context = createContext({
		scene: {
			transparentPackets: [
				{ material: {} },
				{ material: { transmissionFactor: 1 } },
				{ material: {} },
			],
			particleSystems: [],
			environment: {
				backgroundEnabled: false,
				backgroundTexture: null,
			},
		},
	});
	runtime.beginFrame(context);
	runtime.executePass(createPass("main-transparent"), context);

	assert.deepEqual(events.slice(2), [
		"transmission-depth-copy",
		"transparent:0-1",
		"transmission-background-copy",
		"transmission-draw:1",
		"transparent:2-3",
	]);
}

function testRuntimeDebugCapturesUnsupportedStage() {
	const events = [];
	const runtime = createRuntime(events);
	const context = createContext();

	runtime.beginFrame(context);
	runtime.executePass(createPass("unsupported-stage"), context);

	assert.deepEqual(runtime.getDebugState().lastExecutedNodeIds, [
		"webgl-begin-frame:scene-clear:frame",
	]);
}

function testFailedPresentPreservesLastSuccessfulAnalysis() {
	const events = [];
	const options = {};
	const runtime = createRuntime(events, options);
	const context = createContext();

	runtime.beginFrame(context);
	runtime.endFrame(context);
	runtime.commitGraphAnalysis();
	const successful = runtime.getDebugState().graphAnalysis.lastSuccessful;

	const error = new Error("present failed");
	options.presentError = error;
	runtime.beginFrame(context);
	assert.throws(() => runtime.endFrame(context), /present failed/);
	runtime.abortFrame(error);
	const analysis = runtime.getDebugState().graphAnalysis;
	assert.equal(analysis.lastAttempt.state, "aborted");
	assert.equal(analysis.lastSuccessful, successful);
}

async function testWholeFrameCompilesOnceAndUsesPhysicalAlias() {
	const events = [];
	const runtime = createRuntime(events);
	let plannerCalls = 0;
	const originalPlanStage = runtime._planner.planStage.bind(runtime._planner);
	runtime._planner.planStage = (...args) => {
		plannerCalls++;
		return originalPlanStage(...args);
	};
	const passes = [createPass("main-opaque"), createPass("postprocess")];
	passes[1].dependsOn = ["main-opaque"];
	const context = createContext({
		framePlan: {
			stageOrder: [],
			backendPasses: passes,
		},
	});

	runtime.beginFrame(context);
	assert.equal(plannerCalls, 1);
	runtime.executePass(passes[0], context);
	await runtime.executePass(passes[1], context);
	assert.equal(plannerCalls, 1);
	runtime.endFrame(context);

	const debug = runtime.getDebugState();
	assert.deepEqual(debug.compiledStages.map((stage) => stage.pass.stage), [
		"webgl-begin-frame",
		"main-opaque",
		"postprocess",
		"webgl-present",
	]);
	assert.equal(debug.compiledGraph.resources.some((resource) =>
		resource.id === "shadow:atlas" ||
		resource.id === "shadow:transmittance" ||
		resource.id === "shadow:particle-volume"), false);
	assert.equal(debug.compiledGraph.shadowDiagnostics.some((diagnostic) =>
		diagnostic.resourceId === "shadow:atlas" ||
		diagnostic.resourceId === "shadow:transmittance" ||
		diagnostic.resourceId === "shadow:particle-volume"), false);
	assert.equal(debug.compiledGraph.completeness, "complete");
	assert.ok(debug.compiledGraph.dependencies.some((edge) =>
		edge.physicalId === "webgl:scene-color" &&
		edge.toNodeId === "postprocess:pass:test-pass"));
	assert.equal(debug.compiledGraph.transitions.filter((transition) =>
		transition.nodeId === "postprocess:pass:test-pass" &&
		transition.resourceId === "frame:normal").length, 2);
	const presentNodeId = debug.compiledStages
		.find((stage) => stage.pass.stage === "webgl-present")
		.nodes.at(-1).id;
	assert.ok(debug.compiledGraph.dependencies.some((edge) =>
		edge.fromNodeId === "postprocess:pass:test-pass" &&
		edge.toNodeId === presentNodeId));
}

function testWholeFrameShadowCatalogFollowsFramePlan() {
	const events = [];
	const runtime = createRuntime(events);
	const shadowPass = createPass("shadow");
	const opaquePass = createPass("main-opaque");
	opaquePass.dependsOn = ["shadow"];
	const context = createContext({
		framePlan: {
			stageOrder: [],
			backendPasses: [shadowPass, opaquePass],
		},
	});

	runtime.beginFrame(context);
	const graph = runtime.getDebugState().compiledGraph;
	assert.ok(graph.resources.some((resource) => resource.id === "shadow:atlas"));
	assert.ok(graph.resources.some(
		(resource) => resource.id === "shadow:transmittance",
	));
	assert.ok(graph.resources.some(
		(resource) => resource.id === "shadow:particle-volume",
	));
	assert.equal(graph.shadowDiagnostics.some((diagnostic) =>
		diagnostic.code === "read-content-unknown" &&
		(diagnostic.resourceId === "shadow:atlas" ||
			diagnostic.resourceId === "shadow:transmittance" ||
			diagnostic.resourceId === "shadow:particle-volume")), false);
	runtime.abortFrame();
}

function testEmptyPostProcessChainPresentsSceneColorDirectly() {
	const events = [];
	const runtime = createRuntime(events, { emptyPostProcess: true });
	const passes = [createPass("main-opaque"), createPass("postprocess")];
	passes[1].dependsOn = ["main-opaque"];
	const context = createContext({
		framePlan: { stageOrder: [], backendPasses: passes },
	});
	runtime.beginFrame(context);
	const debug = runtime.getDebugState();
	assert.deepEqual(
		debug.compiledStages.find((stage) => stage.pass.stage === "postprocess").nodes,
		[],
	);
	const presentNodeId = debug.compiledStages
		.find((stage) => stage.pass.stage === "webgl-present")
		.nodes.at(-1).id;
	assert.ok(debug.compiledGraph.dependencies.some((edge) =>
		edge.toNodeId === presentNodeId && edge.physicalId === "webgl:scene-color"));
}

function testNodeRegistryRejectsMissingAndDuplicateOwners() {
	assert.throws(
		() => new WebGLFrameNodeExecutorRegistry([]),
		/missing executors/,
	);
	const executor = () => {};
	assert.throws(
		() => new WebGLFrameNodeExecutorRegistry([
			["present", executor],
			["present", executor],
		]),
		/duplicate runtime owners/,
	);
}

function testResourceCatalogPreservesScenePresentAlias() {
	const manager = new WebGLFrameTargetManager({}, 4096, 4096);
	const sceneColor = {};
	manager._targetWidth = 32;
	manager._targetHeight = 16;
	manager._sceneColorTexture = sceneColor;
	manager._presentSourceTexture = sceneColor;
	const catalog = manager.collectGraphResourceCatalog();
	assert.equal(
		catalog.resources.some((entry) => entry.id.startsWith("post:ssao-")),
		false,
		"pass-owned SSAO transients must remain inside the post-process subgraph",
	);
	const sceneBinding = catalog.bindings.find((entry) =>
		entry.resourceId === "frame:scene-color");
	const presentBinding = catalog.bindings.find((entry) =>
		entry.resourceId === "frame:present-source");
	assert.equal(sceneBinding.physicalId, presentBinding.physicalId);
	assert.equal(
		manager.resolveGraphPhysicalResource(sceneBinding.physicalId),
		sceneColor,
	);
	assert.equal(catalog.resources.some((entry) =>
		entry.id === "shadow:atlas" ||
		entry.id === "shadow:transmittance" ||
		entry.id === "shadow:particle-volume"), false);
	assert.equal(catalog.bindings.some((entry) =>
		entry.resourceId === "shadow:atlas" ||
		entry.resourceId === "shadow:transmittance" ||
		entry.resourceId === "shadow:particle-volume"), false);
}

function testPreparedShadowResourcesCompileWithPhysicalBindings() {
	const manager = new WebGLFrameTargetManager({}, 4096, 4096);
	manager._targetWidth = 32;
	manager._targetHeight = 16;
	manager._sceneColorTexture = {};
	manager._sceneMotionTexture = {};
	manager._sceneDepthBuffer = {};
	const frameCatalog = manager.collectGraphResourceCatalog();
	const shadowResources = [
		{ id: "shadow:atlas", format: "depth", width: 1024, height: 768 },
		{ id: "shadow:transmittance", format: "rgba8unorm", width: 1024, height: 768 },
		{ id: "shadow:particle-volume", format: "r32float", width: 512, height: 1024 },
	].map((resource) => ({
		...resource,
		origin: "imported",
		kind: "texture",
		residency: "frame",
		initialContent: "unknown",
		depthOrArrayLayers: 1,
		dimension: "2d",
		sampleCount: 1,
		mipLevelCount: 1,
	}));
	const shadowBindings = shadowResources.map((resource) => ({
		resourceId: resource.id,
		physicalId: `webgl:slot:${resource.id}`,
		kind: "texture",
	}));
	const planner = new WebGLFrameGraphPlanner();
	const shadowPass = createPass("shadow");
	const opaquePass = createPass("main-opaque");
	opaquePass.dependsOn = ["shadow"];
	const context = createContext();
	const state = {
		oitActive: false,
		hasParticleSystems: false,
		hasEnvironmentBackground: false,
	};
	const compiled = new WebGLFrameGraphCompiler().compileFrame({
		resources: [...frameCatalog.resources, ...shadowResources],
		bindings: [...frameCatalog.bindings, ...shadowBindings],
		stages: [
			planner.planStage(shadowPass, context, state),
			planner.planStage(opaquePass, context, state),
		],
	});
	const enforcedErrors = compiled.graph.diagnostics.filter((diagnostic) =>
		diagnostic.enforcement === "enforced" && diagnostic.severity === "error");
	assert.deepEqual(enforcedErrors, []);
	assert.ok(compiled.graph.dependencies.some((edge) =>
		edge.fromNodeId === "shadow:shadow:frame" &&
		edge.toNodeId === "main-opaque:opaque-scene:frame" &&
		edge.resourceId === "shadow:atlas" &&
		edge.physicalId === "webgl:slot:shadow:atlas"));
	assert.ok(compiled.graph.transitions.some((transition) =>
		transition.fromNodeId === "shadow:shadow:frame" &&
		transition.nodeId === "main-opaque:opaque-scene:frame" &&
		transition.resourceId === "shadow:atlas" &&
		transition.reason === "read-after-write"));
	assert.ok(compiled.graph.transitions.some((transition) =>
		transition.fromNodeId === "shadow:shadow:frame" &&
		transition.nodeId === "main-opaque:opaque-scene:frame" &&
		transition.resourceId === "shadow:particle-volume" &&
		transition.reason === "read-after-write"));
}

async function run() {
	testWebGLTemporalStateRollsBackAbortedFrames();
	testRuntimeExecutesOpaqueNodesInOrder();
	testBeginFrameFailureAbortsPreparedServices();
	await testRuntimeDelegatesPostProcessNode();
	await testSkippedPostProcessRecordsExecutionOverlay();
	testRuntimePlansOITParticleFlow();
	testRuntimeExecutesTransmissionPacketsInSortedOrder();
	testRuntimeDebugCapturesUnsupportedStage();
	testFailedPresentPreservesLastSuccessfulAnalysis();
	await testWholeFrameCompilesOnceAndUsesPhysicalAlias();
	testWholeFrameShadowCatalogFollowsFramePlan();
	testEmptyPostProcessChainPresentsSceneColorDirectly();
	testNodeRegistryRejectsMissingAndDuplicateOwners();
	testResourceCatalogPreservesScenePresentAlias();
	testPreparedShadowResourcesCompileWithPhysicalBindings();
	console.log("WebGL frame graph runtime tests passed");
}

await run();
