import assert from "node:assert/strict";
import { WebGLFrameGraphRuntime } from "../../../src/backends/webgl/rendergraph/WebGLFrameGraphRuntime.ts";
import { WebGLFrameNodeExecutorRegistry } from "../../../src/backends/webgl/rendergraph/WebGLFrameNodeExecutorRegistry.ts";
import { WebGLFrameTargetManager } from "../../../src/backends/webgl/WebGLFrameTargetManager.ts";

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

function createExecutor(events, options = {}) {
	return {
		beginFrame() {
			events.push("begin");
		},
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
			];
			if (options.oitActive) {
				resources.push("oit:accum", "oit:reveal");
			}
			return resources;
		},
		collectFrameGraphResourceCatalog() {
			const ids = [
				"frame:scene-color",
				"frame:motion-depth",
				"frame:normal",
				"frame:depth",
				"frame:present-source",
				"post:color",
				"canvas:color",
			];
			if (options.oitActive) ids.push("oit:accum", "oit:reveal");
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
		prepareOITTransparent() {
			events.push("oit-clear:transparent");
		},
		renderOITTransparentAccum() {
			events.push("oit-accum:transparent");
		},
		renderOITTransparentReveal() {
			events.push("oit-reveal:transparent");
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
		execute() {
			events.push("postprocess");
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

function testRuntimeDelegatesPostProcessNode() {
	const events = [];
	const runtime = createRuntime(events);
	const context = createContext();

	runtime.beginFrame(context);
	runtime.executePass(createPass("postprocess"), context);

	assert.ok(events.includes("postprocess"));
	assert.ok(
		runtime
			.getDebugState()
			.lastPlannedNodeIds.includes("postprocess:postprocess:frame")
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
		"oit-resolve",
		"oit-legacy",
		"particles:additive",
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

function testWholeFrameCompilesOnceAndUsesPhysicalAlias() {
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
	assert.equal(plannerCalls, 2);
	runtime.executePass(passes[0], context);
	runtime.executePass(passes[1], context);
	assert.equal(plannerCalls, 2);
	runtime.endFrame(context);

	const debug = runtime.getDebugState();
	assert.deepEqual(debug.compiledStages.map((stage) => stage.pass.stage), [
		"webgl-begin-frame",
		"main-opaque",
		"postprocess",
		"webgl-present",
	]);
	assert.equal(debug.compiledGraph.completeness, "coarse");
	assert.ok(debug.compiledGraph.dependencies.some((edge) =>
		edge.physicalId === "webgl:scene-color" &&
		edge.toNodeId === "postprocess:postprocess:frame"));
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
	const catalog = manager.collectGraphResourceCatalog(null, null);
	const sceneBinding = catalog.bindings.find((entry) =>
		entry.resourceId === "frame:scene-color");
	const presentBinding = catalog.bindings.find((entry) =>
		entry.resourceId === "frame:present-source");
	assert.equal(sceneBinding.physicalId, presentBinding.physicalId);
	assert.equal(
		manager.resolveGraphPhysicalResource(sceneBinding.physicalId),
		sceneColor,
	);
}

function run() {
	testRuntimeExecutesOpaqueNodesInOrder();
	testRuntimeDelegatesPostProcessNode();
	testRuntimePlansOITParticleFlow();
	testRuntimeDebugCapturesUnsupportedStage();
	testFailedPresentPreservesLastSuccessfulAnalysis();
	testWholeFrameCompilesOnceAndUsesPhysicalAlias();
	testNodeRegistryRejectsMissingAndDuplicateOwners();
	testResourceCatalogPreservesScenePresentAlias();
	console.log("WebGL frame graph runtime tests passed");
}

run();
