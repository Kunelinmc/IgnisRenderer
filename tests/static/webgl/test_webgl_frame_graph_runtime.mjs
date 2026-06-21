import assert from "node:assert/strict";
import { WebGLFrameGraphRuntime } from "../../../src/renderers/webgl/rendergraph/WebGLFrameGraphRuntime.ts";

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

function run() {
	testRuntimeExecutesOpaqueNodesInOrder();
	testRuntimeDelegatesPostProcessNode();
	testRuntimePlansOITParticleFlow();
	testRuntimeDebugCapturesUnsupportedStage();
	console.log("WebGL frame graph runtime tests passed");
}

run();
