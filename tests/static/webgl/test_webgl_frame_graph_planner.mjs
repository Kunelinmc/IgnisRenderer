import assert from "node:assert/strict";

import { WebGLFrameGraphCompiler } from "../../../src/backends/webgl/rendergraph/WebGLFrameGraphCompiler.ts";
import { WebGLFrameGraphPlanner } from "../../../src/backends/webgl/rendergraph/WebGLFrameGraphPlanner.ts";

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

function createState(overrides = {}) {
	return {
		oitActive: false,
		hasParticleSystems: false,
		hasEnvironmentBackground: false,
		...overrides,
	};
}

function testBeginFramePlansClearAndEnvironment() {
	const planner = new WebGLFrameGraphPlanner();
	const nodes = planner.planBeginFrame(
		createContext(),
		createState({ hasEnvironmentBackground: true })
	);

	assert.deepEqual(nodes.map((node) => node.kind), [
		"scene-clear",
		"environment",
	]);
}

function testMainOpaquePlansDepthThenColor() {
	const planner = new WebGLFrameGraphPlanner();
	const plan = planner.planStage(
		createPass("main-opaque"),
		createContext(),
		createState()
	);

	assert.deepEqual(plan.nodes.map((node) => node.kind), [
		"opaque-depth-prepass",
		"opaque-scene",
	]);
}

function testTransparentUsesLegacyWithoutOIT() {
	const planner = new WebGLFrameGraphPlanner();
	const plan = planner.planStage(
		createPass("main-transparent"),
		createContext(),
		createState({ oitActive: false })
	);

	assert.deepEqual(plan.nodes.map((node) => node.kind), [
		"transparent-legacy",
	]);
}

function testTransparentOITDefersResolveWhenParticlesExist() {
	const planner = new WebGLFrameGraphPlanner();
	const plan = planner.planStage(
		createPass("main-transparent"),
		createContext(),
		createState({ oitActive: true, hasParticleSystems: true })
	);

	assert.deepEqual(plan.nodes.map((node) => node.kind), [
		"oit-clear",
		"oit-accum",
		"oit-reveal",
	]);
	assert.ok(plan.nodes.every((node) => node.scope === "transparent"));
}

function testTransparentOITResolvesWithoutParticles() {
	const planner = new WebGLFrameGraphPlanner();
	const plan = planner.planStage(
		createPass("main-transparent"),
		createContext(),
		createState({ oitActive: true, hasParticleSystems: false })
	);

	assert.deepEqual(plan.nodes.map((node) => node.kind), [
		"oit-clear",
		"oit-accum",
		"oit-reveal",
		"oit-copy-scene-color",
		"oit-resolve",
		"transparent-legacy",
	]);
	const copyNode = plan.nodes.find((node) =>
		node.kind === "oit-copy-scene-color");
	const resolveNode = plan.nodes.find((node) => node.kind === "oit-resolve");
	assert.deepEqual(copyNode.reads, [{
		id: "frame:scene-color",
		usage: "texture-sampling",
		optional: false,
	}]);
	assert.deepEqual(copyNode.writes, [{
		id: "post:color",
		usage: "framebuffer-color",
		optional: false,
	}]);
	assert.deepEqual(resolveNode.reads.map((read) => read.id), [
		"post:color",
		"oit:accum",
		"oit:reveal",
	]);
	assert.deepEqual(resolveNode.writes.map((write) => write.id), [
		"frame:scene-color",
	]);

	const compiler = new WebGLFrameGraphCompiler();
	compiler.beginFrame([
		"frame:scene-color",
		"frame:depth",
		"post:color",
		"oit:accum",
		"oit:reveal",
	]);
	compiler.compileStage(plan);
	const transitions = compiler.getCompiledFrame().graph.transitions;
	assert.ok(transitions.some((transition) =>
		transition.nodeId === copyNode.id &&
		transition.resourceId === "frame:scene-color" &&
		transition.access === "read"));
	assert.ok(transitions.some((transition) =>
		transition.fromNodeId === copyNode.id &&
		transition.nodeId === resolveNode.id &&
		transition.resourceId === "post:color" &&
		transition.reason === "read-after-write"));
	assert.ok(transitions.some((transition) =>
		transition.fromNodeId === copyNode.id &&
		transition.nodeId === resolveNode.id &&
		transition.resourceId === "frame:scene-color" &&
		transition.reason === "write-after-read"));
	assert.equal(transitions.some((transition) =>
		transition.resourceId === "post:color" &&
		transition.scope === "intra-node"), false);
}

function testParticleOITPlansResolveAndAdditiveParticles() {
	const planner = new WebGLFrameGraphPlanner();
	const plan = planner.planStage(
		createPass("particles"),
		createContext(),
		createState({ oitActive: true, hasParticleSystems: true })
	);

	assert.deepEqual(plan.nodes.map((node) => node.kind), [
		"oit-clear",
		"oit-accum",
		"oit-reveal",
		"oit-copy-scene-color",
		"oit-resolve",
		"transparent-legacy",
		"particles",
	]);
	assert.ok(plan.nodes.every((node) => node.scope === "particles"));
}

function testPostProcessIsComposedByFrameRuntime() {
	const planner = new WebGLFrameGraphPlanner();
	const plan = planner.planStage(
		createPass("postprocess"),
		createContext(),
		createState()
	);

	assert.deepEqual(plan.nodes, []);
}

function run() {
	testBeginFramePlansClearAndEnvironment();
	testMainOpaquePlansDepthThenColor();
	testTransparentUsesLegacyWithoutOIT();
	testTransparentOITDefersResolveWhenParticlesExist();
	testTransparentOITResolvesWithoutParticles();
	testParticleOITPlansResolveAndAdditiveParticles();
	testPostProcessIsComposedByFrameRuntime();
	console.log("WebGL frame graph planner tests passed");
}

run();
