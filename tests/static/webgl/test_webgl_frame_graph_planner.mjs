import assert from "node:assert/strict";
import { WebGLFrameGraphPlanner } from "../../../src/renderers/webgl/rendergraph/WebGLFrameGraphPlanner.ts";

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
		"oit-resolve",
		"transparent-legacy",
	]);
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
		"oit-resolve",
		"transparent-legacy",
		"particles",
	]);
	assert.ok(plan.nodes.every((node) => node.scope === "particles"));
}

function testPostProcessPlansSingleNode() {
	const planner = new WebGLFrameGraphPlanner();
	const plan = planner.planStage(
		createPass("postprocess"),
		createContext(),
		createState()
	);

	assert.deepEqual(plan.nodes.map((node) => node.kind), ["postprocess"]);
}

function run() {
	testBeginFramePlansClearAndEnvironment();
	testMainOpaquePlansDepthThenColor();
	testTransparentUsesLegacyWithoutOIT();
	testTransparentOITDefersResolveWhenParticlesExist();
	testTransparentOITResolvesWithoutParticles();
	testParticleOITPlansResolveAndAdditiveParticles();
	testPostProcessPlansSingleNode();
	console.log("WebGL frame graph planner tests passed");
}

run();
