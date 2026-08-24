import assert from "node:assert/strict";

import { WebGLFrameGraphCompiler } from "../../../src/backends/webgl/rendergraph/WebGLFrameGraphCompiler.ts";
import { WebGLFrameGraphPlanner } from "../../../src/backends/webgl/rendergraph/WebGLFrameGraphPlanner.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

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

function testParticleShadowVolumeGraphOwnershipMatchesShaderConsumers() {
	const planner = new WebGLFrameGraphPlanner();
	const context = createContext();
	const shadow = planner.planStage(
		createPass("shadow"),
		context,
		createState()
	);
	const particleWrite = shadow.nodes[0].writes.find((write) =>
		write.id === "shadow:particle-volume");
	assert.deepEqual(particleWrite, {
		id: "shadow:particle-volume",
		usage: "copy-target",
		optional: true,
	});

	const opaque = planner.planStage(
		createPass("main-opaque"),
		context,
		createState()
	);
	assert.ok(opaque.nodes.find((node) => node.kind === "opaque-scene")
		.reads.some((read) => read.id === "shadow:particle-volume"));
	const transparent = planner.planStage(
		createPass("main-transparent"),
		context,
		createState({ oitActive: true, hasParticleSystems: false })
	);
	for (const node of transparent.nodes.filter((candidate) =>
		candidate.kind === "oit-accum" ||
		candidate.kind === "oit-reveal" ||
		candidate.kind === "transparent-legacy")) {
		assert.ok(node.reads.some((read) =>
			read.id === "shadow:particle-volume"));
	}
	const particles = planner.planStage(
		createPass("particles"),
		context,
		createState({ oitActive: true, hasParticleSystems: true })
	);
	assert.equal(particles.nodes.some((node) =>
		(node.reads ?? []).some((read) =>
			read.id === "shadow:particle-volume")), false);
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

function testTransmissionCreatesOrderedPacketNodes() {
	const planner = new WebGLFrameGraphPlanner();
	const context = createContext({
		scene: {
			transparentPackets: [
				createTestDrawPacket({ material: {} }),
				createTestDrawPacket({ material: { transmissionFactor: 1 } }),
				createTestDrawPacket({ material: {} }),
			],
			particleSystems: [],
			environment: {
				backgroundEnabled: false,
				backgroundTexture: null,
			},
		},
	});
	const plan = planner.planStage(
		createPass("main-transparent"),
		context,
		createState({ oitActive: false }),
	);

	assert.deepEqual(plan.nodes.map((node) => node.kind), [
		"transmission-depth-copy",
		"transparent-legacy",
		"transmission-background-copy",
		"transmission-draw",
		"transparent-legacy",
	]);
	assert.deepEqual(
		plan.nodes.filter((node) => node.kind === "transparent-legacy")
			.map((node) => [node.packetStart, node.packetEnd]),
		[[0, 1], [2, 3]],
	);
	assert.equal(plan.nodes[2].packetIndex, 1);
	assert.equal(plan.nodes[3].packetIndex, 1);
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
	testParticleShadowVolumeGraphOwnershipMatchesShaderConsumers();
	testTransparentUsesLegacyWithoutOIT();
	testTransmissionCreatesOrderedPacketNodes();
	testTransparentOITDefersResolveWhenParticlesExist();
	testTransparentOITResolvesWithoutParticles();
	testParticleOITPlansResolveAndAdditiveParticles();
	testPostProcessIsComposedByFrameRuntime();
	console.log("WebGL frame graph planner tests passed");
}

run();
