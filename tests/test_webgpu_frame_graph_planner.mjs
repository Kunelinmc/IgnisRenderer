import assert from "node:assert/strict";
import { WebGPUFrameGraphPlanner } from "../src/renderers/webgpu/rendergraph/WebGPUFrameGraphPlanner.ts";

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
		deferredActive: false,
		oitActive: false,
		sceneTargetMode: "mrt",
		hasFrameTargets: true,
		hasMSAATargets: false,
		needsPlanarReflectionMask: false,
		...overrides,
	};
}

function run() {
	const planner = new WebGPUFrameGraphPlanner();
	const context = {};

	const opaque = planner.planStage(
		createPass("main-opaque"),
		context,
		createState({ deferredActive: true })
	);
	assert.deepEqual(
		opaque.nodes.map((node) => node.kind),
		["opaque-scene"]
	);
	assert.equal(opaque.nodes[0].label, "WebGPUOpaqueDeferred");

	const transparent = planner.planStage(
		createPass("main-transparent"),
		context,
		createState({ oitActive: true })
	);
	assert.deepEqual(
		transparent.nodes.map((node) => node.kind),
		["oit-transparent"]
	);

	const unknown = planner.planStage(
		createPass("custom-pass"),
		context,
		createState()
	);
	assert.deepEqual(unknown.nodes, []);

	console.log("test_webgpu_frame_graph_planner: ok");
}

run();
