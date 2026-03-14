import assert from "node:assert/strict";
import { RendererStageGraph } from "../src/pipeline/RendererStageGraph.ts";

function testStageOrder() {
	const graph = new RendererStageGraph([
		{ id: "a", dependsOn: [] },
		{ id: "b", dependsOn: ["a"] },
		{ id: "c", dependsOn: ["b"] },
	]);
	const order = graph.getExecutionOrder(
		{
			hasActiveAnimations: true,
			hasParticleSystems: true,
		},
		() => {}
	);
	assert.deepEqual(
		order.map((stage) => stage.id),
		["a", "b", "c"]
	);
}

function testStageCycleAndMissingDependency() {
	const warnings = [];
	const graph = new RendererStageGraph([
		{ id: "a", dependsOn: ["b"] },
		{ id: "b", dependsOn: ["a"] },
		{ id: "c", dependsOn: ["missing"] },
	]);
	const order = graph.getExecutionOrder(
		{
			hasActiveAnimations: true,
			hasParticleSystems: true,
		},
		(key, message) => warnings.push({ key, message })
	);
	assert.deepEqual(order.map((stage) => stage.id), []);
	assert.ok(warnings.length >= 2);
}

function run() {
	testStageOrder();
	testStageCycleAndMissingDependency();
	console.log("Renderer stage graph tests passed");
}

run();
