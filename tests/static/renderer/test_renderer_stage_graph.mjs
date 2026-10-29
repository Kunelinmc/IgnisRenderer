import assert from "node:assert/strict";
import { RenderPipelineRegistry } from "../../../src/pipeline/RenderPipelineRegistry.ts";

function testStageOrder() {
	const graph = new RenderPipelineRegistry({
		stages: [
			{ id: "a", kind: "renderer", dependsOn: [] },
			{ id: "b", kind: "renderer", dependsOn: ["a"] },
			{ id: "c", kind: "renderer", dependsOn: ["b"] },
		],
	});
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
	const graph = new RenderPipelineRegistry({
		stages: [
			{ id: "a", kind: "renderer", dependsOn: ["b"] },
			{ id: "b", kind: "renderer", dependsOn: ["a"] },
			{ id: "c", kind: "renderer", dependsOn: ["missing"] },
		],
	});
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
