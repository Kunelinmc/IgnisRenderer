import assert from "node:assert/strict";
import { RenderPipelineRegistry } from "../../../src/pipeline/RenderPipelineRegistry.ts";
import { Logger } from "../../../src/foundation/Logger.ts";

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
		}
	);
	assert.deepEqual(
		order.map((stage) => stage.id),
		["a", "b", "c"]
	);
}

function testStageCycleAndMissingDependency() {
	const warnings = [];
	Logger.configure({
		resetOnceKeys: true,
		sink: { warn: (...args) => warnings.push(args.join(" ")) },
	});
	const graph = new RenderPipelineRegistry({
		stages: [
			{ id: "a", kind: "renderer", dependsOn: ["b"] },
			{ id: "b", kind: "renderer", dependsOn: ["a"] },
			{ id: "c", kind: "renderer", dependsOn: ["missing"] },
		],
	});
	try {
		const context = { hasActiveAnimations: true, hasParticleSystems: true };
		let order;
		assert.doesNotThrow(() => { order = graph.getExecutionOrder(context); });
		assert.deepEqual(order.map((stage) => stage.id), []);
		assert.equal(warnings.length, 2);
		assert.ok(warnings.some((message) => message.includes("renderer-stage-cycle-a")));
		assert.ok(warnings.some((message) =>
			message.includes("renderer-stage-dependency-missing-c-missing")
		));
		graph.getExecutionOrder(context);
		new RenderPipelineRegistry({
			stages: [{ id: "c", kind: "renderer", dependsOn: ["missing"] }],
		}).getExecutionOrder(context);
		assert.equal(warnings.length, 2, "stage warnings retain global deduplication");
	} finally {
		Logger.reset();
	}
}

function run() {
	testStageOrder();
	testStageCycleAndMissingDependency();
	console.log("Renderer stage graph tests passed");
}

run();
