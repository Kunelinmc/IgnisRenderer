import assert from "node:assert/strict";
import { PostProcessPipeline } from "../../../src/postprocess/index.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createPostProcess(overrides = {}) {
	return createResolvedPostProcess(overrides, "webgl");
}

function createExecutor() {
	return {
		backend: "webgl",
		createResource() {
			throw new Error("Unexpected history allocation in this test");
		},
		destroyResource() {},
		executePass() {
			return { ran: true };
		},
	};
}

function testBuiltInOrderUsesPipelineAuthority() {
	const pipeline = new PostProcessPipeline();
	const order = pipeline.getExecutionOrder(
		createPostProcess({
			ssao: { enabled: true },
			taa: { enabled: true },
			fog: { enabled: true, options: { application: "postprocess" } },
			"motion-blur": { enabled: true },
			dof: { enabled: true },
			bloom: { enabled: true },
			tonemap: { enabled: true },
			"color-filter": { enabled: true },
			fxaa: { enabled: true },
			"interaction-outline": { enabled: true },
			gamma: { enabled: true },
		}),
		createExecutor()
	);
	assert.deepEqual(
		order.map((pass) => pass.id),
		[
			"ssao",
			"taa",
			"fog",
			"motion-blur",
			"dof",
			"bloom",
			"tonemap",
			"color-filter",
			"fxaa",
			"interaction-outline",
			"gamma",
		]
	);
}

function testFogSceneModeSkipsFogInPipelineOrder() {
	const pipeline = new PostProcessPipeline();
	const order = pipeline.getExecutionOrder(
		createPostProcess({
			fog: { enabled: true, options: { application: "scene" } },
			"motion-blur": { enabled: true },
		}),
		createExecutor()
	);
	assert.equal(order.some((pass) => pass.id === "fog"), false);
	assert.ok(order.some((pass) => pass.id === "motion-blur"));
}

function run() {
	testBuiltInOrderUsesPipelineAuthority();
	testFogSceneModeSkipsFogInPipelineOrder();
	console.log("WebGL post-process pipeline-order tests passed");
}

run();
