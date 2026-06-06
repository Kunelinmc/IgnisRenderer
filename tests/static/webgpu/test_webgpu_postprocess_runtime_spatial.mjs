import assert from "node:assert/strict";

import { WebGPUPostProcessRuntime } from "../../../src/renderers/webgpu/WebGPUPostProcessRuntime.ts";
import {
	FakeBackend,
	FakeEncoder,
	createTexture,
} from "../../helpers/webgpu_postprocess_runtime_test_helpers.mjs";

async function testSSAOSSGIAreOwnedByLogicalPassImplementations() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const targets = {
		sceneColor: createTexture(32, 16, "scene"),
		postPing: createTexture(32, 16, "ping"),
		postPong: createTexture(32, 16, "pong"),
	};
	const frameContext = {
		features: {},
		postProcess: {
			options: {},
		},
	};

	assert.deepEqual(
		await runtime.executePass({
			passId: "ssao",
			encoder,
			targets,
			frameContext,
		}),
		{ ran: false }
	);
	assert.deepEqual(
		await runtime.executePass({
			passId: "ssgi",
			encoder,
			targets,
			frameContext,
		}),
		{ ran: false }
	);

	const warmup = await runtime.warmupHints([
		"postprocess:ssao",
		"postprocess:ssgi",
	]);
	assert.deepEqual(warmup, {
		compiled: 0,
		failed: 0,
		errors: [],
	});
	assert.equal(backend.shaderModules.length, 0);
	assert.equal(backend.computePipelines.length, 0);
	assert.equal(backend.buffers.length, 0);
}

await testSSAOSSGIAreOwnedByLogicalPassImplementations();
console.log("WebGPU postprocess spatial runtime tests passed");
