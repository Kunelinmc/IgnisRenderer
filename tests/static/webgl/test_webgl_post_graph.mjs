import assert from "node:assert/strict";
import { resolvePostProcessExecutionOrder } from "../../../src/postprocess/index.ts";
import { WebGLPostProcessExecutor } from "../../../src/backends/webgl/WebGLPostProcessExecutor.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createPostProcess(overrides = {}) {
	return createResolvedPostProcess(overrides, "webgl");
}

function testSyntheticGBufferBridgeDoesNotRequireDeviceServices() {
	let deviceServiceReads = 0;
	const executor = new WebGLPostProcessExecutor({
		getDeviceServices() {
			deviceServiceReads++;
			return null;
		},
	});
	const context = { attachments: { width: 8, height: 4 } };
	const synthetic = executor.createGBufferBridge(context, {
		resourceMode: "synthetic",
	});

	assert.equal(deviceServiceReads, 0);
	assert.equal(synthetic.normalSpace, "world");
	assert.ok(synthetic.channels.depth);
	assert.deepEqual(synthetic.channels.depth.handle, {
		backend: "webgl",
		resource: null,
	});

	executor.createGBufferBridge(context);
	assert.equal(deviceServiceReads, 1);
}

function testBuiltInOrderUsesPipelineAuthority() {
	const order = resolvePostProcessExecutionOrder(
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
			gamma: { enabled: true },
		}),
		{ backend: "webgl" }
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
			"gamma",
		]
	);
}

function testFogSceneModeSkipsFogInPipelineOrder() {
	const order = resolvePostProcessExecutionOrder(
		createPostProcess({
			fog: { enabled: true, options: { application: "scene" } },
			"motion-blur": { enabled: true },
		}),
		{ backend: "webgl" }
	);
	assert.equal(order.some((pass) => pass.id === "fog"), false);
	assert.ok(order.some((pass) => pass.id === "motion-blur"));
}

function run() {
	testSyntheticGBufferBridgeDoesNotRequireDeviceServices();
	testBuiltInOrderUsesPipelineAuthority();
	testFogSceneModeSkipsFogInPipelineOrder();
	console.log("WebGL post-process pipeline-order tests passed");
}

run();
