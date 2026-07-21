import assert from "node:assert/strict";

import { WebGPUPostProcessRuntime } from "../../../src/backends/webgpu/WebGPUPostProcessRuntime.ts";
import { FakeBackend } from "../../helpers/webgpu_postprocess_runtime_test_helpers.mjs";

function testRuntimeOnlyOwnsSharedServices() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	assert.ok(runtime.sharedContext);
	assert.equal("executePass" in runtime, false);
	assert.equal("registerRuntimePass" in runtime, false);
	assert.equal("warmupHints" in runtime, false);
	assert.equal(backend.shaderModules.length, 0);
	assert.equal(backend.computePipelines.length, 0);
	assert.equal(backend.buffers.length, 0);
	runtime.destroy();
}

testRuntimeOnlyOwnsSharedServices();
console.log("WebGPU postprocess spatial runtime tests passed");
