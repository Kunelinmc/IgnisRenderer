import assert from "node:assert/strict";

import { WebGPUPostProcessRuntime } from "../../../src/backends/webgpu/WebGPUPostProcessRuntime.ts";
import { FakeBackend } from "../../helpers/webgpu_postprocess_runtime_test_helpers.mjs";

function testRuntimeOnlyOwnsSharedServices() {
	const backend = new FakeBackend();
	const warnings = [];
	const runtime = new WebGPUPostProcessRuntime(
		backend,
		(key, message) => warnings.push([key, message])
	);
	assert.ok(runtime.compute);
	assert.equal("sharedContext" in runtime, false);
	assert.equal("executePass" in runtime, false);
	assert.equal("registerRuntimePass" in runtime, false);
	assert.equal("warmupHints" in runtime, false);
	runtime.warn("test-warning", "runtime owns diagnostics");
	assert.deepEqual(warnings, [["test-warning", "runtime owns diagnostics"]]);
	assert.equal(backend.shaderModules.length, 0);
	assert.equal(backend.computePipelines.length, 0);
	assert.equal(backend.buffers.length, 0);
	runtime.destroy();
}

testRuntimeOnlyOwnsSharedServices();
console.log("WebGPU postprocess spatial runtime tests passed");
