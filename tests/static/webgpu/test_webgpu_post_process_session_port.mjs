import assert from "node:assert/strict";

import { WebGPUPostProcessExecutor } from "../../../src/backends/webgpu/WebGPUPostProcessExecutor.ts";
import { FakeWebGPUBackend } from "../../helpers/fakes.mjs";

const host = new FakeWebGPUBackend();
const executor = new WebGPUPostProcessExecutor(host);
const context = { attachments: { width: 8, height: 4 } };

assert.equal(executor.gBufferNormalSpace, "view");
assert.equal(executor.createGBufferBridge(context).normalSpace, "view");
assert.deepEqual(executor.createGBufferBridge(context).channels, {});
assert.throws(
	() => executor.createPassExecutionContext({ implementation: {} }),
	/post-process session is not active/,
);

const calls = [];
const port = {
	createGBufferBridge() {
		return { width: 1, height: 1, channels: { color: {} } };
	},
	createPassExecutionContext() {
		calls.push("context");
		return { encoder: {} };
	},
	completePass() {
		calls.push("complete");
	},
	invalidateResourceBindings() {
		calls.push("invalidate");
	},
};
executor.bindSession(port);
assert.throws(() => executor.bindSession(port), /already has an active session/);
assert.deepEqual(executor.createPassExecutionContext({ implementation: {} }), { encoder: {} });
executor.completePass({}, {});
executor.invalidateResourceBindings();
assert.deepEqual(calls, ["context", "complete", "invalidate"]);
executor.unbindSession(port);
assert.throws(
	() => executor.createPassExecutionContext({ implementation: {} }),
	/post-process session is not active/,
);

console.log("WebGPU post-process session port tests passed");
