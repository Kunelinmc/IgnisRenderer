import assert from "node:assert/strict";

import { WebGPUPresentationRuntime } from "../../../src/backends/webgpu/rendergraph/WebGPUPresentationRuntime.ts";

const source = { id: "scene-color" };
const calls = [];
const runtime = new WebGPUPresentationRuntime({}, {
	recording: {
		getFrameTargets: () => ({ sceneColor: source }),
		resolveDirtyRects: (_context, width, height) => [{ x: 0, y: 0, width, height }],
	},
	getOutputColorDomain: () => "display-linear",
});
runtime._pass = {
	warmup() {
		calls.push("warmup");
		return Promise.resolve();
	},
	present(request) {
		calls.push([
			"present",
			request.source.id,
			request.colorDomain,
			request.frameContext.id,
		]);
		return Promise.resolve();
	},
	invalidateBindings() {
		calls.push("invalidate");
	},
	onShaderRuntimeChanged() {
		calls.push("shader-change");
	},
	destroy() {
		calls.push("destroy");
	},
};

const session = {
	context: { id: "frame" },
	encoder: {},
	presented: false,
};
await runtime.executors.presentation({}, session);
assert.equal(session.presented, true);
assert.deepEqual(calls[0], ["present", "scene-color", "display-linear", "frame"]);

await runtime.executors.presentation({}, session);
assert.equal(calls.length, 1, "an already-presented session must not present twice");

await runtime.warmup();
runtime.invalidateFrameResources();
runtime.onShaderRuntimeChanged();
runtime.destroy();
assert.deepEqual(calls.slice(1), [
	"warmup",
	"invalidate",
	"shader-change",
	"destroy",
]);

console.log("WebGPU presentation runtime tests passed");
