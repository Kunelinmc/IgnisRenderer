import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { createWebGPUResourceManager } from "../../helpers/fakes.mjs";
import { Logger } from "../../../src/foundation/Logger.ts";
import { WebGPUPresentPass } from "../../../src/backends/webgpu/rendergraph/WebGPUPresentPass.ts";

import { WebGPUPresentationRuntime } from "../../../src/backends/webgpu/rendergraph/WebGPUPresentationRuntime.ts";

const source = { id: "scene-color" };
const calls = [];
const runtime = new WebGPUPresentationRuntime({}, {
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
	commands: { encoder: {} },
	targets: { frameTargets: { sceneColor: source } },
	dirtyRects: {
		resolveDirtyRects: (_context, width, height) => [{ x: 0, y: 0, width, height }],
	},
};
runtime.beginFrame(session.context);
await runtime.executors.presentation({}, session);
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


// Exercise presentation itself, including repeated domains and alpha mode changes.
const warnings = [];
Logger.configure({ resetOnceKeys: true, sink: { warn: (...args) => warnings.push(args.join(" ")) } });
try {
	const writes = [];
	const buffers = [];
	const store = createWebGPUResourceManager();
	const released = [];
	const pass = new WebGPUPresentPass({
		displayOutputState: { activeDynamicRange: "hdr", requested: { exposure: 7, hdrHeadroom: 4 } },
		writeBuffer: (_buffer, data) => writes.push(Array.from(data)),
		createBindingGroup: () => store.registerBindingGroup({}, "present binding", () => {
			released.push("binding");
		}),
		getCanvasColorTexture: () => ({ width: 1, height: 1 }),
		createBuffer: (descriptor) => {
			buffers.push(descriptor);
			return store.registerBuffer({}, { size: descriptor.size }, () => released.push("params"));
		},
	});
	pass._shaderModule = store.registerShaderModule({}, "present shader", () => released.push("shader"));
	pass._pipeline = store.registerRenderPipeline({}, "present pipeline", () => released.push("pipeline"));
	pass._sampler = store.registerSampler({}, "present sampler", () => released.push("sampler"));
	const encoder = {
		beginRenderPass() {}, setPipeline() {}, setBindingGroup() {},
		draw() {}, endRenderPass() {},
	};
	for (const domain of ["display-encoded", "scene-linear-hdr", "display-linear"]) {
		for (const presentationAlphaMode of ["opaque", "premultiplied"]) {
			await pass.present({
				encoder, source, colorDomain: domain,
				frameContext: { presentationAlphaMode },
				resolveDirtyRects: () => [{ x: 0, y: 0, width: 1, height: 1 }],
			});
		}
	}
	assert.equal(warnings.length, 2);
	assert.match(warnings[0], /presentation-color-domain-incomplete-scene-linear-hdr/);
	assert.match(warnings[0], /tone mapping.*display encoding/);
	assert.match(warnings[1], /presentation-color-domain-incomplete-display-linear/);
	assert.match(warnings[1], /may not match expectations/);
	assert.deepEqual(writes[0], writes[2], "color domain must not alter presentation parameters");
	assert.deepEqual(writes[2], writes[4]);
	assert.notDeepEqual(writes[0], writes[1], "transparent alpha handling must remain active");
	assert.equal(buffers[0].size, 8);
	assert.deepEqual(writes, [
		[1, 0], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1],
	]);
	const shader = readFileSync(new URL("../../../src/shaders/webgpu/utility/present.wgsl", import.meta.url), "utf8");
	assert.equal(shader.match(/struct PresentParams\s*\{([^}]+)\}/)[1]
		.match(/:\s*f32\s*,/g).length * 4, 8);
	assert.doesNotMatch(shader, /acesFitted|hdrSoftShoulder|linearToSrgb|linearSrgbToDisplayP3|exposure/,
		"presentation must not apply missing color conversions");
	pass.destroy();
	assert.deepEqual(released, ["shader", "pipeline", "sampler", "params", "binding"]);
	pass.destroy();
	assert.equal(released.length, 5, "repeated presentation cleanup must remain idempotent");
} finally {
	Logger.reset();
}

// Direct cleanup must propagate the original failure and retain its stop order.
const failureStore = createWebGPUResourceManager();
const failedPass = new WebGPUPresentPass({});
const failure = new Error("shader release failed");
let laterReleases = 0;
failedPass._shaderModule = failureStore.registerShaderModule({}, "failed shader", () => {
	throw failure;
});
failedPass._pipeline = failureStore.registerRenderPipeline({}, "later pipeline", () => {
	laterReleases++;
});
assert.throws(() => failedPass.onShaderRuntimeChanged(), (error) => error === failure);
assert.equal(laterReleases, 0, "direct cleanup must stop when its first release throws");
failedPass.destroy();
assert.equal(laterReleases, 1);

console.log("WebGPU presentation runtime tests passed");
