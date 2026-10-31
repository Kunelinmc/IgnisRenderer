import assert from "node:assert/strict";

import { createWebGPUResourceManager } from "../../helpers/fakes.mjs";
import { WebGPUCommandEncoder } from "../../../src/backends/webgpu/WebGPUCommandEncoder.ts";

function testBindingOffsets(kind) {
	const calls = [];
	const pass = {
		setBindGroup(...args) {
			// Native sequence conversion rejects an explicitly supplied undefined.
			if (args.length > 2 && args[2] === undefined) {
				throw new TypeError("The provided value cannot be converted to a sequence.");
			}
			calls.push(args);
		},
		end() {},
	};
	const native = {
		beginRenderPass() { return pass; },
		beginComputePass() { return pass; },
	};
	const resourceManager = createWebGPUResourceManager();
	const host = {
		resourceManager,
		createPassTimestampWrites() { return undefined; },
		getCanvasColorTexture() { return { width: 1, height: 1 }; },
	};
	const encoder = new WebGPUCommandEncoder(native, host, {});
	if (kind === "render") encoder.beginRenderPass({ colorAttachments: [] });
	else encoder.beginComputePass();
	const nativeGroup = {};
	const group = resourceManager.registerBindingGroup(nativeGroup);
	const offsets = Object.freeze([256, 512]);
	const emptyOffsets = Object.freeze([]);

	assert.doesNotThrow(() => encoder.setBindingGroup(0, group),
		`${kind}: omitted offsets must not reach native sequence conversion`);
	assert.doesNotThrow(() => encoder.setBindingGroup(2, group, undefined),
		`${kind}: explicit undefined must also omit the native argument`);
	encoder.setBindingGroup(1, group, offsets);
	encoder.setBindingGroup(0, group, emptyOffsets);
	assert.deepEqual(calls, [
		[0, nativeGroup],
		[2, nativeGroup],
		[1, nativeGroup, offsets],
		[0, nativeGroup, emptyOffsets],
	]);
	assert.equal(calls[2][2], offsets, `${kind}: preserve the supplied offset sequence`);
	assert.equal(calls[3][2], emptyOffsets, `${kind}: preserve an explicit empty sequence`);
	if (kind === "render") encoder.endRenderPass();
	else encoder.endComputePass();
}

function testResourcesBelongToTheEncoder() {
	const resourceManager = createWebGPUResourceManager();
	const foreignStore = createWebGPUResourceManager();
	const calls = [];
	const pass = { setBindGroup(...args) { calls.push(args); }, end() {} };
	const native = {
		beginRenderPass() { calls.push("begin"); return pass; },
	};
	const host = {
		resourceManager,
		createPassTimestampWrites() {},
		getCanvasColorTexture() { return { width: 1, height: 1 }; },
		getCurrentColorView() { throw new Error("Unexpected canvas fallback"); },
	};
	const encoder = new WebGPUCommandEncoder(native, host, {});
	const texture = (owner) => owner.registerTexture(
		{ texture: {}, view: {} }, { width: 1, height: 1 },
	);
	const attachment = (view) => ({
		colorAttachments: [{ view, loadOp: "clear", storeOp: "store" }],
	});
	assert.throws(() => encoder.beginRenderPass(attachment(texture(foreignStore))),
		/owned|WebGPU|invalid/i, "foreign attachments must fail before native encoding");
	const destroyed = texture(resourceManager);
	destroyed.destroy();
	assert.throws(() => encoder.beginRenderPass(attachment(destroyed)),
		/destroyed|WebGPU|invalid/i, "destroyed attachments cannot fall back to the canvas");
	assert.throws(() => encoder.beginRenderPass(attachment({
		kind: "texture", width: 1, height: 1, destroy() {},
	})), /owned|WebGPU|invalid/i, "texture-shaped objects cannot forge resource ownership");
	assert.equal(calls.length, 0);

	encoder.beginRenderPass({ colorAttachments: [] });
	assert.throws(() => encoder.setBindingGroup(0, foreignStore.registerBindingGroup({})),
		/owned|WebGPU|invalid/i);
	const group = resourceManager.registerBindingGroup({});
	group.destroy();
	assert.throws(() => encoder.setBindingGroup(0, group), /destroyed|WebGPU|invalid/i);
	assert.deepEqual(calls, ["begin"], "invalid groups cannot reach the native pass");
	encoder.endRenderPass();
}

testBindingOffsets("render");
testBindingOffsets("compute");
testResourcesBelongToTheEncoder();
console.log("WebGPU command encoder tests passed");
