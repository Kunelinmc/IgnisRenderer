import assert from "node:assert/strict";

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
	const host = {
		createPassTimestampWrites() { return undefined; },
		getCanvasColorTexture() { return { width: 1, height: 1 }; },
	};
	const encoder = new WebGPUCommandEncoder(native, host, {});
	if (kind === "render") encoder.beginRenderPass({ colorAttachments: [] });
	else encoder.beginComputePass();
	const nativeGroup = {};
	const group = { _gpuResource: nativeGroup };
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

testBindingOffsets("render");
testBindingOffsets("compute");
console.log("WebGPU command encoder tests passed");
