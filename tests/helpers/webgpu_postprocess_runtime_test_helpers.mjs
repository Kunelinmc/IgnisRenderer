import assert from "node:assert/strict";

import { FakeWebGPUBackend as FakeBackend } from "./test_fakes.mjs";

export { FakeBackend };

export class FakeEncoder {
	constructor(backend = null) {
		this.calls = [];
		this.backend = backend;
	}

	beginComputePass(desc = {}) {
		this.calls.push(["beginComputePass", desc.label ?? null]);
	}

	setComputePipeline(pipeline) {
		this.calls.push(["setComputePipeline", pipeline.label]);
	}

	setBindingGroup(index, group) {
		this.calls.push(["setBindingGroup", index, group.label]);
	}

	dispatchWorkgroups(x, y = 1, z = 1) {
		this.calls.push(["dispatchWorkgroups", x, y, z]);
		if (this.backend && this.backend.dispatches) {
			this.backend.dispatches.push([x, y, z]);
		}
	}

	endComputePass() {
		this.calls.push(["endComputePass"]);
	}

	finish() {
		return { _gpuCommandBuffer: {} };
	}
}

export function createTexture(width, height, label) {
	return {
		width,
		height,
		label,
		destroy() {},
	};
}

export function assertClose(actual, expected, epsilon = 1e-6) {
	assert.ok(Math.abs(actual - expected) <= epsilon);
}
