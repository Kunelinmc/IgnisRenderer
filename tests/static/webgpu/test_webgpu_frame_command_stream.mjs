import assert from "node:assert/strict";

import { WebGPUFrameCommandStream } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameCommandStream.ts";

function createHost(failAt = -1) {
	const encoders = [];
	const submitted = [];
	return {
		encoders,
		submitted,
		createCommandEncoder() {
			const id = `encoder-${encoders.length}`;
			const encoder = {
				id,
				finish() { return { label: id }; },
			};
			encoders.push(encoder);
			return encoder;
		},
		submit(commands) {
			if (submitted.length === failAt) throw new Error("submit failed");
			submitted.push(commands[0].label);
		},
	};
}

{
	const host = createHost();
	const stream = new WebGPUFrameCommandStream(host);
	assert.strictEqual(stream.requireEncoder(), host.encoders[0]);
	stream.enqueueCurrent("main:before-reflection");
	stream.enqueueEncoder("reflection:0", {
		finish: () => ({ label: "reflection-command" }),
	});
	assert.strictEqual(stream.resume(), host.encoders[1]);
	await stream.commit("main:final");
	assert.deepEqual(host.submitted, [
		"encoder-0",
		"reflection-command",
		"encoder-1",
	]);
	assert.deepEqual(stream.getDebugState().submittedLabels, [
		"main:before-reflection",
		"reflection:0",
		"main:final",
	]);
}

{
	const host = createHost();
	const stream = new WebGPUFrameCommandStream(host);
	stream.enqueueCurrent("discarded");
	stream.abort();
	assert.equal(stream.encoder, null);
	assert.equal(stream.getDebugState().state, "aborted");
	assert.deepEqual(host.submitted, []);
}

console.log("WebGPU frame command stream tests passed");
