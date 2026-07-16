import assert from "node:assert/strict";

import { WebGPUFramePartialSubmitError } from "../../../src/foundation/Error.ts";
import { WebGPUFrameCommitter } from "../../../src/renderers/webgpu/rendergraph/WebGPUFrameCommitter.ts";

function createHost(failAt = -1) {
	const submitted = [];
	return {
		submitted,
		submit(commands) {
			if (submitted.length === failAt) throw new Error("submit failed");
			submitted.push(commands[0].label);
		},
	};
}

{
	const host = createHost();
	const committer = new WebGPUFrameCommitter(host);
	committer.enqueue("a", { label: "a" });
	committer.enqueue("b", { label: "b" });
	committer.abort();
	assert.deepEqual(host.submitted, []);
	assert.equal(committer.getDebugState().state, "aborted");
}

{
	const host = createHost(1);
	const committer = new WebGPUFrameCommitter(host);
	committer.enqueue("main", { label: "main" });
	committer.enqueue("reflection", { label: "reflection" });
	await assert.rejects(committer.commit(), (error) => {
		assert.ok(error instanceof WebGPUFramePartialSubmitError);
		assert.equal(error.phase, "submit");
		assert.equal(error.submittedCount, 1);
		assert.equal(error.totalCount, 2);
		assert.deepEqual(error.submittedLabels, ["main"]);
		assert.deepEqual(error.pendingLabels, ["reflection"]);
		return true;
	});
}

{
	const host = createHost();
	const committer = new WebGPUFrameCommitter(host);
	committer.enqueue("main", { label: "main" });
	await assert.rejects(
		committer.commit(() => {
			throw new Error("post-submit failed");
		}),
		(error) => {
			assert.ok(error instanceof WebGPUFramePartialSubmitError);
			assert.equal(error.phase, "post-submit");
			assert.deepEqual(error.submittedLabels, ["main"]);
			assert.deepEqual(error.pendingLabels, []);
			return true;
		},
	);
}

console.log("WebGPU frame committer tests passed");
