import assert from "node:assert/strict";

import { WebGPUFrameTransaction } from "../../../src/backends/webgpu/WebGPUFrameTransaction.ts";

function createRuntime(options = {}) {
	const calls = [];
	const port = {};
	const orchestrator = {
		beginFrame() {
			calls.push("orchestrator-begin");
		},
		async endFrame(postSubmit) {
			calls.push("submit");
			if (options.submitError) throw options.submitError;
			await postSubmit?.();
		},
		commitFrameState() {
			calls.push("frame-state-commit");
		},
		abortRecording() {
			calls.push("recording-abort");
		},
		abortFrameState() {
			calls.push("frame-state-abort");
		},
	};
	const postProcess = {
		createSessionPort() {
			calls.push("create-session");
			return port;
		},
	};
	const resources = {
		beginFrameResourceLifecycle() {
			calls.push("resources-begin");
		},
		commitTemporalFrame() {
			calls.push("temporal-commit");
			if (options.temporalCommitError) throw options.temporalCommitError;
		},
		abortTemporalFrame() {
			calls.push("temporal-abort");
		},
	};
	const particleSimulator = {
		beginFrame() {
			calls.push("particle-begin");
		},
		endFrame() {
			calls.push("particle-end");
			if (options.particleEndError) throw options.particleEndError;
		},
	};
	const postProcessRuntime = {
		commitFrame() {
			calls.push("post-process-commit");
			if (options.postProcessCommitError) throw options.postProcessCommitError;
		},
		async abortFrame() {
			calls.push("post-process-abort");
		},
	};
	const postProcessExecutor = {
		bindSession(value) {
			assert.strictEqual(value, port);
			calls.push("bind-session");
		},
		unbindSession(value) {
			assert.strictEqual(value, port);
			calls.push("unbind-session");
		},
	};
	const context = { id: "frame" };
	const transaction = new WebGPUFrameTransaction(context, {
		orchestrator,
		resources,
		particleSimulator,
		postProcessRuntime,
		postProcessExecutor,
		postProcess,
		reportCleanupError(scope, error) {
			calls.push(`cleanup-error:${scope}:${String(error)}`);
		},
	});
	return { calls, context, transaction };
}

{
	const { calls, context, transaction } = createRuntime();
	transaction.begin();
	transaction.assertRecordingContext(context);
	await transaction.commit();
	assert.equal(transaction.state, "committed");
	assert.deepEqual(calls, [
		"particle-begin",
		"resources-begin",
		"create-session",
		"bind-session",
		"orchestrator-begin",
		"submit",
		"particle-end",
		"post-process-commit",
		"temporal-commit",
		"frame-state-commit",
		"unbind-session",
	]);
}

for (const failure of ["particleEndError", "postProcessCommitError", "temporalCommitError"]) {
	const error = new Error(failure);
	const { calls, transaction } = createRuntime({ [failure]: error });
	transaction.begin();
	await assert.rejects(transaction.commit(), (caught) => caught === error);
	assert.equal(transaction.state, "aborted");
	assert.equal(calls.includes("frame-state-commit"), false);
	assert.equal(calls.includes("frame-state-abort"), true);
	assert.equal(calls.includes("unbind-session"), true);
}

{
	const { calls, transaction } = createRuntime();
	transaction.begin();
	transaction.invalidate(new Error("device lost"));
	assert.equal(transaction.state, "invalidated");
	assert.equal(calls.includes("submit"), false);
	assert.equal(calls.includes("post-process-commit"), false);
	assert.equal(calls.includes("temporal-commit"), false);
	assert.equal(calls.includes("frame-state-abort"), true);
	await transaction.abort();
	assert.equal(transaction.state, "invalidated");
}

console.log("WebGPU frame transaction tests passed");
