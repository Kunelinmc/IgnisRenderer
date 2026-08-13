import assert from "node:assert/strict";

import * as frameExecutorFixture from "../../helpers/webgpu_frame_executor_resilience.mjs";

const {
	FakeBackend,
	WebGPUFrameExecutor,
	createFrameContext,
	createResourcesStub,
	getFrameGraphDebugState,
	getFrameTargets,
	initializeIsolatedWebGPUTestState,
} = frameExecutorFixture;

const restoreTestState = initializeIsolatedWebGPUTestState();

async function testZeroSizedFrameSkipsEncoderAndLegacyDepthPath() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(0, 0);

	await executor.beginFrame(context);
	assert.equal(backend.createCommandEncoderCalls, 0);
	assert.equal(getFrameGraphDebugState(executor).active, true);

	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context,
	);
	await executor.endFrame();
	assert.equal(getFrameGraphDebugState(executor).active, false);
	assert.equal(getFrameGraphDebugState(executor).texturePoolOwnerCount, 0);
}

async function testDebugStateRetainsNodesFromEveryCompiledStage() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	await executor.beginFrame(context);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context,
	);
	await executor.endFrame();

	const debugState = getFrameGraphDebugState(executor);
	assert.ok(debugState.lastPlannedNodeIds.includes("main-opaque:scene:opaque-scene"));
	assert.ok(debugState.lastPlannedNodeIds.includes("postprocess:presentation:presentation"));
	assert.deepEqual(
		debugState.compiledStages.map((stage) => stage.pass.stage),
		["main-opaque", "postprocess"],
	);
}

async function testFrameSessionRejectsInvalidLifecycleCalls() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	await assert.rejects(
		executor.executePass({ stage: "main-opaque", executor: "backend", enabled: true }, context),
		/no active frame session/,
	);
	await assert.rejects(executor.endFrame(), /no active frame session/);
	assert.doesNotThrow(() => executor.abortFrame());

	await executor.beginFrame(context);
	await assert.rejects(
		executor.beginFrame(context),
		/already has an active frame session/,
	);
	assert.equal(getFrameGraphDebugState(executor).active, true);
	executor.abortFrame();
}

async function testFrameSessionRequiresOriginalContext() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	const mismatchedContext = createFrameContext(64, 64);

	await executor.beginFrame(context);
	await assert.rejects(
		executor.executePass(
			{ stage: "main-opaque", executor: "backend", enabled: true },
			mismatchedContext,
		),
		/must match the context passed to beginFrame/,
	);
	assert.equal(getFrameGraphDebugState(executor).active, true);
	executor.abortFrame();
}

async function testPassBeforeAsyncBeginSettlesIsRejected() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);
	const pendingBegin = executor.beginFrame(context);
	await assert.rejects(
		executor.executePass(
			{ stage: "main-opaque", executor: "backend", enabled: true },
			context,
		),
		/cannot execute passes in state "preparing"/,
	);
	await pendingBegin;
	executor.abortFrame();
}

async function testAbortFrameClearsActiveStateWithoutSubmit() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	await executor.beginFrame(context);

	executor.abortFrame();

	assert.equal(backend.submits, 0);
	assert.equal(getFrameGraphDebugState(executor).active, false);
	assert.equal(getFrameGraphDebugState(executor).motionHistoryWriteTarget, null);
	assert.equal(getFrameGraphDebugState(executor).oitActive, false);
}

async function testEndFrameFailureClosesActiveState() {
	const backend = new FakeBackend();
	const error = new Error("submit failed");
	backend.submit = () => {
		throw error;
	};
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	await executor.beginFrame(context);
	let caught = null;
	try {
		await executor.endFrame();
	} catch (caughtError) {
		caught = caughtError;
	}

	assert.strictEqual(caught, error);
	assert.equal(getFrameGraphDebugState(executor).active, false);
	assert.equal(getFrameGraphDebugState(executor).motionHistoryWriteTarget, null);
}

async function testInvalidateFrameTargetsDestroysPresentBinding() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());

	executor.invalidateFrameTargets();

	assert.equal(getFrameTargets(executor), null);
}

async function testInvalidateFrameTargetsDefersDuringActiveFrame() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	await executor.beginFrame(context);
	const activeTargets = getFrameTargets(executor);
	assert.ok(activeTargets);

	executor.invalidateFrameTargets();
	assert.strictEqual(getFrameTargets(executor), activeTargets);
	assert.equal(getFrameGraphDebugState(executor).pendingFrameTargetInvalidation, true);

	executor.abortFrame();
	assert.equal(getFrameTargets(executor), null);
	assert.equal(getFrameGraphDebugState(executor).pendingFrameTargetInvalidation, false);
}

async function testShaderRuntimeInvalidationDefersDuringActiveFrame() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(64, 64);

	await executor.beginFrame(context);
	executor.onShaderRuntimeChanged();
	assert.equal(getFrameGraphDebugState(executor).pendingShaderRuntimeInvalidation, true);

	executor.abortFrame();
	assert.equal(getFrameGraphDebugState(executor).pendingShaderRuntimeInvalidation, false);
}

async function run() {
	try {
		await testZeroSizedFrameSkipsEncoderAndLegacyDepthPath();
		await testDebugStateRetainsNodesFromEveryCompiledStage();
		await testFrameSessionRejectsInvalidLifecycleCalls();
		await testFrameSessionRequiresOriginalContext();
		await testPassBeforeAsyncBeginSettlesIsRejected();
		await testAbortFrameClearsActiveStateWithoutSubmit();
		await testEndFrameFailureClosesActiveState();
		await testInvalidateFrameTargetsDestroysPresentBinding();
		await testInvalidateFrameTargetsDefersDuringActiveFrame();
		await testShaderRuntimeInvalidationDefersDuringActiveFrame();
		console.log("WebGPU frame-executor lifecycle/invalidation tests passed");
	} finally {
		restoreTestState();
	}
}

await run();
