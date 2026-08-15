import assert from "node:assert/strict";

import {
	IBL_PREFILTER_EXECUTOR_EXTENSION,
	WEBGPU_COMPUTE_EXTENSION,
} from "../../../src/backends/BackendExtensions.ts";
import { WebGPUBackend } from "../../../src/backends/webgpu/WebGPUBackend.ts";
import { Logger } from "../../../src/foundation/Logger.ts";

function createDeferred() {
	let resolve;
	let reject;
	const promise = new Promise((resolvePromise, rejectPromise) => {
		resolve = resolvePromise;
		reject = rejectPromise;
	});
	return { promise, resolve, reject };
}

function attachBackend(backend = new WebGPUBackend()) {
	backend.attach({
		surface: { canvas: {} },
		events: { emit() {} },
	});
	return backend;
}

function createReadyFrameBackend(options = {}) {
	const calls = [];
	const backend = attachBackend();
	backend._state = "ready";
	backend._device = {};
	backend._queue = {};
	backend._resources = {
		beginFrameResourceLifecycle() {
			calls.push("resources:begin");
		},
		abortTemporalFrame() {
			calls.push("resources:abort");
		},
		commitTemporalFrame() {
			calls.push("resources:commit");
		},
	};
	backend._particleSimulator = {
		beginFrame() {
			calls.push("particles:begin");
		},
		endFrame() {
			calls.push("particles:end");
		},
	};
	backend._frameOrchestrator = {
		beginFrame() {
			calls.push("orchestrator:begin");
			if (options.beginError) {
				throw options.beginError;
			}
		},
		executePass() {
			calls.push("orchestrator:execute");
			if (options.executeError) {
				throw options.executeError;
			}
		},
		async endFrame(postSubmit) {
			calls.push("orchestrator:end");
			if (options.endError) {
				throw options.endError;
			}
			await postSubmit?.();
		},
		abortRecording() {
			calls.push("orchestrator:abort");
		},
		abortFrameState() {
			calls.push("orchestrator:abort-analysis");
		},
		commitFrameState() {
			calls.push("orchestrator:commit-analysis");
		},
	};
	backend._frameRuntime = {
		postProcess: {
			createSessionPort() {
				calls.push("postprocess:create-session");
				return null;
			},
		},
	};
	return { backend, calls };
}

function createFrameContext() {
	return {
		framePlan: { backendPasses: [] },
	};
}

async function testInitializeLifecycleGuards() {
	const detached = new WebGPUBackend();
	await assert.rejects(
		() => detached.initialize(),
		/initialize.*lifecycle state "detached"/i,
	);

	const backend = attachBackend();
	const initialization = createDeferred();
	backend._initializeDeviceRuntime = () => initialization.promise;

	const firstInitialize = backend.initialize();
	assert.equal(backend._state, "initializing");
	await assert.rejects(
		() => backend.initialize(),
		/initialize.*lifecycle state "initializing"/i,
	);

	initialization.resolve();
	await firstInitialize;
	assert.equal(backend._state, "ready");
	await assert.rejects(
		() => backend.initialize(),
		/initialize.*lifecycle state "ready"/i,
	);

	backend.destroy();
	assert.equal(backend._state, "destroyed");
	await assert.rejects(
		() => backend.initialize(),
		/initialize.*lifecycle state "destroyed"/i,
	);
	assert.doesNotThrow(() => backend.destroy());
}

async function testFailedInitializeCanRetry() {
	const backend = attachBackend();
	let attempts = 0;
	backend._initializeDeviceRuntime = async () => {
		attempts++;
		if (attempts === 1) {
			throw new Error("simulated initialization failure");
		}
	};

	await assert.rejects(
		() => backend.initialize(),
		/simulated initialization failure/,
	);
	assert.equal(backend._state, "attached");
	await backend.initialize();
	assert.equal(backend._state, "ready");
	assert.equal(attempts, 2);
}

async function testDestroySupersedesPendingInitialize() {
	const backend = attachBackend();
	const initialization = createDeferred();
	backend._initializeDeviceRuntime = () => initialization.promise;

	const pending = backend.initialize();
	assert.equal(backend._state, "initializing");
	backend.destroy();
	initialization.resolve();

	await assert.rejects(pending, /lifecycle changed.*destroyed/i);
	assert.equal(backend._state, "destroyed");
	assert.equal(backend._device, null);
	assert.equal(backend._queue, null);
}

async function testDeviceLossSupersedesPendingInitialize() {
	const backend = attachBackend();
	const initialization = createDeferred();
	backend._initializeDeviceRuntime = () => initialization.promise;

	const pending = backend.initialize();
	Logger.configure({ level: "silent", resetOnceKeys: true });
	try {
		backend._handleDeviceLost({
			reason: "destroyed",
			message: "simulated loss during initialization",
		});
	} finally {
		Logger.reset();
	}
	initialization.resolve();

	await assert.rejects(pending, /lifecycle changed.*lost/i);
	assert.equal(backend._state, "lost");
	assert.equal(
		backend._deviceLostInfo.message,
		"simulated loss during initialization",
	);
}

async function testReadyRestoreRebuildsSameBackend() {
	const backend = attachBackend();
	const calls = [];
	backend._canvas = {};
	backend._state = "ready";
	backend._device = {};
	backend._queue = {};
	backend._commandScheduler.submitPendingCopyCommands = () => {
		calls.push("submit-pending");
	};
	backend._releaseDeviceRuntime = () => {
		calls.push("release-runtime");
		backend._device = null;
		backend._queue = null;
	};
	backend._initializeDeviceRuntime = async (_epoch, expectedState) => {
		assert.equal(expectedState, "restoring");
		calls.push("initialize-runtime");
	};

	await backend.restore();
	assert.equal(backend._state, "ready");
	assert.deepEqual(calls, [
		"submit-pending",
		"release-runtime",
		"initialize-runtime",
	]);
}

async function testRestoreWaitsForAutomaticRecovery() {
	const backend = attachBackend();
	const recovery = createDeferred();
	backend._canvas = {};
	backend._state = "restoring";
	backend._deviceRecoveryPromise = recovery.promise.finally(() => {
		backend._deviceRecoveryPromise = null;
	});

	const restore = backend.restore();
	backend._device = {};
	backend._queue = {};
	backend._state = "ready";
	recovery.resolve();

	await restore;
	assert.equal(backend._state, "ready");
}

async function testReleaseRuntimeContinuesAfterCleanupFailure() {
	const backend = attachBackend();
	const calls = [];
	backend._postProcessRuntime = {
		destroy() {
			calls.push("postprocess");
			throw new Error("simulated cleanup failure");
		},
	};
	backend._frameOrchestrator = {
		destroy() {
			calls.push("orchestrator");
		},
	};
	backend._resources = {
		destroy() {
			calls.push("resources");
		},
	};

	Logger.configure({ level: "silent", resetOnceKeys: true });
	try {
		backend._releaseDeviceRuntime();
	} finally {
		Logger.reset();
	}
	assert.deepEqual(calls, ["postprocess", "orchestrator", "resources"]);
	assert.equal(backend._postProcessRuntime, null);
	assert.equal(backend._frameOrchestrator, null);
	assert.equal(backend._resources, null);
}

async function testComputeExtensionRejectsUntilReadyAndKeepsIdentity() {
	const backend = attachBackend();
	const extension = backend.extensions.requireBackendExtension(
		WEBGPU_COMPUTE_EXTENSION,
	);
	const iblExecutor = backend.extensions.requireBackendExtension(
		IBL_PREFILTER_EXECUTOR_EXTENSION,
	);
	backend._state = "lost";
	backend._device = {};
	backend._queue = {};

	assert.equal(extension.device, null);
	assert.equal(iblExecutor.getAvailability().acceptsRequests, false);
	assert.throws(
		() => extension.createBuffer({ size: 4, usage: 0 }),
		/compute extension create buffers.*state "ready".*state is "lost"/i,
	);
	backend._state = "restoring";
	assert.throws(
		() => extension.createBuffer({ size: 4, usage: 0 }),
		/compute extension create buffers.*state "ready".*state is "restoring"/i,
	);

	const buffer = { size: 4, destroy() {} };
	backend._resourceManager = {
		createBuffer() {
			return buffer;
		},
	};
	backend._state = "ready";

	assert.strictEqual(
		backend.extensions.requireBackendExtension(WEBGPU_COMPUTE_EXTENSION),
		extension,
	);
	assert.strictEqual(
		backend.extensions.requireBackendExtension(
			IBL_PREFILTER_EXECUTOR_EXTENSION,
		),
		iblExecutor,
	);
	assert.equal(iblExecutor.getAvailability().state, "ready");
	assert.strictEqual(extension.createBuffer({ size: 4, usage: 0 }), buffer);
}

async function testAutomaticRecoveryExhaustionReturnsToLost() {
	const backend = attachBackend();
	backend._canvas = {};
	backend._state = "restoring";
	backend._lifecycleEpoch = 7;
	let attempts = 0;
	backend._initializeDeviceRuntime = async () => {
		attempts++;
		throw new Error(`recovery failure ${attempts}`);
	};
	backend._delayMs = async () => {};

	Logger.configure({ level: "silent", resetOnceKeys: true });
	try {
		await backend._recoverDeviceAfterLoss(7, {
			reason: "unknown",
			message: "simulated recovery exhaustion",
		});
	} finally {
		Logger.reset();
	}

	assert.equal(attempts, 3);
	assert.equal(backend._state, "lost");
}

async function testFrameGuardsRunBeforeSideEffects() {
	const { backend, calls } = createReadyFrameBackend();
	const context = createFrameContext();
	await backend.beginFrame(context);
	const callsBeforeDuplicate = calls.slice();

	await assert.rejects(
		backend.beginFrame(createFrameContext()),
		/requires no active frame/i,
	);
	assert.deepEqual(calls, callsBeforeDuplicate);

	assert.throws(
		() =>
			backend.executePass(
				{ stage: "main-opaque", executor: "backend", enabled: true },
				createFrameContext(),
			),
		/context must match/i,
	);
	assert.equal(calls.includes("orchestrator:execute"), false);
	await backend.abortFrame();
	assert.equal(backend._activeFrameTransaction, null);
}

async function testFrameOperationsRequireActiveFrame() {
	const { backend } = createReadyFrameBackend();
	const context = createFrameContext();
	const pass = {
		stage: "main-opaque",
		executor: "backend",
		enabled: true,
	};

	assert.throws(() => backend.executePass(pass, context), /requires an active frame/i);
	assert.throws(() => backend.skipPass(pass), /requires an active frame/i);
	await assert.rejects(() => backend.endFrame(), /requires an active frame/i);
	await assert.doesNotReject(() => backend.abortFrame());
	await assert.doesNotReject(() => backend.abortFrame());
}

async function testFailedBeginCanBeAborted() {
	const beginError = new Error("simulated begin failure");
	const { backend, calls } = createReadyFrameBackend({ beginError });

	await assert.rejects(backend.beginFrame(createFrameContext()), beginError);
	assert.equal(backend._activeFrameTransaction, null);
	await backend.abortFrame(beginError);

	assert.equal(backend._activeFrameTransaction, null);
	assert.ok(calls.includes("orchestrator:abort"));
	assert.ok(calls.includes("resources:abort"));
	assert.ok(calls.includes("particles:end"));
}

async function testFailedPassAndEndCanBeAborted() {
	const passError = new Error("simulated pass failure");
	const passRuntime = createReadyFrameBackend({ executeError: passError });
	const passContext = createFrameContext();
	await passRuntime.backend.beginFrame(passContext);
	assert.throws(
		() =>
			passRuntime.backend.executePass(
				{ stage: "main-opaque", executor: "backend", enabled: true },
				passContext,
			),
		passError,
	);
	assert.equal(passRuntime.backend._activeFrameTransaction?.isOpen, true);
	await passRuntime.backend.abortFrame(passError);
	assert.equal(passRuntime.backend._activeFrameTransaction, null);

	const endError = new Error("simulated end failure");
	const endRuntime = createReadyFrameBackend({ endError });
	await endRuntime.backend.beginFrame(createFrameContext());
	await assert.rejects(() => endRuntime.backend.endFrame(), endError);
	await endRuntime.backend.abortFrame(endError);
	assert.ok(endRuntime.calls.includes("orchestrator:abort"));
	assert.ok(endRuntime.calls.includes("resources:abort"));
}

async function run() {
	await testInitializeLifecycleGuards();
	await testFailedInitializeCanRetry();
	await testDestroySupersedesPendingInitialize();
	await testDeviceLossSupersedesPendingInitialize();
	await testReadyRestoreRebuildsSameBackend();
	await testRestoreWaitsForAutomaticRecovery();
	await testReleaseRuntimeContinuesAfterCleanupFailure();
	await testComputeExtensionRejectsUntilReadyAndKeepsIdentity();
	await testAutomaticRecoveryExhaustionReturnsToLost();
	await testFrameGuardsRunBeforeSideEffects();
	await testFrameOperationsRequireActiveFrame();
	await testFailedBeginCanBeAborted();
	await testFailedPassAndEndCanBeAborted();
	console.log("WebGPU backend lifecycle tests passed");
}

run();
