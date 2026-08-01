import assert from "node:assert/strict";

import { WebGLContextWorkError } from "../../../src/foundation/Error.ts";
import { WebGLContextWorkQueue } from "../../../src/backends/webgl/WebGLContextWorkQueue.ts";

function deferred() {
	let resolve;
	let reject;
	const promise = new Promise((resolvePromise, rejectPromise) => {
		resolve = resolvePromise;
		reject = rejectPromise;
	});
	return { promise, resolve, reject };
}

function createQueue(log = []) {
	let services = { id: "generation-1" };
	const queue = new WebGLContextWorkQueue({
		resolveServices: () => services,
		restoreBaseline: (scope, frameActive) => {
			log.push(`baseline:${scope.generation}:${frameActive ? "frame" : "idle"}`);
		},
	});
	return {
		queue,
		bind(next = services) {
			services = next;
			return queue.bindContext();
		},
	};
}

async function expectCode(promise, code) {
	await assert.rejects(promise, (error) => {
		assert.ok(error instanceof WebGLContextWorkError);
		assert.equal(error.code, code);
		return true;
	});
}

async function testFramePassAndBoundaryOrdering() {
	const log = [];
	const host = createQueue(log);
	host.bind();
	await host.queue.beginFrame("begin", () => log.push("begin"));
	await host.queue.runFramePass("shadow", () => log.push("shadow"));
	await host.queue.enqueue({
		label: "ibl",
		framePolicy: "between-passes",
		contextLossPolicy: "retain-pending",
		execute: (scope) => log.push(`ibl:${scope.generation}`),
	});
	await host.queue.runFramePass("reflection", () => log.push("reflection"));
	await host.queue.endFrame("end", async () => {
		log.push("present:start");
		await Promise.resolve();
		log.push("present:commit");
	});
	assert.deepEqual(log, [
		"begin",
		"shadow",
		"ibl:1",
		"baseline:1:frame",
		"reflection",
		"present:start",
		"present:commit",
	]);
	assert.equal(host.queue.getDebugSnapshot().frameState, "idle");
}

async function testActivePassAndIdleOnlyRejection() {
	const host = createQueue();
	host.bind();
	await host.queue.beginFrame("begin", () => undefined);
	await host.queue.runFramePass("custom", async () => {
		await expectCode(
			host.queue.enqueue({
				label: "nested",
				framePolicy: "between-passes",
				contextLossPolicy: "reject",
				execute: () => undefined,
			}),
			"active-pass",
		);
		await expectCode(
			host.queue.enqueue({
				label: "warmup",
				framePolicy: "idle-deferred",
				contextLossPolicy: "reject",
				execute: () => undefined,
			}),
			"active-pass",
		);
	});
	await expectCode(
		host.queue.enqueue({
			label: "readback",
			framePolicy: "idle-only",
			contextLossPolicy: "reject",
			execute: () => undefined,
		}),
		"active-frame",
	);
	await host.queue.endFrame("end", () => undefined);
}

async function testFailureRequiresAbort() {
	const host = createQueue();
	host.bind();
	await assert.rejects(
		host.queue.beginFrame("begin", () => {
			throw new Error("begin failed");
		}),
		/begin failed/,
	);
	assert.equal(host.queue.getDebugSnapshot().frameState, "abort-required");
	await expectCode(
		host.queue.enqueue({
			label: "readback",
			framePolicy: "idle-only",
			contextLossPolicy: "reject",
			execute: () => undefined,
		}),
		"active-frame",
	);
	await host.queue.abortFrame("abort");
	assert.equal(host.queue.getDebugSnapshot().frameState, "idle");
}

async function testContextGenerationAndRetention() {
	const log = [];
	const host = createQueue(log);
	host.bind();
	const activeGate = deferred();
	const active = host.queue.enqueue({
		label: "active",
		framePolicy: "idle-only",
		contextLossPolicy: "reject",
		execute: async (scope) => {
			log.push(`active:${scope.generation}`);
			await activeGate.promise;
		},
	});
	await Promise.resolve();
	const retained = host.queue.enqueue({
		label: "retained",
		framePolicy: "between-passes",
		contextLossPolicy: "retain-pending",
		execute: (scope) => log.push(`retained:${scope.generation}:${scope.services.id}`),
	});
	host.queue.suspend();
	await expectCode(active, "context-lost");
	host.bind({ id: "generation-2" });
	await Promise.resolve();
	assert.equal(log.includes("retained:2:generation-2"), false);
	activeGate.resolve();
	await retained;
	assert.ok(log.includes("retained:2:generation-2"));
}

async function testLostPendingAbortAndDestroy() {
	const host = createQueue();
	host.bind();
	host.queue.suspend();
	const controller = new AbortController();
	const pending = host.queue.enqueue({
		label: "waiting-ibl",
		framePolicy: "between-passes",
		contextLossPolicy: "retain-pending",
		signal: controller.signal,
		execute: () => undefined,
	});
	controller.abort();
	await assert.rejects(pending, (error) => error.name === "AbortError");
	const destroyed = host.queue.enqueue({
		label: "destroyed-ibl",
		framePolicy: "between-passes",
		contextLossPolicy: "retain-pending",
		execute: () => undefined,
	});
	host.queue.destroy();
	await expectCode(destroyed, "destroyed");
	host.queue.destroy();
}

async function testBoundedFairness() {
	const log = [];
	const host = createQueue(log);
	host.bind();
	const gate = deferred();
	const firstAux = host.queue.enqueue({
		label: "aux-1",
		framePolicy: "idle-only",
		contextLossPolicy: "reject",
		execute: async () => {
			log.push("aux-1");
			await gate.promise;
		},
	});
	await Promise.resolve();
	const frame = host.queue.beginFrame("frame", () => log.push("frame"));
	const secondAux = host.queue.enqueue({
		label: "aux-2",
		framePolicy: "idle-only",
		contextLossPolicy: "reject",
		execute: () => log.push("aux-2"),
	});
	gate.resolve();
	await firstAux;
	await frame;
	await host.queue.endFrame("end", () => log.push("end"));
	await secondAux;
	assert.ok(log.indexOf("frame") < log.indexOf("aux-2"));
	assert.ok(log.indexOf("end") < log.indexOf("aux-2"));
}

async function testBaselineRestoresAfterFailureAndCancellation() {
	const log = [];
	const host = createQueue(log);
	host.bind();
	await assert.rejects(
		host.queue.enqueue({
			label: "failure",
			framePolicy: "idle-only",
			contextLossPolicy: "reject",
			execute: () => {
				throw new Error("failed");
			},
		}),
		/failed/,
	);
	assert.ok(log.includes("baseline:1:idle"));
	log.length = 0;
	const controller = new AbortController();
	const started = deferred();
	const cancelled = host.queue.enqueue({
		label: "cancelled",
		framePolicy: "idle-only",
		contextLossPolicy: "reject",
		signal: controller.signal,
		execute: (scope) => {
			started.resolve();
			return new Promise((resolve) => {
				scope.signal.addEventListener("abort", resolve, { once: true });
			});
		},
	});
	await started.promise;
	controller.abort();
	await assert.rejects(cancelled, (error) => error.name === "AbortError");
	await Promise.resolve();
	await Promise.resolve();
	assert.ok(log.includes("baseline:1:idle"));
	host.queue.destroy();
}

async function testFrameOperationReservationRejectsConcurrentCalls() {
	const log = [];
	const host = createQueue(log);
	host.bind();
	await host.queue.beginFrame("begin", () => log.push("begin"));
	const boundaryGate = deferred();
	const boundaryStarted = deferred();
	const boundary = host.queue.enqueue({
		label: "boundary",
		framePolicy: "between-passes",
		contextLossPolicy: "reject",
		execute: async () => {
			log.push("boundary:start");
			boundaryStarted.resolve();
			await boundaryGate.promise;
			log.push("boundary:end");
		},
	});
	await boundaryStarted.promise;
	const pass = host.queue.runFramePass("reserved-pass", () => {
		log.push("pass");
	});
	await expectCode(
		host.queue.runFramePass("concurrent-pass", () => log.push("concurrent")),
		"active-pass",
	);
	await expectCode(
		host.queue.endFrame("concurrent-end", () => log.push("concurrent-end")),
		"active-frame",
	);
	boundaryGate.resolve();
	await boundary;
	await pass;
	const endGate = deferred();
	const ending = host.queue.endFrame("end", async () => {
		log.push("end:start");
		await endGate.promise;
		log.push("end:finish");
	});
	await Promise.resolve();
	await expectCode(
		host.queue.endFrame("duplicate-end", () => log.push("duplicate")),
		"active-frame",
	);
	endGate.resolve();
	await ending;
	assert.equal(log.includes("concurrent"), false);
	assert.equal(log.includes("concurrent-end"), false);
	assert.equal(log.includes("duplicate"), false);
}

async function testSuspendRejectsReservedBoundaryWaiter() {
	const log = [];
	const host = createQueue(log);
	host.bind();
	await host.queue.beginFrame("begin", () => undefined);
	const retained = host.queue.enqueue({
		label: "retained-boundary",
		framePolicy: "between-passes",
		contextLossPolicy: "retain-pending",
		execute: (scope) => log.push(`retained:${scope.generation}`),
	});
	const pass = host.queue.runFramePass("waiting-pass", () => log.push("pass"));
	host.queue.suspend();
	await expectCode(pass, "context-lost");
	await host.queue.abortFrame("abort");
	host.bind({ id: "generation-2" });
	await retained;
	assert.equal(log[0], "retained:2");
	assert.ok(log.includes("baseline:2:idle"));
}

async function testDestroyRejectsReservedBoundaryWaiter() {
	const host = createQueue();
	host.bind();
	await host.queue.beginFrame("begin", () => undefined);
	const pending = host.queue.enqueue({
		label: "pending",
		framePolicy: "between-passes",
		contextLossPolicy: "retain-pending",
		execute: () => undefined,
	});
	const pass = host.queue.runFramePass("waiting-pass", () => undefined);
	host.queue.destroy();
	await expectCode(pending, "destroyed");
	await expectCode(pass, "destroyed");
	assert.equal(host.queue.getDebugSnapshot().frameState, "idle");
}

async function testAbortReleasesReservedBoundaryWaiter() {
	const log = [];
	const host = createQueue(log);
	host.bind();
	await host.queue.beginFrame("begin", () => undefined);
	const pending = host.queue.enqueue({
		label: "pending",
		framePolicy: "between-passes",
		contextLossPolicy: "retain-pending",
		execute: () => log.push("pending"),
	});
	const pass = host.queue.runFramePass("waiting-pass", () => log.push("pass"));
	await host.queue.abortFrame("abort");
	await expectCode(pass, "active-frame");
	await pending;
	assert.equal(log.includes("pass"), false);
	assert.equal(host.queue.getDebugSnapshot().frameState, "idle");
}

async function testIdleDeferredBatchDoesNotStarveFrame() {
	const log = [];
	const host = createQueue(log);
	host.bind();
	await host.queue.beginFrame("begin-1", () => log.push("frame-1"));
	const firstGate = deferred();
	const firstStarted = deferred();
	const first = host.queue.enqueue({
		label: "warmup-1",
		framePolicy: "idle-deferred",
		contextLossPolicy: "reject",
		execute: async () => {
			log.push("warmup-1:start");
			firstStarted.resolve();
			await firstGate.promise;
			log.push("warmup-1:end");
		},
	});
	await host.queue.endFrame("end-1", () => log.push("end-1"));
	await firstStarted.promise;
	const nextFrame = host.queue.beginFrame("begin-2", () => log.push("frame-2"));
	const second = host.queue.enqueue({
		label: "warmup-2",
		framePolicy: "idle-deferred",
		contextLossPolicy: "reject",
		execute: () => log.push("warmup-2"),
	});
	firstGate.resolve();
	await first;
	await nextFrame;
	assert.ok(log.indexOf("warmup-1:end") < log.indexOf("frame-2"));
	assert.equal(log.includes("warmup-2"), false);
	await host.queue.endFrame("end-2", () => log.push("end-2"));
	await second;
	assert.ok(log.indexOf("end-2") < log.indexOf("warmup-2"));
}

async function testIdleDeferredRejectsContextLossWithoutReplay() {
	const log = [];
	const host = createQueue(log);
	host.bind();
	const warmup = host.queue.enqueue({
		label: "warmup",
		framePolicy: "idle-deferred",
		contextLossPolicy: "reject",
		execute: () => log.push("warmup"),
	});
	host.queue.suspend();
	await expectCode(warmup, "context-lost");
	host.bind({ id: "generation-2" });
	await Promise.resolve();
	await Promise.resolve();
	assert.equal(log.includes("warmup"), false);
}

async function testMaintenanceCoalescesAndBlocksFrameRelease() {
	const log = [];
	const host = createQueue(log);
	host.bind();
	await host.queue.beginFrame("begin", () => log.push("begin"));
	let size = "first";
	const first = host.queue.enqueueMaintenance({
		key: "resize",
		label: "resize:first",
		contextLossPolicy: "reject",
		execute: () => log.push(`resize:${size}`),
	});
	size = "latest";
	const second = host.queue.enqueueMaintenance({
		key: "resize",
		label: "resize:latest",
		contextLossPolicy: "reject",
		execute: () => log.push(`resize:${size}`),
	});
	await host.queue.endFrame("end", () => log.push("end"));
	await Promise.all([first, second]);
	assert.equal(log.filter((entry) => entry.startsWith("resize:")).length, 1);
	assert.ok(log.indexOf("end") < log.indexOf("resize:latest"));
	assert.ok(log.includes("baseline:1:idle"));

	const maintenanceGate = deferred();
	const idleMaintenance = host.queue.enqueueMaintenance({
		key: "resize",
		label: "resize:idle",
		contextLossPolicy: "reject",
		execute: async () => {
			log.push("resize:idle:start");
			await maintenanceGate.promise;
			log.push("resize:idle:end");
		},
	});
	const frame = host.queue.beginFrame("next-frame", () => log.push("next-frame"));
	await Promise.resolve();
	await Promise.resolve();
	assert.equal(log.includes("next-frame"), false);
	maintenanceGate.resolve();
	await idleMaintenance;
	await frame;
	assert.ok(log.indexOf("resize:idle:end") < log.indexOf("next-frame"));
	await host.queue.abortFrame("abort");
}

async function run() {
	await testFramePassAndBoundaryOrdering();
	await testActivePassAndIdleOnlyRejection();
	await testFailureRequiresAbort();
	await testContextGenerationAndRetention();
	await testLostPendingAbortAndDestroy();
	await testBoundedFairness();
	await testBaselineRestoresAfterFailureAndCancellation();
	await testFrameOperationReservationRejectsConcurrentCalls();
	await testSuspendRejectsReservedBoundaryWaiter();
	await testDestroyRejectsReservedBoundaryWaiter();
	await testAbortReleasesReservedBoundaryWaiter();
	await testIdleDeferredBatchDoesNotStarveFrame();
	await testIdleDeferredRejectsContextLossWithoutReplay();
	await testMaintenanceCoalescesAndBlocksFrameRelease();
	console.log("WebGL context work queue static tests passed");
}

await run();
