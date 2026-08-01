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

async function run() {
	await testFramePassAndBoundaryOrdering();
	await testActivePassAndIdleOnlyRejection();
	await testFailureRequiresAbort();
	await testContextGenerationAndRetention();
	await testLostPendingAbortAndDestroy();
	await testBoundedFairness();
	await testBaselineRestoresAfterFailureAndCancellation();
	console.log("WebGL context work queue static tests passed");
}

await run();
