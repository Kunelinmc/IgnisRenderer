import assert from "node:assert/strict";
import {
	WorkerScheduler,
	globalWorkerScheduler,
} from "../../../src/workers/WorkerScheduler.ts";

import { FakeWorker } from "../../helpers/fakes.mjs";

function delay(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

async function testConcurrencyCap() {
	const scheduler = new WorkerScheduler();
	let activeTasks = 0;
	let maxActiveTasks = 0;

	scheduler.registerPool({
		id: "concurrency",
		size: 2,
		createWorker: () =>
			new FakeWorker((message, worker) => {
				const payload = message.payload;
				activeTasks += 1;
				maxActiveTasks = Math.max(maxActiveTasks, activeTasks);
				setTimeout(() => {
					activeTasks -= 1;
					worker.emitMessage({
						id: message.id,
						result: payload.value,
					});
				}, payload.delayMs);
			}),
	});

	const values = await Promise.all([
		scheduler.schedule("concurrency", { value: 1, delayMs: 20 }),
		scheduler.schedule("concurrency", { value: 2, delayMs: 20 }),
		scheduler.schedule("concurrency", { value: 3, delayMs: 20 }),
		scheduler.schedule("concurrency", { value: 4, delayMs: 20 }),
	]);

	assert.deepEqual(values, [1, 2, 3, 4]);
	assert.equal(maxActiveTasks, 2);
	scheduler.shutdownAll();
}

async function testPriorityScheduling() {
	const scheduler = new WorkerScheduler();
	const executionOrder = [];

	scheduler.registerPool({
		id: "priority",
		size: 1,
		createWorker: () =>
			new FakeWorker((message, worker) => {
				const payload = message.payload;
				executionOrder.push(payload.id);
				setTimeout(() => {
					worker.emitMessage({
						id: message.id,
						result: payload.id,
					});
				}, payload.delayMs);
			}),
	});

	const first = scheduler.schedule("priority", {
		id: "first",
		delayMs: 25,
	});
	const second = scheduler.schedule(
		"priority",
		{ id: "second", delayMs: 0 },
		{ priority: 0 }
	);
	const third = scheduler.schedule(
		"priority",
		{ id: "third", delayMs: 0 },
		{ priority: 10 }
	);

	await Promise.all([first, second, third]);
	assert.deepEqual(executionOrder, ["first", "third", "second"]);
	scheduler.shutdownAll();
}

async function testAbortQueuedTask() {
	const scheduler = new WorkerScheduler();
	const startedTasks = [];

	scheduler.registerPool({
		id: "abort",
		size: 1,
		createWorker: () =>
			new FakeWorker((message, worker) => {
				const payload = message.payload;
				startedTasks.push(payload.id);
				setTimeout(() => {
					worker.emitMessage({
						id: message.id,
						result: payload.id,
					});
				}, payload.delayMs);
			}),
	});

	const firstTask = scheduler.schedule("abort", {
		id: "first",
		delayMs: 40,
	});
	const controller = new AbortController();
	const queuedTask = scheduler.schedule(
		"abort",
		{ id: "second", delayMs: 0 },
		{ signal: controller.signal }
	);
	controller.abort();

	await assert.rejects(queuedTask, (error) => {
		return error instanceof Error && error.name === "AbortError";
	});
	assert.equal(await firstTask, "first");
	assert.deepEqual(startedTasks, ["first"]);
	scheduler.shutdownAll();
}

async function testAbortRunningTaskKeepsWorkerExclusive() {
	const scheduler = new WorkerScheduler();
	const executionTrace = [];
	let activeTasks = 0;
	let maxActiveTasks = 0;

	scheduler.registerPool({
		id: "abort-running",
		size: 1,
		createWorker: () =>
			new FakeWorker((message, worker) => {
				const payload = message.payload;
				executionTrace.push(`${payload.id}:start`);
				activeTasks += 1;
				maxActiveTasks = Math.max(maxActiveTasks, activeTasks);
				setTimeout(() => {
					activeTasks -= 1;
					executionTrace.push(`${payload.id}:end`);
					worker.emitMessage({
						id: message.id,
						result: payload.id,
					});
				}, payload.delayMs);
			}),
	});

	const controller = new AbortController();
	const runningTask = scheduler.schedule(
		"abort-running",
		{ id: "first", delayMs: 35 },
		{ signal: controller.signal }
	);
	await delay(5);
	controller.abort();
	await assert.rejects(runningTask, (error) => {
		return error instanceof Error && error.name === "AbortError";
	});

	const nextTask = scheduler.schedule("abort-running", {
		id: "second",
		delayMs: 0,
	});
	assert.equal(await nextTask, "second");
	assert.equal(maxActiveTasks, 1);
	assert.deepEqual(executionTrace, [
		"first:start",
		"first:end",
		"second:start",
		"second:end",
	]);
	scheduler.shutdownAll();
}

async function testTimeoutRecovery() {
	const scheduler = new WorkerScheduler();
	let createdWorkers = 0;

	scheduler.registerPool({
		id: "timeout",
		size: 1,
		createWorker: () => {
			createdWorkers += 1;
			return new FakeWorker((message, worker) => {
				const payload = message.payload;
				if (payload.hang) return;
				setTimeout(() => {
					worker.emitMessage({
						id: message.id,
						result: payload.value,
					});
				}, payload.delayMs);
			});
		},
	});

	await assert.rejects(
		scheduler.schedule(
			"timeout",
			{ hang: true },
			{ timeoutMs: 10, priority: 1 }
		),
		(error) => {
			return error instanceof Error && error.message.includes("timed out");
		}
	);

	await delay(1);
	const recovery = await scheduler.schedule("timeout", {
		value: "ok",
		delayMs: 0,
	});
	assert.equal(recovery, "ok");
	assert.ok(createdWorkers >= 2);
	scheduler.shutdownAll();
}

function testGlobalSingleton() {
	const fromStatic = WorkerScheduler.getGlobal();
	assert.equal(fromStatic, globalWorkerScheduler);
	fromStatic.shutdownAll();
}

async function run() {
	await testConcurrencyCap();
	await testPriorityScheduling();
	await testAbortQueuedTask();
	await testAbortRunningTaskKeepsWorkerExclusive();
	await testTimeoutRecovery();
	testGlobalSingleton();
	console.log("Worker scheduler tests passed");
}

run().catch((error) => {
	console.error(error);
	process.exit(1);
});
