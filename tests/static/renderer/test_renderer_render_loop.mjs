import assert from "node:assert/strict";

import { Camera } from "../../../src/cameras/Camera.ts";
import { Logger } from "../../../src/foundation/Logger.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

function createRenderer() {
	const canvas = {
		width: 320,
		height: 180,
		getBoundingClientRect() {
			return { width: 320, height: 180 };
		},
	};
	return new Renderer(canvas, new TestRenderBackend(), new Camera());
}

async function flushPromises() {
	await new Promise((resolve) => setTimeout(resolve, 0));
}

async function testRenderLoopHandlesErrorsAndContinues() {
	const renderer = createRenderer();
	const scheduled = new Map();
	const cancelled = [];
	const errors = [];
	const frameTimes = [];
	let nextRequestId = 1;

	globalThis.requestAnimationFrame = (callback) => {
		const requestId = nextRequestId++;
		scheduled.set(requestId, callback);
		return requestId;
	};
	globalThis.cancelAnimationFrame = (requestId) => {
		cancelled.push(requestId);
		scheduled.delete(requestId);
	};
	Logger.configure({
		level: "error",
		resetOnceKeys: true,
		sink: {
			error: (...args) => errors.push(args),
		},
	});
	renderer.renderFrame = async (nowMs) => {
		frameTimes.push(nowMs);
		if (frameTimes.length === 1) {
			throw new Error("frame failed");
		}
		return { rendered: true };
	};

	const stop = renderer.renderLoop();
	assert.strictEqual(renderer.renderLoop(), stop);
	assert.equal(scheduled.size, 1);

	const firstRequest = scheduled.entries().next().value;
	scheduled.delete(firstRequest[0]);
	firstRequest[1](16);
	await flushPromises();

	assert.deepEqual(frameTimes, [16]);
	assert.equal(errors.length, 1);
	assert.match(String(errors[0][1]), /Renderer render loop frame failed/);
	assert.equal(errors[0][2]?.message, "frame failed");
	assert.equal(scheduled.size, 1);

	const secondRequest = scheduled.entries().next().value;
	scheduled.delete(secondRequest[0]);
	secondRequest[1](32);
	await flushPromises();

	assert.deepEqual(frameTimes, [16, 32]);
	assert.equal(scheduled.size, 1);
	const pendingRequestId = scheduled.keys().next().value;
	stop();
	stop();
	assert.deepEqual(cancelled, [pendingRequestId]);
	assert.equal(scheduled.size, 0);
}

async function testRenderLoopCoalescesTicksWhileFrameInFlight() {
	const renderer = createRenderer();
	const scheduled = new Map();
	const errors = [];
	const frameTimes = [];
	let nextRequestId = 1;

	globalThis.requestAnimationFrame = (callback) => {
		const requestId = nextRequestId++;
		scheduled.set(requestId, callback);
		return requestId;
	};
	globalThis.cancelAnimationFrame = (requestId) => {
		scheduled.delete(requestId);
	};
	Logger.configure({
		level: "error",
		resetOnceKeys: true,
		sink: {
			error: (...args) => errors.push(args),
		},
	});

	let releaseFrame;
	const gate = new Promise((resolve) => {
		releaseFrame = resolve;
	});
	renderer.renderFrame = async (nowMs) => {
		frameTimes.push(nowMs);
		if (frameTimes.length === 1) {
			await gate;
		}
		return { rendered: true };
	};

	function fireNextTick(nowMs) {
		const entry = scheduled.entries().next().value;
		scheduled.delete(entry[0]);
		entry[1](nowMs);
	}

	const stop = renderer.renderLoop();
	fireNextTick(16);
	assert.deepEqual(frameTimes, [16]);
	assert.equal(scheduled.size, 1);

	// This tick arrives while frame 1 is still in flight and must be
	// coalesced instead of rejecting or queueing another render.
	fireNextTick(32);
	await flushPromises();
	assert.deepEqual(frameTimes, [16]);
	assert.equal(errors.length, 0);
	assert.equal(scheduled.size, 1);

	// Releasing frame 1 lets the next already-scheduled tick render again.
	releaseFrame();
	await flushPromises();
	fireNextTick(48);
	await flushPromises();
	assert.deepEqual(frameTimes, [16, 48]);

	const pendingRequestId = scheduled.keys().next().value;
	stop();
	stop();
	assert.equal(scheduled.has(pendingRequestId), false);
	assert.equal(scheduled.size, 0);
}

async function testDestroyStopsRenderLoop() {
	const renderer = createRenderer();
	const scheduled = new Map();
	const cancelled = [];
	let nextRequestId = 1;

	globalThis.requestAnimationFrame = (callback) => {
		const requestId = nextRequestId++;
		scheduled.set(requestId, callback);
		return requestId;
	};
	globalThis.cancelAnimationFrame = (requestId) => {
		cancelled.push(requestId);
		scheduled.delete(requestId);
	};

	renderer.renderLoop();
	const pendingRequestId = scheduled.keys().next().value;
	await renderer.destroy();

	assert.deepEqual(cancelled, [pendingRequestId]);
	assert.equal(scheduled.size, 0);
}

const originalRAF = globalThis.requestAnimationFrame;
const originalCancelRAF = globalThis.cancelAnimationFrame;
const originalWindow = globalThis.window;

try {
	globalThis.window = { devicePixelRatio: 1 };
	await testRenderLoopHandlesErrorsAndContinues();
	await testRenderLoopCoalescesTicksWhileFrameInFlight();
	await testDestroyStopsRenderLoop();
	console.log("Renderer render loop tests passed");
} finally {
	Logger.reset();
	globalThis.requestAnimationFrame = originalRAF;
	globalThis.cancelAnimationFrame = originalCancelRAF;
	globalThis.window = originalWindow;
}
