import assert from "node:assert/strict";

import { Camera } from "../../../src/cameras/Camera.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

const originalWindow = globalThis.window;

function createCanvas() {
	return {
		width: 320,
		height: 180,
		getBoundingClientRect() {
			return { width: 320, height: 180 };
		},
	};
}

async function run() {
	globalThis.window = { devicePixelRatio: 1 };
	const backend = new TestRenderBackend();
	const renderer = new Renderer({
		backend,
		canvas: createCanvas(),
		camera: new Camera(),
		displayOutput: {
			mode: "auto",
			exposure: 1.5,
		},
	});

	assert.equal(renderer.getDisplayOutputState(), null);
	await renderer.initialize();
	assert.deepEqual(renderer.getDisplayOutputState(), {
		requested: {
			mode: "auto",
			exposure: 1.5,
			hdrHeadroom: 4,
		},
		activeDynamicRange: "sdr",
		colorSpace: "srgb",
	});

	const changes = [];
	renderer.on("displayoutputchange", (change) => changes.push(change));
	const state = await renderer.setDisplayOutput({
		mode: "hdr",
		hdrHeadroom: 6,
	});
	assert.deepEqual(state, {
		requested: {
			mode: "hdr",
			exposure: 1.5,
			hdrHeadroom: 6,
		},
		activeDynamicRange: "sdr",
		colorSpace: "srgb",
		fallbackReason: "backend-unsupported",
	});
	assert.equal(changes.length, 1);
	assert.equal(changes[0].current, state);

	await assert.rejects(
		renderer.setDisplayOutput({ exposure: 65 }),
		RangeError,
	);
	await renderer.destroy();
	console.log("Renderer display-output tests passed");
}

try {
	await run();
} finally {
	globalThis.window = originalWindow;
}
