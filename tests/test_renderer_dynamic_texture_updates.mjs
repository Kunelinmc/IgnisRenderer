import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Texture } from "../src/core/Texture.ts";
import { Renderer } from "../src/renderers/Renderer.ts";

import { FakeDynamicTexture } from "./helpers/test_fakes.mjs";
import {
	ALL_POST_PROCESS_CAPABILITIES,
	installNoopPostProcessSupport,
} from "./helpers/postprocess.mjs";

class StubBackend {
	constructor() {
		this.type = "stub";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
		};
		installNoopPostProcessSupport(
			this,
			"stub",
			ALL_POST_PROCESS_CAPABILITIES
		);
		this.frameScheduling = "on-demand";
		this.beginFrameCount = 0;
		this.deviceLostInfos = [];
		this.restoreCanvases = [];
	}

	async init(canvas) {
		this.initCanvas = canvas;
	}

	onDeviceLost(info) {
		this.deviceLostInfos.push(info);
	}

	restore(canvas) {
		this.restoreCanvases.push(canvas);
	}

	resize() {}

	getAttachments(width, height) {
		return {
			width,
			height,
			pixels: new Uint8ClampedArray(width * height * 4),
			depthBuffer: new Float32Array(width * height),
			normalBuffer: new Float32Array(width * height * 3),
		};
	}

	beginFrame() {
		this.beginFrameCount++;
	}

	executePass() {}

	endFrame() {}
}

async function run() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

		const backend = new StubBackend();
		const canvas = {
			width: 320,
			height: 180,
			getBoundingClientRect() {
				return { width: 320, height: 180 };
			},
		};
		const camera = new Camera();
		const renderer = new Renderer(backend, canvas, camera);
		await renderer.onDeviceLost({
			reason: "manual-test",
			message: "simulated loss",
		});
		assert.equal(backend.deviceLostInfos.length, 1);
		assert.equal(backend.deviceLostInfos[0].message, "simulated loss");

		await renderer.restore();
		assert.equal(backend.restoreCanvases.length, 1);
		assert.equal(backend.restoreCanvases[0], canvas);

		const dynamicTexture = new FakeDynamicTexture(2);
		const originalWarn = console.warn;
		const warnedMessages = [];
		console.warn = (message) => warnedMessages.push(message);
		try {
			for (let i = 0; i < 1200; i++) {
				renderer.logger.warn(`dynamic warning ${i}`);
			}
		} finally {
			console.warn = originalWarn;
		}

		await renderer.renderScene(0);
		await renderer.renderScene(16);
		await renderer.renderScene(32);

		assert.equal(backend.beginFrameCount, 2);
		assert.equal(warnedMessages.length, 1200);

		dynamicTexture.dispose();
		console.log("Renderer dynamic texture update tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
