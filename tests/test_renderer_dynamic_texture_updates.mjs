import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Texture } from "../src/core/Texture.ts";
import { Renderer } from "../src/renderers/Renderer.ts";

import { FakeDynamicTexture } from "./helpers/test_fakes.mjs";

class StubBackend {
	constructor() {
		this.type = "stub";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			skybox: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
		};
		this.frameScheduling = "on-demand";
		this.beginFrameCount = 0;
	}

	async init() {}

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
		const dynamicTexture = new FakeDynamicTexture(2);
		const originalWarn = console.warn;
		const warnedMessages = [];
		console.warn = (message) => warnedMessages.push(message);
		try {
			for (let i = 0; i < 1200; i++) {
				renderer.warnOnce(`dynamic-warning-${i}`, `dynamic warning ${i}`);
			}
		} finally {
			console.warn = originalWarn;
		}

		await renderer.renderScene(0);
		await renderer.renderScene(16);
		await renderer.renderScene(32);

		assert.equal(backend.beginFrameCount, 2);
		assert.equal(renderer._warnings.size, 1024);
		assert.equal(renderer._warnings.has("dynamic-warning-0"), false);
		assert.equal(renderer._warnings.has("dynamic-warning-1199"), true);
		assert.equal(warnedMessages.length, 1200);

		dynamicTexture.dispose();
		console.log("Renderer dynamic texture update tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
