import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Logger } from "../src/foundation/Logger.ts";
import { Renderer } from "../src/renderers/Renderer.ts";
import {
	ALL_POST_PROCESS_CAPABILITIES,
	installNoopPostProcessSupport,
} from "./helpers/postprocess.mjs";

class StubBackend {
	constructor() {
		this.type = "webgpu";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			clusteredLighting: false,
			oit: false,
		};
		installNoopPostProcessSupport(
			this,
			"webgpu",
			ALL_POST_PROCESS_CAPABILITIES
		);
		this.frameScheduling = "continuous";
		this.passExecutors = {};
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

	beginFrame() {}

	executePass() {}

	endFrame() {}
}

async function run() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;
	const warnings = [];

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;
		Logger.configure({
			level: "warn",
			resetOnceKeys: true,
			sink: {
				warn: (...args) =>
					warnings.push(args.map((arg) => String(arg)).join(" ")),
			},
		});

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
		renderer.features.enableShadows = false;
		renderer.features.enableEnvironment = false;
		renderer.postProcess.disable("gamma");

		await renderer.renderScene(0);
		await renderer.renderScene(16);

		const reflectionWarnings = warnings.filter((warning) =>
			warning.includes("[webgpu-feature-reflection]")
		);
		assert.equal(reflectionWarnings.length, 1);

		console.log("Renderer feature warning once tests passed");
	} finally {
		Logger.reset();
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
