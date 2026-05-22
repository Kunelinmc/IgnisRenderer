import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Renderer } from "../src/renderers/Renderer.ts";
import {
	ALL_POST_PROCESS_CAPABILITIES,
	createNoopPostProcessSupport,
} from "./helpers/postprocess.mjs";

class RegistryBackend {
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
		this.postProcess = createNoopPostProcessSupport(
			"webgpu",
			ALL_POST_PROCESS_CAPABILITIES
		);
		this.frameScheduling = "always";
		this.passExecutors = {};
		this.contexts = [];
		this.executedPasses = [];
		this.skippedPasses = [];
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

	beginFrame(context) {
		this.contexts.push(context);
	}

	executePass(pass) {
		this.executedPasses.push(pass.stage);
	}

	skipPass(pass) {
		this.skippedPasses.push(pass.stage);
	}

	endFrame() {}
}

async function run() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

		const backend = new RegistryBackend();
		const canvas = {
			width: 320,
			height: 180,
			getBoundingClientRect() {
				return { width: 320, height: 180 };
			},
		};
		const renderer = new Renderer(backend, canvas, new Camera());
		renderer.features.enableShadows = false;
		renderer.features.enableReflection = false;
		renderer.features.enableEnvironment = false;

		await renderer.renderScene(0);
		backend.executedPasses.length = 0;
		backend.skippedPasses.length = 0;

		const customPassId = "custom-registry-pass";
		const customReasonId = "custom-registry-dirty";
		renderer.pipeline.registerBackendPass({
			id: customPassId,
			dependsOn: ["main-opaque"],
			shouldRun: () => true,
			incremental: { order: 4.5 },
		});
		renderer.pipeline.registerDirtyReason({
			id: customReasonId,
			firstPass: customPassId,
		});

		try {
			renderer.requestRender(customReasonId);
			await renderer.renderScene(16);

			const stats = renderer.getLastIncrementalFrameStats();
			assert.equal(stats.firstPass, customPassId);
			assert.equal(stats.forceFullFrame, false);
			assert.ok(backend.skippedPasses.includes("main-opaque"));
			assert.ok(backend.skippedPasses.includes("postprocess"));
			assert.ok(backend.executedPasses.includes(customPassId));
		} finally {
			renderer.pipeline.unregisterDirtyReason(customReasonId);
			renderer.pipeline.unregisterBackendPass(customPassId);
		}

		console.log("Renderer pipeline registry tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
