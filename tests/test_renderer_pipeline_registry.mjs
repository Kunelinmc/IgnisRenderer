import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Renderer } from "../src/renderers/Renderer.ts";
import {
	installNoopPostProcessSupport,
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
		installNoopPostProcessSupport(
			this,
			"webgpu"
		);
		this.frameScheduling = "always";
		this.passExecutors = {};
		this.contexts = [];
		this.executedPasses = [];
		this.skippedPasses = [];
		this.beginFrameCalls = 0;
		this.endFrameCalls = 0;
		this.abortFrameCalls = 0;
		this.abortErrors = [];
		this.throwInBeginFrame = null;
		this.throwOnPass = null;
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
		this.beginFrameCalls++;
		this.contexts.push(context);
		if (this.throwInBeginFrame) {
			throw this.throwInBeginFrame;
		}
	}

	executePass(pass) {
		this.executedPasses.push(pass.stage);
		if (this.throwOnPass && pass.stage === this.throwOnPass.stage) {
			throw this.throwOnPass.error;
		}
	}

	skipPass(pass) {
		this.skippedPasses.push(pass.stage);
	}

	endFrame() {
		this.endFrameCalls++;
	}

	abortFrame(error) {
		this.abortFrameCalls++;
		this.abortErrors.push(error);
	}
}

function createRenderer(backend) {
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
	return renderer;
}

async function testRendererAbortsBackendFrameOnPassFailure() {
	const backend = new RegistryBackend();
	const renderer = createRenderer(backend);
	const error = new Error("main pass failed");
	backend.throwOnPass = {
		stage: "main-opaque",
		error,
	};

	let caught = null;
	try {
		await renderer.renderScene(16);
	} catch (caughtError) {
		caught = caughtError;
	}
	assert.strictEqual(caught, error);
	assert.equal(backend.beginFrameCalls, 1);
	assert.equal(backend.endFrameCalls, 0);
	assert.equal(backend.abortFrameCalls, 1);
	assert.strictEqual(backend.abortErrors[0], error);
}

async function testRendererAbortsPartialBeginFrameFailure() {
	const backend = new RegistryBackend();
	const renderer = createRenderer(backend);
	const error = new Error("begin frame failed");
	backend.throwInBeginFrame = error;

	let caught = null;
	try {
		await renderer.renderScene(16);
	} catch (caughtError) {
		caught = caughtError;
	}
	assert.strictEqual(caught, error);
	assert.equal(backend.beginFrameCalls, 1);
	assert.equal(backend.endFrameCalls, 0);
	assert.equal(backend.abortFrameCalls, 1);
	assert.strictEqual(backend.abortErrors[0], error);
}

async function testRendererSuccessfulFrameEndsAndSchedules() {
	const backend = new RegistryBackend();
	const renderer = createRenderer(backend);
	let frameEndEvents = 0;
	let scheduledFrames = 0;
	renderer.on("frameend", () => {
		frameEndEvents++;
	});
	globalThis.requestAnimationFrame = () => {
		scheduledFrames++;
		return scheduledFrames;
	};

	await renderer.renderScene(16);

	assert.equal(backend.beginFrameCalls, 1);
	assert.equal(backend.endFrameCalls, 1);
	assert.equal(backend.abortFrameCalls, 0);
	assert.equal(frameEndEvents, 1);
	assert.equal(scheduledFrames, 1);
}

async function run() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

		await testRendererAbortsBackendFrameOnPassFailure();
		await testRendererAbortsPartialBeginFrameFailure();
		await testRendererSuccessfulFrameEndsAndSchedules();

		const backend = new RegistryBackend();
		const renderer = createRenderer(backend);

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
