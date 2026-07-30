import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { PostProcessPass } from "../../../src/postprocess/index.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";

import { FakeDynamicTexture } from "../../helpers/fakes.mjs";
import {
	installNoopPostProcessAdapter,
} from "../../helpers/postprocess.mjs";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

class StubBackend extends TestRenderBackend {
	constructor() {
		super();
		this.type = "stub";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			clusteredLighting: false,
			oit: false,
			occlusionCulling: false,
			postProcess: false,
		};
		this.postProcessSupport = installNoopPostProcessAdapter(
			this,
			"stub"
		);
		this.frameScheduling = "on-demand";
		this.beginFrameCount = 0;
		this.deviceLostInfos = [];
		this.restoreCanvases = [];
	}

	async initialize() {
		this.initCanvas = this.attachContext.surface.canvas;
	}

	onDeviceLost(info) {
		this.deviceLostInfos.push(info);
	}

	restore() {
		this.restoreCanvases.push(this.attachContext.surface.canvas);
	}

	resize() {}

	getAttachments({ width, height }) {
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

class CleanupPostProcessPass extends PostProcessPass {
	constructor(events) {
		super({
			id: "cleanup-pass",
			enabled: true,
			implementations: {
				stub: {
				describeExecution() {
					return {
						color: { access: "read-write", output: "preserve" },
						histories: [{
							descriptor: { id: "cleanup-history" },
							read: [{ access: "read", usage: "cpu-read" }],
							write: [{ access: "write", usage: "cpu-write" }],
						}],
						transients: [{
							descriptor: { id: "cleanup-transient" },
							uses: [{ access: "write", usage: "cpu-write" }],
						}],
					};
				},
				execute() {
					return { ran: true };
				},
					destroy() {
						events.push("pass-destroy");
					},
				},
			},
		});
	}
}

function createPostProcessFrameContext(postProcess) {
	return {
		viewCamera: {
			type: "perspective",
			fov: 60,
			aspectRatio: 1,
			near: 0.1,
			far: 100,
		},
		attachments: {
			width: 8,
			height: 8,
			pixels: new Uint8ClampedArray(8 * 8 * 4),
			depthBuffer: new Float32Array(8 * 8),
			normalBuffer: new Float32Array(8 * 8 * 3),
		},
		features: {},
		postProcess,
		shadowMaps: [],
		scene: {},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: null,
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width: 8, height: 8 }],
			dirtyTileSize: 8,
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			postProcessStartPass: null,
			reasonMask: 0,
			temporalHistoryReset: false,
		},
		transient: new Map(),
	};
}

async function testRendererPostProcessCleanupBridge(canvas) {
	const backend = new StubBackend();
	const renderer = new Renderer({
		backend,
		canvas,
		camera: new Camera(),
	});
	const events = [];
	renderer.postProcess.registerPass(new CleanupPostProcessPass(events));
	const postProcess = renderer.postProcess.createSnapshot("stub");
	const frameContext = createPostProcessFrameContext(postProcess);

	await backend.postProcessSupport.runtime.execute(frameContext);
	backend.postProcessSupport.runtime.destroy();

	const destroyedIds = backend.postProcessSupport.executor.destroyedResources.map(
		(handle) => handle.id
	);
	assert.ok(destroyedIds.includes("cleanup-history:read"));
	assert.ok(destroyedIds.includes("cleanup-history:write"));
	assert.ok(destroyedIds.includes("cleanup-transient"));
	assert.deepEqual(events, ["pass-destroy"]);
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
		const renderer = new Renderer({ backend, canvas, camera });
		const rendererEvents = [];
		renderer.on("devicelost", (event) => {
			rendererEvents.push([
				"devicelost",
				event.backend,
				event.info?.message,
			]);
		});
		renderer.on("backendresourceevent", (event) => {
			rendererEvents.push([
				"backendresourceevent",
				event.backend,
				event.resource,
				event.action,
			]);
		});
		const deviceLostInfo = {
			reason: "manual-test",
			message: "simulated loss",
		};
		backend.onDeviceLost(deviceLostInfo);
		backend.attachContext.events.emit({
			type: "device-lost",
			info: deviceLostInfo,
		});
		assert.equal(backend.deviceLostInfos.length, 1);
		assert.equal(backend.deviceLostInfos[0].message, "simulated loss");
		assert.deepEqual(rendererEvents[0], [
			"devicelost",
			"stub",
			"simulated loss",
		]);

		await renderer.restore();
		assert.equal(backend.restoreCanvases.length, 1);
		assert.equal(backend.restoreCanvases[0], canvas);

		await testRendererPostProcessCleanupBridge(canvas);
		backend.attachContext.events.emit({
			type: "resource-lifecycle",
			event: {
				resource: "postprocess",
				action: "invalidate",
				backend: "stub",
				reason: "event-test",
			},
		});
		assert.deepEqual(rendererEvents[1], [
			"backendresourceevent",
			"stub",
			"postprocess",
			"invalidate",
		]);

		const dynamicTexture = new FakeDynamicTexture();
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

		await renderer.renderFrame(0);
		dynamicTexture.presentFrame();
		await renderer.renderFrame(16);
		await renderer.renderFrame(32);

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
