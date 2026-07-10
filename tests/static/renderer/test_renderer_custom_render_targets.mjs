import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Renderer } from "../../../src/renderers/Renderer.ts";
import { TextureFormat } from "../../../src/renderers/types.ts";
import { installNoopPostProcessAdapter } from "../../helpers/postprocess.mjs";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

class CustomTargetBackend extends TestRenderBackend {
	constructor() {
		super();
		this.type = "webgpu";
		this.capabilities = {
			...this.capabilities,
			customRenderTargets: true,
			customRenderPasses: true,
			renderTargetReadback: true,
		};
		this.frameScheduling = "always";
		this.contexts = [];
		this.executedPasses = [];
		this.skippedPasses = [];
		this.readRequests = [];
		installNoopPostProcessAdapter(this, "webgpu");
	}

	getAttachments({ width, height }) {
		return { width, height };
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

	readRenderTargetColor(id, attachmentIndex = 0, options = {}) {
		this.readRequests.push({ id, attachmentIndex, options });
		return Promise.resolve({
			bytes: new Uint8Array([1, 2, 3, 4]),
			width: options.width ?? 1,
			height: options.height ?? 1,
			format: options.format ?? TextureFormat.RGBA8Unorm,
			bytesPerPixel: options.bytesPerPixel ?? 4,
			bytesPerRow: options.bytesPerPixel ?? 4,
			toFloat32: () => new Float32Array(1),
			toRGBAFloat32: () => new Float32Array(4),
			toNormalizedRGBA8Float32: () => new Float32Array(4),
		});
	}
}

function createRenderer(backend = new CustomTargetBackend()) {
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
	return { renderer, backend };
}

async function testRegistryAndFrameSnapshot() {
	const { renderer, backend } = createRenderer();
	renderer.renderTargets.register({
		id: "inspect",
		size: { mode: "fixed", width: 64, height: 32 },
		color: [
			{ format: TextureFormat.RGBA8Unorm },
			{ format: TextureFormat.RGBA16Float },
		],
		depth: { format: TextureFormat.Depth32Float },
	});
	assert.throws(
		() =>
			renderer.renderTargets.register({
				id: "inspect",
				size: { mode: "fixed", width: 1, height: 1 },
				color: [{ format: TextureFormat.RGBA8Unorm }],
			}),
		/already registered/
	);

	let executed = 0;
	renderer.renderPasses.register({
		id: "inspect-pass",
		target: "inspect",
		dependsOn: ["main-opaque"],
		incremental: { order: 4.25 },
		execute() {
			executed++;
		},
	});

	await renderer.renderFrame(16);

	const context = backend.contexts.at(-1);
	assert.ok(context.renderTargets.has("inspect"));
	assert.ok(context.customRenderPasses.has("inspect-pass"));
	assert.ok(
		context.framePlan.backendPasses.some(
			(pass) => pass.stage === "inspect-pass" && pass.enabled
		)
	);
	assert.ok(backend.executedPasses.includes("inspect-pass"));
	assert.equal(executed, 0, "fake backend records pass execution only");

	const readback = await renderer.renderTargets.readColor("inspect", 1, {
		width: 2,
		height: 2,
		format: TextureFormat.RGBA16Float,
		bytesPerPixel: 16,
	});
	assert.equal(readback.width, 2);
	assert.equal(backend.readRequests[0].id, "inspect");
	assert.equal(backend.readRequests[0].attachmentIndex, 1);
	await assert.rejects(
		() => renderer.renderTargets.readColor("inspect", 2),
		/unavailable/
	);

	renderer.renderPasses.unregister("inspect-pass");
	renderer.requestRender("unknown");
	await renderer.renderFrame(32);
	const nextPlan = backend.contexts.at(-1).framePlan;
	assert.equal(
		nextPlan.backendPasses.some((pass) => pass.stage === "inspect-pass"),
		false
	);
}

async function run() {
	const originalWindow = globalThis.window;
	try {
		globalThis.window = { devicePixelRatio: 1 };
		await testRegistryAndFrameSnapshot();
		console.log("Renderer custom render target tests passed");
	} finally {
		globalThis.window = originalWindow;
	}
}

await run();

