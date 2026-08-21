import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import { TextureFormat } from "../../../src/backends/types.ts";
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
		const format = attachmentIndex === 1 ?
			TextureFormat.RGBA16Float
		:	TextureFormat.RGBA8Unorm;
		const bytesPerPixel = attachmentIndex === 1 ? 8 : 4;
		return Promise.resolve({
			bytes: new Uint8Array((options.width ?? 1) * (options.height ?? 1) * bytesPerPixel),
			width: options.width ?? 1,
			height: options.height ?? 1,
			format,
			bytesPerPixel,
			bytesPerRow: (options.width ?? 1) * bytesPerPixel,
			origin: "top-left",
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
	const renderer = new Renderer(canvas, backend, new Camera());
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
	});
	assert.equal(readback.width, 2);
	assert.equal(readback.format, TextureFormat.RGBA16Float);
	assert.equal(readback.bytesPerPixel, 8);
	assert.equal(readback.origin, "top-left");
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

function testStrictRegistryTransactionsAndDependencies() {
	const { renderer } = createRenderer();
	assert.throws(
		() => renderer.renderPasses.register({
			id: "missing-target-pass",
			target: "missing",
			execute() {},
		}),
		/target "missing" is not registered/
	);
	assert.equal(renderer.renderPasses.get("missing-target-pass"), null);

	renderer.renderTargets.register({
		id: "strict",
		size: { mode: "fixed", width: 4, height: 4 },
		color: [{ format: TextureFormat.RGBA8Unorm }],
	});
	assert.throws(
		() => renderer.renderPasses.register({
			id: "main-opaque",
			target: "strict",
			execute() {},
		}),
		/Cannot register built-in pipeline stage/
	);
	assert.equal(renderer.renderPasses.get("main-opaque"), null);

	renderer.renderPasses.register({
		id: "strict-pass",
		target: "strict",
		execute() {},
	});
	assert.throws(
		() => renderer.renderTargets.unregister("strict"),
		/referenced by custom render pass "strict-pass"/
	);
	assert.ok(renderer.renderTargets.get("strict"));
	renderer.renderPasses.unregister("strict-pass");
	renderer.renderTargets.unregister("strict");
	assert.equal(renderer.renderTargets.get("strict"), null);

	const multisampled = renderer.renderTargets.register({
		id: "msaa",
		size: { mode: "fixed", width: 4, height: 4 },
		color: [{ format: TextureFormat.RGBA8Unorm }],
		sampleCount: 4.9,
	});
	assert.equal(multisampled.descriptor.sampleCount, 4);
	renderer.renderTargets.unregister("msaa");
	const clamped = renderer.renderTargets.register({
		id: "clamped-samples",
		size: { mode: "fixed", width: 4, height: 4 },
		color: [{ format: TextureFormat.RGBA8Unorm }],
		sampleCount: 0,
	});
	assert.equal(clamped.descriptor.sampleCount, 1);
	renderer.renderTargets.unregister("clamped-samples");
	assert.throws(
		() => renderer.renderTargets.register({
			id: "invalid-samples",
			size: { mode: "fixed", width: 4, height: 4 },
			color: [{ format: TextureFormat.RGBA8Unorm }],
			sampleCount: Number.NaN,
		}),
		/sampleCount must be a finite number/,
	);
	assert.throws(
		() => renderer.renderTargets.register({
			id: "bad-color",
			size: { mode: "fixed", width: 4, height: 4 },
			color: [{ format: TextureFormat.Depth32Float }],
		}),
		/renderable color format/
	);
	assert.throws(
		() => renderer.renderTargets.register({
			id: "bad-depth",
			size: { mode: "fixed", width: 4, height: 4 },
			color: [{ format: TextureFormat.RGBA8Unorm }],
			depth: { format: TextureFormat.RGBA8Unorm },
		}),
		/renderable depth-only format/
	);
}

async function run() {
	const originalWindow = globalThis.window;
	try {
		globalThis.window = { devicePixelRatio: 1 };
		testStrictRegistryTransactionsAndDependencies();
		await testRegistryAndFrameSnapshot();
		console.log("Renderer custom render target tests passed");
	} finally {
		globalThis.window = originalWindow;
	}
}

await run();
