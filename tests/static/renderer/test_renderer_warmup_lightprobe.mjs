import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { LightProbe } from "../../../src/lights/LightProbe.ts";
import { ReflectionProbe } from "../../../src/lights/ReflectionProbe.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { SH } from "../../../src/maths/SH.ts";
import { Renderer } from "../../../src/renderers/Renderer.ts";
import { buildWarmupPlan } from "../../../src/pipeline/WarmupPlanner.ts";
import {
	BloomPass,
	ColorFilterPass,
	PostProcessPass,
} from "../../../src/postprocess/index.ts";
import {
	installNoopPostProcessAdapter,
} from "../../helpers/postprocess.mjs";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

class StubBackend extends TestRenderBackend {
	constructor() {
		super();
		this.type = "stub";
		this.capabilities = {
			sh: true,
			shadows: false,
			reflection: false,
			environment: true,
			postProcess: true,
			clusteredLighting: false,
			oit: false,
			occlusionCulling: false,
		};
		installNoopPostProcessAdapter(
			this,
			"stub"
		);
		this.frameScheduling = "on-demand";
		this.lastWarmupContext = null;
		this.lastWarmupOptions = null;
		this.lastBeginFrameContext = null;
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

	beginFrame(context) {
		this.lastBeginFrameContext = context;
	}

	executePass() {}

	endFrame() {}

	async warmup(context, options) {
		this.lastWarmupContext = context;
		this.lastWarmupOptions = options;
		const now = Date.now();
		return {
			backend: this.type,
			startedAt: now,
			finishedAt: now,
			durationMs: 0,
			total: 0,
			compiled: 0,
			skipped: 0,
			failed: 0,
			phases: [],
			errors: [],
		};
	}
}

function createEnvironmentTexture(width = 32, height = 16) {
	const data = new Uint8ClampedArray(width * height * 4);
	for (let i = 0; i < data.length; i += 4) {
		const pixel = i >> 2;
		data[i] = (pixel * 13) % 255;
		data[i + 1] = (pixel * 29) % 255;
		data[i + 2] = (pixel * 53) % 255;
		data[i + 3] = 255;
	}
	return new Texture(data, width, height, "sRGB");
}

function createRendererFixture() {
	const backend = new StubBackend();
	const camera = new Camera();
	const canvas = {
		width: 320,
		height: 180,
		getBoundingClientRect() {
			return { width: 320, height: 180 };
		},
	};
	const renderer = new Renderer(backend, canvas, camera);
	renderer.features.worldMatrix = Matrix4.identity();
	return { backend, renderer };
}

async function withDOMGlobals(callback) {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;
	const originalIdleCallback = globalThis.requestIdleCallback;
	const originalSetTimeout = globalThis.setTimeout;
	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;
		await callback();
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
		globalThis.requestIdleCallback = originalIdleCallback;
		globalThis.setTimeout = originalSetTimeout;
	}
}

async function testNextFrameWarmupAllowsPaintBeforePreparation() {
	await withDOMGlobals(async () => {
		const { backend, renderer } = createRendererFixture();
		let frameCallback = null;
		let timeoutCallback = null;
		globalThis.requestAnimationFrame = (callback) => {
			frameCallback = callback;
			return 1;
		};
		globalThis.setTimeout = (callback) => {
			timeoutCallback = callback;
			return 1;
		};

		const warmupPromise = renderer.warmup({ scheduling: "next-frame" });
		assert.equal(backend.lastWarmupContext, null);
		assert.equal(typeof frameCallback, "function");

		frameCallback();
		await Promise.resolve();
		assert.equal(backend.lastWarmupContext, null);
		assert.equal(typeof timeoutCallback, "function");

		timeoutCallback();
		await warmupPromise;
		assert.ok(backend.lastWarmupContext);
	});
}

async function testIdleWarmupDefersBackendPreparation() {
	await withDOMGlobals(async () => {
		const { backend, renderer } = createRendererFixture();
		let idleCallback = null;
		globalThis.requestIdleCallback = (callback) => {
			idleCallback = callback;
			return 1;
		};

		const warmupPromise = renderer.warmup({ scheduling: "idle" });
		assert.equal(backend.lastWarmupContext, null);
		assert.equal(typeof idleCallback, "function");

		idleCallback();
		await warmupPromise;
		assert.ok(backend.lastWarmupContext);
	});
}

async function testWarmupDoesNotBakeEnvironmentProbes() {
	await withDOMGlobals(async () => {
		const { renderer } = createRendererFixture();
		renderer.features.enableSH = true;
		const environmentTexture = createEnvironmentTexture();
		renderer.scene.environment.backgroundTexture = environmentTexture;
		renderer.scene.environment.iblTexture = environmentTexture;

		const lightProbe = renderer.scene.add(new LightProbe({ sh: SH.empty() }));
		const reflectionProbe = renderer.scene.add(
			new ReflectionProbe({ source: "environment", prefilteredMap: null })
		);
		const progress = [];
		await renderer.warmup({
			onProgress: (event) => progress.push(event),
		});

		assert.equal(lightProbe.sh[0].r, 0);
		assert.equal(lightProbe.sh[0].g, 0);
		assert.equal(lightProbe.sh[0].b, 0);
		assert.equal(reflectionProbe.prefilteredMap, null);
		assert.equal(
			progress.some((event) => event.phase.startsWith("environment-ibl")),
			false
		);
	});
}

async function testWarmupDoesNotCreateProbeWhenSceneHasNone() {
	await withDOMGlobals(async () => {
		const { renderer } = createRendererFixture();
		const environmentTexture = createEnvironmentTexture();
		renderer.scene.environment.backgroundTexture = environmentTexture;
		renderer.scene.environment.iblTexture = environmentTexture;

		await renderer.warmup();

		const probes = renderer.scene
			.getLights()
			.filter((light) => light.type === "lightProbe");
		assert.equal(probes.length, 0);
	});
}

async function testWarmupAndRenderIncrementalContextContractMatches() {
	await withDOMGlobals(async () => {
		const { backend, renderer } = createRendererFixture();

		await renderer.warmup();
		await renderer.renderScene(0);

		const warmupIncremental = backend.lastWarmupContext?.incremental;
		const renderIncremental = backend.lastBeginFrameContext?.incremental;
		assert.ok(warmupIncremental);
		assert.ok(renderIncremental);
		assert.deepEqual(
			Object.keys(warmupIncremental).sort(),
			Object.keys(renderIncremental).sort()
		);
		assert.equal(warmupIncremental.firstPass, null);
		assert.equal(renderIncremental.firstPass, null);
		assert.equal(warmupIncremental.forceFullFrame, true);
		assert.equal(renderIncremental.forceFullFrame, true);
	});
}

async function testWarmupPostProcessPlanUsesPipelineOrder() {
	await withDOMGlobals(async () => {
		const { backend, renderer } = createRendererFixture();

		renderer.postProcess.getPass("tonemap")?.disable();
		renderer.postProcess.getPass("gamma")?.disable();
		renderer.postProcess.registerPass(new BloomPass({ enabled: true }));
		renderer.postProcess.registerPass(new ColorFilterPass({ enabled: true }));
		renderer.postProcess.registerPass(
			new (class CustomWarmupPass extends PostProcessPass {
				constructor() {
					super({
						id: "custom-warmup-order",
						placement: "overlay",
						enabled: true,
						implementations: {
							stub: {},
						},
					});
				}
			})()
		);

		await renderer.warmup();

		const plan = buildWarmupPlan(backend.lastWarmupContext, {
			includePostProcess: true,
		});
		assert.deepEqual(plan.postProcessPasses, [
			"bloom",
			"color-filter",
			"custom-warmup-order",
		]);
	});
}

async function run() {
	await testNextFrameWarmupAllowsPaintBeforePreparation();
	await testIdleWarmupDefersBackendPreparation();
	await testWarmupDoesNotBakeEnvironmentProbes();
	await testWarmupDoesNotCreateProbeWhenSceneHasNone();
	await testWarmupAndRenderIncrementalContextContractMatches();
	await testWarmupPostProcessPlanUsesPipelineOrder();
	console.log("Renderer warmup light probe tests passed");
}

await run();
