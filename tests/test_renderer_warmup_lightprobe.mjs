import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Texture } from "../src/core/Texture.ts";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { ReflectionProbe } from "../src/lights/ReflectionProbe.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { SH } from "../src/maths/SH.ts";
import { Renderer } from "../src/renderers/Renderer.ts";
import { buildWarmupPlan } from "../src/pipeline/WarmupPlanner.ts";
import {
	ALL_POST_PROCESS_CAPABILITIES,
	installNoopPostProcessSupport,
} from "./helpers/postprocess.mjs";

class StubBackend {
	constructor() {
		this.type = "stub";
		this.capabilities = {
			sh: true,
			shadows: false,
			reflection: false,
			environment: true,
			ssao: false,
			ssgi: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: false,
		};
		installNoopPostProcessSupport(
			this,
			"stub",
			ALL_POST_PROCESS_CAPABILITIES
		);
		this.frameScheduling = "on-demand";
		this.lastWarmupContext = null;
		this.lastWarmupOptions = null;
		this.lastBeginFrameContext = null;
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

async function testWarmupOverwritesOnlyEnvironmentReflectionProbes() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

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
		renderer.features.enableSH = true;
		const environmentTexture = createEnvironmentTexture();
		renderer.scene.environment.backgroundTexture = environmentTexture;
		renderer.scene.environment.iblTexture = environmentTexture;

		const probeA = renderer.scene.add(new LightProbe(SH.empty()));
		const probeB = renderer.scene.add(new LightProbe(SH.empty()));
		const reflectionA = renderer.scene.add(
			new ReflectionProbe({ shape: "box", prefilteredMap: null })
		);
		const reflectionB = renderer.scene.add(
			new ReflectionProbe({ shape: "sphere", prefilteredMap: null })
		);
		const capturedPrefiltered = new Texture(
			new Float32Array([0.25, 0.5, 0.75, 1]),
			1,
			1,
			"HDR"
		);
		const manualPrefiltered = new Texture(
			new Float32Array([0.7, 0.2, 0.1, 1]),
			1,
			1,
			"HDR"
		);
		const reflectionCaptured = renderer.scene.add(
			new ReflectionProbe({
				shape: "sphere",
				source: "capturedScene",
				prefilteredMap: capturedPrefiltered,
			})
		);
		const reflectionManual = renderer.scene.add(
			new ReflectionProbe({
				shape: "sphere",
				source: "manual",
				prefilteredMap: manualPrefiltered,
			})
		);

		const progress = [];
		await renderer.warmup({
			environmentIBLBake: { acceleration: "cpu" },
			onProgress: (event) => progress.push(event),
		});

		const probes = renderer.scene
			.getLights()
			.filter((light) => light.type === "lightProbe");
		assert.equal(probes.length, 2);
		for (const probe of probes) {
			assert.equal(probe.sh.length, 16);
		}
		assert.ok(reflectionA.prefilteredMap);
		assert.ok(reflectionB.prefilteredMap);
		assert.equal(reflectionCaptured.prefilteredMap, capturedPrefiltered);
		assert.equal(reflectionManual.prefilteredMap, manualPrefiltered);
		assert.equal("intensity" in probeA, false);
		assert.equal("intensity" in probeB, false);
		assert.ok(
			progress.some((event) =>
				event.phase.startsWith("environment-ibl-bake:")
			)
		);
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

async function testWarmupCreatesProbeWhenSceneHasNone() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

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
		const environmentTexture = createEnvironmentTexture();
		renderer.scene.environment.backgroundTexture = environmentTexture;
		renderer.scene.environment.iblTexture = environmentTexture;

		await renderer.warmup({
			environmentIBLBake: { acceleration: "cpu" },
		});

		const probes = renderer.scene
			.getLights()
			.filter((light) => light.type === "lightProbe");
		assert.equal(probes.length, 1);
		const reflectionProbes = renderer.scene
			.getLights()
			.filter((light) => light.type === "reflectionProbe");
		assert.equal(reflectionProbes.length, 0);
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

async function testWarmupSkipsLightProbeBakeWhenDisabled() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

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
		const environmentTexture = createEnvironmentTexture();
		renderer.scene.environment.backgroundTexture = environmentTexture;
		renderer.scene.environment.iblTexture = environmentTexture;

		await renderer.warmup({
			includeEnvironmentIBLBake: false,
			environmentIBLBake: { acceleration: "cpu" },
		});

		const probes = renderer.scene
			.getLights()
			.filter((light) => light.type === "lightProbe");
		assert.equal(probes.length, 0);
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

async function testWarmupAndRenderIncrementalContextContractMatches() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

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

		await renderer.warmup({ includeEnvironmentIBLBake: false });
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
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

async function testWarmupWithoutEnvironmentIBLBakeKeepsReflectionProbeUnset() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

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
		const environmentTexture = createEnvironmentTexture();
		renderer.scene.environment.backgroundTexture = environmentTexture;
		renderer.scene.environment.iblTexture = environmentTexture;
		const reflectionProbe = renderer.scene.add(
			new ReflectionProbe({ source: "environment", prefilteredMap: null })
		);

		await renderer.warmup({
			includeEnvironmentIBLBake: false,
		});
		await renderer.renderScene(0);

		assert.equal(reflectionProbe.prefilteredMap, null);
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

async function testWarmupPostProcessPlanUsesPipelineOrder() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

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

		renderer.postProcess.disable("tonemap");
		renderer.postProcess.disable("interaction-outline");
		renderer.postProcess.disable("gamma");
		renderer.postProcess.enable("bloom");
		renderer.postProcess.enable("color-filter");
		renderer.postProcess
			.registerPass({
				id: "custom-warmup-order",
				placement: "overlay",
				isEnabled(postProcess) {
					return postProcess.enabled["custom-warmup-order"];
				},
				implementations: {
					stub: {},
				},
			})
			.enable("custom-warmup-order");

		await renderer.warmup({ includeEnvironmentIBLBake: false });

		const plan = buildWarmupPlan(backend.lastWarmupContext, {
			includePostProcess: true,
		});
		assert.deepEqual(plan.postProcessPasses, [
			"bloom",
			"color-filter",
			"custom-warmup-order",
			"gamma",
		]);
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

async function run() {
	await testWarmupOverwritesOnlyEnvironmentReflectionProbes();
	await testWarmupCreatesProbeWhenSceneHasNone();
	await testWarmupSkipsLightProbeBakeWhenDisabled();
	await testWarmupAndRenderIncrementalContextContractMatches();
	await testWarmupWithoutEnvironmentIBLBakeKeepsReflectionProbeUnset();
	await testWarmupPostProcessPlanUsesPipelineOrder();
	console.log("Renderer warmup light probe tests passed");
}

await run();
