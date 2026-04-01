import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Texture } from "../src/core/Texture.ts";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { SH } from "../src/maths/SH.ts";
import { Renderer } from "../src/renderers/Renderer.ts";

class StubBackend {
	constructor() {
		this.type = "stub";
		this.capabilities = {
			sh: true,
			shadows: false,
			reflection: false,
			skybox: true,
			ssao: false,
			ssgi: false,
			taa: false,
			ssr: false,
			volumetric: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: false,
		};
		this.frameScheduling = "on-demand";
		this.lastWarmupContext = null;
		this.lastWarmupOptions = null;
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

function createSkyboxTexture(width = 32, height = 16) {
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

async function testWarmupOverwritesAllLightProbesFromSkybox() {
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
		renderer.scene.skybox = createSkyboxTexture();

		const probeA = renderer.scene.add(new LightProbe(SH.empty(), 2));
		const probeB = renderer.scene.add(new LightProbe(SH.empty(), 0.5));
		probeA.prefilteredMap = null;
		probeB.prefilteredMap = null;

		const progress = [];
		await renderer.warmup({
			lightProbeBake: { acceleration: "cpu" },
			onProgress: (event) => progress.push(event),
		});

		const probes = renderer.scene
			.getLights()
			.filter((light) => light.type === "lightProbe");
		assert.equal(probes.length, 2);
		for (const probe of probes) {
			assert.ok(probe.prefilteredMap);
			assert.equal(probe.sh.length, 16);
		}
		assert.equal(probeA.intensity, 2);
		assert.equal(probeB.intensity, 0.5);
		assert.ok(
			progress.some((event) =>
				event.phase.startsWith("light-probe-bake:")
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
		renderer.scene.skybox = createSkyboxTexture();

		await renderer.warmup({
			lightProbeBake: { acceleration: "cpu" },
		});

		const probes = renderer.scene
			.getLights()
			.filter((light) => light.type === "lightProbe");
		assert.equal(probes.length, 1);
		assert.ok(probes[0].prefilteredMap);
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
		renderer.scene.skybox = createSkyboxTexture();

		await renderer.warmup({
			includeLightProbeBake: false,
			lightProbeBake: { acceleration: "cpu" },
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

async function run() {
	await testWarmupOverwritesAllLightProbesFromSkybox();
	await testWarmupCreatesProbeWhenSceneHasNone();
	await testWarmupSkipsLightProbeBakeWhenDisabled();
	console.log("Renderer warmup light probe tests passed");
}

await run();
