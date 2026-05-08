import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Scene } from "../src/core/Scene.ts";
import { Renderer } from "../src/renderers/Renderer.ts";

class StubBackend {
	constructor() {
		this.type = "stub";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
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
		this.frameScheduling = "on-demand";
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

function expectReadonlyAssignment(target, key, value) {
	assert.throws(() => {
		target[key] = value;
	}, TypeError);
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

		const replacementCamera = new Camera();
		renderer.scene.add(replacementCamera);
		renderer.setCamera(replacementCamera);
		assert.equal(renderer.camera, replacementCamera);

		const replacementScene = new Scene();
		replacementScene.add(replacementCamera);
		renderer.setScene(replacementScene);
		assert.equal(renderer.scene, replacementScene);

		expectReadonlyAssignment(renderer, "canvas", canvas);
		expectReadonlyAssignment(renderer, "scene", new Scene());
		expectReadonlyAssignment(renderer, "camera", new Camera());
		expectReadonlyAssignment(renderer, "lastTime", 123);
		expectReadonlyAssignment(renderer, "shadowMaps", new Map());
		expectReadonlyAssignment(renderer, "shCoeffs", []);
		expectReadonlyAssignment(renderer, "shAmbientCoeffs", []);

		console.log("Renderer mutable field guard tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
