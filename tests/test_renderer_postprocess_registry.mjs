import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Renderer } from "../src/renderers/Renderer.ts";
import { ALL_POST_PROCESS_CAPABILITIES } from "./helpers/postprocess.mjs";

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
		this.registeredPasses = [];
		this.unregisteredPasses = [];
		this.contexts = [];
		this.executedPasses = [];
		this.postProcess = {
			capabilities: ALL_POST_PROCESS_CAPABILITIES,
			registerPass: (pass) => {
				this.registeredPasses.push(pass);
			},
			unregisterPass: (id) => {
				this.unregisteredPasses.push(id);
			},
		};
		this.frameScheduling = "always";
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

	beginFrame(context) {
		this.contexts.push(context);
	}

	executePass(pass) {
		this.executedPasses.push(pass.stage);
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

		const pass = {
			id: "custom-edge",
			dependsOn: [],
			isEnabled(postProcess) {
				return postProcess.enabled["custom-edge"];
			},
			execute() {},
		};

		renderer.postProcess.registerPass(pass).enable("custom-edge", {
			strength: 0.5,
		});
		renderer.postProcess.disable("tonemap");
		renderer.postProcess.disable("gamma");

		assert.strictEqual(backend.registeredPasses[0], pass);

		await renderer.renderScene(0);

		const postProcess = backend.contexts.at(-1).postProcess;
		assert.equal(postProcess.enabled["custom-edge"], true);
		assert.equal(postProcess.enabled.tonemap, false);
		assert.equal(postProcess.enabled.gamma, false);
		assert.deepEqual(postProcess.options["custom-edge"], { strength: 0.5 });
		assert.ok(backend.executedPasses.includes("gamma"));
		assert.equal(backend.executedPasses.includes("tonemap"), false);

		renderer.postProcess.unregisterPass("custom-edge");
		assert.deepEqual(backend.unregisteredPasses, ["custom-edge"]);
		assert.throws(
			() => renderer.postProcess.enable("custom-edge"),
			/Unknown post-process pass/
		);

		console.log("Renderer postprocess registry tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
