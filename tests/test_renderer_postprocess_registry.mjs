import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Renderer } from "../src/renderers/Renderer.ts";
import { PostProcessPass } from "../src/postprocess/index.ts";
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
		this.contexts = [];
		this.executedPasses = [];
		installNoopPostProcessSupport(
			this,
			"webgpu"
		);
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

		const pass = new (class CustomEdgePass extends PostProcessPass {
			constructor() {
				super({
					id: "custom-edge",
					incremental: {
						firstPass: "tonemap",
						grade: "standard",
						inflationRadius: 18,
					},
					placement: "overlay",
					enabled: true,
					options: { strength: 0.5 },
					implementations: {
						webgpu: {},
					},
				});
			}
		})();

		renderer.postProcess.registerPass(pass);
		renderer.postProcess.getPass("tonemap")?.disable();
		renderer.postProcess.getPass("gamma")?.disable();

		await renderer.renderScene(0);

		const postProcess = backend.contexts.at(-1).postProcess;
		assert.equal(postProcess.isEnabled("custom-edge"), true);
		assert.equal(postProcess.isEnabled("tonemap"), false);
		assert.equal(postProcess.isEnabled("gamma"), false);
		assert.deepEqual(postProcess.getOptions("custom-edge"), { strength: 0.5 });
		assert.equal(
			renderer.pipeline.incremental.resolveFirstEnabledPostProcessStage(
				postProcess
			),
			"tonemap"
		);
		assert.equal(
			renderer.pipeline.incremental.computePostProcessInflationRadius(
				postProcess
			),
			18
		);
		assert.ok(
			backend.postProcessExecutor.executedPasses.includes("custom-edge")
		);
		assert.equal(
			backend.postProcessExecutor.executedPasses.includes("gamma"),
			false
		);
		assert.equal(
			backend.postProcessExecutor.executedPasses.includes("tonemap"),
			false
		);

		renderer.postProcess.unregisterPass("custom-edge");
		assert.throws(
			() => renderer.postProcess.registerPass({ id: "custom-edge" }),
			/requires a PostProcessPass/
		);

		console.log("Renderer postprocess registry tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
