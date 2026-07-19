import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Logger } from "../../../src/foundation/Logger.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import { PostProcessPass } from "../../../src/postprocess/index.ts";
import {
	installNoopPostProcessAdapter,
} from "../../helpers/postprocess.mjs";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

class RegistryBackend extends TestRenderBackend {
	constructor() {
		super();
		this.type = "webgpu";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			clusteredLighting: false,
			oit: false,
			postProcess: false,
		};
		this.contexts = [];
		this.executedPasses = [];
		this.executionEvents = [];
		this.gBufferRequests = 0;
		this.postProcessAbortCalls = 0;
		this.backendAbortCalls = 0;
		this.postProcessSupport = installNoopPostProcessAdapter(
			this,
			"webgpu"
		);
		const createGBufferBridge =
			this.postProcessSupport.executor.createGBufferBridge.bind(
				this.postProcessSupport.executor
			);
		this.postProcessSupport.executor.createGBufferBridge = (context) => {
			this.gBufferRequests++;
			return createGBufferBridge(context);
		};
		const executePostProcessPass =
			this.postProcessSupport.executor.executePass.bind(
				this.postProcessSupport.executor
			);
		this.postProcessSupport.executor.executePass = (passId, request) => {
			this.executionEvents.push(["postprocess", passId]);
			return executePostProcessPass(passId, request);
		};
		this.postProcessSupport.executor.endFrame = () => {
			this.executionEvents.push(["postprocess-frame", "end"]);
		};
		this.postProcessSupport.executor.abortFrame = () => {
			this.postProcessAbortCalls++;
			this.executionEvents.push(["postprocess-frame", "abort"]);
		};
		this.frameScheduling = "always";
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
		this.contexts.push(context);
	}

	executePass(pass) {
		this.executedPasses.push(pass.stage);
		this.executionEvents.push(["backend", pass.stage]);
	}

	endFrame() {
		this.executionEvents.push(["backend", "end"]);
	}

	abortFrame() {
		this.backendAbortCalls++;
		this.executionEvents.push(["backend", "abort"]);
	}
}

class NoAdapterBackend extends TestRenderBackend {
	constructor() {
		super();
		this.type = "missing";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			clusteredLighting: false,
			oit: false,
			postProcess: false,
		};
		this.contexts = [];
		this.executedPasses = [];
		this.frameScheduling = "always";
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

		await renderer.renderFrame(0);

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
			backend.postProcessSupport.executor.executedPasses.includes("custom-edge")
		);
		assert.equal(backend.executedPasses.includes("postprocess"), true);
		assert.equal(
			backend.executionEvents.some(
				(event) => event[0] === "backend" && event[1] === "postprocess"
			),
			true
		);
		assert.ok(
			backend.executionEvents.findIndex(
				(event) => event[0] === "backend" && event[1] === "postprocess"
			) <
				backend.executionEvents.findIndex(
					(event) => event[0] === "postprocess" && event[1] === "custom-edge"
				)
		);
		assert.ok(
			backend.executionEvents.findIndex(
				(event) => event[0] === "postprocess" && event[1] === "custom-edge"
			) <
				backend.executionEvents.findIndex(
					(event) => event[0] === "backend" && event[1] === "end"
				)
		);
		assert.equal(
			backend.postProcessSupport.executor.executedPasses.includes("gamma"),
			false
		);
		assert.equal(
			backend.postProcessSupport.executor.executedPasses.includes("tonemap"),
			false
		);

		renderer.postProcess.unregisterPass("custom-edge");
		assert.throws(
			() => renderer.postProcess.registerPass({ id: "custom-edge" }),
			/requires a PostProcessPass/
		);

		const noopBackend = new RegistryBackend();
		const noopRenderer = new Renderer(noopBackend, canvas, new Camera());
		noopRenderer.features.enableShadows = false;
		noopRenderer.features.enableReflection = false;
		noopRenderer.features.enableEnvironment = false;
		noopRenderer.postProcess.getPass("tonemap")?.disable();
		noopRenderer.postProcess.getPass("gamma")?.disable();
		await noopRenderer.renderFrame(0);
		assert.deepEqual(noopBackend.postProcessSupport.executor.executedPasses, []);
		assert.equal(noopBackend.gBufferRequests, 0);
		assert.equal(noopBackend.executedPasses.includes("postprocess"), false);

		const unsupportedWarnings = [];
		Logger.configure({
			level: "warn",
			resetOnceKeys: true,
			sink: {
				warn: (...args) =>
					unsupportedWarnings.push(
						args.map((arg) => String(arg)).join(" ")
					),
			},
		});
		const missingAdapterBackend = new NoAdapterBackend();
		const missingAdapterRenderer = new Renderer(
			missingAdapterBackend,
			canvas,
			new Camera()
		);
		missingAdapterRenderer.features.enableShadows = false;
		missingAdapterRenderer.features.enableReflection = false;
		missingAdapterRenderer.features.enableEnvironment = false;
		missingAdapterRenderer.postProcess.getPass("tonemap")?.disable();
		missingAdapterRenderer.postProcess.getPass("gamma")?.disable();
		missingAdapterRenderer.postProcess.registerPass(
			new (class MissingAdapterPass extends PostProcessPass {
				constructor() {
					super({
						id: "missing-adapter-pass",
						enabled: true,
						implementations: { missing: {} },
					});
				}
			})()
		);
		await missingAdapterRenderer.renderFrame(0);
		await missingAdapterRenderer.renderFrame(16);
		Logger.reset();
		assert.equal(
			unsupportedWarnings.filter((warning) =>
				warning.includes("postprocess")
			).length,
			0
		);
		assert.equal(
			missingAdapterBackend.executedPasses.includes("postprocess"),
			false
		);

		const historySnapshots = [];
		const historyBackend = new RegistryBackend();
		const historyRenderer = new Renderer(historyBackend, canvas, new Camera());
		historyRenderer.setIncrementalRendering({ enabled: false });
		historyRenderer.features.enableShadows = false;
		historyRenderer.features.enableReflection = false;
		historyRenderer.features.enableEnvironment = false;
		historyRenderer.postProcess.getPass("tonemap")?.disable();
		historyRenderer.postProcess.getPass("gamma")?.disable();
		historyRenderer.postProcess.registerPass(
			new (class HistoryProbePass extends PostProcessPass {
				constructor() {
					super({
						id: "history-probe",
						enabled: true,
						implementations: {
							webgpu: {
								execute: (request) => {
									const slot = request.histories.probe;
									historySnapshots.push({
										valid: slot.valid,
										read: slot.read.id,
										write: slot.write.id,
									});
									return { updatedHistoryIds: ["probe"] };
								},
							},
						},
					});
				}

				getHistoryDescriptors() {
					return [{ id: "probe" }];
				}
			})()
		);
		await historyRenderer.renderFrame(0);
		await historyRenderer.renderFrame(16);
		assert.equal(historySnapshots.length, 2);
		assert.deepEqual(historySnapshots[0], {
			valid: false,
			read: "probe:read",
			write: "probe:write",
		});
		assert.equal(historySnapshots[1].read, "probe:write");
		assert.equal(historySnapshots[1].write, "probe:read");

		const throwingBackend = new RegistryBackend();
		const throwingRenderer = new Renderer(throwingBackend, canvas, new Camera());
		throwingRenderer.features.enableShadows = false;
		throwingRenderer.features.enableReflection = false;
		throwingRenderer.features.enableEnvironment = false;
		throwingRenderer.postProcess.getPass("tonemap")?.disable();
		throwingRenderer.postProcess.getPass("gamma")?.disable();
		throwingRenderer.postProcess.registerPass(
			new (class ThrowingPass extends PostProcessPass {
				constructor() {
					super({
						id: "throwing-pass",
						enabled: true,
						implementations: {
							webgpu: {
								execute: () => {
									throw new Error("postprocess failed");
								},
							},
						},
					});
				}
			})()
		);
		await assert.rejects(
			() => throwingRenderer.renderFrame(0),
			/postprocess failed/
		);
		assert.equal(throwingBackend.postProcessAbortCalls, 1);
		assert.equal(throwingBackend.backendAbortCalls, 1);

		console.log("Renderer postprocess registry tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
