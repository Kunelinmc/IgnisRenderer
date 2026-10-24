import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Logger } from "../../../src/foundation/Logger.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import {
	createPostProcessRegistryFromRequest,
	installNoopPostProcessAdapter,
} from "../../helpers/postprocess.mjs";
import { getDefaultIncrementalRegistry } from "../../../src/pipeline/incremental.ts";
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
		installNoopPostProcessAdapter(
			this,
			"webgpu"
		);
		this.frameScheduling = "always";
		this.contexts = [];
		this.executedPasses = [];
		this.skippedPasses = [];
		this.beginFrameCalls = 0;
		this.endFrameCalls = 0;
		this.abortFrameCalls = 0;
		this.abortErrors = [];
		this.throwInBeginFrame = null;
		this.throwOnPass = null;
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
		this.beginFrameCalls++;
		this.contexts.push(context);
		if (this.throwInBeginFrame) {
			throw this.throwInBeginFrame;
		}
	}

	executePass(pass) {
		this.executedPasses.push(pass.stage);
		if (this.throwOnPass && pass.stage === this.throwOnPass.stage) {
			throw this.throwOnPass.error;
		}
	}

	skipPass(pass) {
		this.skippedPasses.push(pass.stage);
	}

	endFrame() {
		this.endFrameCalls++;
	}

	abortFrame(error) {
		this.abortFrameCalls++;
		this.abortErrors.push(error);
	}
}

function createRenderer(backend) {
	const canvas = {
		width: 320,
		height: 180,
		getBoundingClientRect() {
			return { width: 320, height: 180 };
		},
	};
	const renderer = new Renderer(canvas, backend, new Camera());
	renderer.features.enableReflection = false;
	renderer.features.enableEnvironment = false;
	return renderer;
}

async function testRendererAbortsBackendFrameOnPassFailure() {
	const backend = new RegistryBackend();
	const renderer = createRenderer(backend);
	const error = new Error("main pass failed");
	backend.throwOnPass = {
		stage: "main-opaque",
		error,
	};

	let caught = null;
	try {
		await renderer.renderFrame(16);
	} catch (caughtError) {
		caught = caughtError;
	}
	assert.strictEqual(caught, error);
	assert.equal(backend.beginFrameCalls, 1);
	assert.equal(backend.endFrameCalls, 0);
	assert.equal(backend.abortFrameCalls, 1);
	assert.strictEqual(backend.abortErrors[0], error);
}

async function testRendererAbortsPartialBeginFrameFailure() {
	const backend = new RegistryBackend();
	const renderer = createRenderer(backend);
	const error = new Error("begin frame failed");
	backend.throwInBeginFrame = error;

	let caught = null;
	try {
		await renderer.renderFrame(16);
	} catch (caughtError) {
		caught = caughtError;
	}
	assert.strictEqual(caught, error);
	assert.equal(backend.beginFrameCalls, 1);
	assert.equal(backend.endFrameCalls, 0);
	assert.equal(backend.abortFrameCalls, 1);
	assert.strictEqual(backend.abortErrors[0], error);
}

async function testRendererSuccessfulFrameEndsWithoutScheduling() {
	const backend = new RegistryBackend();
	const renderer = createRenderer(backend);
	let frameEndEvents = 0;
	let scheduledFrames = 0;
	renderer.on("frameend", () => {
		frameEndEvents++;
	});
	globalThis.requestAnimationFrame = () => {
		scheduledFrames++;
		return scheduledFrames;
	};

	const result = await renderer.renderFrame(16);
	assert.equal(result.rendered, true);
	assert.equal(result.incremental.plannedCoverage.mode, "full");
	assert.equal(result.incremental.finalOutputCoverage.mode, "full");

	assert.equal(backend.beginFrameCalls, 1);
	assert.equal(backend.endFrameCalls, 1);
	assert.equal(backend.abortFrameCalls, 0);
	assert.equal(frameEndEvents, 1);
	assert.equal(scheduledFrames, 0);
	const context = backend.contexts.at(-1);
	assert.ok(context.framePlan);
	assert.ok(
		context.framePlan.backendPasses.some(
			(pass) => pass.stage === "main-opaque"
		)
	);
	assert.equal(
		context.framePlan.stageOrder.find(
			(stage) => stage.id === "main-opaque"
		)?.kind,
		"backend-pass"
	);
	assert.deepEqual(
		context.framePlan.backendPasses.find(
			(pass) => pass.stage === "main-opaque"
		)?.dependsOn,
		["reflection", "shadow"]
	);
	assert.equal(
		context.framePlan.backendPasses.some(
			(pass) => pass.stage === "postprocess"
		),
		true
	);
	assert.equal(
		context.framePlan.stageOrder.find(
			(stage) => stage.id === "postprocess"
		)?.kind,
		"backend-pass"
	);
	assert.deepEqual(
		context.framePlan.backendPasses.find(
			(pass) => pass.stage === "postprocess"
		)?.dependsOn,
		["particles"]
	);
}

async function testBackendDirtyRectsProducePartialCoverage() {
	const backend = new RegistryBackend();
	const renderer = createRenderer(backend);
	await renderer.renderFrame(0);

	backend.attachContext.events.emit({
		type: "render-invalidated",
		reason: "texture",
		dirtyRects: [{ x: 64, y: 32, width: 8, height: 8 }],
	});
	const result = await renderer.renderFrame(16);

	assert.equal(result.rendered, true);
	assert.equal(result.incremental.plan.forceFullFrame, false);
	assert.equal(result.incremental.plannedCoverage.mode, "partial");
	assert.ok(result.incremental.plan.dirtyAreaRatio > 0);
	assert.ok(result.incremental.plan.dirtyAreaRatio < 1);
}

async function testRendererReturnsUnchangedCoverageForCleanFrame() {
	const backend = new RegistryBackend();
	backend.frameScheduling = "on-demand";
	const renderer = createRenderer(backend);
	await renderer.renderFrame(0);
	const result = await renderer.renderFrame(16);
	assert.equal(result.rendered, false);
	assert.equal(result.reason, "clean");
	assert.equal(result.incremental.plannedCoverage.mode, "unchanged");
	assert.equal(result.incremental.finalOutputCoverage.mode, "unchanged");
}

async function testFailedFramesRetryWithFullCoverage() {
	const attempts = ["begin", "main-opaque", "end"].flatMap((stage) =>
		[false, true].map((invalidateDuringFailure) => [stage, invalidateDuringFailure]),
	);
	for (const [failureStage, invalidateDuringFailure] of attempts) {
		const backend = new RegistryBackend();
		backend.frameScheduling = "on-demand";
		const renderer = createRenderer(backend);
		try {
			await renderer.renderFrame(0);
			const completedStats = renderer.getLastIncrementalFrameStats();
			backend.attachContext.events.emit({
				type: "render-invalidated",
				reason: "texture",
				dirtyRects: [{ x: 64, y: 32, width: 8, height: 8 }],
			});
			const error = new Error(`failed ${failureStage}`);
			const operation = failureStage === "begin" ? "beginFrame" :
				failureStage === "end" ? "endFrame" : "executePass";
			const original = backend[operation];
			backend[operation] = function (...args) {
				if (operation !== "executePass" || args[0].stage === failureStage) {
					if (invalidateDuringFailure) {
						this.attachContext.events.emit({
							type: "render-invalidated",
							reason: "material",
							dirtyRects: [{ x: 192, y: 96, width: 8, height: 8 }],
						});
					}
					throw error;
				}
				return original.apply(this, args);
			};
			await assert.rejects(() => renderer.renderFrame(16), (caught) => caught === error);
			assert.deepEqual(renderer.getLastIncrementalFrameStats(), completedStats);
			backend[operation] = original;

			const retry = await renderer.renderFrame(32);
			assert.equal(retry.rendered, true, `${failureStage}: failed work must be retried`);
			assert.equal(retry.incremental.plan.forceFullFrame, true);
			assert.equal(retry.incremental.plan.dirtyAreaRatio, 1);
			assert.ok(retry.incremental.plan.reasonMask & (1 << 5), "retain consumed texture work");
			if (invalidateDuringFailure) {
				assert.ok(retry.incremental.plan.reasonMask & (1 << 4),
					"retain requests during failure");
			}
			assert.equal((await renderer.renderFrame(48)).rendered, false);

			backend.attachContext.events.emit({
				type: "render-invalidated", reason: "texture",
				dirtyRects: [{ x: 64, y: 32, width: 8, height: 8 }],
			});
			const partial = await renderer.renderFrame(64);
			assert.equal(partial.incremental.plan.forceFullFrame, false);
			assert.ok(partial.incremental.plan.dirtyAreaRatio < 1);
		} finally {
			await renderer.destroy();
		}
	}
}

async function testRendererRegistrationsRemainIndependent() {
	const backends = [new RegistryBackend(), new RegistryBackend()];
	const renderers = backends.map(createRenderer);
	const passId = "isolated-region-pass";
	try {
		for (const [index, renderer] of renderers.entries()) {
			const passes = createPostProcessRegistryFromRequest({
				[passId]: {
					enabled: true,
					incremental: { grade: "none", inflationRadius: index === 0 ? 80 : 2 },
				},
			}, "webgpu");
			renderer.postProcess.registerPass(passes.getPass(passId));
			await renderer.renderFrame(0);
		}
		const radius = (renderer) => renderer.pipeline.incremental
			.computePostProcessInflationRadius(renderer.postProcess.createSnapshot("webgpu"));
		assert.equal(radius(renderers[0]), 80, "another renderer must not replace pass metadata");
		assert.equal(radius(renderers[1]), 2);
		for (const backend of backends) {
			backend.attachContext.events.emit({
				type: "render-invalidated", reason: "texture",
				dirtyRects: [{ x: 128, y: 64, width: 1, height: 1 }],
			});
		}
		const expanded = await renderers[0].renderFrame(4);
		const local = await renderers[1].renderFrame(4);
		assert.equal(expanded.incremental.plan.forceFullFrame, true,
			"preparation must consume this renderer's larger inflation radius");
		assert.equal(local.incremental.plan.forceFullFrame, false);
		assert.ok(local.incremental.plan.dirtyAreaRatio < 1);
		renderers[1].postProcess.unregisterPass(passId);
		assert.equal(radius(renderers[0]), 80, "another renderer must not unregister this pass");
		await renderers[1].renderFrame(8);

		for (const [index, renderer] of renderers.entries()) {
			const stage = `isolated-stage-${index}`;
			renderer.pipeline.registerPipelineStage({
				id: stage, kind: "backend-pass", dependsOn: ["main-opaque"],
				shouldRun: () => true, incremental: { order: 4.5 },
			});
			renderer.pipeline.registerDirtyReason({ id: "isolated-request", firstPass: stage });
		}
		for (const [index, renderer] of renderers.entries()) {
			renderer.requestRender("isolated-request");
			const result = await renderer.renderFrame(16);
			assert.equal(result.incremental.plan.firstPass, `isolated-stage-${index}`);
			assert.equal(result.incremental.plan.forceFullFrame, false);
		}
	} finally {
		for (const renderer of renderers) await renderer.destroy();
	}
}

async function testSceneReasonIdentitySurvivesLocalMaskAllocation() {
	const renderer = createRenderer(new RegistryBackend());
	const defaults = getDefaultIncrementalRegistry();
	try {
		await renderer.renderFrame(0);
		renderer.pipeline.registerDirtyReason({
			id: "local-mask-owner", firstPass: "main-transparent",
		});
		defaults.registerDirtyReason({
			id: "scene-global-reason", forceFullFrame: true,
		});
		renderer.scene.invalidate("scene-global-reason");
		const result = await renderer.renderFrame(16);
		assert.equal(result.incremental.plan.forceFullFrame, true,
			"a scene reason must not be mistaken for a local reason with the same bit");
	} finally {
		defaults.unregisterDirtyReason("scene-global-reason");
		await renderer.destroy();
	}
}

async function testNonLocalSceneEffectsReplayProducersAndReceivers() {
	for (const effect of ["shadow", "reflection"]) {
		for (const explicitReason of [false, true]) {
			const backend = new RegistryBackend();
			backend.capabilities.shadows = true;
			backend.capabilities.reflection = true;
			const renderer = createRenderer(backend);
			renderer.camera.position.set(0, 0, 4);
			renderer.features.enableReflection = effect === "reflection";
			const mesh = (material) => renderer.scene.add(new MeshInstance({
				mesh: MeshAsset.fromFaces([{
					material,
					vertices: [
						{ x: -0.1, y: -0.1, z: 0 },
						{ x: 0.1, y: -0.1, z: 0 },
						{ x: 0, y: 0.1, z: 0 },
					],
				}]),
			}));
			const moving = mesh(new Material());
			moving.position.x = -0.5;
			const receiver = mesh(new Material(effect === "reflection" ? {
				reflectivity: 0.5,
				mirrorPlane: { normal: { x: 0, y: 0, z: 1 }, constant: 0 },
			} : {}));
			receiver.position.x = 0.5;
			if (effect === "shadow") {
				const light = renderer.scene.add(new DirectionalLight());
				renderer.scene.shadows.bind(light, renderer.scene.shadows.createSingle());
			}
			try {
				await renderer.renderFrame(0);
				backend.executedPasses.length = 0;
				moving.position.x += 0.05;
				if (explicitReason) renderer.requestRender("transform");
				const result = await renderer.renderFrame(16);
				if (!explicitReason) assert.equal(result.incremental.plan.reasonMask, 0);
				assert.equal(result.incremental.plan.forceFullFrame, true,
					`${effect}: receiver influence is not limited to the moving packet's bounds`);
				assert.equal(result.incremental.plan.dirtyAreaRatio, 1);
				assert.ok(backend.executedPasses.includes(effect));
				assert.ok(backend.executedPasses.indexOf(effect) <
					backend.executedPasses.indexOf("main-opaque"));

				if (effect === "shadow") {
					moving.position.x = 100;
					await renderer.renderFrame(32);
					moving.position.x++;
					const offscreen = await renderer.renderFrame(48);
					assert.equal(offscreen.incremental.plan.forceFullFrame, true,
						"offscreen caster changes must still redraw visible receivers");
				}
			} finally {
				await renderer.destroy();
			}
		}
	}
}

async function testRendererWarnsForMissingRendererStageExecutor() {
	const backend = new RegistryBackend();
	const renderer = createRenderer(backend);
	const warnings = [];

	Logger.configure({
		level: "warn",
		resetOnceKeys: true,
		sink: {
			warn: (...args) => {
				warnings.push(args.map((arg) => String(arg)).join(" "));
			},
		},
	});
	try {
		renderer.pipeline.registerPipelineStage({
			id: "custom-renderer-stage",
			kind: "renderer",
			dependsOn: ["postprocess"],
		});
		await renderer.renderFrame(16);
	} finally {
		Logger.reset();
	}

	assert.ok(
		warnings.some((message) =>
			message.includes("renderer-stage-executor-missing-custom-renderer-stage")
		)
	);
}

async function run() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

		await testRendererAbortsBackendFrameOnPassFailure();
		await testRendererAbortsPartialBeginFrameFailure();
		await testRendererSuccessfulFrameEndsWithoutScheduling();
		await testBackendDirtyRectsProducePartialCoverage();
		await testRendererReturnsUnchangedCoverageForCleanFrame();
		await testFailedFramesRetryWithFullCoverage();
		await testRendererRegistrationsRemainIndependent();
		await testSceneReasonIdentitySurvivesLocalMaskAllocation();
		await testNonLocalSceneEffectsReplayProducersAndReceivers();
		await testRendererWarnsForMissingRendererStageExecutor();

		const backend = new RegistryBackend();
		const renderer = createRenderer(backend);

		await renderer.renderFrame(0);
		backend.executedPasses.length = 0;
		backend.skippedPasses.length = 0;

		const customPassId = "custom-registry-pass";
		const customReasonId = "custom-registry-dirty";
		renderer.pipeline.registerPipelineStage({
			id: customPassId,
			kind: "backend-pass",
			dependsOn: ["main-opaque"],
			shouldRun: () => true,
			incremental: { order: 4.5 },
		});
		renderer.pipeline.registerDirtyReason({
			id: customReasonId,
			firstPass: customPassId,
		});

		try {
			renderer.requestRender(customReasonId);
			const result = await renderer.renderFrame(16);

			const stats = renderer.getLastIncrementalFrameStats();
			assert.equal(stats.firstPass, customPassId);
			assert.equal(stats.forceFullFrame, false);
			assert.equal(result.incremental.plannedCoverage.mode, "partial");
			assert.equal(result.incremental.finalOutputCoverage.mode, "full");
			assert.ok(backend.skippedPasses.includes("main-opaque"));
			assert.equal(backend.skippedPasses.includes("postprocess"), true);
			assert.ok(backend.executedPasses.includes(customPassId));
			const framePlan = backend.contexts.at(-1).framePlan;
			assert.ok(framePlan);
			assert.ok(
				framePlan.backendPasses.some(
					(pass) => pass.stage === customPassId && pass.enabled
				)
			);
			assert.equal(
				framePlan.backendPasses.find(
					(pass) => pass.stage === "main-opaque"
				)?.enabled,
				false
			);
		} finally {
			renderer.pipeline.unregisterDirtyReason(customReasonId);
			renderer.pipeline.unregisterPipelineStage(customPassId);
		}

		console.log("Renderer pipeline registry tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
