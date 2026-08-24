import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { Logger } from "../../../src/foundation/Logger.ts";
import { ReflectionProbe } from "../../../src/lights/ReflectionProbe.ts";
import { IBLPrefilter } from "../../../src/lights/ibl/IBLPrefilter.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { ProbeCaptureRuntime } from "../../../src/lights/runtime/ProbeCaptureRuntime.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import {
	installNoopPostProcessAdapter,
} from "../../helpers/postprocess.mjs";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

function createPrefilteredMap(seed = 1) {
	return new Texture({
		data: new Float32Array([seed, seed * 0.5, seed * 0.25, 1]),
		width: 1,
		height: 1,
		colorSpace: "HDR",
	});
}

async function withPrefilterStub(handler, run) {
	const original = IBLPrefilter.prototype.prefilter;
	IBLPrefilter.prototype.prefilter = function prefilterStub(envMap, options) {
		return handler.call(this, envMap, options);
	};
	try {
		return await run();
	} finally {
		IBLPrefilter.prototype.prefilter = original;
	}
}

function createDeferred() {
	let resolve;
	const promise = new Promise((res) => {
		resolve = res;
	});
	return { promise, resolve };
}

async function flushAsyncTasks() {
	for (let i = 0; i < 3; i++) {
		await Promise.resolve();
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
}

async function driveRuntimeUntil(
	runtime,
	createContext,
	predicate,
	maxSteps = 12
) {
	for (let step = 0; step < maxSteps && !predicate(); step++) {
		runtime.execute(createContext(step));
		await flushAsyncTasks();
	}
}

function createCapturedFace(faceSize, seed = 1) {
	const data = new Float32Array(faceSize * faceSize * 4);
	for (let i = 0; i < data.length; i += 4) {
		data[i] = seed;
		data[i + 1] = seed * 0.5;
		data[i + 2] = seed * 0.25;
		data[i + 3] = 1;
	}
	return data;
}

async function testCaptureRuntimeRequiresExplicitRequests() {
	const prefilterCalls = [];
	await withPrefilterStub(
		async () => {
			prefilterCalls.push("call");
			return createPrefilteredMap(prefilterCalls.length);
		},
		async () => {
			const runtime = new ProbeCaptureRuntime();

			const scene = new Scene();
			const probe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 16, height: 8 },
				})
			);
			scene.updateWorldMatrices();

			await runtime.execute({ scene, nowMs: 0 });
			await flushAsyncTasks();
			assert.equal(prefilterCalls.length, 0);
			assert.equal(probe.prefilteredMap, null);

			scene.invalidate("transform");
			await runtime.execute({ scene, nowMs: 60_000 });
			await flushAsyncTasks();
			assert.equal(prefilterCalls.length, 0);

			probe.requestCapture();
			for (let step = 0; step < 12 && prefilterCalls.length < 1; step++) {
				await runtime.execute({ scene, nowMs: 60_016 + step * 16 });
				await flushAsyncTasks();
			}
			assert.equal(prefilterCalls.length, 1);
			assert.ok(probe.prefilteredMap);

			scene.invalidate("lighting");
			await runtime.execute({ scene, nowMs: 120_000 });
			await flushAsyncTasks();
			assert.equal(prefilterCalls.length, 1);

			probe.requestCapture();
			await driveRuntimeUntil(
				runtime,
				(step) => ({ scene, nowMs: 120_016 + step * 16 }),
				() => prefilterCalls.length >= 2
			);
			assert.equal(prefilterCalls.length, 2);
		}
	);
}

async function testCaptureRuntimeThrottlesToOneInFlightPrefilter() {
	const deferredPrefilters = [];
	await withPrefilterStub(
		() => {
			const deferred = createDeferred();
			deferredPrefilters.push(deferred);
			return deferred.promise;
		},
		async () => {
			const runtime = new ProbeCaptureRuntime();

			const scene = new Scene();
			const firstProbe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 16, height: 8 },
				})
			);
			const secondProbe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 16, height: 8 },
				})
			);
			scene.updateWorldMatrices();
			firstProbe.requestCapture();
			secondProbe.requestCapture();

			for (let step = 0; step < 12 && deferredPrefilters.length < 1; step++) {
				runtime.execute({ scene, nowMs: step * 32 });
				runtime.execute({ scene, nowMs: step * 32 + 16 });
				await flushAsyncTasks();
			}
			assert.equal(deferredPrefilters.length, 1);

			deferredPrefilters[0].resolve(createPrefilteredMap(1));
			await flushAsyncTasks();

			assert.equal(deferredPrefilters.length, 1);
			assert.ok(firstProbe.prefilteredMap);
			assert.ok(secondProbe.prefilteredMap);
		}
	);
}

async function testCaptureRuntimeDropsStalePrefilterResults() {
	const deferred = createDeferred();
	let prefilterCallCount = 0;
	await withPrefilterStub(
		() => {
			prefilterCallCount++;
			return deferred.promise;
		},
		async () => {
			const runtime = new ProbeCaptureRuntime();

			const scene = new Scene();
			const probe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 16, height: 8 },
				})
			);
			scene.updateWorldMatrices();

			probe.requestCapture();
			await driveRuntimeUntil(
				runtime,
				(step) => ({ scene, nowMs: step * 16 }),
				() => prefilterCallCount >= 1
			);
			assert.equal(prefilterCallCount, 1);

			probe.position.set(2, 0, 0);
			scene.updateWorldMatrices();

			deferred.resolve(createPrefilteredMap(1));
			await flushAsyncTasks();
			assert.equal(probe.prefilteredMap, null);
		}
	);
}

async function testCaptureRuntimeDropsStalePrefilterResultsWhenCaptureFlagsChange() {
	const deferred = createDeferred();
	let prefilterCallCount = 0;
	await withPrefilterStub(
		() => {
			prefilterCallCount++;
			return deferred.promise;
		},
		async () => {
			const runtime = new ProbeCaptureRuntime();

			const scene = new Scene();
			const probe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 32, height: 16 },
					includeTransparent: true,
				})
			);
			scene.updateWorldMatrices();

			probe.requestCapture();
			await driveRuntimeUntil(
				runtime,
				(step) => ({ scene, nowMs: step * 16 }),
				() => prefilterCallCount >= 1
			);
			assert.equal(prefilterCallCount, 1);

			probe.includeTransparent = false;
			deferred.resolve(createPrefilteredMap(2));
			await flushAsyncTasks();
			assert.equal(probe.prefilteredMap, null);
		}
	);
}

async function testCaptureRuntimeSchedulesNearestProbeFirst() {
	const captureOrder = [];
	await withPrefilterStub(
		async () => createPrefilteredMap(1),
		async () => {
			const runtime = new ProbeCaptureRuntime();
			const scene = new Scene();
			const nearProbe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 32, height: 16 },
				})
			);
			nearProbe.position.set(1, 0, 0);
			const farProbe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 32, height: 16 },
				})
			);
			farProbe.position.set(20, 0, 0);
			scene.updateWorldMatrices();

			nearProbe.requestCapture();
			farProbe.requestCapture();
			runtime.execute({
				scene,
				nowMs: 0,
				frameContext: {},
				cameraWorldPosition: { x: 0, y: 0, z: 0 },
				captureSource: {
					async captureProbeFace(request) {
						captureOrder.push(request.targetId);
						return createCapturedFace(request.faceSize, 1);
					},
				},
			});
			await flushAsyncTasks();

			assert.ok(captureOrder.length > 0);
			assert.equal(captureOrder[0], nearProbe.id);
		}
	);
}

async function testCaptureRuntimeSettlesExplicitRequestsAcrossMultipleProbes() {
	let prefilterCallCount = 0;
	await withPrefilterStub(
		async () => createPrefilteredMap(++prefilterCallCount),
		async () => {
			const runtime = new ProbeCaptureRuntime();
			const scene = new Scene();
			const firstProbe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 16, height: 8 },
				})
			);
			const secondProbe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 16, height: 8 },
				})
			);
			scene.updateWorldMatrices();
			firstProbe.requestCapture();
			secondProbe.requestCapture();

			for (let i = 0; i < 5; i++) {
				await runtime.execute({ scene, nowMs: i * 16 });
				await flushAsyncTasks();
			}

			assert.equal(prefilterCallCount, 1);
			assert.ok(firstProbe.prefilteredMap);
			assert.ok(secondProbe.prefilteredMap);

			await runtime.execute({ scene, nowMs: 96 });
			await flushAsyncTasks();
			assert.equal(prefilterCallCount, 1);
		}
	);
}

async function testCaptureRuntimeDoesNotRecaptureAfterSceneInvalidation() {
	let prefilterCallCount = 0;
	await withPrefilterStub(
		async () => createPrefilteredMap(++prefilterCallCount),
		async () => {
			const runtime = new ProbeCaptureRuntime();
			const scene = new Scene();
			const probe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 16, height: 8 },
				})
			);
			scene.updateWorldMatrices();
			probe.requestCapture();

			await driveRuntimeUntil(
				runtime,
				(step) => ({ scene, nowMs: step * 16 }),
				() => prefilterCallCount >= 1
			);
			assert.equal(prefilterCallCount, 1);

			scene.invalidate("camera");
			await runtime.execute({ scene, nowMs: 16 });
			await flushAsyncTasks();
			assert.equal(prefilterCallCount, 1);

			scene.invalidate("transform");
			await runtime.execute({ scene, nowMs: 60_000 });
			await flushAsyncTasks();
			assert.equal(prefilterCallCount, 1);
		}
	);
}

async function testCaptureRuntimeBudgetDowngradesResolution() {
	const capturedFaceSizes = [];
	let prefilterCallCount = 0;
	await withPrefilterStub(
		async () => {
			prefilterCallCount++;
			return createPrefilteredMap(1);
		},
		async () => {
			const runtime = new ProbeCaptureRuntime({
				captureBudgetMs: 0.2,
			});

			const scene = new Scene();
			const probe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 512, height: 256 },
				})
			);
			scene.updateWorldMatrices();
			probe.requestCapture();

			for (let i = 0; i < 40 && prefilterCallCount === 0; i++) {
				await runtime.execute({
					scene,
					nowMs: i * 16,
					frameContext: {},
					captureSource: {
						async captureProbeFace(request) {
							capturedFaceSizes.push(request.faceSize);
							await new Promise((resolve) => setTimeout(resolve, 1));
							return createCapturedFace(request.faceSize, 0.8);
						},
					},
				});
				if (i === 0) {
					assert.notEqual(scene.dirtyReasonMask, 0);
				}
				await flushAsyncTasks();
			}

			assert.equal(prefilterCallCount, 1);
			assert.ok(probe.prefilteredMap);
			const uniqueSizes = Array.from(new Set(capturedFaceSizes));
			assert.ok(uniqueSizes.length >= 2);
			assert.ok(Math.min(...uniqueSizes) < Math.max(...uniqueSizes));
		}
	);
}

async function testCaptureRuntimeForwardsMeshCaptureFlags() {
	const capturedFlags = [];
	await withPrefilterStub(
		async () => createPrefilteredMap(1),
		async () => {
			const runtime = new ProbeCaptureRuntime();
			const scene = new Scene();
			const probe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 32, height: 16 },
					includeEnvironment: false,
					includeTransparent: false,
					includeParticles: false,
					includeShadows: false,
				})
			);
			scene.updateWorldMatrices();
			probe.requestCapture();

			runtime.execute({
				scene,
				nowMs: 0,
				frameContext: {},
				captureSource: {
					async captureProbeFace(request) {
						capturedFlags.push({
							includeEnvironment: request.includeEnvironment,
							includeTransparent: request.includeTransparent,
							includeParticles: request.includeParticles,
							includeShadows: request.includeShadows,
						});
						return createCapturedFace(request.faceSize, 0.7);
					},
				},
			});
			await flushAsyncTasks();

			assert.ok(capturedFlags.length > 0);
			for (const flags of capturedFlags) {
				assert.equal(flags.includeEnvironment, false);
				assert.equal(flags.includeTransparent, false);
				assert.equal(flags.includeParticles, false);
				assert.equal(flags.includeShadows, false);
			}
		}
	);
}

async function testCaptureRuntimeWarnsWhenMeshCaptureIsUnavailable() {
	const warnings = [];
	Logger.configure({
		level: "warn",
		resetOnceKeys: true,
		sink: {
			warn(...args) {
				warnings.push(args.map((value) => String(value)).join(" "));
			},
		},
	});
	try {
		await withPrefilterStub(
			async (envMap) => envMap,
			async () => {
				const runtime = new ProbeCaptureRuntime();
				const scene = new Scene();
				const probe = scene.add(
					new ReflectionProbe({
						source: "capturedScene",
						includeMeshes: true,
						includeEnvironment: false,
						captureResolution: { width: 16, height: 8 },
					})
				);
				scene.updateWorldMatrices();

				probe.requestCapture();
				await driveRuntimeUntil(
					runtime,
					(step) => ({ scene, nowMs: step * 16 }),
					() => probe.prefilteredMap !== null
				);
				assert.ok(probe.prefilteredMap);
				assert.equal(warnings.length, 1);
				assert.ok(warnings[0].includes("probe-mesh-capture-unsupported"));

				probe.requestCapture();
				await runtime.execute({ scene, nowMs: 16 });
				await flushAsyncTasks();
				assert.equal(warnings.length, 1);
			}
		);
	} finally {
		Logger.reset();
	}
}

class RendererCaptureStageBackendStub extends TestRenderBackend {
	constructor() {
		super();
		this.type = "stub";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: true,
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
		installNoopPostProcessAdapter(
			this,
			"stub"
		);
		this.frameScheduling = "on-demand";
		this.executedStages = [];
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

	beginFrame() {}

	executePass(pass) {
		this.executedStages.push(pass.stage);
	}

	endFrame() {}
}

async function testRendererCaptureStageRunsWithoutReflectivePackets() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;
	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

		const backend = new RendererCaptureStageBackendStub();
		const camera = new Camera();
		const canvas = {
			width: 320,
			height: 180,
			getBoundingClientRect() {
				return { width: 320, height: 180 };
			},
		};
		const renderer = new Renderer(canvas, backend, camera);
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.postProcess.getPass("gamma")?.disable();

		const capturedProbe = renderer.scene.add(
			new ReflectionProbe({
				source: "capturedScene",
			})
		);
		capturedProbe.requestCapture();

		let captureStageCalls = 0;
		let captureContext = null;
		renderer._probeCaptureRuntime = {
			execute(context) {
				captureStageCalls++;
				captureContext = context;
			},
		};

		await renderer.renderFrame(16);
		assert.equal(captureStageCalls, 1);
		assert.equal(captureContext.backend, backend);
		assert.equal(backend.executedStages.includes("reflection"), false);
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

async function run() {
	await testCaptureRuntimeRequiresExplicitRequests();
	await testCaptureRuntimeThrottlesToOneInFlightPrefilter();
	await testCaptureRuntimeDropsStalePrefilterResults();
	await testCaptureRuntimeDropsStalePrefilterResultsWhenCaptureFlagsChange();
	await testCaptureRuntimeSchedulesNearestProbeFirst();
	await testCaptureRuntimeSettlesExplicitRequestsAcrossMultipleProbes();
	await testCaptureRuntimeDoesNotRecaptureAfterSceneInvalidation();
	await testCaptureRuntimeBudgetDowngradesResolution();
	await testCaptureRuntimeForwardsMeshCaptureFlags();
	await testCaptureRuntimeWarnsWhenMeshCaptureIsUnavailable();
	await testRendererCaptureStageRunsWithoutReflectivePackets();
	console.log("Reflection probe capture runtime tests passed");
}

await run();
