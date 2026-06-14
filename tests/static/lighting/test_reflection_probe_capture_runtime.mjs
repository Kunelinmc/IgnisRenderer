import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { Logger } from "../../../src/foundation/Logger.ts";
import { ReflectionProbe } from "../../../src/lights/ReflectionProbe.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { ProbeCaptureRuntime } from "../../../src/lights/runtime/ProbeCaptureRuntime.ts";
import { Renderer } from "../../../src/renderers/Renderer.ts";
import {
	installNoopPostProcessAdapter,
} from "../../helpers/postprocess.mjs";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

function createBakedEnvironment(seed = 1) {
	return {
		sh: Array.from({ length: 16 }, () => ({ r: 0, g: 0, b: 0 })),
		prefilteredMap: new Texture(
			new Float32Array([seed, seed * 0.5, seed * 0.25, 1]),
			1,
			1,
			"HDR"
		),
	};
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

async function testCaptureRuntimeTriggerModes() {
	const bakeCalls = [];
	const runtime = new ProbeCaptureRuntime({
		bakeEnvironmentIBL: async () => {
			bakeCalls.push("call");
			return createBakedEnvironment(bakeCalls.length);
		},
	});

	const manualScene = new Scene();
	const manualProbe = manualScene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "manual",
			captureResolution: { width: 16, height: 8 },
		})
	);
	manualScene.updateWorldMatrices();

	await runtime.execute({ scene: manualScene, nowMs: 0 });
	await flushAsyncTasks();
	assert.equal(bakeCalls.length, 0);
	assert.equal(manualProbe.prefilteredMap, null);

	manualProbe.requestCapture();
	for (let step = 0; step < 12 && bakeCalls.length < 1; step++) {
		await runtime.execute({ scene: manualScene, nowMs: 16 + step * 16 });
		await flushAsyncTasks();
	}
	assert.equal(bakeCalls.length, 1);
	assert.ok(manualProbe.prefilteredMap);

	await runtime.execute({ scene: manualScene, nowMs: 32 });
	await flushAsyncTasks();
	assert.equal(bakeCalls.length, 1);

	const dirtyScene = new Scene();
	const dirtyProbe = dirtyScene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "onSceneDirty",
			captureResolution: { width: 16, height: 8 },
		})
	);
	dirtyScene.updateWorldMatrices();

	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene: dirtyScene, nowMs: 100 + step * 16 }),
		() => bakeCalls.length >= 2
	);
	assert.equal(bakeCalls.length, 2);
	assert.ok(dirtyProbe.prefilteredMap);

	await runtime.execute({ scene: dirtyScene, nowMs: 116 });
	await flushAsyncTasks();
	assert.equal(bakeCalls.length, 2);

	dirtyScene.invalidate("transform");
	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene: dirtyScene, nowMs: 132 + step * 16 }),
		() => bakeCalls.length >= 3
	);
	assert.equal(bakeCalls.length, 3);

	const intervalScene = new Scene();
	const intervalProbe = intervalScene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "interval",
			captureIntervalSeconds: 0.5,
			captureResolution: { width: 16, height: 8 },
		})
	);
	intervalScene.updateWorldMatrices();

	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene: intervalScene, nowMs: step * 16 }),
		() => bakeCalls.length >= 4
	);
	assert.equal(bakeCalls.length, 4);
	assert.ok(intervalProbe.prefilteredMap);

	await runtime.execute({ scene: intervalScene, nowMs: 200 });
	await flushAsyncTasks();
	assert.equal(bakeCalls.length, 4);

	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene: intervalScene, nowMs: 700 + step * 16 }),
		() => bakeCalls.length >= 5
	);
	assert.equal(bakeCalls.length, 5);
}

async function testCaptureRuntimeThrottlesToOneInFlightBake() {
	const deferredBakes = [];
	const runtime = new ProbeCaptureRuntime({
		bakeEnvironmentIBL: () => {
			const deferred = createDeferred();
			deferredBakes.push(deferred);
			return deferred.promise;
		},
	});

	const scene = new Scene();
	const firstProbe = scene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "onSceneDirty",
			captureResolution: { width: 16, height: 8 },
		})
	);
	const secondProbe = scene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "onSceneDirty",
			captureResolution: { width: 16, height: 8 },
		})
	);
	scene.updateWorldMatrices();

	for (let step = 0; step < 12 && deferredBakes.length < 1; step++) {
		runtime.execute({ scene, nowMs: step * 32 });
		runtime.execute({ scene, nowMs: step * 32 + 16 });
		await flushAsyncTasks();
	}
	assert.equal(deferredBakes.length, 1);

	deferredBakes[0].resolve(createBakedEnvironment(1));
	await flushAsyncTasks();

	assert.equal(deferredBakes.length, 1);
	assert.ok(firstProbe.prefilteredMap);
	assert.ok(secondProbe.prefilteredMap);
}

async function testCaptureRuntimeDropsStaleBakeResults() {
	const deferred = createDeferred();
	let bakeCallCount = 0;
	const runtime = new ProbeCaptureRuntime({
		bakeEnvironmentIBL: () => {
			bakeCallCount++;
			return deferred.promise;
		},
	});

	const scene = new Scene();
	const probe = scene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "manual",
			captureResolution: { width: 16, height: 8 },
		})
	);
	scene.updateWorldMatrices();

	probe.requestCapture();
	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene, nowMs: step * 16 }),
		() => bakeCallCount >= 1
	);
	assert.equal(bakeCallCount, 1);

	probe.position.set(2, 0, 0);
	scene.updateWorldMatrices();

	deferred.resolve(createBakedEnvironment(1));
	await flushAsyncTasks();
	assert.equal(probe.prefilteredMap, null);
}

async function testCaptureRuntimeDropsStaleBakeResultsWhenCaptureFlagsChange() {
	const deferred = createDeferred();
	let bakeCallCount = 0;
	const runtime = new ProbeCaptureRuntime({
		bakeEnvironmentIBL: () => {
			bakeCallCount++;
			return deferred.promise;
		},
	});

	const scene = new Scene();
	const probe = scene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "manual",
			captureResolution: { width: 32, height: 16 },
			includeTransparent: true,
		})
	);
	scene.updateWorldMatrices();

	probe.requestCapture();
	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene, nowMs: step * 16 }),
		() => bakeCallCount >= 1
	);
	assert.equal(bakeCallCount, 1);

	probe.includeTransparent = false;
	deferred.resolve(createBakedEnvironment(2));
	await flushAsyncTasks();
	assert.equal(probe.prefilteredMap, null);
}

async function testCaptureRuntimeSchedulesNearestProbeFirst() {
	const captureOrder = [];
	const runtime = new ProbeCaptureRuntime({
		bakeEnvironmentIBL: async () => createBakedEnvironment(1),
	});
	const scene = new Scene();
	const nearProbe = scene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "manual",
			captureResolution: { width: 32, height: 16 },
		})
	);
	nearProbe.position.set(1, 0, 0);
	const farProbe = scene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "manual",
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
		webgpuCaptureSource: {
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

async function testCaptureRuntimeOnSceneDirtySettlesAcrossMultipleProbes() {
	let bakeCallCount = 0;
	const runtime = new ProbeCaptureRuntime({
		bakeEnvironmentIBL: async () => createBakedEnvironment(++bakeCallCount),
	});
	const scene = new Scene();
	const firstProbe = scene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "onSceneDirty",
			captureResolution: { width: 16, height: 8 },
		})
	);
	const secondProbe = scene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "onSceneDirty",
			captureResolution: { width: 16, height: 8 },
		})
	);
	scene.updateWorldMatrices();

	for (let i = 0; i < 5; i++) {
		await runtime.execute({ scene, nowMs: i * 16 });
		await flushAsyncTasks();
	}

	assert.equal(bakeCallCount, 1);
	assert.ok(firstProbe.prefilteredMap);
	assert.ok(secondProbe.prefilteredMap);

	await runtime.execute({ scene, nowMs: 96 });
	await flushAsyncTasks();
	assert.equal(bakeCallCount, 1);
}

async function testCaptureRuntimeIgnoresCameraDirtyForOnSceneDirty() {
	let bakeCallCount = 0;
	const runtime = new ProbeCaptureRuntime({
		bakeEnvironmentIBL: async () => createBakedEnvironment(++bakeCallCount),
	});
	const scene = new Scene();
	scene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "onSceneDirty",
			captureResolution: { width: 16, height: 8 },
		})
	);
	scene.updateWorldMatrices();

	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene, nowMs: step * 16 }),
		() => bakeCallCount >= 1
	);
	assert.equal(bakeCallCount, 1);

	scene.invalidate("camera");
	await runtime.execute({ scene, nowMs: 16 });
	await flushAsyncTasks();
	assert.equal(bakeCallCount, 1);
}

async function testCaptureRuntimeBudgetDowngradesResolution() {
	const capturedFaceSizes = [];
	let bakeCallCount = 0;
	const runtime = new ProbeCaptureRuntime({
		captureBudgetMs: 0.2,
		bakeEnvironmentIBL: async () => {
			bakeCallCount++;
			return createBakedEnvironment(1);
		},
	});

	const scene = new Scene();
	const probe = scene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "manual",
			captureResolution: { width: 512, height: 256 },
		})
	);
	scene.updateWorldMatrices();
	probe.requestCapture();

	for (let i = 0; i < 40 && bakeCallCount === 0; i++) {
		runtime.execute({
			scene,
			nowMs: i * 16,
			frameContext: {},
			webgpuCaptureSource: {
				async captureProbeFace(request) {
					capturedFaceSizes.push(request.faceSize);
					await new Promise((resolve) => setTimeout(resolve, 1));
					return createCapturedFace(request.faceSize, 0.8);
				},
			},
		});
		await flushAsyncTasks();
	}

	assert.equal(bakeCallCount, 1);
	assert.ok(probe.prefilteredMap);
	const uniqueSizes = Array.from(new Set(capturedFaceSizes));
	assert.ok(uniqueSizes.length >= 2);
	assert.ok(Math.min(...uniqueSizes) < Math.max(...uniqueSizes));
}

async function testCaptureRuntimeForwardsMeshCaptureFlags() {
	const capturedFlags = [];
	const runtime = new ProbeCaptureRuntime({
		bakeEnvironmentIBL: async () => createBakedEnvironment(1),
	});
	const scene = new Scene();
	const probe = scene.add(
		new ReflectionProbe({
			source: "capturedScene",
			captureUpdateMode: "manual",
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
		webgpuCaptureSource: {
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
	const runtime = new ProbeCaptureRuntime({
			bakeEnvironmentIBL: async (envMap) => ({
				sh: Array.from({ length: 16 }, () => ({ r: 0, g: 0, b: 0 })),
				prefilteredMap: envMap,
			}),
		});
		const scene = new Scene();
		const probe = scene.add(
			new ReflectionProbe({
				source: "capturedScene",
				captureUpdateMode: "manual",
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
		const renderer = new Renderer(backend, canvas, camera);
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.postProcess.getPass("gamma")?.disable();

		const capturedProbe = renderer.scene.add(
			new ReflectionProbe({
				source: "capturedScene",
				captureUpdateMode: "manual",
			})
		);
		capturedProbe.requestCapture();

		let captureStageCalls = 0;
		renderer._probeCaptureRuntime = {
			execute() {
				captureStageCalls++;
			},
		};

		await renderer.renderScene(16);
		assert.equal(captureStageCalls, 1);
		assert.equal(backend.executedStages.includes("reflection"), false);
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

async function run() {
	await testCaptureRuntimeTriggerModes();
	await testCaptureRuntimeThrottlesToOneInFlightBake();
	await testCaptureRuntimeDropsStaleBakeResults();
	await testCaptureRuntimeDropsStaleBakeResultsWhenCaptureFlagsChange();
	await testCaptureRuntimeSchedulesNearestProbeFirst();
	await testCaptureRuntimeOnSceneDirtySettlesAcrossMultipleProbes();
	await testCaptureRuntimeIgnoresCameraDirtyForOnSceneDirty();
	await testCaptureRuntimeBudgetDowngradesResolution();
	await testCaptureRuntimeForwardsMeshCaptureFlags();
	await testCaptureRuntimeWarnsWhenMeshCaptureIsUnavailable();
	await testRendererCaptureStageRunsWithoutReflectivePackets();
	console.log("Reflection probe capture runtime tests passed");
}

await run();
