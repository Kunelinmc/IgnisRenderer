import assert from "node:assert/strict";
import { CubeTexture } from "../../../src/core/CubeTexture.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { AmbientLight } from "../../../src/lights/AmbientLight.ts";
import { IrradianceProbeGrid } from "../../../src/lights/IrradianceProbeGrid.ts";
import { LightProbe } from "../../../src/lights/LightProbe.ts";
import { ReflectionProbe } from "../../../src/lights/ReflectionProbe.ts";
import { IBLPrefilter } from "../../../src/lights/ibl/IBLPrefilter.ts";
import { ProbeCaptureRuntime } from "../../../src/lights/runtime/ProbeCaptureRuntime.ts";

function createPrefilteredMap(seed = 1) {
	return new Texture(
		new Float32Array([seed, seed * 0.5, seed * 0.25, 1]),
		1,
		1,
		"HDR"
	);
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

function createBoundCubeTexture(seed = 0) {
	return new CubeTexture({
		faces: Array.from(
			{ length: 6 },
			() => new Float32Array([seed, seed, seed, 1])
		),
		size: 1,
		colorSpace: "HDR",
	});
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

async function driveRuntimeUntil(runtime, createContext, predicate, maxSteps = 12) {
	for (let step = 0; step < maxSteps && !predicate(); step++) {
		runtime.execute(createContext(step));
		await flushAsyncTasks();
	}
}

async function testLightProbeManualCaptureProjectsSHWithoutPrefilter() {
	let prefilterCallCount = 0;
	const runtime = new ProbeCaptureRuntime();
	const scene = new Scene();
	scene.add(new AmbientLight({ intensity: 1 }));
	const probe = scene.add(
		new LightProbe({
			source: "capturedScene",
			captureUpdateMode: "manual",
			captureResolution: { width: 16, height: 8 },
			includeMeshes: false,
			includeEnvironment: false,
		})
	);
	scene.updateWorldMatrices();

	await runtime.execute({ scene, nowMs: 0 });
	await flushAsyncTasks();
	assert.equal(probe.sh[0].r, 0);

	probe.requestCapture();
	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene, nowMs: 16 + step * 16 }),
		() => probe.sh[0].r > 0
	);

	assert.ok(probe.sh[0].r > 0);
	assert.equal(prefilterCallCount, 0);
}

async function testLightProbeCaptureWritesBoundTextures() {
	const runtime = new ProbeCaptureRuntime();
	const scene = new Scene();
	const rawTexture = new Texture(null, 0, 0, "HDR");
	const cubeTexture = createBoundCubeTexture();
	const probe = scene.add(
		new LightProbe({
			source: "capturedScene",
			captureUpdateMode: "manual",
			captureResolution: { width: 16, height: 8 },
			includeEnvironment: false,
		})
	);
	probe.capture
		.bindRawTexture(rawTexture)
		.bindCubeTexture(cubeTexture);
	scene.updateWorldMatrices();
	probe.requestCapture();

	await driveRuntimeUntil(
		runtime,
		(step) => ({
			scene,
			nowMs: step * 16,
			frameContext: {},
			webgpuCaptureSource: {
				async captureProbeFace(request) {
					return createCapturedFace(request.faceSize, 0.6);
				},
			},
		}),
		() => probe.sh[0].r > 0
	);

	assert.equal(probe.capture.rawTexture, rawTexture);
	assert.equal(probe.capture.cubeTexture, cubeTexture);
	assert.equal(rawTexture.width, 16);
	assert.equal(rawTexture.height, 8);
	assert.equal(rawTexture.colorSpace, "HDR");
	assert.ok(rawTexture.data instanceof Float32Array);
	assert.equal(cubeTexture.width, 4);
	assert.equal(cubeTexture.height, 4);
	assert.equal(cubeTexture.colorSpace, "HDR");
	assert.equal(cubeTexture.getFaces().length, 6);
	assert.equal(cubeTexture.getFaces()[0].length, 4 * 4 * 4);
}

async function testSharedCaptureUpdatesLightAndReflectionProbe() {
	let prefilterCallCount = 0;
	let faceCaptureCount = 0;
	await withPrefilterStub(
		async () => {
			prefilterCallCount++;
			return createPrefilteredMap(2);
		},
		async () => {
			const runtime = new ProbeCaptureRuntime();
			const scene = new Scene();
			const lightProbe = scene.add(
				new LightProbe({
					source: "capturedScene",
					captureUpdateMode: "manual",
					captureResolution: { width: 16, height: 8 },
					includeEnvironment: false,
				})
			);
			const reflectionProbe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureUpdateMode: "manual",
					captureResolution: { width: 16, height: 8 },
					includeEnvironment: false,
				})
			);
			scene.updateWorldMatrices();
			lightProbe.requestCapture();
			reflectionProbe.requestCapture();

			await driveRuntimeUntil(
				runtime,
				(step) => ({
					scene,
					nowMs: step * 16,
					frameContext: {},
					webgpuCaptureSource: {
						async captureProbeFace(request) {
							faceCaptureCount++;
							return createCapturedFace(request.faceSize, 0.75);
						},
					},
				}),
				() => lightProbe.sh[0].r > 0 && reflectionProbe.prefilteredMap !== null
			);

			assert.equal(faceCaptureCount, 6);
			assert.equal(prefilterCallCount, 1);
			assert.ok(lightProbe.sh[0].r > 0);
			assert.ok(reflectionProbe.prefilteredMap);
		}
	);
}

async function testReflectionProbeCaptureWritesBoundPrefilteredTexture() {
	await withPrefilterStub(
		async () => createPrefilteredMap(4),
		async () => {
			const runtime = new ProbeCaptureRuntime();
			const scene = new Scene();
			const rawTexture = new Texture(null, 0, 0, "HDR");
			const cubeTexture = createBoundCubeTexture();
			const prefilteredTexture = new Texture(null, 0, 0, "HDR");
			const probe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureUpdateMode: "manual",
					captureResolution: { width: 16, height: 8 },
					includeEnvironment: false,
				})
			);
			probe.capture
				.bindRawTexture(rawTexture)
				.bindCubeTexture(cubeTexture)
				.bindPrefilteredTexture(prefilteredTexture);
			scene.updateWorldMatrices();
			probe.requestCapture();

			await driveRuntimeUntil(
				runtime,
				(step) => ({
					scene,
					nowMs: step * 16,
					frameContext: {},
					webgpuCaptureSource: {
						async captureProbeFace(request) {
							return createCapturedFace(request.faceSize, 0.8);
						},
					},
				}),
				() => probe.prefilteredMap !== null
			);

			assert.equal(probe.prefilteredMap, prefilteredTexture);
			assert.equal(probe.capture.prefilteredTexture, prefilteredTexture);
			assert.equal(prefilteredTexture.width, 1);
			assert.equal(prefilteredTexture.height, 1);
			assert.ok(prefilteredTexture.data instanceof Float32Array);
			assert.equal(prefilteredTexture.data[0], 4);
			assert.equal(rawTexture.width, 16);
			assert.equal(cubeTexture.width, 4);
		}
	);
}

async function testSharedCaptureSkipsStaleLightProbeResult() {
	const deferredPrefilter = createDeferred();
	let prefilterCallCount = 0;
	await withPrefilterStub(
		() => {
			prefilterCallCount++;
			return deferredPrefilter.promise;
		},
		async () => {
			const runtime = new ProbeCaptureRuntime();
			const scene = new Scene();
			const lightProbe = scene.add(
				new LightProbe({
					source: "capturedScene",
					captureUpdateMode: "manual",
					captureResolution: { width: 16, height: 8 },
					includeMeshes: false,
					includeEnvironment: false,
				})
			);
			const reflectionProbe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureUpdateMode: "manual",
					captureResolution: { width: 16, height: 8 },
					includeMeshes: false,
					includeEnvironment: false,
				})
			);
			scene.updateWorldMatrices();
			lightProbe.requestCapture();
			reflectionProbe.requestCapture();

			await driveRuntimeUntil(
				runtime,
				(step) => ({ scene, nowMs: step * 16 }),
				() => prefilterCallCount >= 1
			);
			assert.equal(prefilterCallCount, 1);

			lightProbe.position.set(1, 0, 0);
			scene.updateWorldMatrices();
			deferredPrefilter.resolve(createPrefilteredMap(3));
			await flushAsyncTasks();

			assert.equal(lightProbe.sh[0].r, 0);
			assert.ok(reflectionProbe.prefilteredMap);
		}
	);
}

async function testGridManualCellAndWholeGridCaptureRequests() {
	const runtime = new ProbeCaptureRuntime();
	const scene = new Scene();
	scene.add(new AmbientLight({ intensity: 1 }));
	const grid = scene.add(
		new IrradianceProbeGrid({
			dimensions: { x: 2, y: 1, z: 1 },
			source: "capturedScene",
			captureUpdateMode: "manual",
			captureResolution: { width: 16, height: 8 },
			includeMeshes: false,
			includeEnvironment: false,
		})
	);
	scene.updateWorldMatrices();

	grid.requestCapture(1);
	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene, nowMs: step * 16 }),
		() => grid.isCellValid(1)
	);
	assert.equal(grid.isCellValid(0), false);
	assert.equal(grid.isCellValid(1), true);
	assert.ok(grid.getCellSH(1)[0].r > 0);

	grid.requestCapture();
	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene, nowMs: 1000 + step * 16 }),
		() => grid.isCellValid(0) && grid.isCellValid(1),
		32
	);
	assert.equal(grid.isCellValid(0), true);
	assert.equal(grid.isCellValid(1), true);
}

async function testGridOnSceneDirtyCaptureDoesNotSelfTriggerOrStarveCells() {
	const runtime = new ProbeCaptureRuntime({ captureBudgetMs: 100 });
	const scene = new Scene();
	scene.add(new AmbientLight({ intensity: 1 }));
	const grid = scene.add(
		new IrradianceProbeGrid({
			dimensions: { x: 2, y: 1, z: 1 },
			source: "capturedScene",
			captureUpdateMode: "onSceneDirty",
			captureResolution: { width: 16, height: 8 },
			includeMeshes: false,
			includeEnvironment: false,
		})
	);
	scene.updateWorldMatrices();

	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene, nowMs: step * 16 }),
		() => grid.isCellValid(0) && grid.isCellValid(1),
		8
	);
	assert.equal(grid.isCellValid(0), true);
	assert.equal(grid.isCellValid(1), true);
	assert.equal(grid.captureRevision, 2);

	for (let step = 0; step < 4; step++) {
		await runtime.execute({ scene, nowMs: 200 + step * 16 });
		await flushAsyncTasks();
	}
	assert.equal(grid.captureRevision, 2);

	scene.invalidate("lighting");
	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene, nowMs: 1000 + step * 16 }),
		() => grid.captureRevision >= 4,
		8
	);
	assert.equal(grid.captureRevision, 4);
	assert.equal(grid.isCellValid(0), true);
	assert.equal(grid.isCellValid(1), true);
}

async function run() {
	await testLightProbeManualCaptureProjectsSHWithoutPrefilter();
	await testLightProbeCaptureWritesBoundTextures();
	await testSharedCaptureUpdatesLightAndReflectionProbe();
	await testReflectionProbeCaptureWritesBoundPrefilteredTexture();
	await testSharedCaptureSkipsStaleLightProbeResult();
	await testGridManualCellAndWholeGridCaptureRequests();
	await testGridOnSceneDirtyCaptureDoesNotSelfTriggerOrStarveCells();
	console.log("Probe capture runtime tests passed");
}

await run();
