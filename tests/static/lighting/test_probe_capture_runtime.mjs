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
import { directionFromEquirectUV } from "../../../src/lights/runtime/environmentMapRuntime.ts";

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

function createDirectionalCapturedFace(faceIndex, faceSize) {
	const data = new Float32Array(faceSize * faceSize * 4);
	for (let y = 0; y < faceSize; y++) {
		for (let x = 0; x < faceSize; x++) {
			const u = (2 * (x + 0.5)) / faceSize - 1;
			const v = 1 - (2 * (y + 0.5)) / faceSize;
			let direction;
			switch (faceIndex) {
				case 0:
					direction = { x: 1, y: v, z: -u };
					break;
				case 1:
					direction = { x: -1, y: v, z: u };
					break;
				case 2:
					direction = { x: u, y: 1, z: -v };
					break;
				case 3:
					direction = { x: u, y: -1, z: v };
					break;
				case 4:
					direction = { x: u, y: v, z: 1 };
					break;
				default:
					direction = { x: -u, y: v, z: -1 };
					break;
			}
			const length = Math.hypot(direction.x, direction.y, direction.z);
			const index = (y * faceSize + x) * 4;
			data[index] = direction.x / length * 0.5 + 0.5;
			data[index + 1] = direction.y / length * 0.5 + 0.5;
			data[index + 2] = direction.z / length * 0.5 + 0.5;
			data[index + 3] = 1;
		}
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
	const rawTexture = new Texture({ data: null, width: 0, height: 0, colorSpace: "HDR" });
	const cubeTexture = createBoundCubeTexture();
	const probe = scene.add(
		new LightProbe({
			source: "capturedScene",
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
			captureSource: {
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
	let prefilterBackend = null;
	const backend = {
		profile: { id: "probe-test" },
		extensions: {
			getBackendExtension() {
				return undefined;
			},
		},
	};
	await withPrefilterStub(
		async function () {
			prefilterCallCount++;
			prefilterBackend = this._service.backend;
			return createPrefilteredMap(2);
		},
		async () => {
			const runtime = new ProbeCaptureRuntime({ captureBudgetMs: 100 });
			const scene = new Scene();
			const lightProbe = scene.add(
				new LightProbe({
					source: "capturedScene",
					captureResolution: { width: 16, height: 8 },
					includeEnvironment: false,
				})
			);
			const reflectionProbe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
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
					backend,
					captureSource: {
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
			assert.equal(prefilterBackend, backend);
			assert.ok(lightProbe.sh[0].r > 0);
			assert.ok(reflectionProbe.prefilteredMap);
		}
	);
}

async function testReflectionProbeCaptureWritesBoundPrefilteredTexture() {
	await withPrefilterStub(
		async () => createPrefilteredMap(4),
		async () => {
			const runtime = new ProbeCaptureRuntime({ captureBudgetMs: 100 });
			const scene = new Scene();
			const rawTexture = new Texture({ data: null, width: 0, height: 0, colorSpace: "HDR" });
			const cubeTexture = createBoundCubeTexture();
			const prefilteredTexture = new Texture({ data: null, width: 0, height: 0, colorSpace: "HDR" });
			const probe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
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
					captureSource: {
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

async function testSoftwareCapturePreservesHDREnvironmentRadiance() {
	await withPrefilterStub(
		async (envMap) => envMap,
		async () => {
			const runtime = new ProbeCaptureRuntime({ captureBudgetMs: 100 });
			const scene = new Scene();
			scene.environment.backgroundTexture = new Texture({
				data: new Float32Array([
					4, 2, 1, 1,
					4, 2, 1, 1,
				]),
				width: 2,
				height: 1,
				colorSpace: "HDR",
			});
			const rawTexture = new Texture({
				data: null,
				width: 0,
				height: 0,
				colorSpace: "HDR",
			});
			const probe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 16, height: 8 },
					includeMeshes: false,
					includeEnvironment: true,
				})
			);
			probe.capture.bindRawTexture(rawTexture);
			scene.updateWorldMatrices();
			probe.requestCapture();

			await driveRuntimeUntil(
				runtime,
				(step) => ({ scene, nowMs: step * 16 }),
				() => probe.prefilteredMap !== null
			);

			assert.ok(rawTexture.data instanceof Float32Array);
			for (let index = 0; index < rawTexture.data.length; index += 4) {
				assert.ok(Math.abs(rawTexture.data[index] - 4) < 1e-6);
				assert.ok(Math.abs(rawTexture.data[index + 1] - 2) < 1e-6);
				assert.ok(Math.abs(rawTexture.data[index + 2] - 1) < 1e-6);
			}
		}
	);
}

async function testCubemapConversionUsesSeamAwareBilinearSampling() {
	await withPrefilterStub(
		async (envMap) => envMap,
		async () => {
			const runtime = new ProbeCaptureRuntime({ captureBudgetMs: 100 });
			const scene = new Scene();
			const rawTexture = new Texture({
				data: null,
				width: 0,
				height: 0,
				colorSpace: "HDR",
			});
			const probe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
					captureResolution: { width: 32, height: 16 },
					includeMeshes: true,
					includeEnvironment: false,
				})
			);
			probe.capture.bindRawTexture(rawTexture);
			scene.updateWorldMatrices();
			probe.requestCapture();

			await driveRuntimeUntil(
				runtime,
				(step) => ({
					scene,
					nowMs: step * 16,
					frameContext: {},
					captureSource: {
						async captureProbeFace(request) {
							return createDirectionalCapturedFace(
								request.faceIndex,
								request.faceSize
							);
						},
					},
				}),
				() => probe.prefilteredMap !== null
			);

			assert.ok(rawTexture.data instanceof Float32Array);
			let maxError = 0;
			for (let y = 0; y < rawTexture.height; y++) {
				for (let x = 0; x < rawTexture.width; x++) {
					const expected = directionFromEquirectUV(
						(x + 0.5) / rawTexture.width,
						(y + 0.5) / rawTexture.height
					);
					const index = (y * rawTexture.width + x) * 4;
					maxError = Math.max(
						maxError,
						Math.abs(rawTexture.data[index] - (expected.x * 0.5 + 0.5)),
						Math.abs(rawTexture.data[index + 1] - (expected.y * 0.5 + 0.5)),
						Math.abs(rawTexture.data[index + 2] - (expected.z * 0.5 + 0.5))
					);
				}
			}
			assert.ok(maxError < 0.04, `Cubemap conversion max error was ${maxError}`);
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
					captureResolution: { width: 16, height: 8 },
					includeMeshes: false,
					includeEnvironment: false,
				})
			);
			const reflectionProbe = scene.add(
				new ReflectionProbe({
					source: "capturedScene",
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

async function testGridCaptureRequiresExplicitRequest() {
	const runtime = new ProbeCaptureRuntime({ captureBudgetMs: 100 });
	const scene = new Scene();
	scene.add(new AmbientLight({ intensity: 1 }));
	const grid = scene.add(
		new IrradianceProbeGrid({
			dimensions: { x: 2, y: 1, z: 1 },
			source: "capturedScene",
			captureResolution: { width: 16, height: 8 },
			includeMeshes: false,
			includeEnvironment: false,
		})
	);
	scene.updateWorldMatrices();

	for (let step = 0; step < 4; step++) {
		await runtime.execute({ scene, nowMs: 200 + step * 16 });
		await flushAsyncTasks();
	}
	assert.equal(grid.isCellValid(0), false);
	assert.equal(grid.isCellValid(1), false);

	scene.invalidate("lighting");
	for (let step = 0; step < 4; step++) {
		await runtime.execute({ scene, nowMs: 400 + step * 16 });
		await flushAsyncTasks();
	}
	assert.equal(grid.isCellValid(0), false);
	assert.equal(grid.isCellValid(1), false);

	grid.requestCapture();
	await driveRuntimeUntil(
		runtime,
		(step) => ({ scene, nowMs: 1000 + step * 16 }),
		() => grid.isCellValid(0) && grid.isCellValid(1),
		8
	);
	assert.equal(grid.captureRevision, 3);
	assert.equal(grid.isCellValid(0), true);
	assert.equal(grid.isCellValid(1), true);

	scene.invalidate("transform");
	for (let step = 0; step < 4; step++) {
		await runtime.execute({ scene, nowMs: 2000 + step * 16 });
		await flushAsyncTasks();
	}
	assert.equal(grid.captureRevision, 3);
}

async function run() {
	await testLightProbeManualCaptureProjectsSHWithoutPrefilter();
	await testLightProbeCaptureWritesBoundTextures();
	await testSharedCaptureUpdatesLightAndReflectionProbe();
	await testReflectionProbeCaptureWritesBoundPrefilteredTexture();
	await testSoftwareCapturePreservesHDREnvironmentRadiance();
	await testCubemapConversionUsesSeamAwareBilinearSampling();
	await testSharedCaptureSkipsStaleLightProbeResult();
	await testGridManualCellAndWholeGridCaptureRequests();
	await testGridCaptureRequiresExplicitRequest();
	console.log("Probe capture runtime tests passed");
}

await run();
