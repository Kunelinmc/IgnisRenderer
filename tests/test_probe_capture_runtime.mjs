import assert from "node:assert/strict";
import { Scene } from "../src/core/Scene.ts";
import { Texture } from "../src/core/Texture.ts";
import { AmbientLight } from "../src/lights/AmbientLight.ts";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { ReflectionProbe } from "../src/lights/ReflectionProbe.ts";
import { ProbeCaptureRuntime } from "../src/pipeline/ProbeCaptureRuntime.ts";

function createBakedEnvironment(seed = 1) {
	return {
		sh: Array.from({ length: 16 }, (_, index) => ({
			r: seed * (index + 1),
			g: seed * (index + 1) * 0.5,
			b: seed * (index + 1) * 0.25,
		})),
		prefilteredMap: new Texture(
			new Float32Array([seed, seed * 0.5, seed * 0.25, 1]),
			1,
			1,
			"HDR"
		),
	};
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

async function testLightProbeManualCaptureProjectsSHWithoutPrefilterBake() {
	let bakeCallCount = 0;
	const runtime = new ProbeCaptureRuntime({
		bakeEnvironmentIBL: async () => {
			bakeCallCount++;
			return createBakedEnvironment(1);
		},
	});
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
	assert.equal(bakeCallCount, 0);
}

async function testSharedCaptureUpdatesLightAndReflectionProbe() {
	let bakeCallCount = 0;
	let faceCaptureCount = 0;
	const runtime = new ProbeCaptureRuntime({
		bakeEnvironmentIBL: async () => {
			bakeCallCount++;
			return createBakedEnvironment(2);
		},
	});
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
	assert.equal(bakeCallCount, 1);
	assert.ok(lightProbe.sh[0].r > 0);
	assert.ok(reflectionProbe.prefilteredMap);
}

async function testSharedCaptureSkipsStaleLightProbeResult() {
	const deferredBake = createDeferred();
	let bakeCallCount = 0;
	const runtime = new ProbeCaptureRuntime({
		bakeEnvironmentIBL: () => {
			bakeCallCount++;
			return deferredBake.promise;
		},
	});
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
		() => bakeCallCount >= 1
	);
	assert.equal(bakeCallCount, 1);

	lightProbe.position.set(1, 0, 0);
	scene.updateWorldMatrices();
	deferredBake.resolve(createBakedEnvironment(3));
	await flushAsyncTasks();

	assert.equal(lightProbe.sh[0].r, 0);
	assert.ok(reflectionProbe.prefilteredMap);
}

async function run() {
	await testLightProbeManualCaptureProjectsSHWithoutPrefilterBake();
	await testSharedCaptureUpdatesLightAndReflectionProbe();
	await testSharedCaptureSkipsStaleLightProbeResult();
	console.log("Probe capture runtime tests passed");
}

await run();
