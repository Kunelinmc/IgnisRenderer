import assert from "node:assert/strict";
import { Scene } from "../src/core/Scene.ts";
import { Texture } from "../src/core/Texture.ts";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { ReflectionProbe } from "../src/lights/ReflectionProbe.ts";
import {
	EnvironmentIBLUpdateRuntime,
	normalizeEnvironmentIBLUpdateOptions,
} from "../src/pipeline/EnvironmentIBLUpdateRuntime.ts";

function createEnvironmentTexture(width = 16, height = 8, seed = 1) {
	const data = new Uint8ClampedArray(width * height * 4);
	for (let i = 0; i < data.length; i += 4) {
		const pixel = i >> 2;
		data[i] = (pixel * (13 + seed * 3)) % 255;
		data[i + 1] = (pixel * (19 + seed * 5)) % 255;
		data[i + 2] = (pixel * (29 + seed * 7)) % 255;
		data[i + 3] = 255;
	}
	return new Texture(data, width, height, "sRGB");
}

function createPrefilteredTexture(seed = 1, width = 8, height = 4, mipLevels = 3) {
	const texture = new Texture(null, width, height, "HDR");
	texture.mipmaps = [];
	for (let level = 0; level < mipLevels; level++) {
		const mipWidth = Math.max(1, width >> level);
		const mipHeight = Math.max(1, height >> level);
		const mip = new Float32Array(mipWidth * mipHeight * 4);
		for (let i = 0; i < mip.length; i += 4) {
			mip[i] = seed * (1 + level * 0.1);
			mip[i + 1] = seed * 0.5 * (1 + level * 0.1);
			mip[i + 2] = seed * 0.25 * (1 + level * 0.1);
			mip[i + 3] = 1;
		}
		texture.mipmaps.push(mip);
	}
	texture.data = texture.mipmaps[0];
	return texture;
}

function createBakedEnvironment(seed = 1) {
	return {
		sh: Array.from({ length: 16 }, (_, index) => ({
			r: seed * (index + 1) * 0.1,
			g: seed * (index + 1) * 0.05,
			b: seed * (index + 1) * 0.025,
		})),
		prefilteredMap: createPrefilteredTexture(seed),
	};
}

async function flushAsyncTasks() {
	await Promise.resolve();
	await new Promise((resolve) => setTimeout(resolve, 0));
}

async function runUntilIdle(
	runtime,
	contextFactory,
	maxFrames = 128
) {
	let sawComplete = false;
	for (let frame = 0; frame < maxFrames; frame++) {
		const result = runtime.execute(contextFactory());
		if (result.dirtyReason === "environment-ibl-complete") {
			sawComplete = true;
		}
		if (!result.inProgress && sawComplete) {
			return {
				sawComplete,
			};
		}
		await flushAsyncTasks();
	}
	return {
		sawComplete,
	};
}

async function testManualTriggerAndTemporalBlending() {
	let bakeCallCount = 0;
	const runtime = new EnvironmentIBLUpdateRuntime({
		bakeEnvironmentIBL: async () => {
			bakeCallCount++;
			return createBakedEnvironment(1.5);
		},
	});
	const scene = new Scene();
	scene.environment.iblTexture = createEnvironmentTexture();
	const probe = scene.add(new LightProbe({}));
	const manualLightProbe = scene.add(new LightProbe({ source: "manual" }));
	const capturedLightProbe = scene.add(
		new LightProbe({ source: "capturedScene" })
	);
	const initialEnvironmentMap = createPrefilteredTexture(0.05);
	const manualMap = createPrefilteredTexture(0.8);
	const environmentProbe = scene.add(
		new ReflectionProbe({
			source: "environment",
			prefilteredMap: initialEnvironmentMap,
		})
	);
	const manualProbe = scene.add(
		new ReflectionProbe({
			source: "manual",
			prefilteredMap: manualMap,
		})
	);

	const options = normalizeEnvironmentIBLUpdateOptions({
		enabled: true,
		autoUpdate: false,
		mipsPerFrame: 1,
		temporalBlendFactor: 0.5,
		temporalBlendEpsilon: 1e-4,
		acceleration: "cpu",
	});

	let requestToken = 0;
	const initial = runtime.execute({
		scene,
		requestToken,
		options,
	});
	assert.equal(initial.inProgress, false);
	assert.equal(initial.dirtyReason, null);
	assert.equal(bakeCallCount, 0);

	requestToken = 1;
	const started = runtime.execute({
		scene,
		requestToken,
		options,
	});
	assert.equal(started.inProgress, true);
	assert.equal(started.dirtyReason, "environment-ibl");
	await flushAsyncTasks();

	const settled = await runUntilIdle(runtime, () => ({
		scene,
		requestToken,
		options,
	}));
	assert.equal(settled.sawComplete, true);
	assert.equal(bakeCallCount, 1);

	assert.notEqual(environmentProbe.prefilteredMap, initialEnvironmentMap);
	assert.equal(manualProbe.prefilteredMap, manualMap);
	assert.ok(probe.sh[0].r > 0);
	assert.ok(probe.sh[0].g > 0);
	assert.ok(probe.sh[0].b > 0);
	assert.equal(manualLightProbe.sh[0].r, 0);
	assert.equal(capturedLightProbe.sh[0].r, 0);
}

async function testAutoTriggerOnEnvironmentSignatureChange() {
	let bakeCallCount = 0;
	const runtime = new EnvironmentIBLUpdateRuntime({
		bakeEnvironmentIBL: async () => {
			bakeCallCount++;
			return createBakedEnvironment(0.9 + bakeCallCount * 0.1);
		},
	});
	const scene = new Scene();
	scene.environment.iblTexture = createEnvironmentTexture(16, 8, 1);
	scene.add(new LightProbe({}));

	const options = normalizeEnvironmentIBLUpdateOptions({
		enabled: true,
		autoUpdate: true,
		mipsPerFrame: 4,
		temporalBlendFactor: 1,
		temporalBlendEpsilon: 1e-6,
		acceleration: "cpu",
	});
	const requestToken = 0;

	await runUntilIdle(runtime, () => ({
		scene,
		requestToken,
		options,
	}));
	assert.equal(bakeCallCount, 1);

	const stable = runtime.execute({
		scene,
		requestToken,
		options,
	});
	assert.equal(stable.inProgress, false);

	scene.environment.iblTexture?.markNeedsUpdate();
	const restarted = runtime.execute({
		scene,
		requestToken,
		options,
	});
	assert.equal(restarted.inProgress, true);
	assert.equal(restarted.dirtyReason, "environment-ibl");
	await flushAsyncTasks();

	await runUntilIdle(runtime, () => ({
		scene,
		requestToken,
		options,
	}));
	assert.equal(bakeCallCount, 2);
}

async function run() {
	await testManualTriggerAndTemporalBlending();
	await testAutoTriggerOnEnvironmentSignatureChange();
	console.log("Environment IBL update runtime tests passed");
}

await run();
