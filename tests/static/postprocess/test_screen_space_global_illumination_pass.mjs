import assert from "node:assert/strict";

import { CameraType } from "../../../src/cameras/Camera.ts";
import {
	DEFAULT_SSGI_OPTIONS,
	ScreenSpaceGlobalIlluminationPass,
	resolveSSGIHistoryDescriptors,
	resolveSSGIHistoryValid,
	resolveSSGIOptions,
	resolveSSGITransientDescriptors,
	writeSSGIComposeParams,
	writeSSGIDenoiseParams,
	writeSSGITraceParams,
} from "../../../src/postprocess/index.ts";
import { WebGPUPostProcessRuntime } from "../../../src/backends/webgpu/WebGPUPostProcessRuntime.ts";
import {
	FakeBackend,
	FakeEncoder,
	assertClose,
	createTexture,
} from "../../helpers/webgpu_postprocess_runtime_test_helpers.mjs";

function createGBuffer(width = 32, height = 16) {
	return {
		width,
		height,
		normalSpace: "world",
		depthEncoding: "linear-view-z",
		motionEncoding: "ndc-delta",
		channels: {
			depth: {},
			normal: {},
			albedo: {},
			metallic: {},
			motion: {},
		},
		worldPosition: { source: "derived", available: true },
	};
}

function createRequest(options = {}, historyValid = false, cameraType = CameraType.Perspective) {
	return {
		frameContext: {
			viewCamera: {
				type: cameraType,
				fov: 60,
				aspectRatio: 2,
				near: 0.1,
				far: 100,
			},
		},
		gBuffer: createGBuffer(),
		histories: {
			ssgi: { valid: historyValid },
			motion: { valid: historyValid },
		},
		passId: "ssgi",
		options,
		startPassId: null,
	};
}

function createHarness() {
	const backend = new FakeBackend();
	const warnings = [];
	const runtime = new WebGPUPostProcessRuntime(
		backend,
		(key, message) => warnings.push([key, message])
	);
	const encoder = new FakeEncoder();
	const textures = {
		scene: createTexture(32, 16, "scene"),
		output: createTexture(32, 16, "output"),
		albedo: createTexture(32, 16, "g-albedo"),
		normal: createTexture(32, 16, "g-normal"),
		motionDepth: createTexture(32, 16, "g-motion-depth"),
		hiZ: createTexture(32, 16, "hiz"),
		ssgiRead: createTexture(16, 8, "ssgi-read"),
		ssgiWrite: createTexture(16, 8, "ssgi-write"),
		motionRead: createTexture(32, 16, "motion-read"),
		motionWrite: createTexture(32, 16, "motion-write"),
		denoiseA: createTexture(16, 8, "denoise-a"),
		denoiseB: createTexture(16, 8, "denoise-b"),
	};
	const copied = [];
	let failMotionCopy = false;
	const resources = {
		color: { input: textures.scene, output: textures.output },
		getGBuffer(semantic) {
			return {
				albedo: textures.albedo,
				depth: textures.motionDepth,
				metallic: textures.normal,
				motion: textures.motionDepth,
				normal: textures.normal,
			}[semantic] ?? null;
		},
		getHistory(id) {
			return {
				ssgi: { read: textures.ssgiRead, write: textures.ssgiWrite },
				motion: { read: textures.motionRead, write: textures.motionWrite },
			}[id] ?? { read: null, write: null };
		},
		getTransient(id) {
			return {
				"ssgi:denoise-a": textures.denoiseA,
				"ssgi:denoise-b": textures.denoiseB,
			}[id] ?? null;
		},
		getShared(id) {
			return id === "backend:frame-hiz" ? textures.hiZ : null;
		},
		async copyGBufferToHistory(semantic, id) {
			if (failMotionCopy) {
				throw new Error("motion copy failed");
			}
			copied.push([semantic, id]);
		},
	};
	const context = {
		encoder,
		targets: {
			sceneColor: textures.scene,
			postPing: textures.output,
			postPong: createTexture(32, 16, "pong"),
			gAlbedoAlpha: textures.albedo,
			gNormalRoughMetal: textures.normal,
			gMotionDepth: textures.motionDepth,
		},
		shared: runtime,
		frameBinding: { label: "frame-binding" },
		resources,
	};
	return {
		backend,
		context,
		copied,
		encoder,
		resources,
		runtime,
		setFailMotionCopy(value) {
			failMotionCopy = value;
		},
		textures,
		warnings,
	};
}

function getSSGIResources(backend) {
	return {
		pipelines: backend.computePipelines.filter(
			(resource) => resource.label.startsWith("WebGPUSSGI")
		),
		buffers: backend.buffers.filter(
			(resource) => resource.label.startsWith("WebGPUSSGI")
		),
		modules: backend.shaderModules.filter(
			(resource) => resource.label === "WebGPUSSGIShader"
		),
	};
}

function testExecutionDeclaration() {
	const pass = new ScreenSpaceGlobalIlluminationPass({ enabled: true });
	const implementation = pass.getImplementation("webgpu");
	const declaration = implementation.describeExecution({
		options: { downsample: 2 },
	});

	assert.equal(pass.id, "ssgi");
	assert.equal(pass.getImplementation("software"), null);
	assert.deepEqual(
		declaration.gBuffer.map((entry) => entry.semantic),
		["depth", "normal", "albedo", "metallic", "motion"]
	);
	assert.deepEqual(
		declaration.shared.map((entry) => entry.id),
		["backend:frame-hiz"]
	);
	assert.deepEqual(
		declaration.histories.map((entry) => ({
			id: entry.descriptor.id,
			widthScale: entry.descriptor.widthScale ?? 1,
			heightScale: entry.descriptor.heightScale ?? 1,
			usage: entry.descriptor.usage,
			writeAccess: entry.write.map((use) => use.access),
		})),
		[
			{
				id: "ssgi",
				widthScale: 0.5,
				heightScale: 0.5,
				usage: ["sampled", "storage", "render-target"],
				writeAccess: ["write", "read"],
			},
			{
				id: "motion",
				widthScale: 1,
				heightScale: 1,
				usage: ["sampled", "copy-dst", "render-target"],
				writeAccess: ["write"],
			},
		]
	);
	assert.deepEqual(
		declaration.transients.map((entry) => ({
			id: entry.descriptor.id,
			widthScale: entry.descriptor.widthScale,
			heightScale: entry.descriptor.heightScale,
			access: entry.uses.map((use) => use.access),
		})),
		[
			{
				id: "ssgi:denoise-a",
				widthScale: 0.5,
				heightScale: 0.5,
				access: ["write", "read"],
			},
			{
				id: "ssgi:denoise-b",
				widthScale: 0.5,
				heightScale: 0.5,
				access: ["write", "read"],
			},
		]
	);
	assert.equal(pass.schedule.incremental.grade, "cinematic");
	assert.equal(pass.schedule.incremental.firstPass, "ssgi");
	assert.equal(pass.schedule.incremental.inflationRadius, 8);
	pass.destroy();
}

function testOptionsAndParameterPacking() {
	assert.deepEqual(resolveSSGIOptions(), DEFAULT_SSGI_OPTIONS);
	const options = resolveSSGIOptions({
		downsample: 3,
		raysPerPixel: 99.8,
		maxSteps: 3.9,
		binarySearchSteps: 99,
		maxDistance: -1,
		thickness: Number.POSITIVE_INFINITY,
		normalBias: -2,
		distanceFalloffExponent: 99,
		edgeFade: -1,
		intensity: -1,
		historyWeight: 4,
		disocclusionDepthThreshold: 0,
		historyClamp: 99,
		denoiseRadius: 9.8,
		denoiseDepthPhi: 0,
		denoiseNormalPhi: Number.NaN,
	});
	assert.deepEqual(options, {
		downsample: 4,
		raysPerPixel: 4,
		maxSteps: 4,
		binarySearchSteps: 8,
		maxDistance: 8,
		thickness: 0.2,
		normalBias: 0,
		distanceFalloffExponent: 8,
		edgeFade: 0,
		intensity: 0,
		historyWeight: 0.98,
		disocclusionDepthThreshold: 0.001,
		historyClamp: 16,
		denoiseRadius: 4,
		denoiseDepthPhi: 24,
		denoiseNormalPhi: 16,
	});

	const trace = writeSSGITraceParams(
		new Float32Array(20),
		16,
		8,
		resolveSSGIOptions(),
		5,
		true,
		7
	);
	assert.equal(trace.length, 20);
	assertClose(trace[0], 1 / 16);
	assertClose(trace[1], 1 / 8);
	assert.equal(trace[2], 8);
	assert.equal(trace[7], 1);
	assert.equal(trace[8], 24);
	assert.equal(trace[9], 3);
	assert.equal(trace[10], 5);
	assert.equal(trace[11], 7);
	assert.equal(trace[15], 1);

	const denoise = writeSSGIDenoiseParams(
		new Float32Array(8),
		16,
		8,
		resolveSSGIOptions()
	);
	assert.deepEqual(Array.from(denoise.slice(0, 5)), [
		1 / 16,
		1 / 8,
		2,
		24,
		16,
	]);

	const compose = writeSSGIComposeParams(
		new Float32Array(8),
		32,
		16,
		16,
		8,
		resolveSSGIOptions()
	);
	assert.deepEqual(
		Array.from(compose.slice(0, 4)),
		[1 / 32, 1 / 16, 1 / 16, 1 / 8]
	);
	assertClose(compose[4], 0.35);
	assert.deepEqual(Array.from(compose.slice(5)), [24, 16, 0]);
	assert.throws(
		() => writeSSGITraceParams(
			new Float32Array(4),
			1,
			1,
			resolveSSGIOptions(),
			0,
			false,
			0
		),
		/20 floats/
	);
	assert.equal(
		resolveSSGIHistoryValid(
			{ ssgi: { valid: true }, motion: { valid: true } },
			true
		),
		true
	);
	assert.equal(
		resolveSSGIHistoryValid(
			{ ssgi: { valid: true }, motion: { valid: false } },
			true
		),
		false
	);

	const request = { options: { downsample: 4 } };
	assert.deepEqual(
		resolveSSGIHistoryDescriptors(request).map((descriptor) => ({
			id: descriptor.id,
			widthScale: descriptor.widthScale,
		})),
		[
			{ id: "ssgi", widthScale: 0.25 },
			{ id: "motion", widthScale: undefined },
		]
	);
	assert.deepEqual(
		resolveSSGITransientDescriptors(request).map((descriptor) => ({
			id: descriptor.id,
			widthScale: descriptor.widthScale,
		})),
		[
			{ id: "ssgi:denoise-a", widthScale: 0.25 },
			{ id: "ssgi:denoise-b", widthScale: 0.25 },
		]
	);
}

async function testFourStageExecutionAndTemporalContinuity() {
	const harness = createHarness();
	const pass = new ScreenSpaceGlobalIlluminationPass({ enabled: true });
	const implementation = pass.getImplementation("webgpu");
	const firstResult = await implementation.execute(
		createRequest({}, true),
		harness.context
	);

	assert.deepEqual(firstResult, {
		ran: true,
		updatedHistoryIds: ["ssgi", "motion"],
	});
	assert.deepEqual(harness.copied, [["motion", "motion"]]);
	const gpuResources = getSSGIResources(harness.backend);
	assert.equal(gpuResources.modules.length, 1);
	assert.equal(gpuResources.pipelines.length, 2);
	assert.deepEqual(
		gpuResources.pipelines.map((pipeline) => pipeline.desc.compute.entryPoint),
		["csTraceTemporal", "csCompose"]
	);
	assert.deepEqual(
		gpuResources.buffers.map((buffer) => [buffer.label, buffer.size]),
		[
			["WebGPUSSGITraceParams", 80],
			["WebGPUSSGIComposeParams", 32],
		]
	);
	assert.ok(
		gpuResources.modules[0].desc.code.includes("fn cosineHemisphereDirection")
	);
	assert.ok(
		gpuResources.modules[0].desc.code.includes("fn csTraceTemporal")
	);
	assert.equal(
		gpuResources.modules[0].desc.code.includes("csDenoiseHorizontal"),
		false
	);

	const ssgiBindings = harness.backend.bindingGroups.filter(
		(group) => group.label.startsWith("WebGPUSSGI")
	);
	assert.deepEqual(
		ssgiBindings.map((group) => group.desc.entries.length),
		[9, 8]
	);
	assert.equal(ssgiBindings[0].desc.entries[0].resource, harness.textures.scene);
	assert.equal(ssgiBindings[0].desc.entries[3].resource, harness.textures.hiZ);
	assert.equal(
		ssgiBindings[0].desc.entries[4].resource,
		harness.textures.ssgiRead
	);
	assert.equal(
		ssgiBindings[0].desc.entries[8].resource,
		harness.textures.ssgiWrite
	);
	assert.equal(
		ssgiBindings[1].desc.entries[2].resource,
		harness.textures.albedo
	);
	assert.equal(
		ssgiBindings[1].desc.entries[7].resource,
		harness.textures.output
	);
	const denoiseBindings = harness.backend.bindingGroups.filter(
		(group) => group.label.startsWith("WebGPUDenoise_ssgi")
	);
	assert.equal(denoiseBindings.length, 6);
	assert.equal(
		denoiseBindings.every((group) => group.desc.entries.length === 6),
		true
	);

	assert.deepEqual(
		harness.encoder.calls.filter((call) => call[0] === "beginComputePass"),
		[
			["beginComputePass", "WebGPUSSGI_TraceTemporal"],
			["beginComputePass", "WebGPUDenoise_ssgi_quality_H_1"],
			["beginComputePass", "WebGPUDenoise_ssgi_quality_V_1"],
			["beginComputePass", "WebGPUDenoise_ssgi_quality_H_2"],
			["beginComputePass", "WebGPUDenoise_ssgi_quality_V_2"],
			["beginComputePass", "WebGPUDenoise_ssgi_quality_H_4"],
			["beginComputePass", "WebGPUDenoise_ssgi_quality_V_4"],
			["beginComputePass", "WebGPUSSGI_Compose"],
		]
	);
	assert.deepEqual(
		harness.encoder.calls.filter((call) => call[0] === "dispatchWorkgroups"),
		[
			["dispatchWorkgroups", 2, 1, 1],
			["dispatchWorkgroups", 2, 1, 1],
			["dispatchWorkgroups", 2, 1, 1],
			["dispatchWorkgroups", 2, 1, 1],
			["dispatchWorkgroups", 2, 1, 1],
			["dispatchWorkgroups", 2, 1, 1],
			["dispatchWorkgroups", 2, 1, 1],
			["dispatchWorkgroups", 4, 2, 1],
		]
	);
	assert.deepEqual(
		harness.encoder.calls.filter((call) => call[0] === "setBindingGroup"),
		[
			["setBindingGroup", 0, "WebGPUSSGI_TraceBinding"],
			["setBindingGroup", 1, "frame-binding"],
			["setBindingGroup", 0, "WebGPUDenoise_ssgi_horizontal_0"],
			["setBindingGroup", 0, "WebGPUDenoise_ssgi_vertical_0"],
			["setBindingGroup", 0, "WebGPUDenoise_ssgi_horizontal_1"],
			["setBindingGroup", 0, "WebGPUDenoise_ssgi_vertical_1"],
			["setBindingGroup", 0, "WebGPUDenoise_ssgi_horizontal_2"],
			["setBindingGroup", 0, "WebGPUDenoise_ssgi_vertical_2"],
			["setBindingGroup", 0, "WebGPUSSGI_ComposeBinding"],
		]
	);
	assert.equal(gpuResources.buffers[0].lastWrite[15], 0);

	harness.context.encoder = new FakeEncoder();
	await implementation.execute(createRequest({}, true), harness.context);
	assert.equal(gpuResources.buffers[0].lastWrite[15], 1);
	const bindingsAfterSecondFrame = harness.backend.createBindingGroupCalls;
	harness.context.encoder = new FakeEncoder();
	await implementation.execute(createRequest({}, true), harness.context);
	assert.equal(
		harness.backend.createBindingGroupCalls,
		bindingsAfterSecondFrame
	);

	const callsBeforeOrthographic = harness.context.encoder.calls.length;
	const orthographic = await implementation.execute(
		createRequest({}, true, CameraType.Orthographic),
		harness.context
	);
	assert.deepEqual(orthographic, { ran: false });
	assert.equal(harness.context.encoder.calls.length, callsBeforeOrthographic);
	assert.equal(harness.copied.length, 3);
	assert.deepEqual(harness.warnings, [[
		"webgpu-ssgi-orthographic-disabled",
		"WebGPU SSGI is disabled for orthographic cameras.",
	]]);

	harness.context.encoder = new FakeEncoder();
	await implementation.execute(createRequest({}, true), harness.context);
	assert.equal(gpuResources.buffers[0].lastWrite[15], 0);

	pass.destroy();
	harness.runtime.destroy();
}

async function testFailurePathsAndResourceInvalidation() {
	const harness = createHarness();
	const pass = new ScreenSpaceGlobalIlluminationPass({ enabled: true });
	const implementation = pass.getImplementation("webgpu");
	const missingContext = {
		...harness.context,
		resources: {
			...harness.resources,
			getShared: () => null,
		},
	};
	assert.deepEqual(
		await implementation.execute(createRequest({}, true), missingContext),
		{ ran: false }
	);
	assert.deepEqual(harness.copied, []);
	assert.equal(harness.encoder.calls.length, 0);

	await implementation.execute(createRequest({}, true), harness.context);
	harness.context.encoder = new FakeEncoder();
	harness.setFailMotionCopy(true);
	await assert.rejects(
		implementation.execute(createRequest({}, true), harness.context),
		/motion copy failed/
	);
	harness.setFailMotionCopy(false);
	harness.context.encoder = new FakeEncoder();
	await implementation.execute(createRequest({}, true), harness.context);
	const resourcesBeforeInvalidate = getSSGIResources(harness.backend);
	assert.equal(resourcesBeforeInvalidate.buffers[0].lastWrite[15], 0);

	const bindingDestroyCount = harness.backend.bindingGroupDestroyCalls;
	implementation.invalidate();
	assert.equal(
		resourcesBeforeInvalidate.pipelines.every((pipeline) => pipeline.destroyed),
		true
	);
	assert.equal(
		resourcesBeforeInvalidate.buffers.every((buffer) => buffer.destroyed),
		true
	);
	assert.equal(resourcesBeforeInvalidate.modules[0].destroyed, true);
	assert.ok(harness.backend.bindingGroupDestroyCalls > bindingDestroyCount);

	await implementation.warmup(harness.context);
	const resourcesAfterWarmup = getSSGIResources(harness.backend);
	assert.equal(resourcesAfterWarmup.pipelines.length, 4);
	assert.equal(resourcesAfterWarmup.buffers.length, 4);
	assert.equal(resourcesAfterWarmup.modules.length, 2);
	assert.equal(
		resourcesAfterWarmup.pipelines.slice(2).every(
			(pipeline) => !pipeline.destroyed
		),
		true
	);

	implementation.destroy();
	assert.equal(
		resourcesAfterWarmup.pipelines.every((pipeline) => pipeline.destroyed),
		true
	);
	assert.equal(
		resourcesAfterWarmup.buffers.every((buffer) => buffer.destroyed),
		true
	);
	assert.equal(
		resourcesAfterWarmup.modules.every((module) => module.destroyed),
		true
	);
	pass.destroy();
	harness.runtime.destroy();
}

testExecutionDeclaration();
testOptionsAndParameterPacking();
await testFourStageExecutionAndTemporalContinuity();
await testFailurePathsAndResourceInvalidation();
console.log("ScreenSpaceGlobalIlluminationPass tests passed");
