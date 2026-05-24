import assert from "node:assert/strict";

import {
	PostProcessPipeline,
	ScreenSpaceGlobalIlluminationPass,
	createSSGIKernelParams,
	resolveSSGIOptions,
} from "../src/postprocess/index.ts";
import { CameraType } from "../src/cameras/Camera.ts";
import { WebGPUPostProcessRuntime } from "../src/renderers/webgpu/WebGPUPostProcessRuntime.ts";
import {
	FakeBackend,
	FakeEncoder,
	assertClose,
	createTexture,
} from "./helpers/webgpu_postprocess_runtime_test_helpers.mjs";
import {
	ALL_POST_PROCESS_CAPABILITIES,
	createResolvedPostProcess,
} from "./helpers/postprocess.mjs";

const SSGI_ONLY_CAPABILITIES = Object.fromEntries(
	Object.keys(ALL_POST_PROCESS_CAPABILITIES).map((key) => [key, key === "ssgi"])
);

function createWebGPUGBuffer(width = 32, height = 16) {
	return {
		width,
		height,
		normalSpace: "world",
		depthEncoding: "hardware",
		channels: {
			color: {},
			depth: {},
			normal: {},
			albedo: {},
		},
		worldPosition: {
			source: "derived",
			available: true,
		},
	};
}

function createIncremental(width, height) {
	return {
		enabled: false,
		forceFullFrame: true,
		dirtyRects: [{ x: 0, y: 0, width, height }],
		dirtyTileSize: 64,
		dirtyTileColumns: 1,
		dirtyTileRows: 1,
		dirtyTiles: [0],
		dirtyAreaRatio: 1,
		firstPass: null,
		postProcessStartPass: null,
		reasonMask: 0,
		temporalHistoryReset: false,
	};
}

function createRequest(frameContext) {
	const pass = new ScreenSpaceGlobalIlluminationPass({ enabled: true });
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: createWebGPUGBuffer(),
		histories: {},
		pass,
		passId: "ssgi",
		options: frameContext.postProcess.getOptions("ssgi"),
		startPassId: null,
	};
}

async function testSSGIDescriptorAndWebGPUExecution() {
	const pass = new ScreenSpaceGlobalIlluminationPass({ enabled: true });
	assert.equal(pass.id, "ssgi");
	assert.deepEqual(
		pass.getRequirements({}).gBuffer,
		["color", "depth", "normal", "albedo"]
	);
	assert.equal(
		typeof pass.getImplementation("webgpu").execute,
		"function"
	);
	assert.equal(
		pass.getImplementation("software"),
		null
	);

	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(32, 16, "scene");
	const postPing = createTexture(32, 16, "ping");
	const postPong = createTexture(32, 16, "pong");
	const gAlbedoAlpha = createTexture(32, 16, "g-albedo");
	const gNormalRoughMetal = createTexture(32, 16, "g-normal");
	const gMotionDepth = createTexture(32, 16, "g-motion-depth");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
		gAlbedoAlpha,
		gNormalRoughMetal,
		gMotionDepth,
	};
	const frameContext = {
		features: {},
		postProcess: createResolvedPostProcess({
			ssgi: {
				enabled: true,
				options: {
					radius: 4,
					intensity: 0.5,
					falloff: 1.8,
					depthPhi: 1.4,
					normalPhi: 2.5,
					albedoBoost: 1.2,
					samples: 24,
				},
			},
		}),
	};
	const implementation = new ScreenSpaceGlobalIlluminationPass({
		enabled: true,
	}).getImplementation("webgpu");
	const result = await implementation.execute(createRequest(frameContext), {
		encoder,
		targets,
		shared: runtime.sharedContext,
		publishColorTarget(texture) {
			targets.sceneColor = texture;
		},
	});

	assert.equal(result.ran, true);
	assert.equal(backend.samplers.length, 1);
	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.shaderModules[0].label, "WebGPUSSGIShader");
	assert.ok(backend.shaderModules[0].desc.code.includes("SAMPLE_OFFSETS"));
	assert.ok(backend.shaderModules[0].desc.code.includes("MAX_SSGI_SAMPLES"));
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.computePipelines[0].label, "WebGPUSSGIPipeline");
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.buffers[0].desc.label, "WebGPUSSGIParams");
	assert.equal(backend.buffers[0].desc.size, 48);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroups[0].desc.entries.length, 7);
	assert.equal(backend.bindingGroups[0].desc.entries[0].resource, sceneColorMain);
	assert.equal(backend.bindingGroups[0].desc.entries[1].resource, gAlbedoAlpha);
	assert.equal(
		backend.bindingGroups[0].desc.entries[2].resource,
		gNormalRoughMetal
	);
	assert.equal(backend.bindingGroups[0].desc.entries[3].resource, gMotionDepth);
	assert.equal(backend.bindingGroups[0].desc.entries[6].resource, postPong);

	const params = backend.buffers[0].lastWrite;
	assert.equal(params.length, 12);
	assertClose(params[0], 1 / 32);
	assertClose(params[1], 1 / 16);
	assertClose(params[2], 4);
	assertClose(params[3], 0.5);
	assertClose(params[4], 1.8);
	assertClose(params[5], 1.4);
	assertClose(params[6], 2.5);
	assertClose(params[7], 1.2);
	assertClose(params[8], 16);

	assert.deepEqual(encoder.calls, [
		["beginComputePass", "WebGPUSSGI"],
		["setComputePipeline", "WebGPUSSGIPipeline"],
		["setBindingGroup", 0, "WebGPUSSGI_Binding"],
		["dispatchWorkgroups", 4, 2, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPong);
}

async function testSSGIPipelineUsesWebGPUImplementation() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const targets = {
		sceneColor: createTexture(32, 16, "scene"),
		postPing: createTexture(32, 16, "ping"),
		postPong: createTexture(32, 16, "pong"),
		gAlbedoAlpha: createTexture(32, 16, "g-albedo"),
		gNormalRoughMetal: createTexture(32, 16, "g-normal"),
		gMotionDepth: createTexture(32, 16, "g-motion-depth"),
	};
	const frameContext = {
		camera: {
			type: CameraType.Perspective,
			fov: 60,
			aspectRatio: 2,
			near: 0.1,
			far: 100,
		},
		features: {},
		attachments: {
			width: 32,
			height: 16,
		},
		postProcess: createResolvedPostProcess(
			{ ssgi: { enabled: true } },
			SSGI_ONLY_CAPABILITIES,
			"webgpu"
		),
		incremental: createIncremental(32, 16),
		transient: new Map(),
	};
	const executor = {
		backend: "webgpu",
		fallbackCalls: [],
		createResource(desc) {
			return {
				id: desc.id,
				backend: "webgpu",
				width: desc.width,
				height: desc.height,
				format: desc.format,
				resource: createTexture(desc.width, desc.height, desc.id),
			};
		},
		destroyResource() {},
		getPassExecutionContext(passId) {
			if (passId !== "ssgi") {
				return undefined;
			}
			return {
				encoder,
				targets,
				shared: runtime.sharedContext,
				publishColorTarget(texture) {
					targets.sceneColor = texture;
				},
			};
		},
		executePass(passId) {
			this.fallbackCalls.push(passId);
			return { ran: true };
		},
	};
	const pipeline = new PostProcessPipeline();
	const result = await pipeline.execute({
		frameContext,
		executor,
		gBuffer: createWebGPUGBuffer(),
	});

	assert.deepEqual(result.executedPassIds, ["ssgi"]);
	assert.deepEqual(executor.fallbackCalls, []);
	assert.deepEqual(encoder.calls.slice(0, 2), [
		["beginComputePass", "WebGPUSSGI"],
		["setComputePipeline", "WebGPUSSGIPipeline"],
	]);
}

function testSSGIOptionHelpersClampAndPackParams() {
	const options = resolveSSGIOptions({
		samples: 99,
		radius: -3,
		intensity: -1,
		falloff: 0,
		depthPhi: 0,
		normalPhi: 0,
		albedoBoost: -1,
	});
	assert.equal(options.samples, 16);
	assert.equal(options.radius, 1);
	assert.equal(options.intensity, 0);
	assert.equal(options.falloff, 0.1);
	assert.equal(options.depthPhi, 0.01);
	assert.equal(options.normalPhi, 0.1);
	assert.equal(options.albedoBoost, 0);

	const params = createSSGIKernelParams(64, 32, options);
	assert.equal(params.length, 12);
	assert.equal(params[0], 1 / 64);
	assert.equal(params[1], 1 / 32);
	assert.equal(params[2], 1);
	assert.equal(params[8], 16);
}

await testSSGIDescriptorAndWebGPUExecution();
await testSSGIPipelineUsesWebGPUImplementation();
testSSGIOptionHelpersClampAndPackParams();
console.log("ScreenSpaceGlobalIlluminationPass tests passed");
