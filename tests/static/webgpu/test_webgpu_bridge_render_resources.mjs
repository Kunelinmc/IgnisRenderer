import assert from "node:assert/strict";
import {
	WebGPUFrameServiceOwner as WebGPURenderResources
} from "../../../src/backends/webgpu/WebGPUFrameServiceOwner.ts";
import {
	WebGPUFrameOrchestrator as WebGPUFrameExecutor
} from "../../../src/backends/webgpu/rendergraph/WebGPUFrameOrchestrator.ts";
import {
	ShaderSource
} from "../../../src/shaders/ShaderSource.ts";
import {
	WEBGPU_FRAME_CAMERA_UNIFORM_FLOATS,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_FLOATS,
	WEBGPU_FRAME_LIGHT_UNIFORM_FLOATS,
	WEBGPU_FRAME_SHADOW_UNIFORM_FLOATS
} from "../../../src/backends/webgpu/index.ts";
import {
	FramePacketContributorRegistry
} from "../../../src/pipeline/FramePacketContributorRegistry.ts";
import {
	BufferUsage
} from "../../../src/backends/types.ts";
import {
	PointLight
} from "../../../src/lights/PointLight.ts";
import {
	PBRMaterial
} from "../../../src/materials/PBRMaterial.ts";
import {
	Logger
} from "../../../src/foundation/Logger.ts";
import {
	MAX_POINT_LIGHTS
} from "../../../src/backends/constants.ts";
import {
	WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
	WEBGPU_MODEL_BINDING_JOINT_MATRICES,
	WEBGPU_MODEL_BINDING_MORPH_NORMAL,
	WEBGPU_MODEL_BINDING_MORPH_POSITION,
	WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
	WEBGPU_MODEL_BINDING_ANISOTROPY_TEXTURE,
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT
} from "../../../src/backends/webgpu/constants.ts";
import {
	createWebGPUComputeFacade
} from "../../../src/backends/webgpu/ComputeFacade.ts";

import {
	FakeWebGPUBackend as FakeBackend
} from "../../helpers/fakes.mjs";

import {
	createFrame,
	createFrameContextWithFeatures,
	createMainFrameOptions,
	createModel,
	createPacket
} from "../../helpers/webgpu-bridge.mjs";
const previousGPUShaderStage = globalThis.GPUShaderStage;
globalThis.GPUShaderStage = {
	...(previousGPUShaderStage ?? {}),
	VERTEX: previousGPUShaderStage?.VERTEX ?? 1,
	FRAGMENT: previousGPUShaderStage?.FRAGMENT ?? 2,
	COMPUTE: previousGPUShaderStage?.COMPUTE ?? 4,
};
ShaderSource.resetConfiguration();
Logger.reset();

function testRenderResourcesConsumeInjectedComputeFacade() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);

	assert.equal(backend.getComputeFacadeCalls, 0);
	assert.equal(
		typeof resources._computeFacade.createComputePipeline,
		"function"
	);

	resources.destroy();
}

function testRenderResourcesLeaveShaderRuntimeSubscriptionToBackend() {
	const backend = new FakeBackend();
	let listenerCount = 0;
	backend.shaderRuntime = {
		revision: 1,
		getMode: () => "strict",
		onDidChange: () => {
			listenerCount++;
			return () => {
				listenerCount--;
			};
		},
	};
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	assert.equal(listenerCount, 0);

	resources.destroy();
	assert.equal(listenerCount, 0);
}

async function testRenderResourcesLogPointLightLimitOnlyOnce() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);
	const warnings = [];
	Logger.configure({
		level: "warn",
		sink: { warn: (...args) => warnings.push(args) },
		resetOnceKeys: true,
	});

	try {
		await resources.init();
		const model = createModel([new PBRMaterial()]);
		const frame = createFrame(createPacket(model));
		frame.lights = Array.from(
			{ length: MAX_POINT_LIGHTS + 1 },
			() => new PointLight(),
		);
		const context = createFrameContextWithFeatures(
			frame,
			{ enableLighting: true },
			{ clusteredLighting: false },
		);

		resources.prepareFrame(context, createMainFrameOptions());
		resources.prepareFrame(
			context,
			createMainFrameOptions({ temporalStateMode: "reuse" }),
		);

		const pointLimitWarnings = warnings.filter((args) =>
			args.some(
				(value) =>
					typeof value === "string" && value.includes("[webgpu-point-limit]"),
			),
		);
		assert.equal(pointLimitWarnings.length, 1);
	} finally {
		resources.destroy();
		Logger.reset();
	}
}

function testFrameExecutorConsumesComputeFacadeFromHost() {
	const backend = new FakeBackend();
	const resourcesStub = {
		sceneFrameLayout: null,
		createFrameScope() {
			return {
				prepare() { throw new Error("not used by this test"); },
				updateParticleShadowVolumes() {},
				destroy() {},
			};
		},
	};
	const executor = new WebGPUFrameExecutor(
		backend,
		resourcesStub,
		new FramePacketContributorRegistry(),
		resourcesStub,
	);

	assert.equal(backend.getComputeFacadeCalls, 0);
	assert.equal(typeof executor.getDebugState, "function");
}

async function testRenderResourcesUseCopyDstForUploads() {
	const backend = new FakeBackend();
	const model = createModel([
		new PBRMaterial({
			albedo: { r: 255, g: 255, b: 255 },
		}),
	]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	await resources.init();
	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const draw = await resources.getDrawResources(packet, frameResources);

	assert.ok(draw);
	const firstDraw = draw[0];
	assert.ok(firstDraw);
	assert.equal(firstDraw.frameBinding.desc.entries.length, 17);
	assert.ok(
		firstDraw.frameBinding.desc.entries.some((entry) => entry.binding === 7)
	);
	assert.ok(
		firstDraw.frameBinding.desc.entries.some((entry) => entry.binding === 10)
	);
	for (const binding of [0, 14, 15, 16]) {
		assert.ok(
			firstDraw.frameBinding.desc.entries.some(
				(entry) => entry.binding === binding
			)
		);
	}
	assert.equal(
		firstDraw.modelBinding.desc.entries.length,
		1 +
			WEBGPU_TEXTURE_SLOT_COUNT +
			WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT +
			7
	);
	assert.ok(
		firstDraw.modelBinding.desc.entries.some(
			(entry) => entry.binding === WEBGPU_MODEL_BINDING_SHADER_UNIFORMS
		)
	);
	assert.ok(
		firstDraw.modelBinding.desc.entries.some(
			(entry) => entry.binding === WEBGPU_MODEL_BINDING_ANISOTROPY_TEXTURE
		)
	);
	assert.equal(
		firstDraw.pipeline.desc.layout,
		backend.device.pipelineLayouts[0]
	);
	assert.equal(
		firstDraw.pipeline.desc.layout.desc.bindGroupLayouts.length,
		3
	);
	assert.equal(firstDraw.pipeline.desc.fragment.targets.length, 5);
	assert.deepEqual(
		firstDraw.pipeline.desc.fragment.targets.map((target) => target.format),
		["rgba16float", "rgba8unorm", "rgba16float", "rgba16float", "rgba16float"]
	);
	const modelBindingIndices = firstDraw.modelBinding.desc.entries.map(
		(entry) => entry.binding
	);
	assert.ok(modelBindingIndices.includes(29));
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_ANIMATION_PARAMS)
	);
	assert.ok(modelBindingIndices.includes(31));
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_JOINT_MATRICES)
	);
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_MORPH_WEIGHTS)
	);
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_MORPH_POSITION)
	);
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_MORPH_NORMAL)
	);
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_ANISOTROPY_TEXTURE)
	);
	assert.equal(modelBindingIndices.includes(38), false);
	const sceneVertexAttributes =
		firstDraw.pipeline.desc.vertex.buffers[0].attributes;
	assert.ok(
		sceneVertexAttributes.some((attribute) => attribute.shaderLocation === 8)
	);
	assert.ok(
		backend.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Vertex) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	);
	assert.ok(
		backend.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Index) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	);
	assert.ok(
		backend.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Uniform) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	);
	for (const [label, size] of [
		["WebGPUFrameCameraUniforms", WEBGPU_FRAME_CAMERA_UNIFORM_FLOATS * 4],
		["WebGPUFrameLightUniforms", WEBGPU_FRAME_LIGHT_UNIFORM_FLOATS * 4],
		["WebGPUFrameShadowUniforms", WEBGPU_FRAME_SHADOW_UNIFORM_FLOATS * 4],
		["WebGPUFrameEnvironmentUniforms", WEBGPU_FRAME_ENVIRONMENT_UNIFORM_FLOATS * 4],
	]) {
		assert.ok(
			backend.bufferDescs.some(
				(desc) => desc.label === label && desc.size === size
			)
		);
	}
}

async function run() {
	try {
		await testRenderResourcesConsumeInjectedComputeFacade();
		await testRenderResourcesLeaveShaderRuntimeSubscriptionToBackend();
		await testRenderResourcesLogPointLightLimitOnlyOnce();
		await testFrameExecutorConsumesComputeFacadeFromHost();
		await testRenderResourcesUseCopyDstForUploads();
		console.log("WebGPU bridge render resources tests passed");
	} finally {
		ShaderSource.resetConfiguration();
		Logger.reset();
		if (previousGPUShaderStage === undefined) {
			delete globalThis.GPUShaderStage;
		} else {
			globalThis.GPUShaderStage = previousGPUShaderStage;
		}
	}
}
await run();
