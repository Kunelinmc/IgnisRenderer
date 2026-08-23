import assert from "node:assert/strict";
import {
	WebGPUFrameServiceOwner as WebGPURenderResources
} from "../../../src/backends/webgpu/WebGPUFrameServiceOwner.ts";
import {
	ShaderSource
} from "../../../src/shaders/ShaderSource.ts";
import {
	resolveFeatureState
} from "../../../src/pipeline/FeatureResolver.ts";
import {
	Matrix4
} from "../../../src/maths/Matrix4.ts";
import {
	SH
} from "../../../src/maths/SH.ts";
import {
	PBRMaterial
} from "../../../src/materials/PBRMaterial.ts";
import {
	AlphaMode
} from "../../../src/materials/Material.ts";
import {
	ShaderMaterial
} from "../../../src/materials/ShaderMaterial.ts";
import {
	WebGPUScenePipelineResources,
} from "../../../src/backends/webgpu/WebGPUScenePipelineResources.ts";
import { resolveWebGPUScenePassDescriptor } from "../../../src/backends/webgpu/WebGPUScenePassDescriptors.ts";
import { WebGPUPlanarReflectionPass } from "../../../src/backends/webgpu/WebGPUPlanarReflectionPass.ts";
import { WebGPUEnvironmentResources } from "../../../src/backends/webgpu/WebGPUEnvironmentResources.ts";
import { WebGPUDeferredResources } from "../../../src/backends/webgpu/WebGPUDeferredResources.ts";
import {
	Texture
} from "../../../src/core/Texture.ts";
import {
	Logger
} from "../../../src/foundation/Logger.ts";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY
} from "../../../src/pipeline/types.ts";
import {
	ParticleBlendMode
} from "../../../src/particles/types.ts";
import { createBaselineFramePacketSet } from "../../../src/pipeline/FramePackets.ts";
import {
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS
} from "../../../src/backends/webgpu/constants.ts";
import {
	createWebGPUComputeFacade
} from "../../../src/backends/webgpu/ComputeFacade.ts";
import {
	createResolvedPostProcess
} from "../../helpers/postprocess.mjs";


import {
	FakeCommandEncoder as FakeRenderEncoder,
	FakeWebGPUBackend as FakeBackend,
} from "../../helpers/fakes.mjs";
import {
	createFrame,
	createFrameContext,
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

function testWebGPUFrameServiceConstructionDoesNotCompilePipelines() {
	const backend = new FakeBackend();
	const owner = new WebGPURenderResources(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);
	try {
		assert.equal(backend.shaderModules.length, 0);
		assert.equal(backend.pipelines.length, 0);
	} finally {
		owner.destroy();
	}
}

async function testWebGPUBlendMaterialsUseTransparentPipelineState() {
	const backend = new FakeBackend();
	const material = new PBRMaterial({
		albedo: { r: 255, g: 255, b: 255 },
		opacity: 0.6,
	});
	material.alphaMode = AlphaMode.Blend;
	const model = createModel([material]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.opaquePackets = [];
	frame.transparentPackets = [packet];
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

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

	const draw = await resources.getDrawResources(packet, frameResources, {
		sampleCount: 1,
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(pipelineDesc.fragment.targets.length, 5);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.srcFactor,
		"src-alpha"
	);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.dstFactor,
		"one-minus-src-alpha"
	);
	assert.equal(pipelineDesc.fragment.targets[1].writeMask, 0);
	assert.equal(pipelineDesc.fragment.targets[2].writeMask, 0);
	assert.equal(pipelineDesc.fragment.targets[3].writeMask, 0);
	assert.equal(
		pipelineDesc.fragment.targets[4].blend?.alpha?.dstFactor,
		"one-minus-src-alpha"
	);
}

async function testWebGPUTransmissionMaterialsUseTransparentPipelineState() {
	const backend = new FakeBackend();
	const material = new PBRMaterial({
		albedo: { r: 255, g: 255, b: 255 },
		roughness: 0.05,
		metalness: 0,
		transmissionFactor: 1,
		ior: 1.52,
	});
	const model = createModel([material]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.opaquePackets = [];
	frame.transparentPackets = [packet];
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

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

	const draw = await resources.getDrawResources(packet, frameResources, {
		sampleCount: 1,
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(pipelineDesc.fragment.targets.length, 5);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.srcFactor,
		"src-alpha"
	);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.dstFactor,
		"one-minus-src-alpha"
	);
	assert.equal(
		pipelineDesc.fragment.targets[4].blend?.color?.srcFactor,
		"src-alpha"
	);
	assert.equal(
		pipelineDesc.fragment.targets[4].blend?.color?.dstFactor,
		"one-minus-src-alpha"
	);
}

async function testWebGPUEarlyZPrepassOpaquePipelineHasDepthOnlyState() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

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

	const draws = await Promise.all(Array.from({ length: 32 }, () =>
		resources.getDrawResources(packet, frameResources, {
			drawMode: "early-z-prepass",
			sampleCount: 1,
		})));
	const draw = draws[0];
	assert.ok(draw && draw.length > 0);
	assert.ok(draws.every((candidate) => candidate?.[0].pipeline === draw[0].pipeline));
	assert.equal(
		backend.pipelines.filter((candidate) =>
			candidate.label?.startsWith("WebGPUSceneEarlyZPipeline_"),
		).length,
		1,
	);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.layout.desc.bindGroupLayouts.length, 3);
	assert.equal(typeof pipelineDesc.fragment, "undefined");
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, true);
	assert.equal(pipelineDesc.depthStencil.depthCompare, "less");
}

async function testWebGPUEarlyZPrepassMaskPipelineUsesMaskDepthFragment() {
	const backend = new FakeBackend();
	const material = new PBRMaterial();
	material.alphaMode = AlphaMode.Mask;
	const model = createModel([material]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

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

	const draw = await resources.getDrawResources(packet, frameResources, {
		drawMode: "early-z-prepass",
		sampleCount: 1,
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.layout.desc.bindGroupLayouts.length, 3);
	assert.equal(pipelineDesc.fragment.entryPoint, "fsMainDepthMask");
	assert.equal(pipelineDesc.fragment.targets.length, 0);
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, true);
	assert.equal(pipelineDesc.depthStencil.depthCompare, "less");
}

async function testWebGPUEarlyZColorPipelineUsesReadOnlyDepthState() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

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

	const draw = await resources.getDrawResources(packet, frameResources, {
		drawMode: "early-z-color",
		sampleCount: 1,
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(pipelineDesc.depthStencil.depthCompare, "less-equal");
}

async function testWebGPUEarlyZShaderMaterialDepthContract() {
	const backend = new FakeBackend();
	const shaderMaterial = new ShaderMaterial({
		name: "EarlyZShaderMask",
		alphaMode: AlphaMode.Mask,
		vertexEntryPoint: "customVs",
		depthFragmentEntryPoint: "customDepth",
		depthFragmentCode: /* wgsl */ `
@fragment
fn customDepth() {
}
`,
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: /* wgsl */ `
@vertex
fn customVs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(position, 1.0);
}
`,
			},
		],
	});
	const supportedModel = createModel([shaderMaterial]);
	const supportedPacket = createPacket(supportedModel);
	const frame = createFrame(supportedPacket);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

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

	const supportedDraw = await resources.getDrawResources(
		supportedPacket,
		frameResources,
		{
			drawMode: "early-z-prepass",
			sampleCount: 1,
		}
	);
	assert.ok(supportedDraw && supportedDraw.length > 0);
	assert.equal(
		supportedDraw[0].pipeline.desc.fragment.entryPoint,
		"customDepth"
	);

	const missingContractMaterial = new ShaderMaterial({
		name: "EarlyZShaderMaskMissingDepth",
		alphaMode: AlphaMode.Mask,
		vertexEntryPoint: "customVs",
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: /* wgsl */ `
@vertex
fn customVs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(position, 1.0);
}
`,
			},
		],
	});
	const unsupportedModel = createModel([missingContractMaterial]);
	const unsupportedPacket = createPacket(unsupportedModel);
	const unsupportedDraw = await resources.getDrawResources(
		unsupportedPacket,
		frameResources,
		{
			drawMode: "early-z-prepass",
			sampleCount: 1,
		}
	);
	assert.equal(unsupportedDraw, null);
}

async function testWebGPUShaderMaterialDepthWriteFalseSkipsDepthPrepass() {
	const backend = new FakeBackend();
	const shaderMaterial = new ShaderMaterial({
		name: "DepthReadShader",
		depthWrite: false,
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFs",
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: /* wgsl */ `
@vertex
fn customVs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(position, 1.0);
}
`,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: /* wgsl */ `
@fragment
fn customFs() -> @location(0) vec4<f32> {
	return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`,
			},
		],
	});
	const model = createModel([shaderMaterial]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

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

	const prepassDraw = await resources.getDrawResources(packet, frameResources, {
		sceneTargetMode: "single",
		drawMode: "early-z-prepass",
		sampleCount: 1,
	});
	assert.equal(prepassDraw, null);

	const draw = await resources.getDrawResources(packet, frameResources, {
		sceneTargetMode: "single",
		sampleCount: 1,
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(pipelineDesc.depthStencil.depthCompare, "less");

	const earlyZColorDraw = await resources.getDrawResources(
		packet,
		frameResources,
		{
			sceneTargetMode: "single",
			drawMode: "early-z-color",
			sampleCount: 1,
		}
	);
	assert.ok(earlyZColorDraw && earlyZColorDraw.length > 0);
	const earlyZColorPipelineDesc = earlyZColorDraw[0].pipeline.desc;
	assert.equal(earlyZColorPipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(earlyZColorPipelineDesc.depthStencil.depthCompare, "less");
}

async function testWebGPUShaderMaterialCustomUniformBufferBinding() {
	const backend = new FakeBackend();
	const shaderMaterial = new ShaderMaterial({
		name: "CustomUniformShader",
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFs",
		uniformBindings: [
			{ name: "time", type: "f32", value: 1 },
		],
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: /* wgsl */ `
@vertex
fn customVs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(position, 1.0);
}
`,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: /* wgsl */ `
@fragment
fn customFs() -> @location(0) vec4<f32> {
	return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`,
			},
		],
	});
	const model = createModel([shaderMaterial]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

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

	const firstDraw = await resources.getDrawResources(packet, frameResources, {
		sampleCount: 1,
	});
	assert.ok(firstDraw && firstDraw.length > 0);
	const firstUniformEntry = firstDraw[0].modelBinding.desc.entries.find(
		(entry) => entry.binding === WEBGPU_MODEL_BINDING_SHADER_UNIFORMS
	);
	assert.ok(firstUniformEntry);
	assert.ok(
		String(firstUniformEntry.resource.label).startsWith(
			"ShaderMaterialUniform_"
		)
	);
	assert.deepEqual(firstUniformEntry.resource.lastWrite.slice(0, 4), [
		0,
		0,
		128,
		63,
	]);

	const firstResource = firstUniformEntry.resource;
	shaderMaterial.setUniform("time", 2);
	const secondDraw = await resources.getDrawResources(packet, frameResources, {
		sampleCount: 1,
	});
	const secondUniformEntry = secondDraw[0].modelBinding.desc.entries.find(
		(entry) => entry.binding === WEBGPU_MODEL_BINDING_SHADER_UNIFORMS
	);
	assert.strictEqual(secondUniformEntry.resource, firstResource);
	assert.deepEqual(secondUniformEntry.resource.lastWrite.slice(0, 4), [
		0,
		0,
		0,
		64,
	]);

	shaderMaterial.setUniformBinding({
		name: "transform",
		type: "mat4x4f",
		value: Matrix4.identity(),
	});
	const thirdDraw = await resources.getDrawResources(packet, frameResources, {
		sampleCount: 1,
	});
	const thirdUniformEntry = thirdDraw[0].modelBinding.desc.entries.find(
		(entry) => entry.binding === WEBGPU_MODEL_BINDING_SHADER_UNIFORMS
	);
	assert.notStrictEqual(thirdUniformEntry.resource, firstResource);
	assert.ok(thirdUniformEntry.resource.size > firstResource.size);
}

async function testWebGPUOITTransparentPipelineUsesDualTargets() {
	const backend = new FakeBackend();
	const material = new PBRMaterial({
		albedo: { r: 255, g: 255, b: 255 },
		opacity: 0.6,
	});
	material.alphaMode = AlphaMode.Blend;
	const model = createModel([material]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.opaquePackets = [];
	frame.transparentPackets = [packet];
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
				enableOIT: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				oit: true,
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

	const draw = await resources.getDrawResources(packet, frameResources, {
		transparentPipelineMode: "oit",
		sampleCount: 1,
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.depthStencil.depthWriteEnabled, false);
	assert.equal(pipelineDesc.fragment.entryPoint, "fsMainOIT");
	assert.equal(pipelineDesc.fragment.targets.length, 2);
	assert.equal(pipelineDesc.fragment.targets[0].format, "rgba16float");
	assert.equal(pipelineDesc.fragment.targets[1].format, "r8unorm");
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.srcFactor,
		"one"
	);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.dstFactor,
		"one"
	);
	assert.equal(
		pipelineDesc.fragment.targets[1].blend?.color?.srcFactor,
		"zero"
	);
	assert.equal(
		pipelineDesc.fragment.targets[1].blend?.color?.dstFactor,
		"one-minus-src"
	);
}

async function testWebGPUOITTransmissionMaterialsStayLegacyPipeline() {
	const backend = new FakeBackend();
	const material = new PBRMaterial({
		albedo: { r: 255, g: 255, b: 255 },
		roughness: 0.05,
		metalness: 0,
		transmissionFactor: 1,
		ior: 1.52,
	});
	const model = createModel([material]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.opaquePackets = [];
	frame.transparentPackets = [packet];
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
				enableOIT: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				oit: true,
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

	const draw = await resources.getDrawResources(packet, frameResources, {
		transparentPipelineMode: "transmission",
		sampleCount: 1,
	});
	assert.ok(draw && draw.length > 0);
	const pipelineDesc = draw[0].pipeline.desc;
	assert.equal(pipelineDesc.fragment.entryPoint, "fsMain");
	assert.equal(pipelineDesc.fragment.targets.length, 5);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.srcFactor,
		"src-alpha"
	);
	assert.equal(
		pipelineDesc.fragment.targets[0].blend?.color?.dstFactor,
		"one-minus-src-alpha"
	);
}

async function testWebGPUOITParticlePipelinesSplitAlphaAndAdditive() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableShadows: true,
			enableOIT: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			environment: false,
			oit: true,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	resources.beginFrameResourceLifecycle();
	const frameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		createMainFrameOptions()
	);

	const texture = new Texture({
		data: new Uint8Array([255, 255, 255, 255]),
		width: 1,
		height: 1,
		colorSpace: "sRGB",
	});
	const context = {
		camera: frame.camera,
		attachments: { width: 16, height: 16 },
		features,
		postProcess: createResolvedPostProcess(),
		shadowMaps: frame.shadowMaps,
		scene: { ...frame, particleSystems: [] },
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		transient: new Map([
			[
				PARTICLE_TRANSIENT_BATCHES_KEY,
				[
					{
						systemId: "particleSystem-oit-alpha",
						blendMode: ParticleBlendMode.Alpha,
						texture,
						receiveShadows: true,
						particles: [
							{
								position: { x: 0, y: 0, z: 0 },
								size: 1,
								color: { r: 255, g: 255, b: 255, a: 1 },
								rotation: 0,
								depth: 1,
								uvRect: { u0: 0, v0: 0, u1: 1, v1: 1 },
							},
						],
					},
					{
						systemId: "particleSystem-oit-add",
						blendMode: ParticleBlendMode.Additive,
						texture,
						receiveShadows: false,
						particles: [
							{
								position: { x: 0, y: 0, z: 0 },
								size: 1,
								color: { r: 255, g: 255, b: 255, a: 1 },
								rotation: 0,
								depth: 1,
								uvRect: { u0: 0, v0: 0, u1: 1, v1: 1 },
							},
						],
					},
				],
			],
		]),
	};
	const encoder = new FakeRenderEncoder();
	const renderTarget = { width: 16, height: 16, destroy() {} };
	const alphaCount = await resources.getParticleBillboardRenderer().renderParticles(
		encoder,
		context,
		{
			label: "WebGPUParticlesOIT_Test",
			sampleCount: 1,
			colorAttachments: [
				{
					view: renderTarget,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: renderTarget,
					loadOp: "load",
					storeOp: "store",
				},
			],
			depth: renderTarget,
		},
		frameResources,
		"single",
		{
			includeBlendModes: [ParticleBlendMode.Alpha],
			pipelineMode: "oit",
		}
	);
	assert.equal(alphaCount, 1);

	const additiveCount = await resources.getParticleBillboardRenderer().renderParticles(
		encoder,
		context,
		{
			label: "WebGPUParticlesAdd_Test",
			sampleCount: 1,
			colorAttachments: [
				{
					view: renderTarget,
					loadOp: "load",
					storeOp: "store",
				},
			],
			depth: renderTarget,
		},
		frameResources,
		"single",
		{
			includeBlendModes: [ParticleBlendMode.Additive],
			pipelineMode: "legacy",
		}
	);
	assert.equal(additiveCount, 1);

	const oitPipeline = backend.pipelines.find(
		(pipeline) => pipeline.label === "WebGPUParticlePipeline_oit-alpha_single"
	);
	assert.ok(oitPipeline);
	assert.equal(oitPipeline.desc.fragment.entryPoint, "fsMainOIT");
	assert.equal(oitPipeline.desc.fragment.targets.length, 2);
	assert.equal(oitPipeline.desc.fragment.targets[1].format, "r8unorm");

	const additivePipeline = backend.pipelines.find(
		(pipeline) => pipeline.label === "WebGPUParticlePipeline_additive_single"
	);
	assert.ok(additivePipeline);
	assert.equal(additivePipeline.desc.fragment.entryPoint, "fsMain");
	assert.equal(additivePipeline.desc.fragment.targets.length, 1);
	assert.equal(
		additivePipeline.desc.fragment.targets[0].blend?.color?.dstFactor,
		"one"
	);
}

async function testPlanarCompositeUsesReflectionOwnedPipelineAndSharedSnapshot() {
	const backend = new FakeBackend();
	const material = new PBRMaterial();
	const packet = createPacket(createModel([material]));
	const frame = createFrame(packet);
	frame.reflectivePackets = [packet];
	const owner = new WebGPURenderResources(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);
	const context = createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: false,
				enableReflection: true,
			},
			{
				sh: false,
				shadows: false,
				reflection: true,
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
			},
		);
	const frameResources = owner.prepareFrame(
		context,
		createMainFrameOptions(),
	);
	const reflection = owner.createPlanarReflectionDrawResources();
	const reflectionPass = new WebGPUPlanarReflectionPass(
		backend,
		owner,
		reflection,
	);
	try {
		const first = await reflection.getDrawResources(packet, frameResources, {
			sceneTargetMode: "mrt",
			drawMode: "planar-reflection-composite",
			sampleCount: 1,
		});
		const second = await reflection.getDrawResources(packet, frameResources, {
			sceneTargetMode: "mrt",
			drawMode: "planar-reflection-composite",
			sampleCount: 1,
		});
		assert.ok(first?.[0].pipeline.label.startsWith(
			"WebGPUPlanarReflectionCompositePipeline_",
		));
		assert.equal(second?.[0].pipeline, first[0].pipeline);
		assert.deepEqual(owner.getDebugStats().materialSnapshots, {
			frameHits: 1,
			frameResolves: 1,
		});
		assert.ok(
			![...owner._scenePipelines._pipelineCache.keys()].some((key) =>
				key.includes("planar-reflection-composite"),
			),
		);
		const warmup = await reflectionPass.warmup({
			context,
			framePackets: createBaselineFramePacketSet(context),
			sampleCount: 4,
			yieldIfNeeded: async () => {},
		});
		assert.equal(warmup.phase, "webgpu-reflection");
		assert.ok(warmup.compiled > 0);
		assert.ok(backend.pipelines.some((pipeline) =>
			pipeline.label?.startsWith("WebGPUPlanarReflectionCompositePipeline_") &&
			pipeline.desc.sampleCount === 4,
		));
		const createPipeline = backend.createPipeline.bind(backend);
		backend.createPipeline = async (desc) => {
			if (
				desc.label?.startsWith("WebGPUPlanarReflectionCompositePipeline_") &&
				desc.sampleCount === 8
			) throw new Error("reflection warmup compile failure");
			return createPipeline(desc);
		};
		const failedWarmup = await reflectionPass.warmup({
			context,
			framePackets: createBaselineFramePacketSet(context),
			sampleCount: 8,
			yieldIfNeeded: async () => {},
		});
		assert.equal(failedWarmup.failed, 1);
		assert.match(failedWarmup.errors[0].message, /reflection warmup compile failure/);
	} finally {
		reflectionPass.destroy();
		owner.destroy();
	}
}

async function testFeatureResourceWarmupReportsCompilationFailures() {
	const backend = new FakeBackend();
	backend.createPipeline = async () => {
		throw new Error("feature warmup compile failure");
	};
	const environment = new WebGPUEnvironmentResources(backend, {
		environmentPipelineLayout: {},
	});
	const deferred = new WebGPUDeferredResources(backend, {
		deferredLightingPipelineLayout: {},
	});
	try {
		const environmentPhase = await environment.warmup({
			modes: ["single"],
			sampleCount: 1,
			yieldIfNeeded: async () => {},
		});
		assert.equal(environmentPhase.failed, 1);
		assert.match(environmentPhase.errors[0].message, /feature warmup compile failure/);
		const deferredPhase = await deferred.warmup({
			active: true,
			hasDecals: false,
			yieldIfNeeded: async () => {},
		});
		assert.equal(deferredPhase.failed, 1);
		assert.match(deferredPhase.errors[0].message, /feature warmup compile failure/);
	} finally {
		environment.destroy();
		deferred.destroy();
	}
}

async function testFeatureOwnedWarmupCompilesDeferredSceneVariants() {
	const backend = new FakeBackend();
	const packet = createPacket(createModel([new PBRMaterial()]));
	const frame = createFrame(packet);
	const context = createFrameContextWithFeatures(
		frame,
		{
			enableLighting: true,
			enableGamma: true,
			enableShadows: false,
		},
		{
			sh: false,
			shadows: false,
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
		},
	);
	const owner = new WebGPURenderResources(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);
	try {
		const phases = await owner.warmup(
			context,
			{
				materials: [packet.material],
				shaderMaterials: [],
				enableEnvironment: false,
				enableShadows: false,
				enableParticles: false,
				postProcessPasses: [],
				postProcessDescriptors: [],
				sceneTargetMode: "single",
			},
			{},
			createBaselineFramePacketSet(context),
			{
				enableEarlyZPrepass: true,
				enableDeferredLighting: true,
				sampleCount: 1,
			},
		);
		assert.ok(phases.some((phase) => phase.phase === "webgpu-deferred-scene"));
		assert.ok(phases.some((phase) => phase.phase === "webgpu-deferred"));
		assert.ok(backend.pipelines.some((pipeline) =>
			pipeline.label === "WebGPUDeferredLightingPipeline",
		));
		assert.ok(backend.pipelines.some((pipeline) =>
			pipeline.label?.startsWith("WebGPUSceneEarlyZPipeline_"),
		));
	} finally {
		owner.destroy();
	}
}

async function run() {
	try {
		testWebGPUFrameServiceConstructionDoesNotCompilePipelines();
		await testWebGPUBlendMaterialsUseTransparentPipelineState();
		await testWebGPUTransmissionMaterialsUseTransparentPipelineState();
		await testWebGPUEarlyZPrepassOpaquePipelineHasDepthOnlyState();
		await testWebGPUEarlyZPrepassMaskPipelineUsesMaskDepthFragment();
		await testWebGPUEarlyZColorPipelineUsesReadOnlyDepthState();
		await testWebGPUEarlyZShaderMaterialDepthContract();
		await testWebGPUShaderMaterialDepthWriteFalseSkipsDepthPrepass();
		await testWebGPUShaderMaterialCustomUniformBufferBinding();
		await testWebGPUOITTransparentPipelineUsesDualTargets();
		await testWebGPUOITTransmissionMaterialsStayLegacyPipeline();
		await testWebGPUOITParticlePipelinesSplitAlphaAndAdditive();
		await testPlanarCompositeUsesReflectionOwnedPipelineAndSharedSnapshot();
		await testFeatureOwnedWarmupCompilesDeferredSceneVariants();
		await testFeatureResourceWarmupReportsCompilationFailures();
		await testEarlyZInvalidationDiscardsLatePipeline();
		testScenePipelineResourcesExplicitlyDestroyInvalidatedHandles();
		console.log("WebGPU bridge material pipelines tests passed");
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

async function testEarlyZInvalidationDiscardsLatePipeline() {
	const backend = new FakeBackend();
	const createPipeline = backend.createPipeline.bind(backend);
	let release;
	const gate = new Promise((resolve) => {
		release = resolve;
	});
	let delayFirstPipeline = true;
	backend.createPipeline = async (desc) => {
		if (delayFirstPipeline) {
			delayFirstPipeline = false;
			await gate;
		}
		return createPipeline(desc);
	};
	const scenePipelines = new WebGPUScenePipelineResources(backend, {});
	const pending = scenePipelines.resolvePipeline({
		materialState: {
			materialRevision: 1,
			pipelineKey: "early-z-test",
			shaderCacheKey: "builtin-scene|runtime:0|directive:none",
			cullMode: "back",
			depthWrite: true,
			alphaMode: AlphaMode.Opaque,
			transparent: false,
			usesTransmission: false,
			wireframe: false,
			shaderRuntime: {
				revision: 0,
				mode: "strict",
				directiveCacheTag: "none",
				supportsRuntimeInjects: false,
			},
			diagnostic: {
				materialName: "EarlyZTest",
				shaderId: null,
				fallbackReason: null,
			},
			program: { kind: "builtin" },
		},
		pass: resolveWebGPUScenePassDescriptor(
			"single",
			"default",
			"early-z-prepass",
		),
		topology: "triangle-list",
		geometryLayout: { layoutKey: "test", sceneVertexLayouts: [] },
		sampleCount: 1,
	});
	await new Promise((resolve) => setTimeout(resolve, 0));
	scenePipelines.invalidateShaderRuntimeCaches();
	release();
	const pipeline = await pending;
	assert.ok(pipeline);
	assert.equal(pipeline.destroyed, false);
	assert.equal(scenePipelines._earlyZPrepassCache.size, 1);
	assert.equal(backend.renderPipelineDestroyCalls, 1);
}

function testScenePipelineResourcesExplicitlyDestroyInvalidatedHandles() {
	const library = new WebGPUScenePipelineResources({}, {});
	const pipeline = {
		destroyCalls: 0,
		destroy() {
			this.destroyCalls++;
		},
	};
	const shader = {
		destroyCalls: 0,
		destroy() {
			this.destroyCalls++;
		},
	};
	library._pipelineCache.set("scene", pipeline);
	library._earlyZPrepassCache.set("early-z", pipeline);
	library._customShaderModuleCache.set("custom", shader);
	library._sceneShaderModule = shader;
	library.invalidateShaderRuntimeCaches();
	assert.equal(pipeline.destroyCalls, 1);
	assert.equal(shader.destroyCalls, 1);
	assert.equal(library._pipelineCache.size, 0);
	assert.equal(library._earlyZPrepassCache.size, 0);
	assert.equal(library._customShaderModuleCache.size, 0);

	const environment = new WebGPUEnvironmentResources({}, {});
	environment._pipelines.set("environment", pipeline);
	environment._shaderModule = shader;
	environment.onShaderRuntimeChanged();
	assert.equal(pipeline.destroyCalls, 2);
	assert.equal(shader.destroyCalls, 2);
	assert.equal(environment._pipelines.size, 0);

	const deferred = new WebGPUDeferredResources({}, {});
	deferred._deferredLightingPipeline = pipeline;
	deferred._deferredLightingShaderModule = shader;
	deferred.onShaderRuntimeChanged();
	assert.equal(pipeline.destroyCalls, 3);
	assert.equal(shader.destroyCalls, 3);
}
await run();
