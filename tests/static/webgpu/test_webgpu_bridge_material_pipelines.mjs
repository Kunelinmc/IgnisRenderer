import assert from "node:assert/strict";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
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
	assert.equal(pipelineDesc.fragment.entryPoint, "fsMainOITPBR");
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
	assert.equal(pipelineDesc.fragment.entryPoint, "fsMainPBR");
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
		await testSharedStaticDrawPreparation();
		await testRetainedPreparationPinsMaterialBindings();
		await testReleasedScopeDiscardsDrawPreparation();
		await testSharedStaticPreparationLifecycle();
		await testSharedStaticPreparationGeometryReplacement();
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
		await testSceneShaderModuleRequestsCoalesce();
		await testCustomShaderModuleRequestsCoalesce();
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

async function testSharedStaticDrawPreparation() {
	const backend = new FakeBackend();
	const owner = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));
	const material = new PBRMaterial();
	const model = createModel([material]);
	const packets = [createPacket(model), createPacket(new MeshInstance({ mesh: model.mesh }))];
	const frame = createFrame(packets[0]);
	frame.opaquePackets = packets;
	owner.beginFrameResourceLifecycle();
	const context = createFrameContext(frame, resolveFeatureState({}, {}, "webgpu"));
	const prepared = owner.prepareFrame(context, createMainFrameOptions());
	const pipelines = owner._sceneDraws._pipelines;
	const resolvePipeline = pipelines.resolvePipeline.bind(pipelines);
	let pipelineCalls = 0;
	pipelines.resolvePipeline = (request) => {
		pipelineCalls++;
		return resolvePipeline(request);
	};
	try {
		const options = { sampleCount: 1 };
		const first = await owner.getDrawResources(packets[0], prepared, options);
		const second = owner.getDrawResources(packets[1], prepared, options);
		if (second instanceof Promise) await second;
		assert.ok(!(second instanceof Promise), "prepared static hits must be synchronous");
		assert.equal(pipelineCalls, 1, "shared geometry and material prepare once");
		assert.notEqual(first[0], second[0], "instance draw wrappers must remain independent");
		assert.equal(first[0].pipeline, second[0].pipeline);
		assert.equal(first[0].modelBinding, second[0].modelBinding);
		assert.notEqual(first[0].firstInstance, second[0].firstInstance);
		await owner.getDrawResources(packets[1], prepared, { sampleCount: 4 });
		assert.equal(pipelineCalls, 2, "sample count separates preparation");
		await owner.getDrawResources(packets[1], prepared, { ...options, drawMode: "early-z-prepass" });
		assert.equal(pipelineCalls, 3, "pass separates preparation");
		material.roughness = 0.25;
		material.refreshRevision();
		const revised = await owner.getDrawResources(packets[1], prepared, options);
		assert.equal(pipelineCalls, 4, "material revision invalidates preparation");
		assert.notEqual(revised[0].resolvedInputs.materialData, first[0].resolvedInputs.materialData);
		packets[1].submission.geometry.version++;
		await owner.getDrawResources(packets[1], prepared, options);
		assert.equal(pipelineCalls, 5, "geometry revision invalidates preparation");
		pipelines.invalidateShaderRuntimeCaches();
		await owner.getDrawResources(packets[1], prepared, options);
		assert.equal(pipelineCalls, 6, "provider invalidation discards prepared pipelines");
		const auxiliary = { ...prepared };
		await owner.getDrawResources(packets[1], auxiliary, options);
		assert.equal(pipelineCalls, 7, "frame scopes must not share preparation");
		owner.commitTemporalFrame();
		owner.beginFrameResourceLifecycle();
		frame.opaquePackets = [packets[1], packets[0]];
		const nextFrame = owner.prepareFrame(context, createMainFrameOptions());
		nextFrame.frameBinding = { label: "next-frame-binding" };
		const nextPreparation = owner.getDrawResources(packets[1], nextFrame, options);
		const nextDraw = await nextPreparation;
		assert.ok(!(nextPreparation instanceof Promise), "ready preparation survives frame reset");
		assert.equal(pipelineCalls, 7, "new frames reuse immutable preparation");
		assert.equal(nextDraw[0].firstInstance, 0, "reordered instances use current frame indices");
		assert.equal(nextDraw[0].frameBinding, nextFrame.frameBinding);
		owner.commitTemporalFrame();
		material.roughness = 0.5;
		owner.beginFrameResourceLifecycle();
		const changedFrame = owner.prepareFrame(context, createMainFrameOptions());
		const changedDraw = await owner.getDrawResources(packets[1], changedFrame, options);
		assert.equal(pipelineCalls, 8, "new frames refresh direct built-in material changes");
		assert.notEqual(changedDraw[0].resolvedInputs.materialData, nextDraw[0].resolvedInputs.materialData);
		material.alphaMode = AlphaMode.Blend;
		material.refreshRevision();
		await owner.getDrawResources(packets[0], nextFrame, options);
		const transparent = owner.getDrawResources(packets[1], nextFrame, options);
		assert.ok(transparent instanceof Promise,
			"transparent draws retain per-instance preparation");
		await transparent;
	} finally {
		owner.destroy();
	}
}

async function testSharedStaticPreparationLifecycle() {
	const backend = new FakeBackend();
	const owner = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));
	const material = new PBRMaterial();
	const model = createModel([material]);
	const packets = [createPacket(model), createPacket(new MeshInstance({ mesh: model.mesh })),
		createPacket(createModel([material]))];
	const frame = createFrame(packets[0]);
	frame.opaquePackets = packets;
	const context = createFrameContext(frame, resolveFeatureState({}, {}, "webgpu"));
	owner.beginFrameResourceLifecycle();
	const prepared = owner.prepareFrame(context, createMainFrameOptions());
	const options = { sampleCount: 1 };
	const assembler = owner._drawResourceAssembler;
	const pipelines = owner._sceneDraws._pipelines;
	let pipelineCalls = 0;
	let policyCalls = 0;
	let fail = false;
	let gate = null;
	const provider = {
		preparationRevision: 0,
		async resolvePipeline(request) {
			pipelineCalls++;
			if (gate) await gate;
			if (fail) throw new Error("preparation failure");
			return pipelines.resolvePipeline(request);
		},
	};
	const resolvePolicy = owner._materialPipelineResolver.resolve.bind(owner._materialPipelineResolver);
	owner._materialPipelineResolver.resolve = (...args) => {
		policyCalls++;
		return resolvePolicy(...args);
	};
	try {
		const concurrent = await Promise.all(packets.map((packet) =>
			assembler.getDrawResources(packet, prepared, options, provider)));
		assert.equal(pipelineCalls, 2, "concurrent shared geometry requests coalesce");
		assert.equal(policyCalls, 1, "one material policy serves different geometries");
		assert.notEqual(concurrent[0][0].firstInstance, concurrent[1][0].firstInstance);
		await assembler.getDrawResources(packets[0], prepared, options, { ...provider });
		assert.equal(pipelineCalls, 3, "provider identity separates preparation");
		const beforeGrowth = concurrent[0][0].modelBinding;
		owner._staticBatcher._ensureCapacity(512);
		assert.ok(beforeGrowth.destroyed, "arena growth destroys old static bindings");
		const afterGrowth = await assembler.getDrawResources(packets[0], prepared, options, provider);
		assert.notEqual(afterGrowth[0].modelBinding, beforeGrowth);
		assert.equal(pipelineCalls, 4, "arena replacement rebuilds shared bindings");
		let runtimeRevision = 0;
		backend.getShaderRuntimeView = () => ({ revision: runtimeRevision, mode: "strict",
			directiveCacheTag: "test" });
		await assembler.getDrawResources(packets[0], prepared, options, provider);
		runtimeRevision++;
		await assembler.getDrawResources(packets[0], prepared, options, provider);
		assert.equal(pipelineCalls, 6, "runtime fingerprint invalidates preparation");
		assembler.clear();
		fail = true;
		await assert.rejects(assembler.getDrawResources(packets[0], prepared, options, provider),
			/preparation failure/);
		fail = false;
		const retry = await assembler.getDrawResources(packets[0], prepared, options, provider);
		assert.ok(retry, "failed preparation remains retryable");
		assert.equal(pipelineCalls, 8);
		assembler.clear();
		let release;
		gate = new Promise((resolve) => { release = resolve; });
		const stale = assembler.getDrawResources(packets[0], prepared, options, provider);
		// Let material inputs resolve and enter the provider before invalidating it.
		await Promise.resolve();
		await Promise.resolve();
		provider.preparationRevision++;
		release();
		assert.equal(await stale, null, "late invalidated pipelines cannot publish draws");
		gate = null;
		const fresh = await assembler.getDrawResources(packets[0], prepared, options, provider);
		assert.ok(fresh);
		const ready = assembler.getDrawResources(packets[1], prepared, options, provider);
		assert.ok(!(ready instanceof Promise), "fresh preparation replaces stale work");
		prepared.frameBinding = { label: "updated-frame" };
		assert.equal(assembler.getDrawResources(packets[0], prepared, options, provider)[0].frameBinding,
			prepared.frameBinding, "ready draws use current frame bindings");
		let releaseFrameGate;
		let firstEntered;
		let secondEntered;
		let entries = 0;
		const frameGate = new Promise(resolve => { releaseFrameGate = resolve; });
		const firstStarted = new Promise(resolve => { firstEntered = resolve; });
		const secondStarted = new Promise(resolve => { secondEntered = resolve; });
		const resolveFramePipeline = provider.resolvePipeline.bind(provider);
		provider.resolvePipeline = async request => {
			if (++entries === 1) firstEntered(); else secondEntered();
			await frameGate;
			return resolveFramePipeline(request);
		};
		const pendingOldFrame = assembler.getDrawResources(packets[0], prepared,
			{ sampleCount: 4 }, provider);
		await firstStarted;
		assembler.beginFrame();
		const pendingNewFrame = assembler.getDrawResources(packets[1], prepared,
			{ sampleCount: 4 }, provider);
		await secondStarted;
		releaseFrameGate();
		const [oldFrameDraw, newFrameDraw] = await Promise.all([pendingOldFrame, pendingNewFrame]);
		assert.equal(oldFrameDraw, null, "pending work cannot cross the frame boundary");
		assert.ok(newFrameDraw);
		const retained = assembler.getDrawResources(packets[1], prepared, { sampleCount: 4 }, provider);
		if (retained instanceof Promise) await retained;
		assert.ok(!(retained instanceof Promise), "stale cleanup cannot remove a replacement group");
	} finally {
		owner.destroy();
	}
}

async function testSharedStaticPreparationGeometryReplacement() {
	const backend = new FakeBackend();
	const owner = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));
	const model = createModel([new PBRMaterial()]);
	const packets = [createPacket(model), createPacket(new MeshInstance({ mesh: model.mesh }))];
	const frame = createFrame(packets[0]);
	frame.opaquePackets = packets;
	owner.beginFrameResourceLifecycle();
	const prepared = owner.prepareFrame(createFrameContext(frame,
		resolveFeatureState({}, {}, "webgpu")), createMainFrameOptions());
	const assembler = owner._drawResourceAssembler;
	const registry = owner._geometryRegistry;
	const pipelines = owner._sceneDraws._pipelines;
	const options = { sampleCount: 1 };
	let release;
	let entered;
	const gate = new Promise((resolve) => { release = resolve; });
	const started = new Promise((resolve) => { entered = resolve; });
	const provider = {
		preparationRevision: 0,
		async resolvePipeline(request) {
			entered();
			await gate;
			return pipelines.resolvePipeline(request);
		},
	};
	try {
		const oldHandle = registry.getGeometry(packets[0].submission.geometry);
		const old = assembler.getDrawResources(packets[0], prepared, options, provider);
		await started;
		packets[1].submission.geometry.version++;
		const fresh = assembler.getDrawResources(packets[1], prepared, options, provider);
		assert.ok(oldHandle.indexBuffer.destroyed);
		release();
		const [oldDraw, freshDraw] = await Promise.all([old, fresh]);
		assert.equal(oldDraw, null, "shared geometry replacement discards waiting draws");
		assert.ok(!freshDraw[0].indexBuffer.destroyed);
		assembler.clear();
		let releaseAgain;
		let enteredAgain;
		const gateAgain = new Promise((resolve) => { releaseAgain = resolve; });
		const startedAgain = new Promise((resolve) => { enteredAgain = resolve; });
		provider.resolvePipeline = async (request) => {
			enteredAgain();
			await gateAgain;
			return pipelines.resolvePipeline(request);
		};
		const released = assembler.getDrawResources(packets[1], prepared, options, provider);
		await startedAgain;
		registry.releaseGeometry(packets[1].submission.geometry);
		releaseAgain();
		assert.equal(await released, null, "released geometry cannot publish waiting draws");
	} finally {
		owner.destroy();
	}
}

function createStaticPreparationFixture() {
	const backend = new FakeBackend();
	const owner = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));
	const packet = createPacket(createModel([new PBRMaterial()]));
	const frame = createFrame(packet);
	const context = createFrameContext(frame, resolveFeatureState({}, {}, "webgpu"));
	owner.beginFrameResourceLifecycle();
	const prepared = owner.prepareFrame(context, createMainFrameOptions());
	return { owner, packet, context, prepared };
}

async function testReleasedScopeDiscardsDrawPreparation() {
	const { owner, packet, context, prepared } = createStaticPreparationFixture();
	let release;
	let entered;
	const gate = new Promise(resolve => { release = resolve; });
	const started = new Promise(resolve => { entered = resolve; });
	const provider = {
		preparationRevision: 0,
		async resolvePipeline(request) {
			entered();
			await gate;
			return owner._sceneDraws._pipelines.resolvePipeline(request);
		},
	};
	const assembler = owner._drawResourceAssembler;
	try {
		const stale = assembler.getDrawResources(packet, prepared, { sampleCount: 1 }, provider);
		await started;
		owner.releaseScope(prepared.scopeKey);
		const replacement = owner.prepareFrame(context, createMainFrameOptions());
		const fresh = assembler.getDrawResources(packet, replacement, { sampleCount: 1 }, provider);
		release();
		const [oldDraw, newDraw] = await Promise.all([stale, fresh]);
		assert.equal(oldDraw, null, "released scopes cannot publish waiting draws");
		assert.equal(newDraw[0].frameBinding, replacement.frameBinding);
		assert.equal(assembler.getDrawResources(packet, prepared, { sampleCount: 1 }, provider), null,
			"old scope bindings cannot be reused after replacement");
		let publish;
		let enteredPublication;
		const publicationGate = new Promise(resolve => { publish = resolve; });
		const publicationStarted = new Promise(resolve => { enteredPublication = resolve; });
		const publicationProvider = {
			preparationRevision: 0,
			resolvePipeline() {
				enteredPublication();
				return publicationGate;
			},
		};
		const late = assembler.getDrawResources(packet, replacement,
			{ sampleCount: 1 }, publicationProvider);
		await publicationStarted;
		publish(newDraw[0].pipeline);
		queueMicrotask(() => owner.releaseScope(replacement.scopeKey));
		assert.equal(await late, null, "scope release between promise reactions prevents publication");
	} finally {
		owner.destroy();
	}
}

async function testRetainedPreparationPinsMaterialBindings() {
	const { owner, packet, context, prepared } = createStaticPreparationFixture();
	try {
		const draws = await owner.getDrawResources(packet, prepared, { sampleCount: 1 });
		const inputs = draws[0].resolvedInputs;
		const snapshot = { revision: 0, data: inputs.materialData,
			textures: inputs.textures, samplers: inputs.samplers };
		const batcher = owner._staticBatcher;
		const geometry = owner._geometryRegistry.getGeometry(packet.submission.geometry);
		const hotBuffer = batcher._materialBindings.get(snapshot.data).objectUniformBuffer;
		const fillers = Array.from({ length: 4095 }, () => ({ ...snapshot,
			data: { ...snapshot.data } }));
		for (const filler of fillers) {
			batcher.getDrawState(packet, draws[0].pipeline, geometry, filler, "default");
		}
		owner.commitTemporalFrame();
		owner.beginFrameResourceLifecycle();
		const next = owner.prepareFrame(context, createMainFrameOptions());
		const ready = owner.getDrawResources(packet, next, { sampleCount: 1 });
		await ready;
		assert.ok(!(ready instanceof Promise));
		batcher.getDrawState(packet, draws[0].pipeline, geometry,
			{ ...snapshot, data: { ...snapshot.data } }, "default");
		assert.equal(hotBuffer.destroyed, false, "a retained current-frame binding must stay alive");
		for (const filler of fillers) {
			batcher.getDrawState(packet, draws[0].pipeline, geometry, filler, "default");
		}
		assert.equal(hotBuffer.destroyed, false, "active bindings remain alive during cache overflow");
		owner.commitTemporalFrame();
		owner.beginFrameResourceLifecycle();
		assert.ok(batcher.getDebugStats().materialBindings <= 4096,
			"inactive overflow is trimmed at the next frame boundary");
		const newPacket = createPacket(createModel([new PBRMaterial()]));
		const newContext = createFrameContext(createFrame(newPacket),
			resolveFeatureState({}, {}, "webgpu"));
		const newPrepared = owner.prepareFrame(newContext, createMainFrameOptions());
		const freshMaterialDraw = await owner.getDrawResources(newPacket, newPrepared,
			{ sampleCount: 1 });
		assert.ok(freshMaterialDraw, "evicting another inactive material must not discard a fresh draw");
	} finally {
		owner.destroy();
	}
}

async function testSceneShaderModuleRequestsCoalesce() {
	const backend = new FakeBackend();
	const createShaderModule = backend.createShaderModule.bind(backend);
	let release;
	const gate = new Promise((resolve) => {
		release = resolve;
	});
	let createCalls = 0;
	backend.createShaderModule = async (desc) => {
		createCalls++;
		await gate;
		return createShaderModule(desc);
	};
	const scenePipelines = new WebGPUScenePipelineResources(backend, {});
	const first = scenePipelines._getSceneShaderModule();
	const second = scenePipelines._getSceneShaderModule();
	release();
	const [firstModule, secondModule] = await Promise.all([first, second]);

	assert.equal(createCalls, 1);
	assert.equal(firstModule, secondModule);
	assert.equal(scenePipelines._sceneShaderModuleInFlight.size, 0);
	scenePipelines.invalidateShaderRuntimeCaches();
	assert.equal(backend.shaderModuleDestroyCalls, 1);
}

async function testCustomShaderModuleRequestsCoalesce() {
	const backend = new FakeBackend();
	const createShaderModule = backend.createShaderModule.bind(backend);
	let release;
	const gate = new Promise((resolve) => {
		release = resolve;
	});
	let createCalls = 0;
	backend.createShaderModule = async (desc) => {
		createCalls++;
		await gate;
		return createShaderModule(desc);
	};
	const scenePipelines = new WebGPUScenePipelineResources(backend, {});
	const requestModule = () => scenePipelines._getCustomShaderModule(
		"coalesced-custom-module",
		"@vertex fn vsMain() -> @builtin(position) vec4f { return vec4f(); }",
		"WebGPUShaderMaterialVertex_coalesced",
		"vertex",
		"vsMain",
	);
	const first = requestModule();
	const second = requestModule();
	release();
	const [firstModule, secondModule] = await Promise.all([first, second]);

	assert.equal(createCalls, 1);
	assert.equal(firstModule, secondModule);
	assert.equal(scenePipelines._customShaderModuleInFlight.size, 0);
	scenePipelines.invalidateShaderRuntimeCaches();
	assert.equal(backend.shaderModuleDestroyCalls, 1);
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
