import {
	WEBGPU_PARTICLE_BINDING_SAMPLER,
	WEBGPU_PARTICLE_BINDING_TEXTURE,
	WEBGPU_PARTICLE_BINDING_UV_TRANSFORM,
} from "./particleLayout";
import {
	WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
	WEBGPU_MODEL_BINDING_JOINT_MATRICES,
	WEBGPU_MODEL_BINDING_MORPH_NORMAL,
	WEBGPU_MODEL_BINDING_MORPH_POSITION,
	WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
	WEBGPU_GBUFFER_READ_TEXTURE_COUNT,
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "./constants";

export interface WebGPUPipelineLayouts {
	sceneFrameBindGroupLayout: GPUBindGroupLayout;
	clusteredSceneBindGroupLayout: GPUBindGroupLayout;
	gbufferWriteBindGroupLayout: GPUBindGroupLayout;
	gbufferReadBindGroupLayout: GPUBindGroupLayout;
	deferredUnusedBindGroupLayout: GPUBindGroupLayout;
	environmentFrameBindGroupLayout: GPUBindGroupLayout;
	modelBindGroupLayout: GPUBindGroupLayout;
	particleBindGroupLayout: GPUBindGroupLayout;
	scenePipelineLayout: GPUPipelineLayout;
	sceneGBufferPipelineLayout: GPUPipelineLayout;
	sceneDepthPrepassPipelineLayout: GPUPipelineLayout;
	deferredLightingPipelineLayout: GPUPipelineLayout;
	environmentPipelineLayout: GPUPipelineLayout;
	particlePipelineLayout: GPUPipelineLayout;
}

export function createWebGPUPipelineLayouts(
	device: GPUDevice
): WebGPUPipelineLayouts {
	const sceneFrameBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUSceneFrameBindGroupLayout",
		entries: [
			{
				binding: 0,
				visibility:
					GPUShaderStage.VERTEX |
					GPUShaderStage.FRAGMENT |
					GPUShaderStage.COMPUTE,
				buffer: { type: "uniform" },
			},
			{
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				texture: { sampleType: "depth" },
			},
			{
				binding: 2,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				texture: { sampleType: "float" },
			},
			{
				binding: 3,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				sampler: { type: "filtering" },
			},
			{
				binding: 4,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				texture: { sampleType: "float" },
			},
			{
				binding: 5,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				sampler: { type: "filtering" },
			},
			{
				binding: 6,
				visibility:
					GPUShaderStage.VERTEX |
					GPUShaderStage.FRAGMENT |
					GPUShaderStage.COMPUTE,
				buffer: { type: "uniform" },
			},
			{
				binding: 7,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: { type: "read-only-storage" },
			},
			{
				binding: 8,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				texture: { sampleType: "float" },
			},
			{
				binding: 9,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "float" },
			},
		],
	});
	const environmentFrameBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUEnvironmentFrameBindGroupLayout",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: { type: "uniform" },
			},
			{
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "float" },
			},
			{
				binding: 2,
				visibility: GPUShaderStage.FRAGMENT,
				sampler: { type: "filtering" },
			},
			{
				binding: 3,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: { type: "uniform" },
			},
		],
	});
	const clusteredSceneBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUClusteredSceneBindGroupLayout",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: { type: "uniform" },
			},
			{
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: { type: "read-only-storage" },
			},
			{
				binding: 2,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: { type: "read-only-storage" },
			},
			{
				binding: 3,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: { type: "read-only-storage" },
			},
		],
	});
	const gbufferWriteBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUGBufferWriteBindGroupLayout",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.FRAGMENT,
				storageTexture: {
					access: "write-only",
					format: "rgba16float",
					viewDimension: "2d",
				},
			},
			{
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT,
				storageTexture: {
					access: "write-only",
					format: "rgba16float",
					viewDimension: "2d",
				},
			},
			{
				binding: 2,
				visibility: GPUShaderStage.FRAGMENT,
				storageTexture: {
					access: "write-only",
					format: "rgba16float",
					viewDimension: "2d",
				},
			},
		],
	});
	const gbufferReadEntries: GPUBindGroupLayoutEntry[] = [];
	for (let binding = 0; binding < WEBGPU_GBUFFER_READ_TEXTURE_COUNT; binding++) {
		gbufferReadEntries.push({
			binding,
			visibility: GPUShaderStage.FRAGMENT,
			texture: { sampleType: "float" },
		});
	}
	const gbufferReadBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUGBufferReadBindGroupLayout",
		entries: gbufferReadEntries,
	});
	const deferredUnusedBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUDeferredUnusedBindGroupLayout",
		entries: [],
	});

	const modelEntries: GPUBindGroupLayoutEntry[] = [
		{
			binding: 0,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
			buffer: { type: "uniform" },
		},
	];

	for (let i = 0; i < WEBGPU_TEXTURE_SLOT_COUNT; i++) {
		modelEntries.push({
			binding: 1 + i * 2,
			visibility: GPUShaderStage.FRAGMENT,
			texture: { sampleType: "float" },
		});
		if (i < WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT) {
			modelEntries.push({
				binding: 2 + i * 2,
				visibility: GPUShaderStage.FRAGMENT,
				sampler: { type: "filtering" },
			});
		}
	}

	modelEntries.push(
		{
			binding: WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
			visibility: GPUShaderStage.VERTEX,
			buffer: { type: "uniform" },
		},
		{
			binding: WEBGPU_MODEL_BINDING_JOINT_MATRICES,
			visibility: GPUShaderStage.VERTEX,
			buffer: { type: "read-only-storage" },
		},
		{
			binding: WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
			visibility: GPUShaderStage.VERTEX,
			buffer: { type: "read-only-storage" },
		},
		{
			binding: WEBGPU_MODEL_BINDING_MORPH_POSITION,
			visibility: GPUShaderStage.VERTEX,
			buffer: { type: "read-only-storage" },
		},
		{
			binding: WEBGPU_MODEL_BINDING_MORPH_NORMAL,
			visibility: GPUShaderStage.VERTEX,
			buffer: { type: "read-only-storage" },
		}
	);

	const modelBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUModelBindGroupLayout",
		entries: modelEntries,
	});
	const particleBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUParticleBindGroupLayout",
		entries: [
			{
				binding: WEBGPU_PARTICLE_BINDING_TEXTURE,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "float" },
			},
			{
				binding: WEBGPU_PARTICLE_BINDING_SAMPLER,
				visibility: GPUShaderStage.FRAGMENT,
				sampler: { type: "filtering" },
			},
			{
				binding: WEBGPU_PARTICLE_BINDING_UV_TRANSFORM,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: { type: "uniform" },
			},
		],
	});

	const scenePipelineLayout = device.createPipelineLayout({
		label: "WebGPUScenePipelineLayout",
		bindGroupLayouts: [
			sceneFrameBindGroupLayout,
			modelBindGroupLayout,
			clusteredSceneBindGroupLayout,
		],
	});
	const sceneGBufferPipelineLayout = device.createPipelineLayout({
		label: "WebGPUSceneGBufferPipelineLayout",
		bindGroupLayouts: [
			sceneFrameBindGroupLayout,
			modelBindGroupLayout,
			clusteredSceneBindGroupLayout,
			gbufferWriteBindGroupLayout,
		],
	});
	const sceneDepthPrepassPipelineLayout = device.createPipelineLayout({
		label: "WebGPUSceneDepthPrepassPipelineLayout",
		bindGroupLayouts: [
			sceneFrameBindGroupLayout,
			modelBindGroupLayout,
			clusteredSceneBindGroupLayout,
		],
	});
	const deferredLightingPipelineLayout = device.createPipelineLayout({
		label: "WebGPUDeferredLightingPipelineLayout",
		bindGroupLayouts: [
			sceneFrameBindGroupLayout,
			deferredUnusedBindGroupLayout,
			clusteredSceneBindGroupLayout,
			gbufferReadBindGroupLayout,
		],
	});
	const environmentPipelineLayout = device.createPipelineLayout({
		label: "WebGPUEnvironmentPipelineLayout",
		bindGroupLayouts: [environmentFrameBindGroupLayout],
	});
	const particlePipelineLayout = device.createPipelineLayout({
		label: "WebGPUParticlePipelineLayout",
		bindGroupLayouts: [sceneFrameBindGroupLayout, particleBindGroupLayout],
	});

	return {
		sceneFrameBindGroupLayout,
		clusteredSceneBindGroupLayout,
		gbufferWriteBindGroupLayout,
		gbufferReadBindGroupLayout,
		deferredUnusedBindGroupLayout,
		environmentFrameBindGroupLayout,
		modelBindGroupLayout,
		particleBindGroupLayout,
		scenePipelineLayout,
		sceneGBufferPipelineLayout,
		sceneDepthPrepassPipelineLayout,
		deferredLightingPipelineLayout,
		environmentPipelineLayout,
		particlePipelineLayout,
	};
}
