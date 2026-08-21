import {
	WEBGPU_GBUFFER_READ_TEXTURE_COUNT,
	WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
	WEBGPU_MODEL_BINDING_JOINT_MATRICES,
	WEBGPU_MODEL_BINDING_MORPH_NORMAL,
	WEBGPU_MODEL_BINDING_MORPH_POSITION,
	WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	WEBGPU_MODEL_BINDING_STATIC_INSTANCES,
	WEBGPU_PARTICLE_BINDING_SAMPLER,
	WEBGPU_PARTICLE_BINDING_TEXTURE,
	WEBGPU_PARTICLE_BINDING_UV_TRANSFORM,
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "./constants";

const WEBGPU_DECAL_BATCH_STORAGE_TEXTURE_COUNT = WEBGPU_GBUFFER_READ_TEXTURE_COUNT;
const WEBGPU_DECAL_BATCH_STORAGE_BUFFER_COUNT = 3;

export interface WebGPUPipelineLayouts {
	sceneFrameBindGroupLayout: GPUBindGroupLayout;
	decalFrameBindGroupLayout: GPUBindGroupLayout;
	clusteredSceneBindGroupLayout: GPUBindGroupLayout;
	gbufferWriteBindGroupLayout: GPUBindGroupLayout;
	gbufferReadBindGroupLayout: GPUBindGroupLayout;
	decalBindGroupLayout: GPUBindGroupLayout;
	decalOutputBindGroupLayout: GPUBindGroupLayout;
	decalBatchBindGroupLayout: GPUBindGroupLayout;
	planarReflectionBindGroupLayout: GPUBindGroupLayout;
	deferredUnusedBindGroupLayout: GPUBindGroupLayout;
	environmentFrameBindGroupLayout: GPUBindGroupLayout;
	modelBindGroupLayout: GPUBindGroupLayout;
	particleBindGroupLayout: GPUBindGroupLayout;
	scenePipelineLayout: GPUPipelineLayout;
	sceneGBufferPipelineLayout: GPUPipelineLayout;
	sceneDepthPrepassPipelineLayout: GPUPipelineLayout;
	planarReflectionPipelineLayout: GPUPipelineLayout;
	deferredLightingPipelineLayout: GPUPipelineLayout;
	decalPipelineLayout: GPUPipelineLayout;
	decalBatchPipelineLayout: GPUPipelineLayout;
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
			{
				binding: 10,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "float" },
			},
			{
				binding: 11,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				texture: { sampleType: "uint" },
			},
			{
				binding: 12,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				texture: { sampleType: "depth" },
			},
			{
				binding: 13,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				sampler: { type: "comparison" },
			},
			{
				binding: 14,
				visibility:
					GPUShaderStage.VERTEX |
					GPUShaderStage.FRAGMENT |
					GPUShaderStage.COMPUTE,
				buffer: { type: "uniform" },
			},
			{
				binding: 15,
				visibility:
					GPUShaderStage.VERTEX |
					GPUShaderStage.FRAGMENT |
					GPUShaderStage.COMPUTE,
				buffer: { type: "uniform" },
			},
			{
				binding: 16,
				visibility:
					GPUShaderStage.VERTEX |
					GPUShaderStage.FRAGMENT |
					GPUShaderStage.COMPUTE,
				buffer: { type: "uniform" },
			},
		],
	});
	const decalFrameBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUDecalFrameBindGroupLayout",
		entries: [
			{
				binding: 0,
				visibility:
					GPUShaderStage.VERTEX |
					GPUShaderStage.FRAGMENT |
					GPUShaderStage.COMPUTE,
				buffer: { type: "uniform" },
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
		entries: Array.from({ length: 8 }, (_, binding) => ({
			binding,
			visibility: GPUShaderStage.FRAGMENT,
			buffer: {
				type: binding === 0 ? "uniform" : "read-only-storage",
			} as GPUBufferBindingLayout,
		})),
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
					format: "rgba16uint",
					viewDimension: "2d",
				},
			},
		],
	});
	const gbufferReadEntries: GPUBindGroupLayoutEntry[] = [];
	for (let binding = 0; binding < WEBGPU_GBUFFER_READ_TEXTURE_COUNT - 1; binding++) {
		gbufferReadEntries.push({
			binding,
			visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
			texture: { sampleType: "float" },
		});
	}
	gbufferReadEntries.push({
		binding: WEBGPU_GBUFFER_READ_TEXTURE_COUNT - 1,
		visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
		texture: { sampleType: "uint" },
	});
	const gbufferReadBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUGBufferReadBindGroupLayout",
		entries: gbufferReadEntries,
	});
	const decalEntries: GPUBindGroupLayoutEntry[] = [
		{
			binding: 0,
			visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
			buffer: { type: "uniform" },
		},
	];
	for (let i = 0; i < WEBGPU_TEXTURE_SLOT_COUNT; i++) {
		decalEntries.push({
			binding: 1 + i * 2,
			visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
			texture: { sampleType: "float" },
		});
		if (i < WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT) {
			decalEntries.push({
				binding: 2 + i * 2,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				sampler: { type: "filtering" },
			});
		}
	}
	const decalBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUDecalBindGroupLayout",
		entries: decalEntries,
	});
	const decalOutputEntries: GPUBindGroupLayoutEntry[] = [
		{
			binding: 11,
			visibility: GPUShaderStage.FRAGMENT,
			storageTexture: {
				access: "write-only",
				format: "rgba16float",
				viewDimension: "2d",
			},
		},
		{
			binding: 12,
			visibility: GPUShaderStage.FRAGMENT,
			storageTexture: {
				access: "write-only",
				format: "rgba16uint",
				viewDimension: "2d",
			},
		},
	];
	const decalOutputBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUDecalOutputBindGroupLayout",
		entries: decalOutputEntries,
	});
	const decalBatchEntries: GPUBindGroupLayoutEntry[] =
		deviceSupportsDecalBatchBindGroupLayout(device) ?
			[
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "uniform" },
				},
				{
					binding: 1,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "read-only-storage" },
				},
				{
					binding: 2,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "read-only-storage" },
				},
				{
					binding: 3,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "read-only-storage" },
				},
				{
					binding: 4,
					visibility: GPUShaderStage.COMPUTE,
					storageTexture: {
						access: "write-only",
						format: "rgba8unorm",
						viewDimension: "2d",
					},
				},
			]
		:	[];
	if (decalBatchEntries.length > 0) {
		const batchFormats = [
			"rgba8unorm",
			"rgba16float",
			"rgba16float",
			"rgba16float",
			"rgba16float",
			"rgba8unorm",
			"rgba16float",
			"rgba16uint",
		] as const;
		for (let index = 0; index < batchFormats.length; index++) {
			decalBatchEntries.push({
				binding: 5 + index,
				visibility: GPUShaderStage.COMPUTE,
				storageTexture: {
					access: "write-only",
					format: batchFormats[index],
					viewDimension: "2d",
				},
			});
		}
	}
	const decalBatchBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUDecalBatchBindGroupLayout",
		entries: decalBatchEntries,
	});
	const deferredUnusedBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUDeferredUnusedBindGroupLayout",
		entries: [],
	});
	const planarReflectionBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUPlanarReflectionBindGroupLayout",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "float" },
			},
		],
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
		},
		{
			binding: WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
			buffer: { type: "uniform" },
		},
		{
			binding: WEBGPU_MODEL_BINDING_STATIC_INSTANCES,
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
	const planarReflectionPipelineLayout = device.createPipelineLayout({
		label: "WebGPUPlanarReflectionPipelineLayout",
		bindGroupLayouts: [
			sceneFrameBindGroupLayout,
			modelBindGroupLayout,
			planarReflectionBindGroupLayout,
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
	const decalPipelineLayout = device.createPipelineLayout({
		label: "WebGPUDecalPipelineLayout",
		bindGroupLayouts: [
			decalFrameBindGroupLayout,
			gbufferReadBindGroupLayout,
			decalBindGroupLayout,
			decalOutputBindGroupLayout,
		],
	});
	const decalBatchPipelineLayout = device.createPipelineLayout({
		label: "WebGPUDecalBatchPipelineLayout",
		bindGroupLayouts: [
			decalFrameBindGroupLayout,
			gbufferReadBindGroupLayout,
			decalBindGroupLayout,
			decalBatchBindGroupLayout,
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
		decalFrameBindGroupLayout,
		clusteredSceneBindGroupLayout,
		gbufferWriteBindGroupLayout,
		gbufferReadBindGroupLayout,
		decalBindGroupLayout,
		decalOutputBindGroupLayout,
		decalBatchBindGroupLayout,
		planarReflectionBindGroupLayout,
		deferredUnusedBindGroupLayout,
		environmentFrameBindGroupLayout,
		modelBindGroupLayout,
		particleBindGroupLayout,
		scenePipelineLayout,
		sceneGBufferPipelineLayout,
		sceneDepthPrepassPipelineLayout,
		planarReflectionPipelineLayout,
		deferredLightingPipelineLayout,
		decalPipelineLayout,
		decalBatchPipelineLayout,
		environmentPipelineLayout,
		particlePipelineLayout,
	};
}

function deviceSupportsDecalBatchBindGroupLayout(device: GPUDevice): boolean {
	const limits = device.limits;
	const maxStorageTextures =
		limits?.maxStorageTexturesPerShaderStage ?? Number.POSITIVE_INFINITY;
	const maxStorageBuffers =
		limits?.maxStorageBuffersPerShaderStage ?? Number.POSITIVE_INFINITY;
	return (
		maxStorageTextures >= WEBGPU_DECAL_BATCH_STORAGE_TEXTURE_COUNT &&
		maxStorageBuffers >= WEBGPU_DECAL_BATCH_STORAGE_BUFFER_COUNT
	);
}
