import {
	WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
	WEBGPU_MODEL_BINDING_JOINT_MATRICES,
	WEBGPU_MODEL_BINDING_MORPH_NORMAL,
	WEBGPU_MODEL_BINDING_MORPH_POSITION,
	WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
	WEBGPU_MODEL_BINDING_MATERIAL_COMMON,
	WEBGPU_MODEL_BINDING_FLAT_MATERIAL,
	WEBGPU_MODEL_BINDING_PBR_MATERIAL,
	WEBGPU_MODEL_BINDING_PHONG_MATERIAL,
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	WEBGPU_MODEL_BINDING_STATIC_INSTANCES,
	WEBGPU_FLAT_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_MATERIAL_COMMON_UNIFORM_BYTE_SIZE,
	WEBGPU_OBJECT_UNIFORM_BYTE_SIZE,
	WEBGPU_PBR_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_PHONG_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_PARTICLE_BINDING_SAMPLER,
	WEBGPU_PARTICLE_BINDING_TEXTURE,
	WEBGPU_PARTICLE_BINDING_UV_TRANSFORM,
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "./constants";
import type { WebGPUShadingFamily } from "./types";

export interface WebGPUPipelineLayouts {
	sceneFrameBindGroupLayout: GPUBindGroupLayout;
	depthFrameBindGroupLayout: GPUBindGroupLayout;
	depthModelBindGroupLayout: GPUBindGroupLayout;
	depthMaterialBindGroupLayout: GPUBindGroupLayout;
	clusteredSceneBindGroupLayout: GPUBindGroupLayout;
	planarReflectionBindGroupLayout: GPUBindGroupLayout;
	environmentFrameBindGroupLayout: GPUBindGroupLayout;
	modelBindGroupLayouts: Record<WebGPUShadingFamily, GPUBindGroupLayout>;
	particleFrameBindGroupLayout: GPUBindGroupLayout;
	particleBindGroupLayout: GPUBindGroupLayout;
	scenePipelineLayouts: Record<WebGPUShadingFamily, GPUPipelineLayout>;
	sceneDepthPrepassPipelineLayout: GPUPipelineLayout;
	planarReflectionPipelineLayouts: Record<WebGPUShadingFamily, GPUPipelineLayout>;
	environmentPipelineLayout: GPUPipelineLayout;
	particlePipelineLayout: GPUPipelineLayout;
}

export function createWebGPUPipelineLayouts(
	device: GPUDevice
): WebGPUPipelineLayouts {
	const depthFrameBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUDepthFrameBindGroupLayout",
		entries: [{
			binding: 0,
			visibility: GPUShaderStage.VERTEX,
			buffer: { type: "uniform" },
		}],
	});
	const depthModelBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUDepthModelBindGroupLayout",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX,
				buffer: {
					type: "uniform",
					minBindingSize: WEBGPU_OBJECT_UNIFORM_BYTE_SIZE,
					hasDynamicOffset: true,
				},
			},
			{
				binding: WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
				visibility: GPUShaderStage.VERTEX,
				buffer: { type: "uniform" },
			},
			...[
				WEBGPU_MODEL_BINDING_JOINT_MATRICES,
				WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
				WEBGPU_MODEL_BINDING_MORPH_POSITION,
				WEBGPU_MODEL_BINDING_STATIC_INSTANCES,
			].map((binding): GPUBindGroupLayoutEntry => ({
				binding,
				visibility: GPUShaderStage.VERTEX,
				buffer: { type: "read-only-storage" },
			})),
		],
	});
	const depthMaterialBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUDepthMaterialBindGroupLayout",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: { type: "uniform", minBindingSize: 48 },
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
		],
	});
	const sceneDepthPrepassPipelineLayout = device.createPipelineLayout({
		label: "WebGPUSceneDepthPrepassPipelineLayout",
		bindGroupLayouts: [
			depthFrameBindGroupLayout,
			depthModelBindGroupLayout,
			depthMaterialBindGroupLayout,
		],
	});
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
				binding: 6,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				texture: { sampleType: "float" },
			},
			{
				binding: 7,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "float" },
			},
			{
				binding: 8,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: "float" },
			},
			{
				binding: 9,
				visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
				sampler: { type: "comparison" },
			},
			{
				binding: 10,
				visibility:
					GPUShaderStage.VERTEX |
					GPUShaderStage.FRAGMENT |
					GPUShaderStage.COMPUTE,
				buffer: { type: "uniform" },
			},
			{
				binding: 11,
				visibility:
					GPUShaderStage.VERTEX |
					GPUShaderStage.FRAGMENT |
					GPUShaderStage.COMPUTE,
				buffer: { type: "uniform" },
			},
			{
				binding: 12,
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
			buffer: {
				type: "uniform",
				minBindingSize: WEBGPU_OBJECT_UNIFORM_BYTE_SIZE,
				hasDynamicOffset: true,
			},
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
		},
		{
			binding: WEBGPU_MODEL_BINDING_MATERIAL_COMMON,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
			buffer: {
				type: "uniform",
				minBindingSize: WEBGPU_MATERIAL_COMMON_UNIFORM_BYTE_SIZE,
			},
		}
	);

	const modelBindGroupLayouts = {} as Record<
		WebGPUShadingFamily,
		GPUBindGroupLayout
	>;
	for (const family of ["pbr", "phong", "flat", "unlit"] as const) {
		const familyEntries = modelEntries.slice();
		const lightingEntry = createLightingMaterialLayoutEntry(family);
		if (lightingEntry) familyEntries.push(lightingEntry);
		modelBindGroupLayouts[family] = device.createBindGroupLayout({
			label: `WebGPUModelBindGroupLayout_${family}`,
			entries: familyEntries,
		});
	}
	const particleFrameBindGroupLayout = device.createBindGroupLayout({
		label: "WebGPUParticleFrameBindGroupLayout",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: { type: "uniform" },
			},
		],
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

	const scenePipelineLayouts = {} as Record<WebGPUShadingFamily, GPUPipelineLayout>;
	const planarReflectionPipelineLayouts = {} as Record<
		WebGPUShadingFamily,
		GPUPipelineLayout
	>;
	for (const family of ["pbr", "phong", "flat", "unlit"] as const) {
		const modelLayout = modelBindGroupLayouts[family];
		scenePipelineLayouts[family] = device.createPipelineLayout({
			label: `WebGPUScenePipelineLayout_${family}`,
			bindGroupLayouts: [
				sceneFrameBindGroupLayout,
				modelLayout,
				clusteredSceneBindGroupLayout,
			],
		});
		planarReflectionPipelineLayouts[family] = device.createPipelineLayout({
			label: `WebGPUPlanarReflectionPipelineLayout_${family}`,
			bindGroupLayouts: [
				sceneFrameBindGroupLayout,
				modelLayout,
				planarReflectionBindGroupLayout,
			],
		});
	}
	const environmentPipelineLayout = device.createPipelineLayout({
		label: "WebGPUEnvironmentPipelineLayout",
		bindGroupLayouts: [environmentFrameBindGroupLayout],
	});
	const particlePipelineLayout = device.createPipelineLayout({
		label: "WebGPUParticlePipelineLayout",
		bindGroupLayouts: [particleFrameBindGroupLayout, particleBindGroupLayout],
	});

	return {
		sceneFrameBindGroupLayout,
		depthFrameBindGroupLayout,
		depthModelBindGroupLayout,
		depthMaterialBindGroupLayout,
		clusteredSceneBindGroupLayout,
		planarReflectionBindGroupLayout,
		environmentFrameBindGroupLayout,
		modelBindGroupLayouts,
		particleFrameBindGroupLayout,
		particleBindGroupLayout,
		scenePipelineLayouts,
		sceneDepthPrepassPipelineLayout,
		planarReflectionPipelineLayouts,
		environmentPipelineLayout,
		particlePipelineLayout,
	};
}

function createLightingMaterialLayoutEntry(
	family: WebGPUShadingFamily,
): GPUBindGroupLayoutEntry | null {
	switch (family) {
		case "pbr":
			return {
				binding: WEBGPU_MODEL_BINDING_PBR_MATERIAL,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: {
					type: "uniform",
					minBindingSize: WEBGPU_PBR_MATERIAL_UNIFORM_BYTE_SIZE,
				},
			};
		case "phong":
			return {
				binding: WEBGPU_MODEL_BINDING_PHONG_MATERIAL,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: {
					type: "uniform",
					minBindingSize: WEBGPU_PHONG_MATERIAL_UNIFORM_BYTE_SIZE,
				},
			};
		case "flat":
			return {
				binding: WEBGPU_MODEL_BINDING_FLAT_MATERIAL,
				visibility: GPUShaderStage.FRAGMENT,
				buffer: {
					type: "uniform",
					minBindingSize: WEBGPU_FLAT_MATERIAL_UNIFORM_BYTE_SIZE,
				},
			};
		case "unlit":
			return null;
	}
}
