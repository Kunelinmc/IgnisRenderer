import { WEBGPU_SCENE_SHADER } from "../../../shaders/webgpu/sceneShader";
import { WEBGPU_SKYBOX_SHADER } from "../../../shaders/webgpu/skyboxShader";
import { createWebGPUMaterialUniformData } from "./";
import { TextureFormat } from "../types";
import type { Material } from "../../../materials/Material";
import type { IRenderPipeline, IShaderModule } from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";

export type WebGPUSceneTargetMode = "single" | "mrt";

export class WebGPUPipelineLibrary {
	private _backend: WebGPUBackend;
	private _layouts: WebGPUPipelineLayouts;
	private _sceneShaderModule: IShaderModule | null = null;
	private _skyboxShaderModule: IShaderModule | null = null;
	private _skyboxPipelines = new Map<WebGPUSceneTargetMode, IRenderPipeline>();
	private _materialPipelineCache = new WeakMap<
		Material,
		{ key: string; mode: WebGPUSceneTargetMode; pipeline: IRenderPipeline }
	>();
	private _pipelineCache = new Map<string, IRenderPipeline>();

	constructor(backend: WebGPUBackend, layouts: WebGPUPipelineLayouts) {
		this._backend = backend;
		this._layouts = layouts;
	}

	public async init(): Promise<void> {
		await Promise.all([
			this._getSceneShaderModule(),
			this._getSkyboxShaderModule(),
		]);
	}

	public async getPipeline(
		material: Material,
		mode: WebGPUSceneTargetMode = "mrt"
	): Promise<IRenderPipeline> {
		const materialData = createWebGPUMaterialUniformData(material);
		const cached = this._materialPipelineCache.get(material);
		if (
			cached &&
			cached.key === materialData.pipelineKey &&
			cached.mode === mode
		) {
			return cached.pipeline;
		}

		const cacheKey = `${materialData.pipelineKey}|${mode}`;
		let pipeline = this._pipelineCache.get(cacheKey);
		if (!pipeline) {
			pipeline = await this._createPipeline(
				material,
				materialData.pipelineKey,
				mode
			);
			this._pipelineCache.set(cacheKey, pipeline);
		}

		this._materialPipelineCache.set(material, {
			key: materialData.pipelineKey,
			mode,
			pipeline,
		});

		return pipeline;
	}

	private async _createPipeline(
		material: Material,
		pipelineKey: string,
		mode: WebGPUSceneTargetMode
	): Promise<IRenderPipeline> {
		const shaderModule = await this._getSceneShaderModule();
		const fragmentTargets =
			mode === "mrt"
				? [
						{ format: TextureFormat.RGBA16Float },
						{ format: TextureFormat.RGBA8Unorm },
						{ format: TextureFormat.RGBA16Float },
						{ format: TextureFormat.RGBA16Float },
						{ format: TextureFormat.RGBA16Float },
					]
				: [{ format: this._backend.canvasFormat as any }];
		const fragmentEntryPoint = mode === "mrt" ? "fsMain" : "fsMainSingle";

		return this._backend.createPipeline({
			layout: this._layouts.scenePipelineLayout,
			label: `WebGPUScenePipeline_${pipelineKey}_${mode}`,
			vertex: {
				module: shaderModule,
				entryPoint: "vsMain",
				buffers: [
					{
						arrayStride: 56,
						attributes: [
							{ format: "float32x3", offset: 0, shaderLocation: 0 },
							{ format: "float32x2", offset: 24, shaderLocation: 1 },
							{ format: "float32x3", offset: 12, shaderLocation: 2 },
							{ format: "float32x4", offset: 32, shaderLocation: 3 },
							{ format: "float32x2", offset: 48, shaderLocation: 4 },
						],
					},
				],
			},
			fragment: {
				module: shaderModule,
				entryPoint: fragmentEntryPoint,
				targets: fragmentTargets as any,
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: material.doubleSided ? "none" : "back",
				frontFace: "ccw",
			},
			depthStencil: {
				format: TextureFormat.Depth32Float,
				depthWriteEnabled: true,
				depthCompare: "less",
			},
		} as any);
	}

	public async getSkyboxPipeline(
		mode: WebGPUSceneTargetMode = "single"
	): Promise<IRenderPipeline> {
		const cached = this._skyboxPipelines.get(mode);
		if (cached) {
			return cached;
		}

		const shaderModule = await this._getSkyboxShaderModule();
		const targetFormat =
			mode === "mrt" ? TextureFormat.RGBA16Float : (this._backend.canvasFormat as any);
		const depthFormat =
			mode === "mrt" ? TextureFormat.Depth32Float : TextureFormat.Depth24Plus;
		const pipeline = this._backend.createPipeline({
			layout: this._layouts.skyboxPipelineLayout,
			label: `WebGPUSkyboxPipeline_${mode}`,
			vertex: {
				module: shaderModule,
				entryPoint: "vsMain",
			},
			fragment: {
				module: shaderModule,
				entryPoint: "fsMain",
				targets: [{ format: targetFormat }],
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: "none",
				frontFace: "ccw",
			},
			depthStencil: {
				format: depthFormat,
				depthWriteEnabled: false,
				depthCompare: "always",
			},
		} as any);
		this._skyboxPipelines.set(mode, pipeline);
		return pipeline;
	}

	private async _getSceneShaderModule(): Promise<IShaderModule> {
		if (!this._sceneShaderModule) {
			this._sceneShaderModule = await this._backend.createShaderModule({
				code: WEBGPU_SCENE_SHADER,
				label: "WebGPUSceneShader",
			});
		}

		return this._sceneShaderModule;
	}

	private async _getSkyboxShaderModule(): Promise<IShaderModule> {
		if (!this._skyboxShaderModule) {
			this._skyboxShaderModule = await this._backend.createShaderModule({
				code: WEBGPU_SKYBOX_SHADER,
				label: "WebGPUSkyboxShader",
			});
		}

		return this._skyboxShaderModule;
	}
}
