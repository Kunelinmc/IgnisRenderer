import { WEBGPU_SCENE_SHADER } from "../../../shaders/webgpu/sceneShader";
import { WEBGPU_SKYBOX_SHADER } from "../../../shaders/webgpu/skyboxShader";
import { createWebGPUMaterialUniformData } from "./";
import { TextureFormat } from "../types";
import type { Material } from "../../../materials/Material";
import type { IRenderPipeline, IShaderModule } from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";

export class WebGPUPipelineLibrary {
	private _backend: WebGPUBackend;
	private _layouts: WebGPUPipelineLayouts;
	private _sceneShaderModule: IShaderModule | null = null;
	private _skyboxShaderModule: IShaderModule | null = null;
	private _skyboxPipeline: IRenderPipeline | null = null;
	private _materialPipelineCache = new WeakMap<
		Material,
		{ key: string; pipeline: IRenderPipeline }
	>();
	private _pipelineCache = new Map<string, IRenderPipeline>();

	constructor(backend: WebGPUBackend, layouts: WebGPUPipelineLayouts) {
		this._backend = backend;
		this._layouts = layouts;
	}

	public async init(): Promise<void> {
		await Promise.all([this._getSceneShaderModule(), this._getSkyboxShaderModule()]);
	}

	public async getPipeline(material: Material): Promise<IRenderPipeline> {
		const materialData = createWebGPUMaterialUniformData(material);
		const cached = this._materialPipelineCache.get(material);
		if (cached && cached.key === materialData.pipelineKey) {
			return cached.pipeline;
		}

		let pipeline = this._pipelineCache.get(materialData.pipelineKey);
		if (!pipeline) {
			pipeline = await this._createPipeline(material, materialData.pipelineKey);
			this._pipelineCache.set(materialData.pipelineKey, pipeline);
		}

		this._materialPipelineCache.set(material, {
			key: materialData.pipelineKey,
			pipeline,
		});

		return pipeline;
	}

	private async _createPipeline(
		material: Material,
		pipelineKey: string
	): Promise<IRenderPipeline> {
		const shaderModule = await this._getSceneShaderModule();
		return this._backend.createPipeline({
			layout: this._layouts.scenePipelineLayout,
			label: `WebGPUScenePipeline_${pipelineKey}`,
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
				entryPoint: "fsMain",
				targets: [{ format: this._backend.canvasFormat as any }],
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: material.doubleSided ? "none" : "back",
				frontFace: "ccw",
			},
			depthStencil: {
				format: TextureFormat.Depth24Plus,
				depthWriteEnabled: true,
				depthCompare: "less",
			},
		} as any);
	}

	public async getSkyboxPipeline(): Promise<IRenderPipeline> {
		if (this._skyboxPipeline) {
			return this._skyboxPipeline;
		}

		const shaderModule = await this._getSkyboxShaderModule();
		this._skyboxPipeline = this._backend.createPipeline({
			layout: this._layouts.skyboxPipelineLayout,
			label: "WebGPUSkyboxPipeline",
			vertex: {
				module: shaderModule,
				entryPoint: "vsMain",
			},
			fragment: {
				module: shaderModule,
				entryPoint: "fsMain",
				targets: [{ format: this._backend.canvasFormat as any }],
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: "none",
				frontFace: "ccw",
			},
			depthStencil: {
				format: TextureFormat.Depth24Plus,
				depthWriteEnabled: false,
				depthCompare: "always",
			},
		} as any);

		return this._skyboxPipeline;
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
