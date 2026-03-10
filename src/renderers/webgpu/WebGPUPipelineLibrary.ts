import { getWebGPUSceneShader } from "../../shaders/webgpu/sceneShader";
import { getWebGPUSkyboxShader } from "../../shaders/webgpu/skyboxShader";
import { createWebGPUMaterialUniformData } from "./";
import { WEBGPU_SCENE_VERTEX_STRIDE } from "./constants";
import { TextureFormat } from "../types";
import type { Material } from "../../materials/Material";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import type { IRenderPipeline, IShaderModule } from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";

export type WebGPUSceneTargetMode = "single" | "mrt";

interface CachedPipelineEntry {
	key: string;
	mode: WebGPUSceneTargetMode;
	shaderKey: string;
	pipeline: IRenderPipeline;
}

interface WebGPUSceneProgram {
	vertexModule: IShaderModule;
	fragmentModule: IShaderModule;
	vertexEntryPoint: string;
	fragmentEntryPoint: string;
}

export class WebGPUPipelineLibrary {
	private _backend: WebGPUBackend;
	private _layouts: WebGPUPipelineLayouts;
	private _sceneShaderModule: IShaderModule | null = null;
	private _skyboxShaderModule: IShaderModule | null = null;
	private _customShaderModuleCache = new Map<string, IShaderModule>();
	private _skyboxPipelines = new Map<WebGPUSceneTargetMode, IRenderPipeline>();
	private _materialPipelineCache = new WeakMap<Material, CachedPipelineEntry>();
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
		mode: WebGPUSceneTargetMode = "single",
		isWireframe = false
	): Promise<IRenderPipeline> {
		const { pipelineKey } = createWebGPUMaterialUniformData(
			material,
			isWireframe
		);
		const shaderKey = this._getShaderCacheKey(material);
		const cached = this._materialPipelineCache.get(material);
		if (
			cached &&
			cached.key === pipelineKey &&
			cached.mode === mode &&
			cached.shaderKey === shaderKey
		) {
			return cached.pipeline;
		}

		const cacheKey = `${pipelineKey}|${mode}|${shaderKey}`;
		let pipeline = this._pipelineCache.get(cacheKey);
		if (!pipeline) {
			pipeline = await this._createPipeline(
				material,
				mode,
				isWireframe
			);
			this._pipelineCache.set(cacheKey, pipeline);
		}

		this._materialPipelineCache.set(material, {
			key: pipelineKey,
			mode,
			shaderKey,
			pipeline,
		});

		return pipeline;
	}

	private async _createPipeline(
		material: Material,
		mode: WebGPUSceneTargetMode,
		isWireframe: boolean
	): Promise<IRenderPipeline> {
		const { pipelineKey } = createWebGPUMaterialUniformData(material, isWireframe);
		const sceneProgram = await this._resolveSceneProgram(material, mode);
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

		return this._backend.createPipeline({
			layout: this._layouts.scenePipelineLayout,
			label: `WebGPUScenePipeline_${pipelineKey}_${mode}`,
			vertex: {
				module: sceneProgram.vertexModule,
				entryPoint: sceneProgram.vertexEntryPoint,
				buffers: [
					{
						arrayStride: WEBGPU_SCENE_VERTEX_STRIDE,
						attributes: [
							{ format: "float32x3", offset: 0, shaderLocation: 0 },
							{ format: "float32x2", offset: 24, shaderLocation: 1 },
							{ format: "float32x3", offset: 12, shaderLocation: 2 },
							{ format: "float32x4", offset: 32, shaderLocation: 3 },
							{ format: "float32x2", offset: 48, shaderLocation: 4 },
							{ format: "float32x4", offset: 56, shaderLocation: 5 },
							{ format: "float32x4", offset: 72, shaderLocation: 6 },
							{ format: "float32x4", offset: 88, shaderLocation: 7 },
							{ format: "float32x4", offset: 104, shaderLocation: 8 },
						],
					},
				],
			},
			fragment: {
				module: sceneProgram.fragmentModule,
				entryPoint: sceneProgram.fragmentEntryPoint,
				targets: fragmentTargets as any,
			},
			primitive: {
				topology: (isWireframe ? "line-list" : "triangle-list") as any,
				cullMode: isWireframe ? "none" : (material.cullMode as any),
				frontFace: "ccw",
			},
			depthStencil: {
				format: TextureFormat.Depth32Float,
				depthWriteEnabled: true,
				depthCompare: "less",
			},
		} as any);
	}

	private async _resolveSceneProgram(
		material: Material,
		mode: WebGPUSceneTargetMode
	): Promise<WebGPUSceneProgram> {
		if (!(material instanceof ShaderMaterial)) {
			const shaderModule = await this._getSceneShaderModule();
			return {
				vertexModule: shaderModule,
				fragmentModule: shaderModule,
				vertexEntryPoint: "vsMain",
				fragmentEntryPoint: mode === "mrt" ? "fsMain" : "fsMainSingle",
			};
		}

		const program = material.resolveWebGPUProgram(mode);
		const shaderCacheKey = material.getWebGPUCacheKey();
		const vertexModule = await this._getCustomShaderModule(
			`${shaderCacheKey}:${mode}:vertex`,
			program.vertexCode,
			`WebGPUShaderMaterialVertex_${shaderCacheKey}`
		);
		const fragmentModule = await this._getCustomShaderModule(
			`${shaderCacheKey}:${mode}:fragment`,
			program.fragmentCode,
			`WebGPUShaderMaterialFragment_${shaderCacheKey}_${mode}`
		);

		return {
			vertexModule,
			fragmentModule,
			vertexEntryPoint: program.vertexEntryPoint,
			fragmentEntryPoint: program.fragmentEntryPoint,
		};
	}

	private async _getCustomShaderModule(
		key: string,
		code: string,
		label: string
	): Promise<IShaderModule> {
		let module = this._customShaderModuleCache.get(key);
		if (!module) {
			module = await this._backend.createShaderModule({
				code,
				label,
			});
			this._customShaderModuleCache.set(key, module);
		}
		return module;
	}

	private _getShaderCacheKey(material: Material): string {
		if (material instanceof ShaderMaterial) {
			return `shader:${material.getWebGPUCacheKey()}`;
		}
		return "builtin-scene";
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
			mode === "mrt"
				? TextureFormat.RGBA16Float
				: (this._backend.canvasFormat as any);
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
			const shaderCode = await getWebGPUSceneShader();
			this._sceneShaderModule = await this._backend.createShaderModule({
				code: shaderCode,
				label: "WebGPUSceneShader",
			});
		}

		return this._sceneShaderModule;
	}

	private async _getSkyboxShaderModule(): Promise<IShaderModule> {
		if (!this._skyboxShaderModule) {
			const shaderCode = await getWebGPUSkyboxShader();
			this._skyboxShaderModule = await this._backend.createShaderModule({
				code: shaderCode,
				label: "WebGPUSkyboxShader",
			});
		}

		return this._skyboxShaderModule;
	}
}
