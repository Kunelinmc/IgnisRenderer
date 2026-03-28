import { createInlineCompositeShaderSource } from "../../shaders/runtime";
import { getWebGPUSceneShaderComposite } from "../../shaders/webgpu/sceneShader";
import { getWebGPUSkyboxShaderComposite } from "../../shaders/webgpu/skyboxShader";
import { createWebGPUMaterialUniformData } from "./";
import { WEBGPU_SCENE_VERTEX_STRIDE } from "./constants";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { TextureFormat } from "../types";
import type { PrimitiveDrawTopology } from "../../core/types";
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
	sampleCount: number;
	topology: PrimitiveDrawTopology;
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
	private _disposeShaderRuntimeListener: (() => void) | null = null;
	private _sceneShaderModule: IShaderModule | null = null;
	private _skyboxShaderModule: IShaderModule | null = null;
	private _customShaderModuleCache = new Map<string, IShaderModule>();
	private _skyboxPipelines = new Map<string, IRenderPipeline>();
	private _materialPipelineCache = new WeakMap<Material, CachedPipelineEntry>();
	private _pipelineCache = new Map<string, IRenderPipeline>();

	constructor(backend: WebGPUBackend, layouts: WebGPUPipelineLayouts) {
		this._backend = backend;
		this._layouts = layouts;
		const shaderRuntime = this._getShaderRuntime();
		if (shaderRuntime && typeof shaderRuntime.onDidChange === "function") {
			this._disposeShaderRuntimeListener = shaderRuntime.onDidChange(() =>
				this.invalidateShaderRuntimeCaches()
			);
		}
	}

	public destroy(): void {
		this._disposeShaderRuntimeListener?.();
		this._disposeShaderRuntimeListener = null;
		this.invalidateShaderRuntimeCaches();
	}

	public invalidateShaderRuntimeCaches(): void {
		this._sceneShaderModule = null;
		this._skyboxShaderModule = null;
		this._customShaderModuleCache.clear();
		this._skyboxPipelines.clear();
		this._materialPipelineCache = new WeakMap<Material, CachedPipelineEntry>();
		this._pipelineCache.clear();
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
		isWireframe = false,
		topology: PrimitiveDrawTopology = DEFAULT_PRIMITIVE_DRAW_TOPOLOGY
	): Promise<IRenderPipeline> {
		const sampleCount = this._resolveSampleCount(mode);
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
			cached.shaderKey === shaderKey &&
			cached.sampleCount === sampleCount &&
			cached.topology === topology
		) {
			return cached.pipeline;
		}

		const cacheKey =
			`${pipelineKey}|${mode}|${shaderKey}` +
			`|topology:${topology}|msaa:${sampleCount}`;
		let pipeline = this._pipelineCache.get(cacheKey);
		if (!pipeline) {
			pipeline = await this._createPipeline(
				material,
				mode,
				isWireframe,
				topology
			);
			this._pipelineCache.set(cacheKey, pipeline);
		}

		this._materialPipelineCache.set(material, {
			key: pipelineKey,
			mode,
			shaderKey,
			sampleCount,
			topology,
			pipeline,
		});

		return pipeline;
	}

	private async _createPipeline(
		material: Material,
		mode: WebGPUSceneTargetMode,
		isWireframe: boolean,
		topology: PrimitiveDrawTopology
	): Promise<IRenderPipeline> {
		const sampleCount = this._resolveSampleCount(mode);
		const { pipelineKey } = createWebGPUMaterialUniformData(
			material,
			isWireframe
		);
		const sceneProgram = await this._resolveSceneProgram(material, mode);
		const fragmentTargets =
			mode === "mrt" ?
				[
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA8Unorm },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
				]
			:	[{ format: this._backend.canvasFormat as any }];

		const effectiveTopology = isWireframe ? "line-list" : topology;
		const triangleTopology =
			effectiveTopology === DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;

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
				topology: effectiveTopology as any,
				cullMode:
					isWireframe || !triangleTopology ? "none" : (material.cullMode as any),
				frontFace: "ccw",
			},
			depthStencil: {
				format: TextureFormat.Depth32Float,
				depthWriteEnabled: true,
				depthCompare: "less",
			},
			sampleCount,
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

		try {
			const program = material.resolveWebGPUProgram(mode);
			const shaderCacheKey = material.getWebGPUCacheKey();
			const vertexModule = await this._getCustomShaderModule(
				`${shaderCacheKey}:${mode}:vertex`,
				program.vertexCode,
				`WebGPUShaderMaterialVertex_${shaderCacheKey}`,
				"vertex",
				program.vertexEntryPoint
			);
			const fragmentModule = await this._getCustomShaderModule(
				`${shaderCacheKey}:${mode}:fragment`,
				program.fragmentCode,
				`WebGPUShaderMaterialFragment_${shaderCacheKey}_${mode}`,
				"fragment",
				program.fragmentEntryPoint
			);

			return {
				vertexModule,
				fragmentModule,
				vertexEntryPoint: program.vertexEntryPoint,
				fragmentEntryPoint: program.fragmentEntryPoint,
			};
		} catch (error) {
			if (!this._isWarnMode()) {
				throw error;
			}
			this._warnOnce(
				`webgpu-shader-material-compile-failed-${material.shaderId}`,
				`ShaderMaterial ${material.name} custom WebGPU shader compile failed; ` +
					`using built-in scene shader. ${String(error)}`
			);
			const shaderModule = await this._getSceneShaderModule();
			return {
				vertexModule: shaderModule,
				fragmentModule: shaderModule,
				vertexEntryPoint: "vsMain",
				fragmentEntryPoint: mode === "mrt" ? "fsMain" : "fsMainSingle",
			};
		}
	}

	private async _getCustomShaderModule(
		key: string,
		code: string,
		label: string,
		stage: "vertex" | "fragment",
		entryPoint: string
	): Promise<IShaderModule> {
		let module = this._customShaderModuleCache.get(key);
		if (!module) {
			const composite = createInlineCompositeShaderSource(
				code,
				`<shader-material:${key}>`,
				"source"
			);
			module = await this._backend.createShaderModule({
				code: composite.code,
				sourceMap: composite.sourceMap,
				label,
				language: "wgsl",
				stage,
				entryPoint,
				sourceKind: "custom-material",
			});
			this._customShaderModuleCache.set(key, module);
		}
		return module;
	}

	private _getShaderCacheKey(material: Material): string {
		if (material instanceof ShaderMaterial) {
			return (
				`shader:${material.getWebGPUCacheKey()}` +
				`|runtime:${this._getShaderRuntimeRevision()}`
			);
		}
		return `builtin-scene|runtime:${this._getShaderRuntimeRevision()}`;
	}

	public async getSkyboxPipeline(
		mode: WebGPUSceneTargetMode = "single"
	): Promise<IRenderPipeline> {
		const sampleCount = this._resolveSampleCount(mode);
		const cacheKey = `${mode}|msaa:${sampleCount}`;
		const cached = this._skyboxPipelines.get(cacheKey);
		if (cached) {
			return cached;
		}

		const shaderModule = await this._getSkyboxShaderModule();
		const targetFormat =
			mode === "mrt" ?
				TextureFormat.RGBA16Float
			:	(this._backend.canvasFormat as any);
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
			sampleCount,
		} as any);
		this._skyboxPipelines.set(cacheKey, pipeline);
		return pipeline;
	}

	private _resolveSampleCount(mode: WebGPUSceneTargetMode): number {
		if (mode !== "mrt") {
			return 1;
		}
		const getter = (this._backend as { getMSAASampleCount?: () => number })
			.getMSAASampleCount;
		if (typeof getter !== "function") {
			return 1;
		}
		const sampleCount = getter.call(this._backend);
		if (!Number.isFinite(sampleCount)) {
			return 1;
		}
		return Math.max(1, Math.floor(sampleCount));
	}

	private async _getSceneShaderModule(): Promise<IShaderModule> {
		if (!this._sceneShaderModule) {
			const shader = await getWebGPUSceneShaderComposite();
			this._sceneShaderModule = await this._backend.createShaderModule({
				code: shader.code,
				sourceMap: shader.sourceMap,
				label: "WebGPUSceneShader",
				language: "wgsl",
				stage: "unknown",
				sourceKind: "builtin-scene",
			});
		}

		return this._sceneShaderModule;
	}

	private async _getSkyboxShaderModule(): Promise<IShaderModule> {
		if (!this._skyboxShaderModule) {
			const shader = await getWebGPUSkyboxShaderComposite();
			this._skyboxShaderModule = await this._backend.createShaderModule({
				code: shader.code,
				sourceMap: shader.sourceMap,
				label: "WebGPUSkyboxShader",
				language: "wgsl",
				stage: "unknown",
				sourceKind: "builtin-skybox",
			});
		}

		return this._skyboxShaderModule;
	}

	private _isWarnMode(): boolean {
		const shaderRuntime = this._getShaderRuntime();
		if (!shaderRuntime || typeof shaderRuntime.getMode !== "function") {
			return false;
		}
		return shaderRuntime.getMode() === "warn";
	}

	private _warnOnce(key: string, message: string): void {
		const backend = this._backend as unknown as {
			warnOnce?: (warnKey: string, warnMessage: string) => void;
		};
		if (typeof backend.warnOnce === "function") {
			backend.warnOnce(key, message);
			return;
		}
		console.warn(message);
	}

	private _getShaderRuntimeRevision(): number {
		const shaderRuntime = this._getShaderRuntime();
		if (!shaderRuntime || typeof shaderRuntime.revision !== "number") {
			return 0;
		}
		return shaderRuntime.revision;
	}

	private _getShaderRuntime():
		| {
				revision?: number;
				getMode?: () => "strict" | "warn" | "silent";
				onDidChange?: (listener: () => void) => () => void;
		  }
		| null {
		const backend = this._backend as unknown as {
			shaderRuntime?: {
				revision?: number;
				getMode?: () => "strict" | "warn" | "silent";
				onDidChange?: (listener: () => void) => () => void;
			};
		};
		return backend.shaderRuntime ?? null;
	}
}
