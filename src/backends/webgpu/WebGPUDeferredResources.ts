import { Logger } from "../../foundation/Logger";
import { ShaderSource } from "../../shaders/ShaderSource";
import { TextureFormat } from "../../core/TextureFormat";
import {
	TextureUsage,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderPipeline,
	type IShaderModule,
	type IRenderTexture,
} from "../types";
import type { WebGPUDeferredResourceProvider } from "./WebGPUResourceContracts";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import {
	toShaderCompileError,
	type WarmupPhaseCounters,
} from "../../pipeline/WarmupPlanner";

export interface WebGPUDeferredWarmupRequest {
	readonly active: boolean;
	readonly hasDecals: boolean;
	yieldIfNeeded(): Promise<void>;
}

/**
 * Owns WebGPU deferred layouts, pipelines, and the deferred placeholder group.
 *
 * @internal Owned by the WebGPU frame service owner.
 */
export class WebGPUDeferredResources implements WebGPUDeferredResourceProvider {
	private _deferredLightingShaderModule: IShaderModule | null = null;
	private _deferredLightingPipeline: IRenderPipeline | null = null;
	private _deferredLightingInFlight: Promise<IRenderPipeline> | null = null;
	private _shaderGeneration = 0;
	private _destroyed = false;
	private _decalShaderModule: IShaderModule | null = null;
	private _decalPipeline: IRenderPipeline | null = null;
	private _decalBatchPipeline: IComputePipeline | null = null;
	private _deferredUnusedBinding: IBindingGroup | null = null;
	private _placeholderTextures: {
		rgba16Float: IRenderTexture;
		rgba8Unorm: IRenderTexture;
		rgba16Uint: IRenderTexture;
	} | null = null;

	constructor(
		private readonly _backend: WebGPUDeviceResourceHost,
		private readonly _layouts: WebGPUPipelineLayouts,
	) {}

	public getGBufferReadLayout(): GPUBindGroupLayout {
		return this._layouts.gbufferReadBindGroupLayout;
	}

	public getDecalBindGroupLayout(): GPUBindGroupLayout {
		return this._layouts.decalBindGroupLayout;
	}

	public getDecalOutputBindGroupLayout(): GPUBindGroupLayout {
		return this._layouts.decalOutputBindGroupLayout;
	}

	public getDecalBatchBindGroupLayout(): GPUBindGroupLayout {
		return this._layouts.decalBatchBindGroupLayout;
	}

	public getDeferredUnusedBinding(): IBindingGroup {
		if (!this._deferredUnusedBinding) {
			this._deferredUnusedBinding = this._backend.createBindingGroup({
				layout: this._layouts.deferredUnusedBindGroupLayout,
				entries: [],
				label: "WebGPUDeferredUnusedBinding",
			});
		}
		return this._deferredUnusedBinding;
	}

	public getDeferredPlaceholderTextures(): {
		readonly rgba16Float: IRenderTexture;
		readonly rgba8Unorm: IRenderTexture;
		readonly rgba16Uint: IRenderTexture;
	} {
		if (!this._placeholderTextures) {
			const usage = TextureUsage.TextureBinding | TextureUsage.StorageBinding;
			this._placeholderTextures = {
				rgba16Float: this._backend.createTexture({
					width: 1,
					height: 1,
					format: TextureFormat.RGBA16Float,
					usage,
					label: "WebGPUDeferredPlaceholderRGBA16Float",
				}),
				rgba8Unorm: this._backend.createTexture({
					width: 1,
					height: 1,
					format: TextureFormat.RGBA8Unorm,
					usage,
					label: "WebGPUDeferredPlaceholderRGBA8Unorm",
				}),
				rgba16Uint: this._backend.createTexture({
					width: 1,
					height: 1,
					format: TextureFormat.RGBA16Uint,
					usage,
					label: "WebGPUDeferredPlaceholderRGBA16Uint",
				}),
			};
		}
		return this._placeholderTextures;
	}

	/**
	 * Resolves the shared built-in deferred lighting pipeline.
	 * @internal WebGPU frame orchestration; applications use `Renderer.renderFrame()`.
	 * @returns A cached pipeline, compiled lazily and retained until shader-runtime invalidation.
	 */
	public async getDeferredLightingPipeline(): Promise<IRenderPipeline> {
		if (this._destroyed) throw new Error("WebGPU deferred resources are destroyed.");
		if (this._deferredLightingPipeline) return this._deferredLightingPipeline;
		if (this._deferredLightingInFlight) return this._deferredLightingInFlight;
		const generation = this._shaderGeneration;
		const creation = this._createDeferredLightingPipeline(generation);
		this._deferredLightingInFlight = creation;
		try {
			return await creation;
		} finally {
			if (this._deferredLightingInFlight === creation) {
				this._deferredLightingInFlight = null;
			}
		}
	}

	private async _createDeferredLightingPipeline(
		generation: number,
	): Promise<IRenderPipeline> {
		const shader = await ShaderSource.load("webgpu.deferredLighting");
		if (generation !== this._shaderGeneration || this._destroyed) {
			return this.getDeferredLightingPipeline();
		}
		const shaderModule = await this._backend.createShaderModule({
			code: shader.source.code,
			sourceMap: shader.source.sourceMap,
			label: "WebGPUDeferredLightingShader",
			language: "wgsl", stage: "unknown", sourceKind: "builtin-scene",
		});
		let pipeline: IRenderPipeline;
		try {
			pipeline = await this._backend.createPipeline({
				layout: this._layouts.deferredLightingPipelineLayout,
				label: "WebGPUDeferredLightingPipeline",
				vertex: {
					module: shaderModule,
					entryPoint: "vsMainDeferredLighting",
				},
				fragment: {
					module: shaderModule,
					entryPoint: "fsMainDeferredLighting",
					targets: [{ format: TextureFormat.RGBA16Float }],
				},
				primitive: {
					topology: "triangle-list" as any,
					cullMode: "none",
					frontFace: "ccw",
				},
				sampleCount: 1,
			} as any);
		} catch (error) {
			this._destroyResource(shaderModule);
			throw error;
		}
		if (generation !== this._shaderGeneration || this._destroyed) {
			this._destroyResource(pipeline);
			this._destroyResource(shaderModule);
			return this.getDeferredLightingPipeline();
		}
		this._deferredLightingShaderModule = shaderModule;
		this._deferredLightingPipeline = pipeline;
		return pipeline;
	}

	public async getDecalPipeline(): Promise<IRenderPipeline> {
		if (this._decalPipeline) return this._decalPipeline;
		const shaderModule = await this._getDecalShaderModule();
		this._decalPipeline = await this._backend.createPipeline({
			layout: this._layouts.decalPipelineLayout,
			label: "WebGPUDeferredDecalPipeline",
			vertex: { module: shaderModule, entryPoint: "vsMain" },
			fragment: {
				module: shaderModule,
				entryPoint: "fsMain",
				targets: [
					{ format: TextureFormat.RGBA8Unorm },
					{ format: TextureFormat.RGBA8Unorm },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA8Unorm },
				],
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: "none",
				frontFace: "ccw",
			},
			sampleCount: 1,
		} as any);
		return this._decalPipeline;
	}

	public async getDecalBatchPipeline(): Promise<IComputePipeline> {
		if (this._decalBatchPipeline) return this._decalBatchPipeline;
		this._decalBatchPipeline = await this._backend.createComputePipeline({
			layout: this._layouts.decalBatchPipelineLayout,
			label: "WebGPUDeferredDecalBatchPipeline",
			compute: {
				module: await this._getDecalShaderModule(),
				entryPoint: "csMainBatch",
			},
		} as any);
		return this._decalBatchPipeline;
	}

	public async warmup(
		options: WebGPUDeferredWarmupRequest,
	): Promise<WarmupPhaseCounters> {
		if (!options.active) {
			return {
				phase: "webgpu-deferred",
				total: 0,
				compiled: 0,
				skipped: 0,
				failed: 0,
				errors: [],
			};
		}
		const tasks: Array<{
			readonly label: string;
			readonly run: () => Promise<unknown>;
		}> = [{
			label: "WebGPUDeferredLightingWarmup",
			run: () => this.getDeferredLightingPipeline(),
		}];
		if (options.hasDecals) {
			tasks.push({
				label: "WebGPUDeferredDecalWarmup",
				run: () => this.getDecalPipeline(),
			}, {
				label: "WebGPUDeferredDecalBatchWarmup",
				run: () => this.getDecalBatchPipeline(),
			});
		}
		let compiled = 0;
		let failed = 0;
		const errors = [];
		for (const task of tasks) {
			try {
				await task.run();
				compiled++;
			} catch (error) {
				failed++;
				errors.push(toShaderCompileError(error, "webgpu", task.label));
			}
			await options.yieldIfNeeded();
		}
		return {
			phase: "webgpu-deferred",
			total: tasks.length,
			compiled,
			skipped: 0,
			failed,
			errors,
		};
	}

	public onShaderRuntimeChanged(): void {
		this._shaderGeneration++;
		this._destroyResource(this._deferredLightingPipeline);
		this._destroyResource(this._decalPipeline);
		this._destroyResource(this._decalBatchPipeline);
		this._destroyResource(this._deferredLightingShaderModule);
		this._destroyResource(this._decalShaderModule);
		this._deferredLightingShaderModule = null;
		this._deferredLightingPipeline = null;
		this._deferredLightingInFlight = null;
		this._decalShaderModule = null;
		this._decalPipeline = null;
		this._decalBatchPipeline = null;
		this._deferredUnusedBinding?.destroy();
		this._deferredUnusedBinding = null;
		if (this._placeholderTextures) {
			this._placeholderTextures.rgba16Float.destroy();
			this._placeholderTextures.rgba8Unorm.destroy();
			this._placeholderTextures.rgba16Uint.destroy();
			this._placeholderTextures = null;
		}
	}

	public destroy(): void {
		this._destroyed = true;
		this.onShaderRuntimeChanged();
	}

	private async _getDecalShaderModule(): Promise<IShaderModule> {
		if (!this._decalShaderModule) {
			const shader = await ShaderSource.load("webgpu.utility.decal");
			this._decalShaderModule = await this._backend.createShaderModule({
				code: shader.source.code,
				sourceMap: shader.source.sourceMap,
				label: "WebGPUDecalShader",
				language: "wgsl",
				stage: "unknown",
				sourceKind: "decal",
			});
		}
		return this._decalShaderModule;
	}

	private _destroyResource(
		resource: IRenderPipeline | IShaderModule | IComputePipeline | null,
	): void {
		if (!resource) return;
		const kind = resource.kind === "shader-module" ? "shader module" : "pipeline";
		try {
			resource.destroy();
		} catch (error) {
			Logger.warn(
				`[webgpu-cache-resource-destroy-failed] Failed to destroy cached ${kind}: ` +
					String(error),
				{ scope: "WebGPUDeferredResources" },
			);
		}
	}
}
