import { ShaderSource } from "../../shaders/ShaderSource";
import type { WebGPUBackend } from "../WebGPUBackend";
import { TextureFormat, type IBindingGroup, type IComputePipeline, type IRenderPipeline, type IShaderModule } from "../types";
import type { WebGPUDeferredResourceProvider } from "./WebGPUResourceContracts";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";

/**
 * Owns WebGPU deferred layouts, pipelines, and the deferred placeholder group.
 *
 * @internal Owned by the WebGPU frame service owner.
 */
export class WebGPUDeferredResources implements WebGPUDeferredResourceProvider {
	private _decalShaderModule: IShaderModule | null = null;
	private _decalPipeline: IRenderPipeline | null = null;
	private _decalBatchPipeline: IComputePipeline | null = null;
	private _deferredUnusedBinding: IBindingGroup | null = null;

	public constructor(
		private readonly _backend: WebGPUBackend,
		private readonly _layouts: WebGPUPipelineLayouts,
		private readonly _getDeferredLightingPipeline: () => Promise<IRenderPipeline>,
	) {}

	public getGBufferWriteLayout(): GPUBindGroupLayout {
		return this._layouts.gbufferWriteBindGroupLayout;
	}

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

	public async getDeferredLightingPipeline(): Promise<IRenderPipeline> {
		return this._getDeferredLightingPipeline();
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
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
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

	public onShaderRuntimeChanged(): void {
		this._decalShaderModule = null;
		this._decalPipeline = null;
		this._decalBatchPipeline = null;
		this._destroyBindingGroup(this._deferredUnusedBinding);
		this._deferredUnusedBinding = null;
	}

	public destroy(): void {
		this.onShaderRuntimeChanged();
	}

	private async _getDecalShaderModule(): Promise<IShaderModule> {
		if (!this._decalShaderModule) {
			const shader = await ShaderSource.load("webgpu.utility.decal.composite");
			this._decalShaderModule = await this._backend.createShaderModule({
				code: shader.code,
				sourceMap: shader.sourceMap,
				label: "WebGPUDecalShader",
				language: "wgsl",
				stage: "unknown",
				sourceKind: "decal",
			});
		}
		return this._decalShaderModule;
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroy = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroy === "function") destroy.call(group);
	}
}
