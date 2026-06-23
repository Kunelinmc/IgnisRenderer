import type { TextureFilter } from "../../core/Texture";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	AddressMode,
	FilterMode,
	type IBindingGroup,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
	PrimitiveTopology,
	TextureFormat,
} from "../types";
import { getTextureFormatInfo } from "../TextureFormatInfo";
import type { WebGPUBackend } from "../WebGPUBackend";
import { getWebGPURenderPipeline } from "./WebGPUResourceAccess";

interface WebGPUMipmapPipelineEntry {
	pipeline: IRenderPipeline;
}

const GPU_SHADER_STAGE_FRAGMENT =
	(globalThis as typeof globalThis & { GPUShaderStage?: { FRAGMENT?: number } })
		.GPUShaderStage?.FRAGMENT ?? 0x2;

/**
 * Generates 2D texture mip chains by rendering each mip from the previous mip.
 */
export class WebGPUMipmapGenerator {
	private _backend: WebGPUBackend;
	private _shaderModule: IShaderModule | null = null;
	private _shaderModulePromise: Promise<IShaderModule | null> | null = null;
	private _bindGroupLayout: GPUBindGroupLayout | null = null;
	private _pipelineLayout: GPUPipelineLayout | null = null;
	private _sampler: ISampler | null = null;
	private _pipelines = new Map<TextureFormat, WebGPUMipmapPipelineEntry>();
	private _pipelinePromises = new Map<
		TextureFormat,
		Promise<IRenderPipeline | null>
	>();
	private _viewCache = new WeakMap<object, GPUTextureView[]>();
	private _lifecycleGeneration = 0;

	constructor(backend: WebGPUBackend) {
		this._backend = backend;
	}

	/**
	 * Generates all missing mip levels for a 2D texture.
	 *
	 * @param texture Texture with level 0 already populated.
	 * @param format Actual WebGPU texture format.
	 * @param mipLevelCount Number of allocated mip levels.
	 * @returns True when generation commands were recorded and submitted.
	 * @sideEffects Allocates cached WebGPU pipeline resources and submits GPU work.
	 */
	public async generate(
		texture: IRenderTexture,
		format: TextureFormat,
		mipLevelCount: number
	): Promise<boolean> {
		const levelCount = Math.max(1, Math.floor(mipLevelCount));
		if (levelCount <= 1 || !canGenerateWebGPUMipmaps(format)) {
			return false;
		}

		const device = this._backend.device;
		const queue = this._backend.queue;
		if (!device || !queue) {
			return false;
		}

		const pipeline = await this._getPipeline(device, format);
		const sampler = this._getSampler();
		const views = this._getMipViews(texture, levelCount);
		const bindGroupLayout = this._bindGroupLayout;
		if (!pipeline || !sampler || !bindGroupLayout || views.length < levelCount) {
			return false;
		}

		const encoder = device.createCommandEncoder({
			label: "WebGPUMipmapGeneratorEncoder",
		});
		const transientBindings: IBindingGroup[] = [];
		try {
			for (let level = 1; level < levelCount; level++) {
				const sourceView = views[level - 1];
				const targetView = views[level];
				const binding = this._backend.createBindingGroup({
					layout: bindGroupLayout,
					entries: [
						{ binding: 0, resource: sourceView },
						{ binding: 1, resource: sampler },
					],
					label: `WebGPUMipmapBinding_${level}`,
				});
				transientBindings.push(binding);
				const pass = encoder.beginRenderPass({
					label: `WebGPUMipmapPass_${level}`,
					colorAttachments: [
						{
							view: targetView,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "clear",
							storeOp: "store",
						},
					],
				});
				pass.setPipeline(getWebGPURenderPipeline(pipeline));
				pass.setBindGroup(0, getNativeBindGroup(binding));
				pass.draw(3, 1, 0, 0);
				pass.end();
			}
			queue.submit([encoder.finish()]);
			return true;
		} finally {
			for (const binding of transientBindings) {
				this._destroyManagedResource(binding);
			}
		}
	}

	/**
	 * Releases helper-owned resources and cached views.
	 */
	public destroy(): void {
		this._lifecycleGeneration++;
		this._destroyManagedResource(this._shaderModule);
		this._destroyManagedResource(this._sampler);
		for (const entry of this._pipelines.values()) {
			this._destroyManagedResource(entry.pipeline);
		}
		this._sampler = null;
		this._shaderModule = null;
		this._shaderModulePromise = null;
		this._bindGroupLayout = null;
		this._pipelineLayout = null;
		this._pipelines.clear();
		this._pipelinePromises.clear();
		this._viewCache = new WeakMap<object, GPUTextureView[]>();
	}

	private async _getPipeline(
		device: GPUDevice,
		format: TextureFormat
	): Promise<IRenderPipeline | null> {
		const cached = this._pipelines.get(format);
		if (cached) {
			return cached.pipeline;
		}
		const pending = this._pipelinePromises.get(format);
		if (pending) {
			return pending;
		}

		const generation = this._lifecycleGeneration;
		const creationPromise = this._createPipeline(device, format, generation)
			.finally(() => {
				if (this._pipelinePromises.get(format) === creationPromise) {
					this._pipelinePromises.delete(format);
				}
			});
		this._pipelinePromises.set(format, creationPromise);
		return creationPromise;
	}

	private async _createPipeline(
		device: GPUDevice,
		format: TextureFormat,
		generation: number
	): Promise<IRenderPipeline | null> {
		if (this._lifecycleGeneration !== generation) {
			return null;
		}
		const shaderModule = await this._getShaderModule(generation);
		if (this._lifecycleGeneration !== generation || !shaderModule) {
			return null;
		}
		const bindGroupLayout = this._getBindGroupLayout(device);
		const pipelineLayout = this._getPipelineLayout(device, bindGroupLayout);
		if (!bindGroupLayout || !pipelineLayout) {
			return null;
		}

		const pipeline = await this._backend.createPipeline({
			label: `WebGPUMipmapPipeline_${format}`,
			layout: pipelineLayout,
			vertex: {
				module: shaderModule,
				entryPoint: "vsMain",
			},
			fragment: {
				module: shaderModule,
				entryPoint: "fsMain",
				targets: [{ format }],
			},
			primitive: {
				topology: PrimitiveTopology.TriangleList,
				cullMode: "none",
				frontFace: "ccw",
			},
		});
		if (this._lifecycleGeneration !== generation) {
			this._destroyManagedResource(pipeline);
			return null;
		}
		const existing = this._pipelines.get(format);
		if (existing) {
			this._destroyManagedResource(pipeline);
			return existing.pipeline;
		}
		this._pipelines.set(format, { pipeline });
		return pipeline;
	}

	private async _getShaderModule(
		generation: number
	): Promise<IShaderModule | null> {
		if (this._shaderModule) {
			return this._shaderModule;
		}
		if (this._shaderModulePromise) {
			return this._shaderModulePromise;
		}
		const creationPromise = this._backend
			.createShaderModule({
				label: "WebGPUMipmapShader",
				code: ShaderSource.getSync("webgpu.utility.mipmapBlit.raw"),
				language: "wgsl",
				stage: "unknown",
				sourceKind: "unknown",
			})
			.then((module) => {
				if (this._lifecycleGeneration !== generation) {
					this._destroyManagedResource(module);
					return null;
				}
				if (this._shaderModule) {
					this._destroyManagedResource(module);
					return this._shaderModule;
				}
				this._shaderModule = module;
				return module;
			})
			.finally(() => {
				if (this._shaderModulePromise === creationPromise) {
					this._shaderModulePromise = null;
				}
			});
		this._shaderModulePromise = creationPromise;
		return creationPromise;
	}

	private _getBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
		if (!this._bindGroupLayout) {
			this._bindGroupLayout = device.createBindGroupLayout({
				label: "WebGPUMipmapBindGroupLayout",
				entries: [
					{
						binding: 0,
						visibility: GPU_SHADER_STAGE_FRAGMENT,
						texture: {
							sampleType: "float",
							viewDimension: "2d",
							multisampled: false,
						},
					},
					{
						binding: 1,
						visibility: GPU_SHADER_STAGE_FRAGMENT,
						sampler: { type: "filtering" },
					},
				],
			});
		}
		return this._bindGroupLayout;
	}

	private _getPipelineLayout(
		device: GPUDevice,
		bindGroupLayout: GPUBindGroupLayout
	): GPUPipelineLayout {
		if (!this._pipelineLayout) {
			this._pipelineLayout = device.createPipelineLayout({
				label: "WebGPUMipmapPipelineLayout",
				bindGroupLayouts: [bindGroupLayout],
			});
		}
		return this._pipelineLayout;
	}

	private _getSampler(): ISampler {
		if (!this._sampler) {
			this._sampler = this._backend.createSampler({
				label: "WebGPUMipmapSampler",
				addressModeU: AddressMode.ClampToEdge,
				addressModeV: AddressMode.ClampToEdge,
				magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear,
				mipmapFilter: FilterMode.Nearest,
			});
		}
		return this._sampler;
	}

	private _getMipViews(
		texture: IRenderTexture,
		mipLevelCount: number
	): GPUTextureView[] {
		const cached = this._viewCache.get(texture as object);
		if (cached && cached.length >= mipLevelCount) {
			return cached;
		}
		const views: GPUTextureView[] = [];
		for (let level = 0; level < mipLevelCount; level++) {
			views.push(
				this._backend.createTextureView(texture, {
					baseMipLevel: level,
					mipLevelCount: 1,
				})
			);
		}
		this._viewCache.set(texture as object, views);
		return views;
	}

	private _destroyManagedResource(resource: unknown): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(resource);
		}
	}
}

export function textureFilterRequiresMipmaps(
	value: TextureFilter | string | undefined
): boolean {
	switch (value) {
		case "NearestMipmapNearest":
		case "NearestMipmapLinear":
		case "LinearMipmapNearest":
		case "LinearMipmapLinear":
			return true;
		default:
			return false;
	}
}

export function resolveWebGPUMipmapLevelCount(
	width: number,
	height: number
): number {
	const maxDimension = Math.max(1, width | 0, height | 0);
	return Math.max(1, Math.floor(Math.log2(maxDimension)) + 1);
}

export function canGenerateWebGPUMipmaps(format: TextureFormat): boolean {
	try {
		const info = getTextureFormatInfo(format);
		return (
			info.formatClass === "color" &&
			info.sampleType === "float" &&
			info.isFilterable &&
			info.isRenderable &&
			!info.isCompressed &&
			!info.hasDepth &&
			!info.hasStencil
		);
	} catch {
		return false;
	}
}

function getNativeBindGroup(binding: IBindingGroup): GPUBindGroup {
	return (
		(binding as { _gpuResource?: GPUBindGroup })._gpuResource ??
		(binding as unknown as GPUBindGroup)
	);
}
