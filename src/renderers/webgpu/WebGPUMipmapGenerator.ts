import { Platform } from "../../foundation/Platform";
import type { TextureFilter } from "../../core/Texture";
import {
	AddressMode,
	FilterMode,
	type IBindingGroup,
	type IRenderTexture,
	type ISampler,
	TextureFormat,
} from "../types";
import { getTextureFormatInfo } from "../TextureFormatInfo";
import type { WebGPUBackend } from "../WebGPUBackend";

interface WebGPUMipmapPipelineEntry {
	pipeline: GPURenderPipeline;
}

const GPU_SHADER_STAGE_FRAGMENT =
	(globalThis as typeof globalThis & { GPUShaderStage?: { FRAGMENT?: number } })
		.GPUShaderStage?.FRAGMENT ?? 0x2;

const browserMipmapShaderSources: Record<string, string> = Platform.isNodeRuntime()
	? {}
	: import.meta.glob<string>("../../shaders/webgpu/utility/mipmapBlit.wgsl", {
			query: "?raw",
			import: "default",
			eager: true,
		});

/**
 * Generates 2D texture mip chains by rendering each mip from the previous mip.
 */
export class WebGPUMipmapGenerator {
	private _backend: WebGPUBackend;
	private _shaderModule: GPUShaderModule | null = null;
	private _bindGroupLayout: GPUBindGroupLayout | null = null;
	private _pipelineLayout: GPUPipelineLayout | null = null;
	private _sampler: ISampler | null = null;
	private _pipelines = new Map<TextureFormat, WebGPUMipmapPipelineEntry>();
	private _viewCache = new WeakMap<object, GPUTextureView[]>();

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
	public generate(
		texture: IRenderTexture,
		format: TextureFormat,
		mipLevelCount: number
	): boolean {
		const levelCount = Math.max(1, Math.floor(mipLevelCount));
		if (levelCount <= 1 || !canGenerateWebGPUMipmaps(format)) {
			return false;
		}

		const device = this._backend.device;
		const queue = this._backend.queue;
		if (!device || !queue) {
			return false;
		}

		const pipeline = this._getPipeline(device, format);
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
				pass.setPipeline(pipeline);
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
		this._destroyManagedResource(this._sampler);
		this._sampler = null;
		this._shaderModule = null;
		this._bindGroupLayout = null;
		this._pipelineLayout = null;
		this._pipelines.clear();
		this._viewCache = new WeakMap<object, GPUTextureView[]>();
	}

	private _getPipeline(
		device: GPUDevice,
		format: TextureFormat
	): GPURenderPipeline | null {
		const cached = this._pipelines.get(format);
		if (cached) {
			return cached.pipeline;
		}

		const shaderModule = this._getShaderModule(device);
		const bindGroupLayout = this._getBindGroupLayout(device);
		const pipelineLayout = this._getPipelineLayout(device, bindGroupLayout);
		if (!shaderModule || !bindGroupLayout || !pipelineLayout) {
			return null;
		}

		const pipeline = device.createRenderPipeline({
			label: `WebGPUMipmapPipeline_${format}`,
			layout: pipelineLayout,
			vertex: {
				module: shaderModule,
				entryPoint: "vsMain",
			},
			fragment: {
				module: shaderModule,
				entryPoint: "fsMain",
				targets: [{ format: format as GPUTextureFormat }],
			},
			primitive: {
				topology: "triangle-list",
				cullMode: "none",
				frontFace: "ccw",
			},
		});
		this._pipelines.set(format, { pipeline });
		return pipeline;
	}

	private _getShaderModule(device: GPUDevice): GPUShaderModule | null {
		if (!this._shaderModule) {
			this._shaderModule = device.createShaderModule({
				label: "WebGPUMipmapShader",
				code: getWebGPUMipmapBlitShaderSource(),
			});
		}
		return this._shaderModule;
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

function getWebGPUMipmapBlitShaderSource(): string {
	if (Platform.isNodeRuntime()) {
		const nodeProcess = (
			globalThis as typeof globalThis & {
				process?: {
					getBuiltinModule?: (specifier: string) => unknown;
				};
			}
		).process;
		const fs = nodeProcess?.getBuiltinModule?.("fs") as
			| {
					readFileSync: (path: URL, encoding: "utf8") => string;
			  }
			| undefined;
		if (!fs) {
			throw new Error("Node fs module is unavailable for WebGPU mipmap shader.");
		}
		return fs.readFileSync(
			new URL("../../shaders/webgpu/utility/mipmapBlit.wgsl", import.meta.url),
			"utf8"
		);
	}

	const source =
		browserMipmapShaderSources[
			"../../shaders/webgpu/utility/mipmapBlit.wgsl"
		];
	if (!source) {
		throw new Error("WebGPU mipmap shader source was not bundled.");
	}
	return source;
}

function getNativeBindGroup(binding: IBindingGroup): GPUBindGroup {
	return (
		(binding as { _gpuResource?: GPUBindGroup })._gpuResource ??
		(binding as unknown as GPUBindGroup)
	);
}
