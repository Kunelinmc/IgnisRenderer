/// <reference types="@webgpu/types" />
import {
	type ComputePassDesc,
	type ICommandEncoder,
	type RenderPassDesc,
} from "../ral/ICommandEncoder";
import type { Renderer } from "../Renderer";
import type { IRenderBackend } from "./IRenderBackend";
import type { FrameContext, FramePass } from "../pipeline/types";
import { WebGPUFrameExecutor } from "./webgpu/WebGPUFrameExecutor";
import { RenderResources } from "../resources/RenderResources";
import {
	BufferUsage,
	type BindingGroupDesc,
	type BufferDesc,
	type ComputePipelineDesc,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
	type IndexFormat,
	type PipelineDesc,
	type SamplerDesc,
	type ShaderModuleDesc,
	type TextureDataLayout,
	type TextureDesc,
	TextureFormat,
	TextureUsage,
} from "../ral/types";

export class WebGPUBackend implements IRenderBackend {
	public readonly type = "webgpu";
	public readonly capabilities = {
		sh: false,
		shadows: true,
		reflection: false,
		skybox: false,
		ssao: false,
		volumetric: false,
	};

	public canvas: HTMLCanvasElement | null = null;
	public context: GPUCanvasContext | null = null;
	public device!: GPUDevice;
	public queue!: GPUQueue;
	public canvasFormat: GPUTextureFormat = "bgra8unorm";

	private _depthTexture: IRenderTexture | null = null;
	private _currentCanvasView: GPUTextureView | null = null;
	private _renderer: Renderer | null = null;
	private _resources: RenderResources | null = null;
	private _frameExecutor: WebGPUFrameExecutor | null = null;

	constructor(canvas?: HTMLCanvasElement) {
		this.canvas = canvas ?? null;
	}

	public setRenderer(renderer: Renderer): void {
		this._renderer = renderer;
	}

	public getAttachments(width: number, height: number): any {
		return {
			width,
			height,
		};
	}

	public async init(canvas: HTMLCanvasElement): Promise<void> {
		this.canvas = canvas;

		if (!navigator.gpu) {
			throw new Error("WebGPU not supported on this browser.");
		}

		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) {
			throw new Error("No appropriate GPUAdapter found.");
		}

		try {
			this.device = await adapter.requestDevice();
			this.device.lost.then((info) => {
				console.error(`WebGPU device was lost: ${info.message}`);
			});
		} catch (error) {
			throw new Error(`Failed to request WebGPU device: ${error}`);
		}

		this.queue = this.device.queue;
		this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
		this.context = canvas.getContext("webgpu");
		if (!this.context) {
			throw new Error("Failed to acquire WebGPU canvas context.");
		}

		this._configureContext();
		this._recreateDepthTexture();

		if (!this._renderer) {
			throw new Error("WebGPU backend requires a renderer before init().");
		}

		this._resources = new RenderResources(this._renderer, this);
		await this._resources.init();
		this._frameExecutor = new WebGPUFrameExecutor(this, this._resources);
	}

	public resize(_width: number, _height: number): void {
		if (!this.device || !this.context || !this.canvas) {
			return;
		}

		this._configureContext();
		this._recreateDepthTexture();
	}

	public beginFrame(context: FrameContext): void {
		if (!this._resources || !this._frameExecutor) {
			throw new Error("WebGPU backend has not been initialized.");
		}

		this._resources.prepareFrame(context);
		this._frameExecutor.beginFrame();
	}

	public executePass(
		pass: FramePass,
		context: FrameContext
	): Promise<void> | void {
		if (!this._frameExecutor) {
			throw new Error("WebGPU backend has not been initialized.");
		}

		return this._frameExecutor.executePass(pass, context);
	}

	public endFrame(): void {
		this._frameExecutor?.endFrame();
	}

	public createBuffer(desc: BufferDesc): IRenderBuffer {
		const gpuBuffer = this.device.createBuffer({
			size: desc.size,
			usage: this._mapBufferUsage(desc.usage),
			label: desc.label,
		});

		return {
			size: desc.size,
			destroy: () => gpuBuffer.destroy(),
			_gpuResource: gpuBuffer,
		} as any;
	}

	public createTexture(desc: TextureDesc): IRenderTexture {
		const gpuTexture = this.device.createTexture({
			size: [desc.width, desc.height, 1],
			format: desc.format as GPUTextureFormat,
			usage: this._mapTextureUsage(desc.usage),
			label: desc.label,
		});
		const gpuView = gpuTexture.createView();

		return {
			width: desc.width,
			height: desc.height,
			destroy: () => gpuTexture.destroy(),
			_gpuResource: gpuTexture,
			_gpuTexture: gpuTexture,
			_gpuView: gpuView,
		} as any;
	}

	public createSampler(desc: SamplerDesc): ISampler {
		const gpuSampler = this.device.createSampler({
			addressModeU: desc.addressModeU as GPUAddressMode | undefined,
			addressModeV: desc.addressModeV as GPUAddressMode | undefined,
			magFilter: desc.magFilter as GPUFilterMode | undefined,
			minFilter: desc.minFilter as GPUFilterMode | undefined,
			mipmapFilter: desc.mipmapFilter as GPUFilterMode | undefined,
			label: desc.label,
		});

		return {
			label: desc.label,
			_gpuResource: gpuSampler,
		} as any;
	}

	public async createShaderModule(
		desc: ShaderModuleDesc
	): Promise<IShaderModule> {
		const gpuModule = this.device.createShaderModule({
			code: desc.code,
			label: desc.label,
		});

		const info = await gpuModule.getCompilationInfo();
		if (info.messages.length > 0) {
			console.group(
				`WebGPU Shader Compilation Info [${desc.label || "unnamed"}]`
			);
			for (const message of info.messages) {
				const logType =
					message.type === "error"
						? "error"
						: message.type === "warning"
							? "warn"
							: "log";
				console[logType](
					`${message.message} (at line ${message.lineNum}, col ${message.linePos})`
				);
			}
			console.groupEnd();
		}

		return {
			label: desc.label,
			_gpuResource: gpuModule,
		} as any;
	}

	public createPipeline(desc: PipelineDesc): IRenderPipeline {
		this.device.pushErrorScope("validation");

		const gpuPipeline = this.device.createRenderPipeline({
			layout: desc.layout ?? "auto",
			vertex: {
				module: (desc.vertex.module as any)._gpuResource ?? desc.vertex.module,
				entryPoint: desc.vertex.entryPoint,
				buffers:
					desc.vertex.buffers?.map((buffer) => ({
						arrayStride: buffer.arrayStride,
						stepMode: buffer.stepMode ?? "vertex",
						attributes: buffer.attributes.map((attribute) => ({
							format: attribute.format as GPUVertexFormat,
							offset: attribute.offset,
							shaderLocation: attribute.shaderLocation,
						})),
					})) ?? [],
			},
			fragment: desc.fragment
				? {
						module:
							(desc.fragment.module as any)._gpuResource ??
							desc.fragment.module,
						entryPoint: desc.fragment.entryPoint,
						targets: desc.fragment.targets.map((target) => ({
							format: target.format as GPUTextureFormat,
							blend: target.blend,
						})),
					}
				: undefined,
			primitive: {
				topology: desc.primitive?.topology ?? "triangle-list",
				cullMode: desc.primitive?.cullMode ?? "none",
				frontFace: desc.primitive?.frontFace ?? "ccw",
			},
			depthStencil: desc.depthStencil
				? {
						format: desc.depthStencil.format as GPUTextureFormat,
						depthWriteEnabled: desc.depthStencil.depthWriteEnabled,
						depthCompare: desc.depthStencil.depthCompare as GPUCompareFunction,
					}
				: undefined,
			label: desc.label,
		});

		this.device.popErrorScope().then((error) => {
			if (error) {
				console.error(
					`WebGPU Pipeline validation error [${desc.label}]: ${error.message}`
				);
			}
		});

		return {
			label: desc.label,
			_gpuResource: gpuPipeline,
		} as any;
	}

	public createComputePipeline(desc: ComputePipelineDesc): IComputePipeline {
		const gpuPipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module:
					(desc.compute.module as any)._gpuResource ?? desc.compute.module,
				entryPoint: desc.compute.entryPoint,
			},
			label: desc.label,
		});

		return {
			label: desc.label,
			_gpuResource: gpuPipeline,
		} as any;
	}

	public createBindingGroup(desc: BindingGroupDesc): IBindingGroup {
		const pipeline = (desc.pipeline as any)?._gpuResource as
			| GPURenderPipeline
			| undefined;
		const layout =
			(desc.layout as GPUBindGroupLayout | undefined) ??
			pipeline?.getBindGroupLayout(desc.layoutIndex ?? 0);

		if (!layout) {
			throw new Error(
				`WebGPU binding group ${desc.label ?? "(unnamed)"} requires an explicit layout or pipeline`
			);
		}

		const gpuBindGroup = this.device.createBindGroup({
			layout,
			entries: desc.entries.map((entry) => ({
				binding: entry.binding,
				resource: this._mapBindingResource(entry.resource),
			})),
			label: desc.label,
		});

		return {
			label: desc.label,
			_gpuResource: gpuBindGroup,
		} as any;
	}

	public createCommandEncoder(): ICommandEncoder {
		return new WebGPUCommandEncoder(this.device.createCommandEncoder(), this);
	}

	public writeBuffer(
		buffer: IRenderBuffer,
		data: BufferSource,
		offset: number = 0
	): void {
		this.queue.writeBuffer((buffer as any)._gpuResource, offset, data);
	}

	public writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		desc: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number }
	): void {
		this.queue.writeTexture(
			{
				texture: (texture as any)._gpuTexture ?? (texture as any)._gpuResource,
			},
			data,
			{
				offset: desc.offset ?? 0,
				bytesPerRow: desc.bytesPerRow,
				rowsPerImage: desc.rowsPerImage,
			},
			size
		);
	}

	public copyTextureToTexture(
		source: {
			texture: IRenderTexture;
			origin?: GPUOrigin3D;
			aspect?: GPUTextureAspect;
		},
		destination: {
			texture: IRenderTexture;
			origin?: GPUOrigin3D;
			aspect?: GPUTextureAspect;
		},
		copySize: { width: number; height: number; depthOrArrayLayers?: number }
	): void {
		const commandEncoder = this.device.createCommandEncoder();

		commandEncoder.copyTextureToTexture(
			{
				texture:
					(source.texture as any)._gpuTexture ??
					(source.texture as any)._gpuResource,
				origin: source.origin,
				aspect: source.aspect,
			},
			{
				texture:
					(destination.texture as any)._gpuTexture ??
					(destination.texture as any)._gpuResource,
				origin: destination.origin,
				aspect: destination.aspect,
			},
			copySize
		);

		this.queue.submit([commandEncoder.finish()]);
	}

	public submit(commands: any[]): void {
		this.device.pushErrorScope("validation");
		this.queue.submit(commands.map((command) => command._gpuCommandBuffer));
		this._currentCanvasView = null;
		this.device.popErrorScope().then((error) => {
			if (error) {
				console.error(`WebGPU Submit validation error: ${error.message}`);
			}
		});
	}

	public getCanvasColorTexture(): IRenderTexture {
		if (!this.context || !this.canvas) {
			throw new Error("WebGPU not initialized");
		}

		const gpuTexture = this.context.getCurrentTexture();
		const gpuView = gpuTexture.createView();
		return {
			width: this.canvas.width,
			height: this.canvas.height,
			destroy: () => {},
			_gpuResource: gpuTexture,
			_gpuTexture: gpuTexture,
			_gpuView: gpuView,
		} as any;
	}

	public getCanvasDepthTexture(): IRenderTexture {
		if (!this._depthTexture) {
			throw new Error("Depth texture not initialized");
		}
		return this._depthTexture;
	}

	public getCurrentColorView(): GPUTextureView {
		if (!this.context) {
			throw new Error("WebGPU canvas context is not initialized.");
		}

		if (!this._currentCanvasView) {
			this._currentCanvasView = this.context.getCurrentTexture().createView();
		}

		return this._currentCanvasView;
	}

	public getCurrentDepthView(): GPUTextureView {
		if (!this._depthTexture) {
			throw new Error("WebGPU depth texture is not initialized.");
		}

		return (this._depthTexture as any)._gpuView;
	}

	private _configureContext(): void {
		if (!this.context || !this.canvas) {
			return;
		}

		this.context.configure({
			device: this.device,
			format: this.canvasFormat,
			alphaMode: "premultiplied",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
		});
	}

	private _recreateDepthTexture(): void {
		if (!this.device || !this.canvas) {
			return;
		}

		if (this.canvas.width <= 0 || this.canvas.height <= 0) {
			return;
		}

		this._depthTexture?.destroy();
		this._depthTexture = this.createTexture({
			width: this.canvas.width,
			height: this.canvas.height,
			format: TextureFormat.Depth24Plus,
			usage: TextureUsage.RenderAttachment,
			label: "WebGPUCanvasDepth",
		});
	}

	private _mapBindingResource(resource: any): GPUBindingResource {
		if (resource?._gpuView) {
			return resource._gpuView;
		}

		if (resource?._gpuTexture) {
			return resource._gpuTexture.createView();
		}

		if (typeof resource?.size === "number" && resource?._gpuResource) {
			return { buffer: resource._gpuResource };
		}

		if (resource?._gpuResource) {
			return resource._gpuResource;
		}

		return resource;
	}

	private _mapBufferUsage(usage: number): GPUBufferUsageFlags {
		let flags = 0;
		if (usage & BufferUsage.Vertex) flags |= GPUBufferUsage.VERTEX;
		if (usage & BufferUsage.Index) flags |= GPUBufferUsage.INDEX;
		if (usage & BufferUsage.Uniform) flags |= GPUBufferUsage.UNIFORM;
		if (usage & BufferUsage.Storage) flags |= GPUBufferUsage.STORAGE;
		if (usage & BufferUsage.CopySrc) flags |= GPUBufferUsage.COPY_SRC;
		if (usage & BufferUsage.CopyDst) flags |= GPUBufferUsage.COPY_DST;
		if (usage & BufferUsage.MapRead) flags |= GPUBufferUsage.MAP_READ;
		if (usage & BufferUsage.MapWrite) flags |= GPUBufferUsage.MAP_WRITE;
		return flags;
	}

	private _mapTextureUsage(usage: number): GPUTextureUsageFlags {
		let flags = 0;
		if (usage & TextureUsage.CopySrc) flags |= GPUTextureUsage.COPY_SRC;
		if (usage & TextureUsage.CopyDst) flags |= GPUTextureUsage.COPY_DST;
		if (usage & TextureUsage.TextureBinding) {
			flags |= GPUTextureUsage.TEXTURE_BINDING;
		}
		if (usage & TextureUsage.StorageBinding) {
			flags |= GPUTextureUsage.STORAGE_BINDING;
		}
		if (usage & TextureUsage.RenderAttachment) {
			flags |= GPUTextureUsage.RENDER_ATTACHMENT;
		}
		if (usage & TextureUsage.ComputeStorage) {
			flags |= GPUTextureUsage.STORAGE_BINDING;
		}
		return flags;
	}
}

class WebGPUCommandEncoder implements ICommandEncoder {
	private _encoder: GPUCommandEncoder;
	private _backend: WebGPUBackend;
	private _passEncoder: GPURenderPassEncoder | GPUComputePassEncoder | null =
		null;

	constructor(encoder: GPUCommandEncoder, backend: WebGPUBackend) {
		this._encoder = encoder;
		this._backend = backend;
	}

	public beginRenderPass(desc: RenderPassDesc): void {
		this._passEncoder = this._encoder.beginRenderPass({
			colorAttachments: desc.colorAttachments.map((attachment) => ({
				view:
					(attachment.view as any)?._gpuView ??
					this._backend.getCurrentColorView(),
				clearValue: attachment.clearValue,
				loadOp: attachment.loadOp,
				storeOp: attachment.storeOp,
			})),
			depthStencilAttachment: desc.depthStencilAttachment
				? {
						view:
							(desc.depthStencilAttachment.view as any)?._gpuView ??
							this._backend.getCurrentDepthView(),
						depthClearValue: desc.depthStencilAttachment.depthClearValue ?? 1,
						depthLoadOp: desc.depthStencilAttachment.depthLoadOp ?? "clear",
						depthStoreOp: desc.depthStencilAttachment.depthStoreOp ?? "store",
					}
				: undefined,
			label: desc.label,
		});
	}

	public beginComputePass(desc?: ComputePassDesc): void {
		this._passEncoder = this._encoder.beginComputePass({
			label: desc?.label,
		});
	}

	public setComputePipeline(pipeline: IComputePipeline): void {
		(this._passEncoder as GPUComputePassEncoder | null)?.setPipeline(
			(pipeline as any)._gpuResource
		);
	}

	public dispatchWorkgroups(x: number, y: number = 1, z: number = 1): void {
		(this._passEncoder as GPUComputePassEncoder | null)?.dispatchWorkgroups(
			x,
			y,
			z
		);
	}

	public endComputePass(): void {
		this._passEncoder?.end();
		this._passEncoder = null;
	}

	public setPipeline(pipeline: IRenderPipeline): void {
		(this._passEncoder as GPURenderPassEncoder | null)?.setPipeline(
			(pipeline as any)._gpuResource
		);
	}

	public setBindingGroup(index: number, group: IBindingGroup): void {
		(
			this._passEncoder as GPURenderPassEncoder | GPUComputePassEncoder | null
		)?.setBindGroup(index, (group as any)._gpuResource);
	}

	public setVertexBuffer(slot: number, buffer: IRenderBuffer): void {
		(this._passEncoder as GPURenderPassEncoder | null)?.setVertexBuffer(
			slot,
			(buffer as any)._gpuResource
		);
	}

	public setIndexBuffer(buffer: IRenderBuffer, format: IndexFormat): void {
		(this._passEncoder as GPURenderPassEncoder | null)?.setIndexBuffer(
			(buffer as any)._gpuResource,
			format
		);
	}

	public drawIndexed(
		indexCount: number,
		instanceCount: number = 1,
		firstIndex: number = 0
	): void {
		(this._passEncoder as GPURenderPassEncoder | null)?.drawIndexed(
			indexCount,
			instanceCount,
			firstIndex
		);
	}

	public draw(
		vertexCount: number,
		instanceCount: number = 1,
		firstVertex: number = 0
	): void {
		(this._passEncoder as GPURenderPassEncoder | null)?.draw(
			vertexCount,
			instanceCount,
			firstVertex
		);
	}

	public endRenderPass(): void {
		this._passEncoder?.end();
		this._passEncoder = null;
	}

	public finish(): any {
		return {
			_gpuCommandBuffer: this._encoder.finish(),
		};
	}
}
