/// <reference types="@webgpu/types" />
import {
	type ComputePassDesc,
	type ICommandBuffer,
	type ICommandEncoder,
	type RenderPassDesc,
} from "./ICommandEncoder";
import type { IRenderBackend, RendererBackendBridge } from "./IRenderBackend";
import {
	type FrameAttachments,
	type FrameContext,
	type FramePass,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
} from "../pipeline/types";
import { WebGPUErrorScopeHelper } from "./webgpu/WebGPUErrorScopeHelper";
import { WebGPUFrameExecutor } from "./webgpu/WebGPUFrameExecutor";
import { WebGPURenderResources } from "./webgpu/WebGPURenderResources";
import {
	attachWebGPUTexture,
	createWebGPUTexture,
	getWebGPUBindGroup,
	getWebGPUBuffer,
	getWebGPUComputePipeline,
	getWebGPUPipeline,
	getWebGPURenderPipeline,
	getWebGPUShaderModule,
	getWebGPUTexture,
	tryGetWebGPUBuffer,
	tryGetWebGPUTexture,
	type WebGPUTexture,
} from "./webgpu/WebGPUResourceAccess";
import type { WebGPUPostProcessPassPlugin } from "./webgpu/WebGPUPostProcessGraph";
import { DefaultParticleSimulator } from "../simulation/particles/DefaultParticleSimulator";
import {
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
	WEBGPU_BINDING_GROUP_CACHE_LIMIT,
	WEBGPU_BINDING_GROUP_CACHE_TTL_FRAMES,
	WEBGPU_PIPELINE_CACHE_LIMIT,
	WEBGPU_PIPELINE_LAYOUT_CACHE_LIMIT,
	WEBGPU_COPY_BATCH_SIZE,
	WEBGPU_TIMESTAMP_QUERY_CAPACITY,
	WEBGPU_DEFAULT_MSAA_SAMPLE_COUNT,
} from "./webgpu/constants";
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
} from "./types";

interface InternalRenderBuffer extends IRenderBuffer {
	_gpuResource: GPUBuffer;
}

interface InternalTexture extends IRenderTexture {
	_gpuResource: GPUTexture;
	_gpuTexture: GPUTexture;
	_gpuView: GPUTextureView;
	_webgpuTexture: WebGPUTexture;
}

interface InternalSampler extends ISampler {
	_gpuResource: GPUSampler;
}

interface InternalShaderModule extends IShaderModule {
	_gpuResource: GPUShaderModule;
}

interface InternalRenderPipeline extends IRenderPipeline {
	_gpuResource: GPURenderPipeline;
}

interface InternalComputePipeline extends IComputePipeline {
	_gpuResource: GPUComputePipeline;
}

interface InternalBindingGroup extends IBindingGroup {
	_gpuResource: GPUBindGroup;
}

interface InternalCommandBuffer {
	_backendCommandBuffer?: GPUCommandBuffer;
	_gpuCommandBuffer: GPUCommandBuffer;
}

interface CachedBindingGroupEntry {
	group: InternalBindingGroup;
	lastUsedFrame: number;
}

interface TimestampPairEntry {
	label: string;
	startIndex: number;
	endIndex: number;
}

export class WebGPUBackend implements IRenderBackend {
	public readonly type = "webgpu";
	public readonly frameScheduling = "on-demand";
	public readonly passExecutors = {
		"animation-sim": "shared",
		"particle-sim": "backend",
	} as const;
	public readonly capabilities = {
		sh: true,
		shadows: true,
		reflection: false,
		skybox: true,
		ssao: true,
		taa: true,
		ssr: true,
		volumetric: true,
	};

	public canvas: HTMLCanvasElement | null = null;
	public context: GPUCanvasContext | null = null;
	public device!: GPUDevice;
	public queue!: GPUQueue;
	public canvasFormat: GPUTextureFormat = "bgra8unorm";
	public canvasDepthFormat: TextureFormat = TextureFormat.Depth24Plus;

	private _depthTexture: IRenderTexture | null = null;
	private _currentCanvasTexture: GPUTexture | null = null;
	private _currentCanvasView: GPUTextureView | null = null;
	private _errorScopes: WebGPUErrorScopeHelper | null = null;
	private _renderer: RendererBackendBridge | null = null;
	private _resources: WebGPURenderResources | null = null;
	private _frameExecutor: WebGPUFrameExecutor | null = null;
	private _particleSimulator: DefaultParticleSimulator | null = null;
	private _samplerCache = new Map<string, InternalSampler>();
	private _shaderModuleCache = new Map<string, Promise<InternalShaderModule>>();
	private _renderPipelineCache = new Map<string, InternalRenderPipeline>();
	private _computePipelineCache = new Map<string, InternalComputePipeline>();
	private _bindingGroupCache = new Map<string, CachedBindingGroupEntry>();
	private _pipelineBindGroupLayoutCache = new Map<string, GPUBindGroupLayout>();
	private _resourceIds = new WeakMap<object, number>();
	private _nextResourceId = 1;
	private _frameSerial = 0;
	private _copyCommandEncoder: GPUCommandEncoder | null = null;
	private _copyPendingCount = 0;
	private _copyFlushScheduled = false;
	private _timestampSupported = false;
	private _timestampQuerySet: GPUQuerySet | null = null;
	private _timestampResolveBuffer: GPUBuffer | null = null;
	private _timestampReadBuffer: GPUBuffer | null = null;
	private _timestampQueryCursor = 0;
	private _timestampPairs: TimestampPairEntry[] = [];
	private _timestampReadPending = false;
	private _timestampPeriodNs = 1;
	private _timestampResults = new Map<string, number>();
	private _pendingPostProcessPasses = new Map<
		string,
		WebGPUPostProcessPassPlugin
	>();
	private _msaaSampleCount = 1;

	constructor(canvas?: HTMLCanvasElement) {
		this.canvas = canvas ?? null;
	}

	public setRenderer(renderer: RendererBackendBridge): void {
		this._renderer = renderer;
	}

	public getAttachments(width: number, height: number): FrameAttachments {
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
			const requiredLimits: Record<string, number> = {};
			const requiredFeatures: GPUFeatureName[] = [];
			if (
				(adapter.limits?.maxColorAttachments ?? 0) >=
				WEBGPU_MRT_COLOR_TARGET_COUNT
			) {
				requiredLimits.maxColorAttachments = WEBGPU_MRT_COLOR_TARGET_COUNT;
			}
			if (
				(adapter.limits?.maxColorAttachmentBytesPerSample ?? 0) >=
				WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE
			) {
				requiredLimits.maxColorAttachmentBytesPerSample =
					WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE;
			}
			if (
				typeof adapter.features?.has === "function" &&
				adapter.features.has("timestamp-query" as GPUFeatureName)
			) {
				requiredFeatures.push("timestamp-query" as GPUFeatureName);
			}

			this.device = await adapter.requestDevice({
				requiredFeatures:
					requiredFeatures.length > 0 ? requiredFeatures : undefined,
				requiredLimits:
					Object.keys(requiredLimits).length > 0 ?
						(requiredLimits as any)
					:	undefined,
			});
			this.device.lost.then((info) => {
				console.error(`WebGPU device was lost: ${info.message}`);
			});
		} catch (error) {
			throw new Error(`Failed to request WebGPU device: ${error}`);
		}

		this.queue = this.device.queue;
		this._errorScopes = new WebGPUErrorScopeHelper(this.device);
		this.canvasDepthFormat = this._selectCanvasDepthFormat();
		this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
		this._msaaSampleCount = this._selectMSAASampleCount();
		this._initTimestampResources();
		this.context = canvas.getContext("webgpu");
		if (!this.context) {
			throw new Error("Failed to acquire WebGPU canvas context.");
		}

		this._configureContext();
		this._recreateDepthTexture();

		if (!this._renderer) {
			throw new Error("WebGPU backend requires a renderer before init().");
		}

		this._resources = new WebGPURenderResources(this._renderer, this);
		await this._resources.init();
		this._frameExecutor = new WebGPUFrameExecutor(this, this._resources);
		this._particleSimulator = new DefaultParticleSimulator({
			backendTag: this.type,
			maxParticlesPerSystem: 300000,
		});
		for (const pass of this._pendingPostProcessPasses.values()) {
			this._frameExecutor.registerPostProcessPass(pass);
		}
	}

	public resize(_width: number, _height: number): void {
		if (!this.device || !this.context || !this.canvas) {
			return;
		}

		this._submitPendingCopyCommands();
		this._configureContext();
		this._resetCurrentCanvasTargets();
		this._bindingGroupCache.clear();
		this._recreateDepthTexture();
		this._frameExecutor?.invalidateFrameTargets();
	}

	public beginFrame(context: FrameContext): void {
		if (!this._resources || !this._frameExecutor) {
			throw new Error("WebGPU backend has not been initialized.");
		}

		this._frameSerial++;
		this._submitPendingCopyCommands();
		this._evictStaleBindingGroups();
		this._particleSimulator?.beginFrame(context);
		this._resources.prepareFrame(context);
		this._frameExecutor.beginFrame(context);
	}

	public executePass(
		pass: FramePass,
		context: FrameContext
	): Promise<void> | void {
		if (!this._frameExecutor) {
			throw new Error("WebGPU backend has not been initialized.");
		}

		if (pass.stage === "animation-sim") {
			return;
		}

		if (pass.stage === "particle-sim") {
			this._particleSimulator?.simulate(
				context,
				this._resolveParticleDeltaTime(context)
			);
			this._particleSimulator?.emitRenderBatches(context);
			return;
		}

		return this._frameExecutor.executePass(pass, context);
	}

	public async endFrame(): Promise<void> {
		await this._frameExecutor?.endFrame();
		this._particleSimulator?.endFrame();
	}

	public registerPostProcessPass(pass: WebGPUPostProcessPassPlugin): void {
		this._pendingPostProcessPasses.set(pass.id, pass);
		this._frameExecutor?.registerPostProcessPass(pass);
	}

	public unregisterPostProcessPass(id: string): void {
		this._pendingPostProcessPasses.delete(id);
		this._frameExecutor?.unregisterPostProcessPass(id);
	}

	public destroy(): void {
		this._submitPendingCopyCommands();
		this._frameExecutor?.destroy();
		this._frameExecutor = null;
		this._resources = null;
		this._particleSimulator = null;
		this._depthTexture?.destroy();
		this._depthTexture = null;
		this._releaseTimestampResources();
		this._resetCurrentCanvasTargets();
		this._errorScopes = null;
		this._samplerCache.clear();
		this._shaderModuleCache.clear();
		this._renderPipelineCache.clear();
		this._computePipelineCache.clear();
		this._bindingGroupCache.clear();
		this._pipelineBindGroupLayoutCache.clear();
		this._resourceIds = new WeakMap<object, number>();
		this._nextResourceId = 1;
		this._frameSerial = 0;
		this._pendingPostProcessPasses.clear();
		this._msaaSampleCount = 1;
		if (this.context) {
			this.context.unconfigure();
			this.context = null;
		}
		if (this.device) {
			this.device.destroy();
		}
	}

	public createBuffer(desc: BufferDesc): IRenderBuffer {
		const hasInitialData = !!desc.initialData;
		const mappedAtCreation = hasInitialData || !!desc.mappedAtCreation;
		const gpuBuffer = this.device.createBuffer({
			size: desc.size,
			usage: this._mapBufferUsage(desc.usage),
			mappedAtCreation,
			label: desc.label,
		});
		if (hasInitialData) {
			const source = desc.initialData as BufferSource;
			const mappedRange = gpuBuffer.getMappedRange();
			const target = new Uint8Array(mappedRange);
			const srcView = this._toUint8View(source);
			const copyLength = Math.min(target.byteLength, srcView.byteLength);
			target.set(srcView.subarray(0, copyLength), 0);
			gpuBuffer.unmap();
		}

		return {
			size: desc.size,
			destroy: () => gpuBuffer.destroy(),
			_gpuResource: gpuBuffer,
		} as InternalRenderBuffer;
	}

	public createTexture(desc: TextureDesc): IRenderTexture {
		const dimension = (desc.dimension ?? "2d") as GPUTextureDimension;
		const depthOrArrayLayers = Math.max(1, desc.depthOrArrayLayers ?? 1);
		const sampleCount = Math.max(1, desc.sampleCount ?? 1);
		const size: GPUExtent3DStrict =
			dimension === "1d" ?
				{
					width: Math.max(1, desc.width | 0),
				}
			:	{
					width: Math.max(1, desc.width | 0),
					height: Math.max(1, desc.height | 0),
					depthOrArrayLayers,
				};
		const gpuTexture = this.device.createTexture({
			size,
			dimension,
			sampleCount,
			format: desc.format as GPUTextureFormat,
			usage: this._mapTextureUsage(desc.usage),
			mipLevelCount: Math.max(1, desc.mipLevelCount ?? 1),
			viewFormats: desc.viewFormats as GPUTextureFormat[] | undefined,
			label: desc.label,
		});
		const webgpuTexture = createWebGPUTexture(gpuTexture);
		const texture: InternalTexture = {
			width: desc.width,
			height: desc.height,
			destroy: () => gpuTexture.destroy(),
			_gpuResource: gpuTexture,
			_gpuTexture: gpuTexture,
			_gpuView: webgpuTexture.view,
			_webgpuTexture: webgpuTexture,
		};
		attachWebGPUTexture(texture, webgpuTexture);
		return texture;
	}

	public createSampler(desc: SamplerDesc): ISampler {
		const cacheKey = this._getSamplerCacheKey(desc);
		const cached = this._samplerCache.get(cacheKey);
		if (cached) {
			return cached;
		}

		const gpuSampler = this.device.createSampler({
			addressModeU: desc.addressModeU as GPUAddressMode | undefined,
			addressModeV: desc.addressModeV as GPUAddressMode | undefined,
			magFilter: desc.magFilter as GPUFilterMode | undefined,
			minFilter: desc.minFilter as GPUFilterMode | undefined,
			mipmapFilter: desc.mipmapFilter as GPUFilterMode | undefined,
			label: desc.label,
		});

		const sampler = {
			label: desc.label,
			_gpuResource: gpuSampler,
		} as InternalSampler;
		this._samplerCache.set(cacheKey, sampler);
		return sampler;
	}

	public async createShaderModule(
		desc: ShaderModuleDesc
	): Promise<IShaderModule> {
		const cacheKey = desc.code;
		const cached = this._shaderModuleCache.get(cacheKey);
		if (cached) {
			return cached;
		}

		const creationPromise = (async () => {
			const gpuModule = this.device.createShaderModule({
				code: desc.code,
				label: desc.label,
			});

			if (
				desc.logCompilationInfo === true &&
				typeof gpuModule.getCompilationInfo === "function"
			) {
				void gpuModule
					.getCompilationInfo()
					.then((info) => {
						if (info.messages.length <= 0) {
							return;
						}
						console.group(
							`WebGPU Shader Compilation Info [${desc.label || "unnamed"}]`
						);
						for (const message of info.messages) {
							const logType =
								message.type === "error" ? "error"
								: message.type === "warning" ? "warn"
								: "log";
							console[logType](
								`${message.message} (at line ${message.lineNum}, col ${message.linePos})`
							);
						}
						console.groupEnd();
					})
					.catch((error) => {
						console.warn(
							`WebGPU shader compilation info unavailable [${desc.label ?? "unnamed"}]: ${String(error)}`
						);
					});
			}

			return {
				label: desc.label,
				destroy: () => {},
				_gpuResource: gpuModule,
			} as InternalShaderModule;
		})();

		this._shaderModuleCache.set(cacheKey, creationPromise);
		try {
			return await creationPromise;
		} catch (error) {
			if (this._shaderModuleCache.get(cacheKey) === creationPromise) {
				this._shaderModuleCache.delete(cacheKey);
			}
			throw error;
		}
	}

	public createPipeline(desc: PipelineDesc): IRenderPipeline {
		const cacheKey = this._getRenderPipelineCacheKey(desc);
		const cached = this._getLruCacheEntry(this._renderPipelineCache, cacheKey);
		if (cached) {
			return cached;
		}
		const sampleCount = Math.max(1, Math.floor(desc.sampleCount ?? 1));

		const gpuPipeline = this._runValidationScope(
			`createRenderPipeline:${desc.label ?? "unnamed"}`,
			() =>
				this.device.createRenderPipeline({
					layout: desc.layout ?? "auto",
					vertex: {
						module: getWebGPUShaderModule(desc.vertex.module),
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
					fragment:
						desc.fragment ?
							{
								module: getWebGPUShaderModule(desc.fragment.module),
								entryPoint: desc.fragment.entryPoint,
								targets: desc.fragment.targets.map((target) => ({
									format: target.format as GPUTextureFormat,
									blend: target.blend,
								})),
							}
						:	undefined,
					primitive: {
						topology: desc.primitive?.topology ?? "triangle-list",
						cullMode: desc.primitive?.cullMode ?? "none",
						frontFace: desc.primitive?.frontFace ?? "ccw",
					},
					depthStencil:
						desc.depthStencil ?
							{
								format: desc.depthStencil.format as GPUTextureFormat,
								depthWriteEnabled: desc.depthStencil.depthWriteEnabled,
								depthCompare: desc.depthStencil
									.depthCompare as GPUCompareFunction,
							}
						:	undefined,
					multisample: {
						count: sampleCount,
					},
					label: desc.label,
				})
		);

		const pipeline = {
			label: desc.label,
			destroy: () => {},
			_gpuResource: gpuPipeline,
		} as InternalRenderPipeline;
		this._renderPipelineCache.set(cacheKey, pipeline);
		this._trimCache(this._renderPipelineCache, WEBGPU_PIPELINE_CACHE_LIMIT);
		return pipeline;
	}

	public createComputePipeline(desc: ComputePipelineDesc): IComputePipeline {
		const cacheKey = this._getComputePipelineCacheKey(desc);
		const cached = this._getLruCacheEntry(this._computePipelineCache, cacheKey);
		if (cached) {
			return cached;
		}

		const gpuPipeline = this._runValidationScope(
			`createComputePipeline:${desc.label ?? "unnamed"}`,
			() =>
				this.device.createComputePipeline({
					layout: desc.layout ?? "auto",
					compute: {
						module: getWebGPUShaderModule(desc.compute.module),
						entryPoint: desc.compute.entryPoint,
					},
					label: desc.label,
				})
		);

		const pipeline = {
			label: desc.label,
			destroy: () => {},
			_gpuResource: gpuPipeline,
		} as InternalComputePipeline;
		this._computePipelineCache.set(cacheKey, pipeline);
		this._trimCache(this._computePipelineCache, WEBGPU_PIPELINE_CACHE_LIMIT);
		return pipeline;
	}

	public createBindingGroup(desc: BindingGroupDesc): IBindingGroup {
		const pipeline =
			desc.pipeline ?
				getWebGPUPipeline(desc.pipeline as IRenderPipeline | IComputePipeline)
			:	null;
		const layout =
			(desc.layout as GPUBindGroupLayout | undefined) ??
			this._getPipelineBindGroupLayout(
				pipeline as GPURenderPipeline | GPUComputePipeline | null,
				desc.layoutIndex ?? 0
			);

		if (!layout) {
			throw new Error(
				`WebGPU binding group ${desc.label ?? "(unnamed)"} requires an explicit layout or pipeline`
			);
		}

		const cacheKey = this._getBindingGroupCacheKey(desc, layout);
		const cached = this._bindingGroupCache.get(cacheKey);
		if (cached) {
			cached.lastUsedFrame = this._frameSerial;
			this._bindingGroupCache.delete(cacheKey);
			this._bindingGroupCache.set(cacheKey, cached);
			return cached.group;
		}

		const gpuBindGroup = this._runValidationScope(
			`createBindGroup:${desc.label ?? "unnamed"}`,
			() =>
				this.device.createBindGroup({
					layout,
					entries: desc.entries.map((entry) => ({
						binding: entry.binding,
						resource: this._mapBindingResource(entry.resource),
					})),
					label: desc.label,
				})
		);

		const group = {
			label: desc.label,
			destroy: () => {},
			_gpuResource: gpuBindGroup,
		} as InternalBindingGroup;
		this._bindingGroupCache.set(cacheKey, {
			group,
			lastUsedFrame: this._frameSerial,
		});
		this._trimBindingGroupCache();
		return group;
	}

	public createCommandEncoder(): ICommandEncoder {
		return new WebGPUCommandEncoder(this.device.createCommandEncoder(), this);
	}

	public writeBuffer(
		buffer: IRenderBuffer,
		data: BufferSource,
		offset: number = 0
	): void {
		this.queue.writeBuffer(getWebGPUBuffer(buffer), offset, data);
	}

	public writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		desc: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number }
	): void {
		const gpuTexture = getWebGPUTexture(texture).texture;
		this.queue.writeTexture(
			{
				texture: gpuTexture,
				mipLevel: Math.max(0, desc.mipLevel ?? 0),
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
		const commandEncoder = this._getCopyCommandEncoder();
		const sourceTexture = getWebGPUTexture(source.texture).texture;
		const destinationTexture = getWebGPUTexture(destination.texture).texture;

		commandEncoder.copyTextureToTexture(
			{
				texture: sourceTexture,
				origin: source.origin,
				aspect: source.aspect,
			},
			{
				texture: destinationTexture,
				origin: destination.origin,
				aspect: destination.aspect,
			},
			copySize
		);
		this._copyPendingCount++;
		if (this._copyPendingCount >= WEBGPU_COPY_BATCH_SIZE) {
			this._submitPendingCopyCommands();
			return;
		}
		this._scheduleCopyFlush();
	}

	public submit(commands: ICommandBuffer[]): void {
		const submitted: GPUCommandBuffer[] = [];
		const copyCommandBuffer = this._flushPendingCopyCommandBuffer();
		if (copyCommandBuffer) {
			submitted.push(copyCommandBuffer);
		}
		for (const command of commands) {
			submitted.push(this._toInternalCommandBuffer(command)._gpuCommandBuffer);
		}
		if (submitted.length <= 0) {
			return;
		}
		const timestampResolve = this._buildTimestampResolveCommand();
		if (timestampResolve) {
			submitted.push(timestampResolve.commandBuffer);
		}
		this._runValidationScope("queue.submit", () => {
			this.queue.submit(submitted);
		});
		if (timestampResolve) {
			this._readTimestampResultsAsync(
				timestampResolve.queryCount,
				timestampResolve.pairs
			);
		}
		this._resetCurrentCanvasTargets();
	}

	public getCanvasColorTexture(): IRenderTexture {
		if (!this.context || !this.canvas) {
			throw new Error("WebGPU not initialized");
		}

		const current = this._getCurrentCanvasTexture();
		const texture: InternalTexture = {
			width: this.canvas.width,
			height: this.canvas.height,
			destroy: () => {},
			_gpuResource: current.texture,
			_gpuTexture: current.texture,
			_gpuView: current.view,
			_webgpuTexture: current,
		};
		attachWebGPUTexture(texture, current);
		return texture;
	}

	public getCanvasDepthTexture(): IRenderTexture {
		if (!this._depthTexture) {
			// fallback/safeguard for 0-dimension canvas
			throw new Error(
				"Depth texture not initialized (possibly zero dimension canvas)"
			);
		}
		return this._depthTexture;
	}

	public getCurrentColorView(): GPUTextureView {
		if (!this.context) {
			throw new Error("WebGPU canvas context is not initialized.");
		}

		return this._getCurrentCanvasTexture().view;
	}

	public getCurrentDepthView(): GPUTextureView {
		if (!this._depthTexture) {
			throw new Error("WebGPU depth texture is not initialized.");
		}

		return getWebGPUTexture(this._depthTexture).view;
	}

	public getMSAASampleCount(): number {
		return this._msaaSampleCount;
	}

	public getTimestampDurationsMs(): ReadonlyMap<string, number> {
		return this._timestampResults;
	}

	public createPassTimestampWrites(label: string):
		| {
				querySet: GPUQuerySet;
				beginningOfPassWriteIndex: number;
				endOfPassWriteIndex: number;
		  }
		| undefined {
		if (!this._timestampSupported || !this._timestampQuerySet) {
			return undefined;
		}
		if (this._timestampQueryCursor + 1 >= WEBGPU_TIMESTAMP_QUERY_CAPACITY) {
			return undefined;
		}
		const startIndex = this._timestampQueryCursor++;
		const endIndex = this._timestampQueryCursor++;
		this._timestampPairs.push({
			label: label || "pass",
			startIndex,
			endIndex,
		});
		return {
			querySet: this._timestampQuerySet,
			beginningOfPassWriteIndex: startIndex,
			endOfPassWriteIndex: endIndex,
		};
	}

	private _runValidationScope<T>(label: string, operation: () => T): T {
		if (!this._errorScopes) {
			return operation();
		}
		return this._errorScopes.run("validation", label, operation);
	}

	private _getCurrentCanvasTexture(): WebGPUTexture {
		if (!this.context) {
			throw new Error("WebGPU canvas context is not initialized.");
		}

		if (!this._currentCanvasTexture || !this._currentCanvasView) {
			this._currentCanvasTexture = this.context.getCurrentTexture();
			this._currentCanvasView = this._currentCanvasTexture.createView();
		}

		return createWebGPUTexture(
			this._currentCanvasTexture,
			this._currentCanvasView
		);
	}

	private _resetCurrentCanvasTargets(): void {
		this._currentCanvasTexture = null;
		this._currentCanvasView = null;
	}

	private _getSamplerCacheKey(desc: SamplerDesc): string {
		return [
			desc.addressModeU ?? "",
			desc.addressModeV ?? "",
			desc.magFilter ?? "",
			desc.minFilter ?? "",
			desc.mipmapFilter ?? "",
		].join("|");
	}

	private _getRenderPipelineCacheKey(desc: PipelineDesc): string {
		const parts: string[] = [];
		parts.push(`layout:${this._getCacheToken(desc.layout ?? "auto")}`);
		parts.push(`vs.module:${this._getCacheToken(desc.vertex.module)}`);
		parts.push(`vs.entry:${desc.vertex.entryPoint}`);

		const vertexBuffers = desc.vertex.buffers ?? [];
		parts.push(`vs.buffers:${vertexBuffers.length}`);
		for (let i = 0; i < vertexBuffers.length; i++) {
			const buffer = vertexBuffers[i];
			parts.push(
				`vsb${i}.stride:${buffer.arrayStride}`,
				`vsb${i}.step:${buffer.stepMode ?? "vertex"}`,
				`vsb${i}.attrs:${buffer.attributes.length}`
			);
			for (let j = 0; j < buffer.attributes.length; j++) {
				const attribute = buffer.attributes[j];
				parts.push(
					`vsa${i}.${j}.fmt:${attribute.format}`,
					`vsa${i}.${j}.off:${attribute.offset}`,
					`vsa${i}.${j}.loc:${attribute.shaderLocation}`
				);
			}
		}

		if (desc.fragment) {
			parts.push(`fs.module:${this._getCacheToken(desc.fragment.module)}`);
			parts.push(`fs.entry:${desc.fragment.entryPoint}`);
			parts.push(`fs.targets:${desc.fragment.targets.length}`);
			for (let i = 0; i < desc.fragment.targets.length; i++) {
				const target = desc.fragment.targets[i];
				parts.push(`fst${i}.fmt:${target.format}`);
				parts.push(`fst${i}.blend:${this._serializeBlendState(target.blend)}`);
			}
		} else {
			parts.push("fs:none");
		}

		parts.push(
			`primitive.topology:${desc.primitive?.topology ?? "triangle-list"}`
		);
		parts.push(`primitive.cull:${desc.primitive?.cullMode ?? "none"}`);
		parts.push(`primitive.front:${desc.primitive?.frontFace ?? "ccw"}`);
		parts.push(
			`multisample.count:${Math.max(1, Math.floor(desc.sampleCount ?? 1))}`
		);
		if (desc.depthStencil) {
			parts.push(`depth.format:${desc.depthStencil.format}`);
			parts.push(`depth.write:${desc.depthStencil.depthWriteEnabled ? 1 : 0}`);
			parts.push(`depth.compare:${desc.depthStencil.depthCompare}`);
		} else {
			parts.push("depth:none");
		}
		return parts.join("|");
	}

	private _getComputePipelineCacheKey(desc: ComputePipelineDesc): string {
		return [
			`layout:${this._getCacheToken(desc.layout ?? "auto")}`,
			`cs.module:${this._getCacheToken(desc.compute.module)}`,
			`cs.entry:${desc.compute.entryPoint}`,
		].join("|");
	}

	private _getBindingGroupCacheKey(
		desc: BindingGroupDesc,
		layout: GPUBindGroupLayout
	): string {
		const layoutKey =
			desc.layout ?
				`layout:${this._getCacheToken(desc.layout)}`
			:	`pipeline:${this._getCacheToken(desc.pipeline ?? null)}:${desc.layoutIndex ?? 0}:resolved:${this._getCacheToken(layout)}`;
		const entries = desc.entries
			.map(
				(entry) =>
					`${entry.binding}:${this._getBindingResourceCacheToken(entry.resource)}`
			)
			.join("|");
		return `${layoutKey}|${entries}`;
	}

	private _getPipelineBindGroupLayout(
		pipeline: GPURenderPipeline | GPUComputePipeline | null,
		layoutIndex: number
	): GPUBindGroupLayout | undefined {
		if (!pipeline) {
			return undefined;
		}
		const cacheKey = `${this._getCacheToken(pipeline)}:${layoutIndex}`;
		const cached = this._getLruCacheEntry(
			this._pipelineBindGroupLayoutCache,
			cacheKey
		);
		if (cached) {
			return cached;
		}
		const layout = pipeline.getBindGroupLayout(layoutIndex);
		this._pipelineBindGroupLayoutCache.set(cacheKey, layout);
		this._trimCache(
			this._pipelineBindGroupLayoutCache,
			WEBGPU_PIPELINE_LAYOUT_CACHE_LIMIT
		);
		return layout;
	}

	private _getBindingResourceCacheToken(resource: unknown): string {
		const texture = tryGetWebGPUTexture(resource);
		if (texture) {
			return `tex:${this._getCacheToken(texture.texture)}:${this._getCacheToken(texture.view)}`;
		}

		const buffer = tryGetWebGPUBuffer(resource);
		if (buffer) {
			return `buf:${this._getCacheToken(buffer)}`;
		}

		if (resource && typeof resource === "object") {
			const binding = resource as GPUBufferBinding;
			if (
				binding.buffer &&
				typeof (binding.buffer as GPUBuffer).destroy === "function"
			) {
				return `bufBinding:${this._getCacheToken(binding.buffer)}:${binding.offset ?? 0}:${binding.size ?? -1}`;
			}
		}

		return this._getCacheToken(resource);
	}

	private _serializeBlendState(blend: unknown): string {
		if (!blend || typeof blend !== "object") {
			return "none";
		}
		const asBlend = blend as {
			color?: {
				srcFactor?: string;
				dstFactor?: string;
				operation?: string;
			};
			alpha?: {
				srcFactor?: string;
				dstFactor?: string;
				operation?: string;
			};
		};
		return [
			`c:${asBlend.color?.srcFactor ?? "none"},${asBlend.color?.dstFactor ?? "none"},${asBlend.color?.operation ?? "add"}`,
			`a:${asBlend.alpha?.srcFactor ?? "none"},${asBlend.alpha?.dstFactor ?? "none"},${asBlend.alpha?.operation ?? "add"}`,
		].join("/");
	}

	private _getCacheToken(value: unknown): string {
		if (value === null) {
			return "null";
		}
		if (value === undefined) {
			return "undefined";
		}

		const type = typeof value;
		if (type === "string" || type === "number" || type === "boolean") {
			return `${type}:${String(value)}`;
		}
		if (type === "bigint") {
			return `bigint:${String(value)}`;
		}
		if (type === "symbol") {
			return `symbol:${String(value)}`;
		}
		if (type === "function" || type === "object") {
			return `obj:${this._getObjectId(value as object)}`;
		}
		return `${type}:${String(value)}`;
	}

	private _getObjectId(value: object): number {
		let id = this._resourceIds.get(value);
		if (id !== undefined) {
			return id;
		}
		id = this._nextResourceId++;
		this._resourceIds.set(value, id);
		return id;
	}

	private _getLruCacheEntry<T>(cache: Map<string, T>, key: string): T | undefined {
		const cached = cache.get(key);
		if (cached === undefined && !cache.has(key)) {
			return undefined;
		}
		cache.delete(key);
		cache.set(key, cached as T);
		return cached as T;
	}

	private _trimCache<T>(cache: Map<string, T>, maxSize: number): void {
		if (cache.size <= maxSize) {
			return;
		}

		const toEvict = cache.size - maxSize;
		let evicted = 0;
		for (const key of cache.keys()) {
			cache.delete(key);
			evicted++;
			if (evicted >= toEvict) {
				break;
			}
		}
	}

	private _trimBindingGroupCache(): void {
		this._evictStaleBindingGroups();
		if (this._bindingGroupCache.size <= WEBGPU_BINDING_GROUP_CACHE_LIMIT) {
			return;
		}
		const toEvict =
			this._bindingGroupCache.size - WEBGPU_BINDING_GROUP_CACHE_LIMIT;
		let evicted = 0;
		for (const key of this._bindingGroupCache.keys()) {
			this._bindingGroupCache.delete(key);
			evicted++;
			if (evicted >= toEvict) {
				break;
			}
		}
	}

	private _evictStaleBindingGroups(): void {
		if (this._bindingGroupCache.size <= 0) {
			return;
		}
		const minAliveFrame =
			this._frameSerial - WEBGPU_BINDING_GROUP_CACHE_TTL_FRAMES;
		for (const [key, entry] of this._bindingGroupCache) {
			if (entry.lastUsedFrame < minAliveFrame) {
				this._bindingGroupCache.delete(key);
			}
		}
	}

	private _getCopyCommandEncoder(): GPUCommandEncoder {
		if (!this._copyCommandEncoder) {
			this._copyCommandEncoder = this.device.createCommandEncoder({
				label: "WebGPUCopyBatchEncoder",
			});
		}
		return this._copyCommandEncoder;
	}

	private _flushPendingCopyCommandBuffer(): GPUCommandBuffer | null {
		if (!this._copyCommandEncoder || this._copyPendingCount <= 0) {
			this._copyCommandEncoder = null;
			this._copyPendingCount = 0;
			this._copyFlushScheduled = false;
			return null;
		}
		const commandBuffer = this._copyCommandEncoder.finish();
		this._copyCommandEncoder = null;
		this._copyPendingCount = 0;
		this._copyFlushScheduled = false;
		return commandBuffer;
	}

	private _submitPendingCopyCommands(): void {
		const commandBuffer = this._flushPendingCopyCommandBuffer();
		if (!commandBuffer) {
			return;
		}
		this._runValidationScope("queue.submit.copyBatch", () => {
			this.queue.submit([commandBuffer]);
		});
	}

	private _scheduleCopyFlush(): void {
		if (this._copyFlushScheduled) {
			return;
		}
		this._copyFlushScheduled = true;
		const scheduleMicrotask =
			typeof queueMicrotask === "function" ? queueMicrotask : (
				(callback: () => void) => {
					void Promise.resolve().then(callback);
				}
			);
		scheduleMicrotask(() => {
			if (!this._copyFlushScheduled) {
				return;
			}
			this._submitPendingCopyCommands();
		});
	}

	private _selectCanvasDepthFormat(): TextureFormat {
		const candidates: TextureFormat[] = [
			TextureFormat.Depth24Plus,
			TextureFormat.Depth32Float,
		];
		for (const candidate of candidates) {
			try {
				const probe = this.device.createTexture({
					size: [1, 1, 1],
					format: candidate as GPUTextureFormat,
					usage: GPUTextureUsage.RENDER_ATTACHMENT,
					label: "WebGPUDepthFormatProbe",
				});
				probe.destroy();
				return candidate;
			} catch {
				// Try next candidate
			}
		}
		return TextureFormat.Depth24Plus;
	}

	private _selectMSAASampleCount(): number {
		const preferred = Math.max(
			1,
			Math.floor(WEBGPU_DEFAULT_MSAA_SAMPLE_COUNT)
		);
		if (preferred <= 1) {
			return 1;
		}
		const candidates = Array.from(new Set([preferred, 4, 2])).filter(
			(sampleCount) => sampleCount > 1
		);
		for (const sampleCount of candidates) {
			if (this._supportsMSAASampleCount(sampleCount)) {
				return sampleCount;
			}
		}
		console.warn(
			`WebGPU MSAA sample counts [${candidates.join(", ")}] are not supported for current render target formats; falling back to 1x`
		);
		return 1;
	}

	private _supportsMSAASampleCount(sampleCount: number): boolean {
		const count = Math.max(1, Math.floor(sampleCount));
		if (count <= 1) {
			return true;
		}
		try {
			const descriptors: GPUTextureDescriptor[] = [
				{
					size: [1, 1, 1],
					sampleCount: count,
					format: TextureFormat.RGBA16Float as GPUTextureFormat,
					usage: GPUTextureUsage.RENDER_ATTACHMENT,
					label: "WebGPUMSAAProbeColor16F",
				},
				{
					size: [1, 1, 1],
					sampleCount: count,
					format: TextureFormat.RGBA8Unorm as GPUTextureFormat,
					usage: GPUTextureUsage.RENDER_ATTACHMENT,
					label: "WebGPUMSAAProbeColor8",
				},
				{
					size: [1, 1, 1],
					sampleCount: count,
					format: TextureFormat.Depth32Float as GPUTextureFormat,
					usage: GPUTextureUsage.RENDER_ATTACHMENT,
					label: "WebGPUMSAAProbeDepth",
				},
			];
			for (const descriptor of descriptors) {
				const probe = this.device.createTexture(descriptor);
				probe.destroy();
			}
			return true;
		} catch {
			return false;
		}
	}

	private _initTimestampResources(): void {
		this._timestampSupported = false;
		this._timestampQuerySet = null;
		this._timestampResolveBuffer = null;
		this._timestampReadBuffer = null;
		this._timestampQueryCursor = 0;
		this._timestampPairs = [];
		this._timestampResults.clear();
		this._timestampReadPending = false;
		const queueWithTimestamp = this.queue as GPUQueue & {
			getTimestampPeriod?: () => number;
		};
		this._timestampPeriodNs =
			typeof queueWithTimestamp.getTimestampPeriod === "function" ?
				queueWithTimestamp.getTimestampPeriod()
			:	1;
		if (!this.device || typeof this.device.createQuerySet !== "function") {
			return;
		}
		if (
			typeof this.device.features?.has === "function" &&
			!this.device.features.has("timestamp-query" as GPUFeatureName)
		) {
			return;
		}
		try {
			this._timestampQuerySet = this.device.createQuerySet({
				type: "timestamp",
				count: WEBGPU_TIMESTAMP_QUERY_CAPACITY,
			});
			this._timestampResolveBuffer = this.device.createBuffer({
				label: "WebGPUTimestampResolveBuffer",
				size:
					WEBGPU_TIMESTAMP_QUERY_CAPACITY * BigUint64Array.BYTES_PER_ELEMENT,
				usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
			});
			this._timestampReadBuffer = this.device.createBuffer({
				label: "WebGPUTimestampReadBuffer",
				size:
					WEBGPU_TIMESTAMP_QUERY_CAPACITY * BigUint64Array.BYTES_PER_ELEMENT,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			});
			this._timestampSupported = true;
		} catch {
			this._releaseTimestampResources();
		}
	}

	private _buildTimestampResolveCommand():
		| {
				commandBuffer: GPUCommandBuffer;
				queryCount: number;
				pairs: TimestampPairEntry[];
		  }
		| undefined {
		if (
			!this._timestampSupported ||
			!this._timestampQuerySet ||
			!this._timestampResolveBuffer ||
			!this._timestampReadBuffer
		) {
			return undefined;
		}
		const queryCount = this._timestampQueryCursor;
		if (queryCount <= 0) {
			return undefined;
		}
		if (this._timestampReadPending) {
			this._dropPendingTimestampSamples();
			return undefined;
		}
		this._tryUnmapBuffer(this._timestampReadBuffer);
		if (this._isBufferMapped(this._timestampReadBuffer)) {
			this._dropPendingTimestampSamples();
			return undefined;
		}
		const pairs = this._timestampPairs.slice();
		const resolveEncoder = this.device.createCommandEncoder({
			label: "WebGPUTimestampResolveEncoder",
		});
		resolveEncoder.resolveQuerySet(
			this._timestampQuerySet,
			0,
			queryCount,
			this._timestampResolveBuffer,
			0
		);
		resolveEncoder.copyBufferToBuffer(
			this._timestampResolveBuffer,
			0,
			this._timestampReadBuffer,
			0,
			queryCount * BigUint64Array.BYTES_PER_ELEMENT
		);
		this._timestampQueryCursor = 0;
		this._timestampPairs = [];
		return {
			commandBuffer: resolveEncoder.finish(),
			queryCount,
			pairs,
		};
	}

	private _readTimestampResultsAsync(
		queryCount: number,
		pairs: TimestampPairEntry[]
	): void {
		if (
			this._timestampReadPending ||
			!this._timestampReadBuffer ||
			queryCount <= 0
		) {
			return;
		}
		this._tryUnmapBuffer(this._timestampReadBuffer);
		if (this._isBufferMapped(this._timestampReadBuffer)) {
			return;
		}
		this._timestampReadPending = true;
		const byteLength = queryCount * BigUint64Array.BYTES_PER_ELEMENT;
		void this._timestampReadBuffer
			.mapAsync(GPUMapMode.READ, 0, byteLength)
			.then(() => {
				if (!this._timestampReadBuffer) {
					return;
				}
				const view = this._timestampReadBuffer.getMappedRange(0, byteLength);
				const data = new BigUint64Array(view.slice(0));
				const result = new Map<string, number>();
				for (let i = 0; i < pairs.length; i++) {
					const pair = pairs[i];
					if (pair.endIndex >= data.length || pair.startIndex >= data.length) {
						continue;
					}
					const start = data[pair.startIndex];
					const end = data[pair.endIndex];
					const deltaTicks = end >= start ? end - start : 0n;
					const durationMs =
						(Number(deltaTicks) * this._timestampPeriodNs) / 1_000_000;
					result.set(`${pair.label}#${i}`, durationMs);
				}
				this._timestampResults = result;
				this._timestampReadBuffer.unmap();
			})
			.catch((error) => {
				console.warn(`WebGPU timestamp readback failed: ${String(error)}`);
				if (this._timestampReadBuffer) {
					try {
						this._timestampReadBuffer.unmap();
					} catch {
						// ignore
					}
				}
			})
			.finally(() => {
				this._timestampReadPending = false;
			});
	}

	private _releaseTimestampResources(): void {
		this._timestampSupported = false;
		this._timestampQueryCursor = 0;
		this._timestampPairs = [];
		this._timestampReadPending = false;
		this._timestampResults.clear();
		if (this._timestampReadBuffer) {
			this._tryUnmapBuffer(this._timestampReadBuffer);
			try {
				this._timestampReadBuffer.destroy();
			} catch {
				// ignore
			}
			this._timestampReadBuffer = null;
		}
		if (this._timestampResolveBuffer) {
			try {
				this._timestampResolveBuffer.destroy();
			} catch {
				// ignore
			}
			this._timestampResolveBuffer = null;
		}
		if (this._timestampQuerySet) {
			try {
				this._timestampQuerySet.destroy();
			} catch {
				// ignore
			}
			this._timestampQuerySet = null;
		}
	}

	private _dropPendingTimestampSamples(): void {
		this._timestampQueryCursor = 0;
		this._timestampPairs = [];
	}

	private _isBufferMapped(buffer: GPUBuffer | null): boolean {
		if (!buffer) {
			return false;
		}
		const state = (buffer as GPUBuffer & { mapState?: GPUBufferMapState })
			.mapState;
		return (state ?? "unmapped") !== "unmapped";
	}

	private _tryUnmapBuffer(buffer: GPUBuffer | null): void {
		if (!this._isBufferMapped(buffer) || !buffer) {
			return;
		}
		try {
			buffer.unmap();
		} catch {
			// ignore
		}
	}

	private _toUint8View(data: BufferSource): Uint8Array {
		if (data instanceof ArrayBuffer) {
			return new Uint8Array(data);
		}
		return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
	}

	private _configureContext(): void {
		if (!this.context || !this.canvas) {
			return;
		}

		this._resetCurrentCanvasTargets();
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
			if (this._depthTexture) {
				this._depthTexture.destroy();
				this._depthTexture = null;
			}
			return;
		}

		this._depthTexture?.destroy();
		this._depthTexture = this.createTexture({
			width: this.canvas.width,
			height: this.canvas.height,
			format: this.canvasDepthFormat,
			usage: TextureUsage.RenderAttachment,
			label: "WebGPUCanvasDepth",
		});
	}

	private _mapBindingResource(resource: any): GPUBindingResource {
		const texture = tryGetWebGPUTexture(resource);
		if (texture) {
			return texture.view;
		}

		if (resource && typeof (resource as GPUTexture).createView === "function") {
			return (resource as GPUTexture).createView();
		}

		const buffer = tryGetWebGPUBuffer(resource);
		if (buffer) {
			return { buffer };
		}

		if (
			resource?.buffer &&
			typeof (resource.buffer as GPUBuffer).destroy === "function"
		) {
			return resource as GPUBindingResource;
		}

		if (resource?._gpuResource) {
			return resource._gpuResource;
		}

		return resource;
	}

	private _mapBufferUsage(usage: number): GPUBufferUsageFlags {
		let flags = 0;
		if (usage & BufferUsage.Vertex) {
			flags |= GPUBufferUsage.VERTEX;
		}
		if (usage & BufferUsage.Index) {
			flags |= GPUBufferUsage.INDEX;
		}
		if (usage & BufferUsage.Uniform) {
			flags |= GPUBufferUsage.UNIFORM;
		}
		if (usage & BufferUsage.Storage) {
			flags |= GPUBufferUsage.STORAGE;
		}
		if (usage & BufferUsage.CopySrc) {
			flags |= GPUBufferUsage.COPY_SRC;
		}
		if (usage & BufferUsage.CopyDst) {
			flags |= GPUBufferUsage.COPY_DST;
		}
		if (usage & BufferUsage.MapRead) {
			flags |= GPUBufferUsage.MAP_READ;
		}
		if (usage & BufferUsage.MapWrite) {
			flags |= GPUBufferUsage.MAP_WRITE;
		}
		return flags;
	}

	private _mapTextureUsage(usage: number): GPUTextureUsageFlags {
		let flags = 0;
		if (usage & TextureUsage.CopySrc) {
			flags |= GPUTextureUsage.COPY_SRC;
		}
		if (usage & TextureUsage.CopyDst) {
			flags |= GPUTextureUsage.COPY_DST;
		}
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

	private _toInternalCommandBuffer(
		command: ICommandBuffer
	): InternalCommandBuffer {
		const internal = command as InternalCommandBuffer;
		if (!internal._gpuCommandBuffer) {
			throw new Error("Invalid command buffer for WebGPU submit().");
		}
		return internal;
	}

	private _resolveParticleDeltaTime(context: FrameContext): number {
		const value = context.transient.get(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY);
		if (typeof value !== "number" || !Number.isFinite(value)) {
			return 0;
		}
		return Math.max(0, value);
	}
}

class WebGPUCommandEncoder implements ICommandEncoder {
	private _encoder: GPUCommandEncoder;
	private _backend: WebGPUBackend;
	private _renderPass: GPURenderPassEncoder | null = null;
	private _computePass: GPUComputePassEncoder | null = null;

	constructor(encoder: GPUCommandEncoder, backend: WebGPUBackend) {
		this._encoder = encoder;
		this._backend = backend;
	}

	public beginRenderPass(desc: RenderPassDesc): void {
		const timestampWrites = this._backend.createPassTimestampWrites(
			desc.label ?? "render-pass"
		);
		this._renderPass = this._encoder.beginRenderPass({
			colorAttachments: desc.colorAttachments.map((attachment) => ({
				view:
					tryGetWebGPUTexture(attachment.view)?.view ??
					this._backend.getCurrentColorView(),
				resolveTarget:
					attachment.resolveTarget ?
						tryGetWebGPUTexture(attachment.resolveTarget)?.view
					:	undefined,
				clearValue:
					attachment.loadOp === "clear" ? attachment.clearValue : undefined,
				loadOp: attachment.loadOp,
				storeOp: attachment.storeOp,
			})),
			depthStencilAttachment:
				desc.depthStencilAttachment ?
					{
						view:
							tryGetWebGPUTexture(desc.depthStencilAttachment.view)?.view ??
							this._backend.getCurrentDepthView(),
						depthClearValue:
							(desc.depthStencilAttachment.depthLoadOp ?? "clear") === "clear" ?
								(desc.depthStencilAttachment.depthClearValue ?? 1)
							:	undefined,
						depthLoadOp: desc.depthStencilAttachment.depthLoadOp ?? "clear",
						depthStoreOp: desc.depthStencilAttachment.depthStoreOp ?? "store",
					}
				:	undefined,
			label: desc.label,
			timestampWrites,
		});
	}

	public beginComputePass(desc?: ComputePassDesc): void {
		const timestampWrites = this._backend.createPassTimestampWrites(
			desc?.label ?? "compute-pass"
		);
		this._computePass = this._encoder.beginComputePass({
			label: desc?.label,
			timestampWrites,
		});
	}

	public setComputePipeline(pipeline: IComputePipeline): void {
		this._computePass?.setPipeline(getWebGPUComputePipeline(pipeline));
	}

	public dispatchWorkgroups(x: number, y: number = 1, z: number = 1): void {
		this._computePass?.dispatchWorkgroups(x, y, z);
	}

	public endComputePass(): void {
		this._computePass?.end();
		this._computePass = null;
	}

	public setPipeline(pipeline: IRenderPipeline): void {
		this._renderPass?.setPipeline(getWebGPURenderPipeline(pipeline));
	}

	public setBindingGroup(index: number, group: IBindingGroup): void {
		const groupResource = getWebGPUBindGroup(group);
		if (this._renderPass) {
			this._renderPass.setBindGroup(index, groupResource);
		} else if (this._computePass) {
			this._computePass.setBindGroup(index, groupResource);
		}
	}

	public setVertexBuffer(slot: number, buffer: IRenderBuffer): void {
		this._renderPass?.setVertexBuffer(slot, getWebGPUBuffer(buffer));
	}

	public setIndexBuffer(buffer: IRenderBuffer, format: IndexFormat): void {
		this._renderPass?.setIndexBuffer(getWebGPUBuffer(buffer), format);
	}

	public drawIndexed(
		indexCount: number,
		instanceCount: number = 1,
		firstIndex: number = 0,
		baseVertex: number = 0,
		firstInstance: number = 0
	): void {
		this._renderPass?.drawIndexed(
			indexCount,
			instanceCount,
			firstIndex,
			baseVertex,
			firstInstance
		);
	}

	public draw(
		vertexCount: number,
		instanceCount: number = 1,
		firstVertex: number = 0,
		firstInstance: number = 0
	): void {
		this._renderPass?.draw(
			vertexCount,
			instanceCount,
			firstVertex,
			firstInstance
		);
	}

	public endRenderPass(): void {
		this._renderPass?.end();
		this._renderPass = null;
	}

	public finish(): InternalCommandBuffer {
		if (this._renderPass || this._computePass) throw Error("Pass still active");
		return {
			_gpuCommandBuffer: this._encoder.finish(),
		};
	}
}
