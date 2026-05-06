/// <reference types="@webgpu/types" />
import {
	type ICommandBuffer,
	type ICommandEncoder,
} from "./ICommandEncoder";
import type {
	IRenderBackend,
	RendererBackendBridge,
	WarmupOptions,
	WarmupReport,
} from "./IRenderBackend";
import {
	FRAME_PASS_DEPENDENCIES,
	type FrameAttachments,
	type FrameContext,
	type FramePass,
	INTERACTION_TRANSIENT_STATE_KEY,
	isFogPostProcessEnabled,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
} from "../pipeline/types";
import { hasParticleShadowCasters } from "../pipeline/ParticleShadowVolume";
import { WebGPUErrorScopeHelper } from "./webgpu/WebGPUErrorScopeHelper";
import { WebGPUFrameExecutor } from "./webgpu/WebGPUFrameExecutor";
import { WebGPUCommandEncoder } from "./webgpu/WebGPUCommandEncoder";
import {
	WebGPUReflectionProbeCapturePass,
	type WebGPUReflectionProbeCaptureFaceRequest,
} from "./webgpu/WebGPUReflectionProbeCapturePass";
import { WebGPURenderResources } from "./webgpu/WebGPURenderResources";
import {
	attachWebGPUTexture,
	createWebGPUTexture,
	getWebGPUBuffer,
	getWebGPUPipeline,
	getWebGPUShaderModule,
	getWebGPUTexture,
	tryGetWebGPUBuffer,
	tryGetWebGPUTexture,
	type WebGPUTexture,
} from "./webgpu/WebGPUResourceAccess";
import type { WebGPUPostProcessPassPlugin } from "./webgpu/WebGPUPostProcessGraph";
import type { IParticleSimulator } from "../simulation/particles/IParticleSimulator";
import { WebGPUParticleSimulator } from "../simulation/particles/WebGPUParticleSimulator";
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
	WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
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
	type PipelineDesc,
	type SamplerDesc,
	type ShaderModuleDesc,
	type TextureDataLayout,
	type TextureDesc,
	TextureFormat,
	TextureUsage,
} from "./types";
import {
	formatShaderCompilerMessages,
	mapShaderCompilerMessages,
	normalizeWebGPUCompilationMessages,
	ShaderBackendCompileStage,
	ShaderCompileError,
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	ShaderRuntime,
} from "../shaders/runtime";
import type {
	ShaderBackendCompileResult,
	ShaderCompilerMessage,
	ShaderDirectiveCompileHook,
	ShaderProcessResult,
	ShaderRuntimeMode,
} from "../shaders/runtime";
import {
	addWarmupPhase,
	buildWarmupPlan,
	createWarmupReport,
	finalizeWarmupReport,
	toShaderCompileError,
} from "../pipeline/WarmupPlanner";
import type { Texture } from "../core/Texture";
import {
	createWebGPUComputeFacade,
	invalidateWebGPUComputeFacade,
	type IWebGPUComputeFacade,
} from "./webgpu/ComputeFacade";
import { Logger } from "../foundation/Logger";

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
	destroy(): void;
	_gpuResource: GPUSampler;
}

interface InternalShaderModule extends IShaderModule {
	destroy(): void;
	_gpuResource: GPUShaderModule;
}

interface InternalRenderPipeline extends IRenderPipeline {
	destroy(): void;
	_gpuResource: GPURenderPipeline;
}

interface InternalComputePipeline extends IComputePipeline {
	destroy(): void;
	_gpuResource: GPUComputePipeline;
}

interface InternalBindingGroup extends IBindingGroup {
	destroy(): void;
	_gpuResource: GPUBindGroup;
}

interface InternalCommandBuffer {
	_backendCommandBuffer: GPUCommandBuffer;
	_ownerToken: object;
	_submitted: boolean;
}

interface CachedSamplerEntry {
	key: string;
	label?: string;
	gpuResource: GPUSampler;
	refCount: number;
}

interface CachedShaderModuleEntry {
	key: string;
	label?: string;
	refCount: number;
	gpuResource: GPUShaderModule;
}

interface CachedRenderPipelineEntry {
	key: string;
	label?: string;
	refCount: number;
	gpuResource: GPURenderPipeline;
}

interface CachedComputePipelineEntry {
	key: string;
	label?: string;
	refCount: number;
	gpuResource: GPUComputePipeline;
}

interface BindingResourceSignature {
	binding: number;
	kind: number;
	primaryId: number;
	secondaryId: number;
	offset: number;
	size: number;
}

interface CachedBindingGroupEntry {
	hashKey: bigint;
	layoutId: number;
	signatures: BindingResourceSignature[];
	group: InternalBindingGroup;
	lastUsedFrame: number;
	lastTouchedTick: number;
	refCount: number;
}

interface TimestampPairEntry {
	label: string;
	startIndex: number;
	endIndex: number;
}

type BindingResourceInput = BindingGroupDesc["entries"][number]["resource"];
type PipelineDescLayout = PipelineDesc["layout"] | undefined;
type ComputePipelineDescLayout = ComputePipelineDesc["layout"] | undefined;

const HASH64_OFFSET_BASIS = 0xcbf29ce484222325n;
const HASH64_PRIME = 0x100000001b3n;
const HASH64_MASK = 0xffffffffffffffffn;

const SHADER_CODE_HASH_CACHE_LIMIT = 128;
const RESOURCE_ID_REBASE_THRESHOLD = 0x40000000;
const DEVICE_RECOVERY_MAX_ATTEMPTS = 3;
const DEVICE_RECOVERY_BASE_DELAY_MS = 100;
const WEBGPU_MSAA_SAMPLE_CANDIDATES = [16, 8, 4, 2, 1];
const WEBGPU_EXPLICIT_MSAA_ENABLE_SAMPLE_COUNT = 4;

export interface WebGPUBackendOptions {
	canvas?: HTMLCanvasElement;
	shaderMode?: ShaderRuntimeMode;
	directiveHook?: ShaderDirectiveCompileHook | null;
	enableMSAA?: boolean;
	enableEarlyZPrepass?: boolean;
}

type WebGPUPassHandler = (
	pass: FramePass,
	context: FrameContext
) => void | Promise<void>;

function isWebGPUBackendOptions(
	value: unknown
): value is WebGPUBackendOptions {
	return typeof value === "object" && value !== null;
}

function resolveWebGPUBackendCtorArgs(
	canvasOrOptions?: HTMLCanvasElement | WebGPUBackendOptions,
	options?: WebGPUBackendOptions
): { canvas: HTMLCanvasElement | null; options: WebGPUBackendOptions } {
	const fallbackOptions = options ?? {};
	if (
		typeof HTMLCanvasElement !== "undefined" &&
		canvasOrOptions instanceof HTMLCanvasElement
	) {
		return {
			canvas: canvasOrOptions,
			options: fallbackOptions,
		};
	}
	if (isWebGPUBackendOptions(canvasOrOptions)) {
		return {
			canvas: canvasOrOptions.canvas ?? null,
			options: {
				...canvasOrOptions,
				...fallbackOptions,
			},
		};
	}
	return {
		canvas: null,
		options: fallbackOptions,
	};
}

function resolveGPUTextureExtent(
	texture: GPUTexture,
	fallbackWidth: number,
	fallbackHeight: number
): { width: number; height: number } {
	const textureWithExtent = texture as GPUTexture & {
		width?: unknown;
		height?: unknown;
	};
	const width =
		typeof textureWithExtent.width === "number" &&
		Number.isFinite(textureWithExtent.width) ?
			textureWithExtent.width
		:	fallbackWidth;
	const height =
		typeof textureWithExtent.height === "number" &&
		Number.isFinite(textureWithExtent.height) ?
			textureWithExtent.height
		:	fallbackHeight;
	return {
		width: Math.max(1, Math.floor(width)),
		height: Math.max(1, Math.floor(height)),
	};
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
		ssgi: true,
		taa: true,
		ssr: true,
		volumetric: true,
		fog: true,
		motionBlur: true,
		dof: true,
		bloom: true,
		colorFilter: true,
		clusteredLighting: true,
		oit: true,
	};

	private _canvas: HTMLCanvasElement | null = null;
	private _context: GPUCanvasContext | null = null;
	private _device: GPUDevice | null = null;
	private _queue: GPUQueue | null = null;

	/**
	 * Returns the current presentation canvas.
	 * The backend owns this reference and does not allow external replacement.
	 */
	public get canvas(): HTMLCanvasElement | null {
		return this._canvas;
	}

	/**
	 * Returns the active WebGPU canvas context.
	 * The backend owns context lifecycle and configuration.
	 */
	public get context(): GPUCanvasContext | null {
		return this._context;
	}

	/**
	 * Returns the active GPU device handle for diagnostics and advanced tooling.
	 * The backend owns device lifecycle and prevents external reassignment.
	 */
	public get device(): GPUDevice | null {
		return this._device;
	}

	/**
	 * Returns the active queue associated with `device`.
	 * Queue ownership remains internal to keep backend state coherent.
	 */
	public get queue(): GPUQueue | null {
		return this._queue;
	}

	public canvasFormat: GPUTextureFormat = "bgra8unorm";
	public canvasDepthFormat: TextureFormat = TextureFormat.Depth24Plus;
	public readonly shaderRuntime: ShaderRuntime;

	private _depthTexture: IRenderTexture | null = null;
	private _currentCanvasTexture: GPUTexture | null = null;
	private _currentCanvasView: GPUTextureView | null = null;
	private _errorScopes: WebGPUErrorScopeHelper | null = null;
	private _resources: WebGPURenderResources | null = null;
	private _frameExecutor: WebGPUFrameExecutor | null = null;
	private _reflectionProbeCapturePass: WebGPUReflectionProbeCapturePass | null = null;
	private _particleSimulator: IParticleSimulator | null = null;
	private _deviceLost = false;
	private _deviceLostInfo: GPUDeviceLostInfo | null = null;
	private _deviceLossPromise: Promise<GPUDeviceLostInfo> | null = null;
	private _samplerCache = new Map<string, CachedSamplerEntry>();
	private _shaderModuleCache = new Map<string, CachedShaderModuleEntry>();
	private _shaderCodeHashCache = new Map<string, string>();
	private _shaderModuleInFlight = new Map<string, Promise<CachedShaderModuleEntry>>();
	private _renderPipelineCache = new Map<string, CachedRenderPipelineEntry>();
	private _computePipelineCache = new Map<string, CachedComputePipelineEntry>();
	private _bindingGroupCache = new Map<bigint, CachedBindingGroupEntry[]>();
	private _bindingGroupCacheEntryCount = 0;
	private _pipelineBindGroupLayoutCache = new Map<string, GPUBindGroupLayout>();
	private _autoRenderPipelineLayoutCache = new Map<string, GPUPipelineLayout>();
	private _autoComputePipelineLayoutCache = new Map<string, GPUPipelineLayout>();
	private _commandBufferOwnerToken: object = {};
	private _resourceIds = new WeakMap<object, number>();
	private _nextResourceId = 1;
	private _destroyRequested = false;
	private _deviceRecoveryNonce = 0;
	private _deviceRecoveryPromise: Promise<void> | null = null;
	private _frameSerial = 0;
	private _bindingGroupTouchTick = 0;
	private _executedPasses = new Set<FramePass["stage"]>();
	private _plannedPasses = new Set<FramePass["stage"]>();
	private _plannedPassOrder = new Map<FramePass["stage"], number>();
	private _autoDisposeRegistry: FinalizationRegistry<string> | null =
		typeof FinalizationRegistry === "function"
			? new FinalizationRegistry<string>((label) => {
					const key = `webgpu-resource-gc:${label}`;
					Logger.warn(
						`[${key}] WebGPU resource "${label}" was garbage collected without explicit destroy().`,
						{ scope: "WebGPUBackend", onceKey: key },
					);
				})
			: null;
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
	private _pendingPostProcessPasses = new Map<string, WebGPUPostProcessPassPlugin>();
	private _warmupLogCompilationInfo = false;
	private _msaaSelectionCache = new Map<string, number>();
	private readonly _defaultMSAASampleCount: number;
	private _preferredMSAASampleCount = WEBGPU_DEFAULT_MSAA_SAMPLE_COUNT;
	private _msaaSampleCount = 1;
	private _enableEarlyZPrepass = true;
	private _shaderCompileStage: ShaderBackendCompileStage;
	private readonly _passHandlers: Map<FramePass["stage"], WebGPUPassHandler>;

	constructor(
		canvasOrOptions?: HTMLCanvasElement | WebGPUBackendOptions,
		options?: WebGPUBackendOptions,
	) {
		const resolved = resolveWebGPUBackendCtorArgs(canvasOrOptions, options);
		const shaderMode = resolved.options.shaderMode ?? "strict";
		this._defaultMSAASampleCount =
			resolved.options.enableMSAA === false ? 1 : WEBGPU_DEFAULT_MSAA_SAMPLE_COUNT;
		this._preferredMSAASampleCount = this._defaultMSAASampleCount;
		this._enableEarlyZPrepass = resolved.options.enableEarlyZPrepass !== false;
		this._canvas = resolved.canvas ?? null;
		this.shaderRuntime = new ShaderRuntime({
			mode: shaderMode,
		});
		this._shaderCompileStage = new ShaderBackendCompileStage({
			backend: "webgpu",
			runtime: this.shaderRuntime,
			profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
			hook: resolved.options.directiveHook ?? null,
			mode: shaderMode,
		});
		this._passHandlers = this._createPassHandlers();
		this.shaderRuntime.onDidChange(() => {
			this._onShaderRuntimeChanged();
		});
	}

	public setRenderer(renderer: RendererBackendBridge): void {
		void renderer;
	}

	public getComputeFacade(): IWebGPUComputeFacade {
		return createWebGPUComputeFacade(this);
	}

	public getShaderDirectiveCacheTag(): string {
		return this._shaderCompileStage.getCacheFingerprintTag();
	}

	public isEarlyZPrepassEnabled(): boolean {
		return this._enableEarlyZPrepass;
	}

	public getAttachments(width: number, height: number): FrameAttachments {
		return {
			width,
			height,
		};
	}

	public async init(canvas: HTMLCanvasElement): Promise<void> {
		this._canvas = canvas;
		this._destroyRequested = false;

		if (!navigator.gpu) {
			throw new Error("WebGPU not supported on this browser.");
		}

		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) {
			throw new Error("No appropriate GPUAdapter found.");
		}

		let requestedDevice: GPUDevice;
		try {
			const requiredLimits: Record<string, number> = {};
			const requiredFeatures: GPUFeatureName[] = [];
			const adapterMaxTextureDimension2D = adapter.limits?.maxTextureDimension2D ?? 0;
			const adapterMaxSampledTexturesPerShaderStage =
				adapter.limits?.maxSampledTexturesPerShaderStage;
			const requiredSampledTexturesPerShaderStage =
				WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT;
			if ((adapter.limits?.maxColorAttachments ?? 0) >= WEBGPU_MRT_COLOR_TARGET_COUNT) {
				requiredLimits.maxColorAttachments = WEBGPU_MRT_COLOR_TARGET_COUNT;
			}
			if (
				(adapter.limits?.maxColorAttachmentBytesPerSample ?? 0) >=
				WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE
			) {
				requiredLimits.maxColorAttachmentBytesPerSample = WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE;
			}
			if (adapterMaxTextureDimension2D > 0) {
				requiredLimits.maxTextureDimension2D = adapterMaxTextureDimension2D;
			}
			requiredLimits.maxSampledTexturesPerShaderStage = requiredSampledTexturesPerShaderStage;
			if (
				typeof adapterMaxSampledTexturesPerShaderStage === "number" &&
				adapterMaxSampledTexturesPerShaderStage < requiredSampledTexturesPerShaderStage
			) {
				throw new Error(
					"WebGPU adapter maxSampledTexturesPerShaderStage " +
						`(${adapterMaxSampledTexturesPerShaderStage}) is below required ` +
						"scene pipeline sampled texture count " +
						`(${requiredSampledTexturesPerShaderStage}).`,
				);
			}
			if (
				typeof adapter.features?.has === "function" &&
				adapter.features.has("timestamp-query" as GPUFeatureName)
			) {
				requiredFeatures.push("timestamp-query" as GPUFeatureName);
			}

			requestedDevice = await adapter.requestDevice({
				requiredFeatures: requiredFeatures.length > 0 ? requiredFeatures : undefined,
				requiredLimits:
					Object.keys(requiredLimits).length > 0 ? (requiredLimits as any) : undefined,
			});
			const deviceMaxSampledTexturesPerShaderStage =
				requestedDevice.limits?.maxSampledTexturesPerShaderStage;
			if (
				typeof deviceMaxSampledTexturesPerShaderStage === "number" &&
				deviceMaxSampledTexturesPerShaderStage < requiredSampledTexturesPerShaderStage
			) {
				throw new Error(
					"Requested WebGPU device maxSampledTexturesPerShaderStage " +
						`(${deviceMaxSampledTexturesPerShaderStage}) is below required ` +
						"scene pipeline sampled texture count " +
						`(${requiredSampledTexturesPerShaderStage}).`,
				);
			}
		} catch (error) {
			throw new Error(`Failed to request WebGPU device: ${error}`);
		}

		const context = canvas.getContext("webgpu");
		if (!context) {
			throw new Error("Failed to acquire WebGPU canvas context.");
		}

		this._deviceLost = false;
		this._deviceLostInfo = null;
		this._device = requestedDevice;
		this._queue = requestedDevice.queue;
		this._deviceLossPromise = requestedDevice.lost.then((info) => {
			if (this.device !== requestedDevice) {
				return info;
			}
			this._handleDeviceLost(info);
			return info;
		});

		try {
			this._errorScopes = new WebGPUErrorScopeHelper(requestedDevice);
			this.canvasDepthFormat = this._selectCanvasDepthFormat();
			this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
			this._msaaSampleCount = this._selectMSAASampleCount();
			this._initTimestampResources();
			this._context = context;
			this._configureContext();
			this._recreateDepthTexture();

			this._resources = new WebGPURenderResources(this);
			await this._resources.init();
			this._frameExecutor = new WebGPUFrameExecutor(this, this._resources);
			this._reflectionProbeCapturePass = new WebGPUReflectionProbeCapturePass(
				this,
				this._resources,
			);
			this._particleSimulator = new WebGPUParticleSimulator({
				backend: this,
				backendTag: this.type,
				maxParticlesPerSystem: 300000,
			});
			for (const pass of this._pendingPostProcessPasses.values()) {
				this._frameExecutor.registerPostProcessPass(pass);
			}
		} catch (error) {
			this._rollbackInitializationState();
			throw error;
		}
	}

	public resize(width: number, height: number): void {
		if (!this.device || !this.context || !this.canvas) {
			return;
		}
		const resolvedWidth = Number.isFinite(width)
			? Math.max(0, Math.floor(width))
			: this.canvas.width;
		const resolvedHeight = Number.isFinite(height)
			? Math.max(0, Math.floor(height))
			: this.canvas.height;
		if (this.canvas.width !== resolvedWidth || this.canvas.height !== resolvedHeight) {
			this.canvas.width = resolvedWidth;
			this.canvas.height = resolvedHeight;
		}

		this._submitPendingCopyCommands();
		this._configureContext();
		this._resetCurrentCanvasTargets();
		this._bindingGroupCache.clear();
		this._bindingGroupCacheEntryCount = 0;
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
		this._prepareFramePassPlan(context);
		this._executedPasses.clear();
		this._particleSimulator?.beginFrame(context);
		this._frameExecutor.beginFrame(context);
		this._resources.prepareFrame(context);
	}

	public executePass(pass: FramePass, context: FrameContext): Promise<void> | void {
		if (!this._frameExecutor) {
			throw new Error("WebGPU backend has not been initialized.");
		}

		this._validatePassDependencies(pass);
		const handler = this._passHandlers.get(pass.stage);
		const result = handler
			? handler(pass, context)
			: this._frameExecutor.executePass(pass, context);
		if (result && typeof (result as Promise<void>).then === "function") {
			return (result as Promise<void>).then(() => {
				this._markPassExecuted(pass.stage);
			});
		}
		this._markPassExecuted(pass.stage);
		return result;
	}

	public skipPass(pass: FramePass): void {
		this._markPassExecuted(pass.stage);
	}

	public async warmup(context: FrameContext, options: WarmupOptions = {}): Promise<WarmupReport> {
		const report = createWarmupReport(this.type);
		if (!this._resources || !this._frameExecutor) {
			throw new Error("WebGPU backend has not been initialized.");
		}

		const plan = buildWarmupPlan(context, options);
		this._warmupLogCompilationInfo = options.logCompilationInfo === true;
		try {
			const framePhase = await this._frameExecutor.warmup(context, plan);
			addWarmupPhase(report, framePhase);
			const resourcePhase = await this._resources.warmup(context, plan);
			addWarmupPhase(report, resourcePhase);
		} catch (error) {
			addWarmupPhase(report, {
				phase: "webgpu-warmup",
				total: 1,
				compiled: 0,
				skipped: 0,
				failed: 1,
				errors: [toShaderCompileError(error, this.type, "WebGPUWarmup")],
			});
		} finally {
			this._warmupLogCompilationInfo = false;
		}
		return finalizeWarmupReport(report);
	}

	public async endFrame(): Promise<void> {
		await this._frameExecutor?.endFrame();
		this._particleSimulator?.endFrame();
		this._executedPasses.clear();
		this._plannedPasses.clear();
		this._plannedPassOrder.clear();
	}

	public registerPostProcessPass(pass: WebGPUPostProcessPassPlugin): void {
		this._pendingPostProcessPasses.set(pass.id, pass);
		this._frameExecutor?.registerPostProcessPass(pass);
	}

	public unregisterPostProcessPass(id: string): void {
		this._pendingPostProcessPasses.delete(id);
		this._frameExecutor?.unregisterPostProcessPass(id);
	}

	public getTextureForSlot(texture: Texture | null, slotIndex: number): IRenderTexture {
		this._assertDeviceOperational("resolve texture resources");
		if (!this._resources) {
			throw new Error(
				"WebGPU resources are not initialized; cannot resolve texture resources.",
			);
		}
		return this._resources.getTextureForSlot(texture, slotIndex);
	}

	public registerExternalTexture(
		texture: Texture,
		resource: IRenderTexture,
		uploadedVersion: number = texture.version,
		mipLevelCount: number = 1,
	): void {
		this._assertDeviceOperational("register external textures");
		if (!this._resources) {
			throw new Error(
				"WebGPU resources are not initialized; cannot register external textures.",
			);
		}
		this._resources.registerExternalTexture(texture, resource, uploadedVersion, mipLevelCount);
	}

	public unregisterExternalTexture(texture: Texture): void {
		if (!this._resources) {
			return;
		}
		this._resources.unregisterExternalTexture(texture);
	}

	public getFrameSceneTargetMode(): "mrt" | "single" {
		return this._frameExecutor?.getSceneTargetModeForFrame() ?? "single";
	}

	public async captureReflectionProbeFace(
		request: WebGPUReflectionProbeCaptureFaceRequest,
	): Promise<Float32Array | null> {
		if (!this._reflectionProbeCapturePass) {
			return null;
		}
		return this._reflectionProbeCapturePass.captureFace(request);
	}

	public destroy(): void {
		this._destroyRequested = true;
		this._deviceRecoveryNonce++;
		this._deviceRecoveryPromise = null;
		if (!this._deviceLost && this.queue) {
			this._submitPendingCopyCommands();
		}
		this._rollbackInitializationState();
		this._pendingPostProcessPasses.clear();
		this._deviceLost = false;
		this._deviceLostInfo = null;
		this._deviceLossPromise = null;
	}

	public createBuffer(desc: BufferDesc): IRenderBuffer {
		this._assertDeviceOperational("create buffers");
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

		const buffer = {
			size: desc.size,
			destroy: () => {},
			unmap: () => {},
			_gpuResource: gpuBuffer,
		} as InternalRenderBuffer;
		buffer.unmap = () => {
			this._tryUnmapBuffer(gpuBuffer);
		};
		buffer.destroy = this._createManagedDestroy(buffer, {
			label: desc.label ?? "WebGPUBuffer",
			dispose: () => {
				this._tryUnmapBuffer(gpuBuffer);
				gpuBuffer.destroy();
			},
		});
		return buffer;
	}

	public createTexture(desc: TextureDesc): IRenderTexture {
		this._assertDeviceOperational("create textures");
		const dimension = (desc.dimension ?? "2d") as GPUTextureDimension;
		const resolvedWidth = this._resolvePositiveInteger(desc.width, 1);
		const resolvedHeight = this._resolvePositiveInteger(desc.height, 1);
		const depthOrArrayLayers = this._resolvePositiveInteger(desc.depthOrArrayLayers ?? 1, 1);
		const requestedSampleCount = Math.max(1, Math.floor(desc.sampleCount ?? 1));
		const sampleCount =
			dimension === "2d"
				? this._resolveSupportedMSAASampleCount(requestedSampleCount, [
						desc.format as GPUTextureFormat,
					])
				: 1;
		const size: GPUExtent3DStrict =
			dimension === "1d"
				? {
						width: resolvedWidth,
					}
				: {
						width: resolvedWidth,
						height: resolvedHeight,
						depthOrArrayLayers,
					};
		const baseDescriptor: GPUTextureDescriptor = {
			size,
			dimension,
			sampleCount,
			format: desc.format as GPUTextureFormat,
			usage: this._mapTextureUsage(desc.usage),
			mipLevelCount: Math.max(1, desc.mipLevelCount ?? 1),
			viewFormats: desc.viewFormats as GPUTextureFormat[] | undefined,
			label: desc.label,
		};
		const gpuTexture = this.device.createTexture(baseDescriptor);
		const webgpuTexture = createWebGPUTexture(gpuTexture);
		const texture: InternalTexture = {
			width: resolvedWidth,
			height: dimension === "1d" ? 1 : resolvedHeight,
			destroy: () => {},
			_gpuResource: gpuTexture,
			_gpuTexture: gpuTexture,
			_gpuView: webgpuTexture.view,
			_webgpuTexture: webgpuTexture,
		};
		texture.destroy = this._createManagedDestroy(texture, {
			label: desc.label ?? "WebGPUTexture",
			dispose: () => gpuTexture.destroy(),
		});
		attachWebGPUTexture(texture, webgpuTexture);
		return texture;
	}

	public createSampler(desc: SamplerDesc): ISampler {
		this._assertDeviceOperational("create samplers");
		const cacheKey = this._getSamplerCacheKey(desc);
		const cached = this._getLruCacheEntry(this._samplerCache, cacheKey);
		if (cached) {
			return this._acquireSamplerHandle(cached);
		}

		const gpuSampler = this.device.createSampler({
			addressModeU: desc.addressModeU as GPUAddressMode | undefined,
			addressModeV: desc.addressModeV as GPUAddressMode | undefined,
			magFilter: desc.magFilter as GPUFilterMode | undefined,
			minFilter: desc.minFilter as GPUFilterMode | undefined,
			mipmapFilter: desc.mipmapFilter as GPUFilterMode | undefined,
			label: desc.label,
		});

		const entry: CachedSamplerEntry = {
			key: cacheKey,
			refCount: 0,
			label: desc.label,
			gpuResource: gpuSampler,
		};
		this._samplerCache.set(cacheKey, entry);
		const sampler = this._acquireSamplerHandle(entry);
		this._trimRefCountedCache(this._samplerCache, WEBGPU_PIPELINE_LAYOUT_CACHE_LIMIT);
		return sampler;
	}

	public async createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule> {
		this._assertDeviceOperational("create shader modules");
		const processed = await this._processShaderSource(desc);
		if (processed.hasErrors) {
			this._reportShaderRuntimeDiagnostics(desc, processed);
		}
		const effectiveSourceMap = processed.sourceMap ?? desc.sourceMap ?? null;
		const effectiveCodeHash = processed.code === desc.code ? desc.codeHash : undefined;
		const effectiveDesc: ShaderModuleDesc = {
			...desc,
			code: processed.code,
			sourceMap: effectiveSourceMap,
			codeHash: effectiveCodeHash,
			directiveFingerprint: processed.directiveFingerprint,
			logCompilationInfo: desc.logCompilationInfo ?? this._warmupLogCompilationInfo,
		};
		const cacheKey = this._getShaderModuleCacheKey(effectiveDesc);
		const cached = this._shaderModuleCache.get(cacheKey);
		if (cached) {
			this._touchCacheEntry(this._shaderModuleCache, cacheKey, cached);
			return this._acquireShaderModuleHandleAndTrim(cached);
		}

		const inFlight = this._shaderModuleInFlight.get(cacheKey);
		if (inFlight) {
			const entry = await inFlight;
			return this._acquireShaderModuleHandleAndTrim(entry);
		}

		const creationPromise = (async () => {
			let lastError: unknown = null;
			for (let attempt = 0; attempt < 2; attempt++) {
				try {
					const gpuModule = this.device.createShaderModule({
						code: effectiveDesc.code,
						label: effectiveDesc.label,
					});
					let compileMessages: ShaderCompilerMessage[] = [];
					if (typeof gpuModule.getCompilationInfo === "function") {
						try {
							const info = await gpuModule.getCompilationInfo();
							compileMessages = normalizeWebGPUCompilationMessages(info.messages);
						} catch (error) {
							Logger.warn(
								`WebGPU shader compilation info unavailable [${effectiveDesc.label ?? "unnamed"}]: ${String(error)}`,
								{ scope: "WebGPUBackend" },
							);
						}
					}
					if (compileMessages.length > 0) {
						const mappedMessages = mapShaderCompilerMessages(
							compileMessages,
							effectiveDesc.code,
							effectiveDesc.sourceMap,
						);
						if (effectiveDesc.logCompilationInfo === true) {
							const label = effectiveDesc.label ?? "unnamed";
							console.group(`WebGPU Shader Compilation Info [${label}]`);
							console.log(formatShaderCompilerMessages(mappedMessages));
							console.groupEnd();
						}
						const hasErrors = mappedMessages.some(
							(message) => message.type === "error",
						);
						if (hasErrors) {
							throw new ShaderCompileError({
								backend: "webgpu",
								language: effectiveDesc.language ?? "wgsl",
								stage: effectiveDesc.stage ?? "unknown",
								label: effectiveDesc.label,
								sourceKind: effectiveDesc.sourceKind ?? "unknown",
								variantKey: effectiveDesc.variantKey,
								materialId: effectiveDesc.materialId,
								code: effectiveDesc.code,
								sourceMap: effectiveDesc.sourceMap,
								messages: compileMessages,
							});
						}
					}

					const entry: CachedShaderModuleEntry = {
						key: cacheKey,
						refCount: 0,
						label: effectiveDesc.label,
						gpuResource: gpuModule,
					};
					this._shaderModuleCache.set(cacheKey, entry);
					return entry;
				} catch (error) {
					lastError = this._createShaderModuleError(error, effectiveDesc);
					if (attempt === 0) {
						continue;
					}
				}
			}
			throw lastError;
		})();
		this._shaderModuleInFlight.set(cacheKey, creationPromise);
		try {
			const entry = await creationPromise;
			return this._acquireShaderModuleHandleAndTrim(entry);
		} catch (error) {
			throw error;
		} finally {
			if (this._shaderModuleInFlight.get(cacheKey) === creationPromise) {
				this._shaderModuleInFlight.delete(cacheKey);
			}
		}
	}

	public createPipeline(desc: PipelineDesc): IRenderPipeline {
		this._assertDeviceOperational("create render pipelines");
		const layout = this._resolveRenderPipelineLayout(desc);
		const cacheKey = this._getRenderPipelineCacheKey(desc, layout);
		const cached = this._getLruCacheEntry(this._renderPipelineCache, cacheKey);
		if (cached) {
			return this._acquireRenderPipelineHandle(cached);
		}

		const gpuPipeline = this._runValidationScope(
			`createRenderPipeline:${desc.label ?? "unnamed"}`,
			() =>
				this.device.createRenderPipeline(
					this._createRenderPipelineDescriptor(desc, layout),
				),
		);

		const entry: CachedRenderPipelineEntry = {
			key: cacheKey,
			refCount: 0,
			label: desc.label,
			gpuResource: gpuPipeline,
		};
		this._renderPipelineCache.set(cacheKey, entry);
		const pipeline = this._acquireRenderPipelineHandle(entry);
		this._trimRefCountedCache(this._renderPipelineCache, WEBGPU_PIPELINE_CACHE_LIMIT);
		return pipeline;
	}

	public createComputePipeline(desc: ComputePipelineDesc): IComputePipeline {
		this._assertDeviceOperational("create compute pipelines");
		const layout = this._resolveComputePipelineLayout(desc);
		const cacheKey = this._getComputePipelineCacheKey(desc, layout);
		const cached = this._getLruCacheEntry(this._computePipelineCache, cacheKey);
		if (cached) {
			return this._acquireComputePipelineHandle(cached);
		}

		const gpuPipeline = this._runValidationScope(
			`createComputePipeline:${desc.label ?? "unnamed"}`,
			() =>
				this.device.createComputePipeline(
					this._createComputePipelineDescriptor(desc, layout),
				),
		);

		const entry: CachedComputePipelineEntry = {
			key: cacheKey,
			refCount: 0,
			label: desc.label,
			gpuResource: gpuPipeline,
		};
		this._computePipelineCache.set(cacheKey, entry);
		const pipeline = this._acquireComputePipelineHandle(entry);
		this._trimRefCountedCache(this._computePipelineCache, WEBGPU_PIPELINE_CACHE_LIMIT);
		return pipeline;
	}

	public createBindingGroup(desc: BindingGroupDesc): IBindingGroup {
		this._assertDeviceOperational("create binding groups");
		const pipeline = desc.pipeline
			? getWebGPUPipeline(desc.pipeline as IRenderPipeline | IComputePipeline)
			: null;
		const layout =
			(desc.layout as GPUBindGroupLayout | undefined) ??
			this._getPipelineBindGroupLayout(
				pipeline as GPURenderPipeline | GPUComputePipeline | null,
				desc.layoutIndex ?? 0,
			);

		if (!layout) {
			throw new Error(
				`WebGPU binding group ${desc.label ?? "(unnamed)"} requires an explicit layout or pipeline`,
			);
		}

		const layoutId = this._getObjectId(layout);
		const signatures = desc.entries.map((entry) =>
			this._getBindingResourceSignature(entry.binding, entry.resource),
		);
		const cacheKey = this._getBindingGroupCacheKey(layoutId, signatures);
		const cached = this._findBindingGroupCacheEntry(cacheKey, layoutId, signatures);
		if (cached) {
			cached.lastUsedFrame = this._frameSerial;
			cached.lastTouchedTick = ++this._bindingGroupTouchTick;
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
				}),
		);

		const group = {
			label: desc.label,
			destroy: () => {},
			_gpuResource: gpuBindGroup,
		} as InternalBindingGroup;
		group.destroy = this._createManagedDestroy(group, {
			label: desc.label ?? "WebGPUBindGroup",
			dispose: () => {
				this._releaseBindingGroupCacheEntry(cacheKey, group as InternalBindingGroup);
			},
		});
		const entry: CachedBindingGroupEntry = {
			hashKey: cacheKey,
			layoutId,
			signatures: signatures.map((signature) => ({ ...signature })),
			group,
			lastUsedFrame: this._frameSerial,
			lastTouchedTick: ++this._bindingGroupTouchTick,
			refCount: 1,
		};
		const bucket = this._bindingGroupCache.get(cacheKey);
		if (bucket) {
			bucket.push(entry);
		} else {
			this._bindingGroupCache.set(cacheKey, [entry]);
		}
		this._bindingGroupCacheEntryCount++;
		this._trimBindingGroupCache();
		return group;
	}

	public createCommandEncoder(): ICommandEncoder {
		this._assertDeviceOperational("create command encoders");
		return new WebGPUCommandEncoder(
			this.device.createCommandEncoder(),
			this,
			this._commandBufferOwnerToken,
		);
	}

	public writeBuffer(buffer: IRenderBuffer, data: BufferSource, offset: number = 0): void {
		this._assertDeviceOperational("write buffers");
		this.queue.writeBuffer(getWebGPUBuffer(buffer), offset, data);
	}

	public writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		desc: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number },
	): void {
		this._assertDeviceOperational("write textures");
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
			size,
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
		copySize: { width: number; height: number; depthOrArrayLayers?: number },
	): void {
		this._assertDeviceOperational("copy textures");
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
			copySize,
		);
		this._copyPendingCount++;
		if (this._copyPendingCount >= WEBGPU_COPY_BATCH_SIZE) {
			this._submitPendingCopyCommands();
			return;
		}
		this._scheduleCopyFlush();
	}

	public submit(commands: ICommandBuffer[]): void {
		this._assertDeviceOperational("submit command buffers");
		const submitted: GPUCommandBuffer[] = [];
		const copyCommandBuffer = this._flushPendingCopyCommandBuffer();
		if (copyCommandBuffer) {
			submitted.push(copyCommandBuffer);
		}
		for (const command of commands) {
			const internal = this._toInternalCommandBuffer(command);
			internal._submitted = true;
			submitted.push(internal._backendCommandBuffer);
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
			this._readTimestampResultsAsync(timestampResolve.queryCount, timestampResolve.pairs);
		}
		this._resetCurrentCanvasTargets();
	}

	public getCanvasColorTexture(): IRenderTexture {
		if (!this.context || !this.canvas) {
			throw new Error("WebGPU not initialized");
		}

		const current = this._getCurrentCanvasTexture();
		const size = resolveGPUTextureExtent(
			current.texture,
			this.canvas.width,
			this.canvas.height,
		);
		const texture: InternalTexture = {
			width: size.width,
			height: size.height,
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
			throw new Error("Depth texture not initialized (possibly zero dimension canvas)");
		}
		return this._depthTexture;
	}

	public createTextureView(
		texture: IRenderTexture,
		desc?: GPUTextureViewDescriptor,
	): GPUTextureView {
		this._assertDeviceOperational("create texture views");
		const gpuTexture = getWebGPUTexture(texture);
		return this._runValidationScope("device.createTextureView", () => {
			if (desc) {
				return gpuTexture.texture.createView(desc);
			}
			return gpuTexture.view;
		});
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

	public setMSAAEnabled(enabled: boolean): void {
		if (enabled !== true) {
			this.setMSAASampleCount(1);
			return;
		}
		const sampleCount =
			this._preferredMSAASampleCount > 1
				? this._preferredMSAASampleCount
				: WEBGPU_EXPLICIT_MSAA_ENABLE_SAMPLE_COUNT;
		this.setMSAASampleCount(sampleCount);
	}

	public setMSAASampleCount(sampleCount: number): void {
		if (!Number.isFinite(sampleCount)) {
			return;
		}
		const requested = Math.max(1, Math.floor(sampleCount));
		this._preferredMSAASampleCount = requested;
		const resolved = this._resolveSupportedMSAASampleCount(requested);
		if (resolved === this._msaaSampleCount) {
			return;
		}
		this._msaaSampleCount = resolved;
		this._onMSAASampleCountChanged();
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

	private _handleDeviceLost(info: GPUDeviceLostInfo): void {
		if (this._deviceLost) {
			return;
		}
		this._deviceLost = true;
		this._deviceLostInfo = info;
		const reason =
			typeof info.reason === "string" && info.reason.length > 0 ? ` (${info.reason})` : "";
		Logger.error(`WebGPU device was lost${reason}: ${info.message}`, {
			scope: "WebGPUBackend",
		});
		this._rollbackInitializationState();
		if (this._destroyRequested || info.reason === "destroyed") {
			return;
		}
		this._scheduleDeviceRecovery(info);
	}

	private _scheduleDeviceRecovery(info: GPUDeviceLostInfo): void {
		if (this._deviceRecoveryPromise || this._destroyRequested) {
			return;
		}
		const canvas = this.canvas;
		if (!canvas) {
			Logger.error("WebGPU device recovery skipped: backend is missing canvas.", {
				scope: "WebGPUBackend",
			});
			return;
		}
		const nonce = ++this._deviceRecoveryNonce;
		this._deviceRecoveryPromise = this._recoverDeviceAfterLoss(canvas, nonce, info).finally(
			() => {
				if (this._deviceRecoveryNonce === nonce) {
					this._deviceRecoveryPromise = null;
				}
			},
		);
	}

	private async _recoverDeviceAfterLoss(
		canvas: HTMLCanvasElement,
		nonce: number,
		info: GPUDeviceLostInfo,
	): Promise<void> {
		let lastError: unknown = null;
		for (let attempt = 1; attempt <= DEVICE_RECOVERY_MAX_ATTEMPTS; attempt++) {
			if (this._destroyRequested || nonce !== this._deviceRecoveryNonce) {
				return;
			}
			try {
				await this.init(canvas);
				if (this._destroyRequested || nonce !== this._deviceRecoveryNonce) {
					this._rollbackInitializationState();
					return;
				}
				Logger.warn(`WebGPU device recovery succeeded on attempt ${attempt}.`, {
					scope: "WebGPUBackend",
				});
				this._deviceLostInfo = null;
				return;
			} catch (error) {
				lastError = error;
				this._reportNonFatalError(`device recovery attempt ${attempt}`, error);
				if (attempt < DEVICE_RECOVERY_MAX_ATTEMPTS) {
					await this._delayMs(DEVICE_RECOVERY_BASE_DELAY_MS * attempt);
				}
			}
		}
		const reason =
			typeof info.reason === "string" && info.reason.length > 0 ? ` (${info.reason})` : "";
		Logger.error(`WebGPU device recovery failed${reason}: ${info.message}`, {
			scope: "WebGPUBackend",
		});
		if (lastError) {
			this._reportNonFatalError("device recovery exhausted", lastError);
		}
	}

	private _rollbackInitializationState(): void {
		this._copyCommandEncoder = null;
		this._copyPendingCount = 0;
		this._copyFlushScheduled = false;
		invalidateWebGPUComputeFacade(this);
		this._reflectionProbeCapturePass?.destroy();
		this._reflectionProbeCapturePass = null;
		this._frameExecutor?.destroy();
		this._frameExecutor = null;
		this._resources?.destroy();
		this._resources = null;
		const particleSimulator = this._particleSimulator as
			| ({ destroy?: () => void } & IParticleSimulator)
			| null;
		particleSimulator?.destroy?.();
		this._particleSimulator = null;
		this._depthTexture?.destroy();
		this._depthTexture = null;
		this._releaseTimestampResources();
		this._resetCurrentCanvasTargets();
		this._errorScopes = null;
		this._samplerCache.clear();
		this._shaderModuleCache.clear();
		this._shaderCodeHashCache.clear();
		this._shaderModuleInFlight.clear();
		this._renderPipelineCache.clear();
		this._computePipelineCache.clear();
		this._bindingGroupCache.clear();
		this._bindingGroupCacheEntryCount = 0;
		this._pipelineBindGroupLayoutCache.clear();
		this._autoRenderPipelineLayoutCache.clear();
		this._autoComputePipelineLayoutCache.clear();
		this._resourceIds = new WeakMap<object, number>();
		this._nextResourceId = 1;
		this._commandBufferOwnerToken = {};
		this._frameSerial = 0;
		this._bindingGroupTouchTick = 0;
		this._executedPasses.clear();
		this._plannedPasses.clear();
		this._plannedPassOrder.clear();
		this._msaaSelectionCache.clear();
		this._preferredMSAASampleCount = this._defaultMSAASampleCount;
		this._msaaSampleCount = 1;
		if (this.context) {
			try {
				this.context.unconfigure();
			} catch (error) {
				this._reportNonFatalError("context unconfigure", error);
			}
			this._context = null;
		}
		if (this.device) {
			try {
				this.device.destroy();
			} catch (error) {
				this._reportNonFatalError("device destroy", error);
			}
		}
		this._deviceLossPromise = null;
		this._device = null;
		this._queue = null;
	}

	private _assertDeviceOperational(
		operation: string,
	): asserts this is this & { device: GPUDevice; queue: GPUQueue } {
		if (this._deviceLost) {
			const reason =
				typeof this._deviceLostInfo?.reason === "string" &&
				this._deviceLostInfo.reason.length > 0
					? ` (${this._deviceLostInfo.reason})`
					: "";
			const message = this._deviceLostInfo?.message ?? "unknown cause";
			throw new Error(`WebGPU device is lost${reason}; cannot ${operation}: ${message}`);
		}
		if (!this.device || !this.queue) {
			throw new Error(`WebGPU backend is not initialized; cannot ${operation}.`);
		}
	}

	private _resolvePositiveInteger(value: number, fallback: number): number {
		if (!Number.isFinite(value)) {
			return Math.max(1, Math.floor(fallback));
		}
		return Math.max(1, Math.floor(value));
	}

	private _getCurrentCanvasTexture(): WebGPUTexture {
		if (!this.context) {
			throw new Error("WebGPU canvas context is not initialized.");
		}

		if (!this._currentCanvasTexture || !this._currentCanvasView) {
			this._currentCanvasTexture = this.context.getCurrentTexture();
			this._currentCanvasView = this._currentCanvasTexture.createView();
		}

		return createWebGPUTexture(this._currentCanvasTexture, this._currentCanvasView);
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

	private _getShaderModuleCacheKey(desc: ShaderModuleDesc): string {
		const directiveFingerprint = desc.directiveFingerprint ?? "none";
		const explicitHash = desc.codeHash;
		if (explicitHash) {
			return `directive:${directiveFingerprint}|codeHash:${explicitHash}`;
		}
		const hash = this._getCachedShaderCodeHash(desc.code);
		return `directive:${directiveFingerprint}|len:${desc.code.length}|hash:${hash}`;
	}

	private _acquireSamplerHandle(entry: CachedSamplerEntry): InternalSampler {
		this._bumpRefCount(entry);
		const sampler = {
			label: entry.label,
			destroy: () => {},
			_gpuResource: entry.gpuResource,
		} as InternalSampler;
		sampler.destroy = this._createManagedDestroy(sampler, {
			label: entry.label ?? "WebGPUSampler",
			dispose: () => {
				this._releaseSamplerCacheEntry(entry.key, entry.gpuResource);
			},
		});
		return sampler;
	}

	private _acquireShaderModuleHandle(entry: CachedShaderModuleEntry): InternalShaderModule {
		this._bumpRefCount(entry);
		const module = {
			label: entry.label,
			destroy: () => {},
			_gpuResource: entry.gpuResource,
		} as InternalShaderModule;
		module.destroy = this._createManagedDestroy(module, {
			label: entry.label ?? "WebGPUShaderModule",
			dispose: () => {
				this._releaseShaderModuleCacheEntry(entry.key, entry.gpuResource);
			},
		});
		return module;
	}

	private _acquireShaderModuleHandleAndTrim(
		entry: CachedShaderModuleEntry,
	): InternalShaderModule {
		const module = this._acquireShaderModuleHandle(entry);
		this._trimRefCountedCache(this._shaderModuleCache, WEBGPU_PIPELINE_CACHE_LIMIT);
		return module;
	}

	private _acquireRenderPipelineHandle(entry: CachedRenderPipelineEntry): InternalRenderPipeline {
		this._bumpRefCount(entry);
		const pipeline = {
			label: entry.label,
			destroy: () => {},
			_gpuResource: entry.gpuResource,
		} as InternalRenderPipeline;
		pipeline.destroy = this._createManagedDestroy(pipeline, {
			label: entry.label ?? "WebGPURenderPipeline",
			dispose: () => {
				this._releasePipelineCacheEntry(
					this._renderPipelineCache,
					entry.key,
					entry.gpuResource,
				);
			},
		});
		return pipeline;
	}

	private _acquireComputePipelineHandle(
		entry: CachedComputePipelineEntry,
	): InternalComputePipeline {
		this._bumpRefCount(entry);
		const pipeline = {
			label: entry.label,
			destroy: () => {},
			_gpuResource: entry.gpuResource,
		} as InternalComputePipeline;
		pipeline.destroy = this._createManagedDestroy(pipeline, {
			label: entry.label ?? "WebGPUComputePipeline",
			dispose: () => {
				this._releasePipelineCacheEntry(
					this._computePipelineCache,
					entry.key,
					entry.gpuResource,
				);
			},
		});
		return pipeline;
	}

	private _releaseSamplerCacheEntry(key: string, sampler: GPUSampler): void {
		const cached = this._samplerCache.get(key);
		if (!cached || cached.gpuResource !== sampler) {
			return;
		}
		cached.refCount = Math.max(0, cached.refCount - 1);
		if (cached.refCount <= 0) {
			this._samplerCache.delete(key);
		}
	}

	private _releaseShaderModuleCacheEntry(key: string, module: GPUShaderModule): void {
		const cached = this._shaderModuleCache.get(key);
		if (!cached || cached.gpuResource !== module) {
			return;
		}
		cached.refCount = Math.max(0, cached.refCount - 1);
		if (cached.refCount <= 0) {
			this._shaderModuleCache.delete(key);
		}
	}

	private _bumpRefCount(entry: { refCount: number }): void {
		entry.refCount = Math.min(65535, entry.refCount + 1);
	}

	private _createManagedDestroy(
		target: object,
		options: {
			label: string;
			dispose: () => void;
		},
	): () => void {
		let disposed = false;
		const token = {};
		if (this._autoDisposeRegistry) {
			this._autoDisposeRegistry.register(target, options.label, token);
		}
		return () => {
			if (disposed) {
				return;
			}
			disposed = true;
			if (this._autoDisposeRegistry) {
				this._autoDisposeRegistry.unregister(token);
			}
			options.dispose();
		};
	}

	private _releasePipelineCacheEntry<TPipeline extends object>(
		cache: Map<string, { refCount: number; gpuResource: TPipeline }>,
		key: string,
		pipeline: TPipeline,
	): void {
		const cached = cache.get(key);
		if (!cached || cached.gpuResource !== pipeline) {
			return;
		}
		cached.refCount = Math.max(0, cached.refCount - 1);
		if (cached.refCount <= 0) {
			cache.delete(key);
		}
	}

	private _resolveRenderPipelineLayout(
		desc: PipelineDesc,
	): GPUPipelineLayout | GPUAutoLayoutMode {
		const explicitLayout = this._resolveExplicitPipelineLayout(desc.layout);
		if (explicitLayout) {
			return explicitLayout;
		}
		return this._resolveAutoRenderPipelineLayout();
	}

	private _resolveComputePipelineLayout(
		desc: ComputePipelineDesc,
	): GPUPipelineLayout | GPUAutoLayoutMode {
		const explicitLayout = this._resolveExplicitPipelineLayout(desc.layout);
		if (explicitLayout) {
			return explicitLayout;
		}
		return this._resolveAutoComputePipelineLayout();
	}

	private _resolveAutoRenderPipelineLayout(): GPUAutoLayoutMode {
		// Keep implicit layout ownership inside the driver/runtime.
		// Some implementations reject creating a new pipeline layout from
		// bind-group layouts returned by a pipeline that used `layout: "auto"`.
		return "auto";
	}

	private _resolveAutoComputePipelineLayout(): GPUAutoLayoutMode {
		// Keep implicit layout ownership inside the driver/runtime.
		// Some implementations reject creating a new pipeline layout from
		// bind-group layouts returned by a pipeline that used `layout: "auto"`.
		return "auto";
	}

	private _resolveExplicitPipelineLayout(
		layout: PipelineDescLayout | ComputePipelineDescLayout,
	): GPUPipelineLayout | null {
		if (layout === null || layout === undefined || layout === "auto") {
			return null;
		}
		return layout as GPUPipelineLayout;
	}

	private _createRenderPipelineDescriptor(
		desc: PipelineDesc,
		layout: GPUPipelineLayout | GPUAutoLayoutMode,
	): GPURenderPipelineDescriptor {
		const probeFormats = this._getRenderPipelineProbeFormats(desc);
		const sampleCount = this._resolveSupportedMSAASampleCount(
			Math.max(1, Math.floor(desc.sampleCount ?? 1)),
			probeFormats,
		);
		return {
			layout,
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
			fragment: desc.fragment
				? {
						module: getWebGPUShaderModule(desc.fragment.module),
						entryPoint: desc.fragment.entryPoint,
						targets: desc.fragment.targets.map((target) => ({
							format: target.format as GPUTextureFormat,
							blend: target.blend,
							writeMask: target.writeMask,
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
			multisample: {
				count: sampleCount,
			},
			label: desc.label,
		};
	}

	private _createComputePipelineDescriptor(
		desc: ComputePipelineDesc,
		layout: GPUPipelineLayout | GPUAutoLayoutMode,
	): GPUComputePipelineDescriptor {
		return {
			layout,
			compute: {
				module: getWebGPUShaderModule(desc.compute.module),
				entryPoint: desc.compute.entryPoint,
			},
			label: desc.label,
		};
	}

	private _getRenderPipelineProbeFormats(desc: PipelineDesc): GPUTextureFormat[] {
		const formats: GPUTextureFormat[] = [];
		if (desc.fragment?.targets) {
			for (const target of desc.fragment.targets) {
				formats.push(target.format as GPUTextureFormat);
			}
		}
		if (desc.depthStencil?.format) {
			formats.push(desc.depthStencil.format as GPUTextureFormat);
		}
		return formats;
	}

	private _getRenderPipelineCacheKey(
		desc: PipelineDesc,
		layout: GPUPipelineLayout | GPUAutoLayoutMode,
	): string {
		const parts: string[] = [];
		parts.push(`layout:${this._getCacheToken(layout)}`);
		parts.push(`vs.module:${this._getCacheToken(desc.vertex.module)}`);
		parts.push(`vs.entry:${desc.vertex.entryPoint}`);

		const vertexBuffers = desc.vertex.buffers ?? [];
		parts.push(`vs.buffers:${vertexBuffers.length}`);
		for (let i = 0; i < vertexBuffers.length; i++) {
			const buffer = vertexBuffers[i];
			parts.push(
				`vsb${i}.stride:${buffer.arrayStride}`,
				`vsb${i}.step:${buffer.stepMode ?? "vertex"}`,
				`vsb${i}.attrs:${buffer.attributes.length}`,
			);
			for (let j = 0; j < buffer.attributes.length; j++) {
				const attribute = buffer.attributes[j];
				parts.push(
					`vsa${i}.${j}.fmt:${attribute.format}`,
					`vsa${i}.${j}.off:${attribute.offset}`,
					`vsa${i}.${j}.loc:${attribute.shaderLocation}`,
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
				parts.push(`fst${i}.write:${this._serializeWriteMask(target.writeMask)}`);
			}
		} else {
			parts.push("fs:none");
		}

		parts.push(`primitive.topology:${desc.primitive?.topology ?? "triangle-list"}`);
		parts.push(`primitive.cull:${desc.primitive?.cullMode ?? "none"}`);
		parts.push(`primitive.front:${desc.primitive?.frontFace ?? "ccw"}`);
		const probeFormats = this._getRenderPipelineProbeFormats(desc);
		const sampleCount = this._resolveSupportedMSAASampleCount(
			Math.max(1, Math.floor(desc.sampleCount ?? 1)),
			probeFormats,
		);
		parts.push(`multisample.count:${sampleCount}`);
		if (desc.depthStencil) {
			parts.push(`depth.format:${desc.depthStencil.format}`);
			parts.push(`depth.write:${desc.depthStencil.depthWriteEnabled ? 1 : 0}`);
			parts.push(`depth.compare:${desc.depthStencil.depthCompare}`);
		} else {
			parts.push("depth:none");
		}
		return parts.join("|");
	}

	private _getComputePipelineCacheKey(
		desc: ComputePipelineDesc,
		layout: GPUPipelineLayout | GPUAutoLayoutMode,
	): string {
		return [
			`layout:${this._getCacheToken(layout)}`,
			`cs.module:${this._getCacheToken(desc.compute.module)}`,
			`cs.entry:${desc.compute.entryPoint}`,
		].join("|");
	}

	private _getBindingGroupCacheKey(
		layoutId: number,
		signatures: BindingResourceSignature[],
	): bigint {
		let hash = HASH64_OFFSET_BASIS;
		hash = this._hash64Combine(hash, layoutId);
		hash = this._hash64Combine(hash, signatures.length);
		for (const signature of signatures) {
			hash = this._hash64Combine(hash, signature.binding);
			hash = this._hash64Combine(hash, signature.kind);
			hash = this._hash64Combine(hash, signature.primaryId);
			hash = this._hash64Combine(hash, signature.secondaryId);
			hash = this._hash64Combine(hash, signature.offset);
			hash = this._hash64Combine(hash, signature.size);
		}
		return hash;
	}

	private _findBindingGroupCacheEntry(
		hashKey: bigint,
		layoutId: number,
		signatures: BindingResourceSignature[],
	): CachedBindingGroupEntry | null {
		const bucket = this._bindingGroupCache.get(hashKey);
		if (!bucket) {
			return null;
		}
		for (const candidate of bucket) {
			if (
				candidate.layoutId === layoutId &&
				this._isBindingSignaturesMatch(candidate.signatures, signatures)
			) {
				return candidate;
			}
		}
		return null;
	}

	private _releaseBindingGroupCacheEntry(hashKey: bigint, group: InternalBindingGroup): void {
		const bucket = this._bindingGroupCache.get(hashKey);
		if (!bucket) {
			return;
		}
		for (let i = 0; i < bucket.length; i++) {
			const candidate = bucket[i];
			if (candidate.group === group) {
				candidate.refCount = Math.max(0, candidate.refCount - 1);
				if (candidate.refCount <= 0) {
					this._removeBindingGroupCacheEntry(hashKey, candidate);
				}
				return;
			}
		}
	}

	private _isBindingSignaturesMatch(
		left: BindingResourceSignature[],
		right: BindingResourceSignature[],
	): boolean {
		if (left.length !== right.length) {
			return false;
		}
		for (let i = 0; i < left.length; i++) {
			const a = left[i];
			const b = right[i];
			if (
				a.binding !== b.binding ||
				a.kind !== b.kind ||
				a.primaryId !== b.primaryId ||
				a.secondaryId !== b.secondaryId ||
				a.offset !== b.offset ||
				a.size !== b.size
			) {
				return false;
			}
		}
		return true;
	}

	private _getBindingResourceSignature(
		binding: number,
		resource: BindingResourceInput,
	): BindingResourceSignature {
		const texture = tryGetWebGPUTexture(resource);
		if (texture) {
			return {
				binding,
				kind: 1,
				primaryId: this._getObjectId(texture.texture),
				secondaryId: this._getObjectId(texture.view),
				offset: 0,
				size: -1,
			};
		}

		const buffer = tryGetWebGPUBuffer(resource);
		if (buffer) {
			return {
				binding,
				kind: 2,
				primaryId: this._getObjectId(buffer),
				secondaryId: 0,
				offset: 0,
				size: -1,
			};
		}

		if (resource && typeof resource === "object") {
			const asBufferBinding = resource as GPUBufferBinding;
			if (
				asBufferBinding.buffer &&
				typeof (asBufferBinding.buffer as GPUBuffer).destroy === "function"
			) {
				return {
					binding,
					kind: 3,
					primaryId: this._getObjectId(asBufferBinding.buffer),
					secondaryId: 0,
					offset: Math.max(0, asBufferBinding.offset ?? 0),
					size: Math.max(-1, asBufferBinding.size ?? -1),
				};
			}

			if (typeof (resource as GPUTexture).createView === "function") {
				return {
					binding,
					kind: 4,
					primaryId: this._getObjectId(resource),
					secondaryId: 0,
					offset: 0,
					size: -1,
				};
			}

			const resourceWithHandle = resource as {
				_gpuResource?: unknown;
			};
			if (
				resourceWithHandle._gpuResource &&
				typeof resourceWithHandle._gpuResource === "object"
			) {
				return {
					binding,
					kind: 5,
					primaryId: this._getObjectId(resourceWithHandle._gpuResource as object),
					secondaryId: 0,
					offset: 0,
					size: -1,
				};
			}

			return {
				binding,
				kind: 6,
				primaryId: this._getObjectId(resource),
				secondaryId: 0,
				offset: 0,
				size: -1,
			};
		}

		throw new Error(
			`Unsupported binding resource for binding ${binding}: expected object-backed WebGPU resource.`,
		);
	}

	private _getPipelineBindGroupLayout(
		pipeline: GPURenderPipeline | GPUComputePipeline | null,
		layoutIndex: number,
	): GPUBindGroupLayout | undefined {
		if (!pipeline) {
			return undefined;
		}
		const cacheKey = `${this._getCacheToken(pipeline)}:${layoutIndex}`;
		const cached = this._getLruCacheEntry(this._pipelineBindGroupLayoutCache, cacheKey);
		if (cached) {
			return cached;
		}
		const layout = pipeline.getBindGroupLayout(layoutIndex);
		this._pipelineBindGroupLayoutCache.set(cacheKey, layout);
		this._trimCache(this._pipelineBindGroupLayoutCache, WEBGPU_PIPELINE_LAYOUT_CACHE_LIMIT);
		return layout;
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

	private _serializeWriteMask(writeMask: unknown): string {
		if (typeof writeMask !== "number" || !Number.isFinite(writeMask)) {
			return "all";
		}
		return String(Math.max(0, Math.floor(writeMask)));
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
			const backendHandle = value as { _gpuResource?: unknown };
			if (backendHandle._gpuResource && typeof backendHandle._gpuResource === "object") {
				return `obj:${this._getObjectId(backendHandle._gpuResource as object)}`;
			}
			return `obj:${this._getObjectId(value as object)}`;
		}
		return `${type}:${String(value)}`;
	}

	private _getObjectId(value: object): number {
		let id = this._resourceIds.get(value);
		if (id !== undefined) {
			return id;
		}
		if (this._nextResourceId >= RESOURCE_ID_REBASE_THRESHOLD) {
			this._rebaseObjectIds();
		}
		id = this._nextResourceId++;
		this._resourceIds.set(value, id);
		return id;
	}

	private _rebaseObjectIds(): void {
		this._resourceIds = new WeakMap<object, number>();
		this._nextResourceId = 1;
		this._renderPipelineCache.clear();
		this._computePipelineCache.clear();
		this._pipelineBindGroupLayoutCache.clear();
		this._bindingGroupCache.clear();
		this._bindingGroupCacheEntryCount = 0;
		this._msaaSelectionCache.clear();
		Logger.warn(
			"WebGPU object-id space rebased; related caches were cleared to avoid unbounded growth.",
			{ scope: "WebGPUBackend" },
		);
	}

	private _getCachedShaderCodeHash(code: string): string {
		const cached = this._getLruCacheEntry(this._shaderCodeHashCache, code);
		if (cached !== undefined) {
			return cached;
		}
		const hash = this._hashString64(code).toString(16);
		this._shaderCodeHashCache.set(code, hash);
		this._trimCache(this._shaderCodeHashCache, SHADER_CODE_HASH_CACHE_LIMIT);
		return hash;
	}

	private _hashString64(value: string): bigint {
		let hash = HASH64_OFFSET_BASIS;
		for (let i = 0; i < value.length; i++) {
			hash = this._hash64Combine(hash, value.charCodeAt(i));
		}
		return hash;
	}

	private _hash64Combine(hash: bigint, value: number): bigint {
		const normalized = BigInt(value >>> 0);
		return ((hash ^ normalized) * HASH64_PRIME) & HASH64_MASK;
	}

	private _touchCacheEntry<K, T>(cache: Map<K, T>, key: K, value: T): void {
		cache.delete(key);
		cache.set(key, value);
	}

	private _getLruCacheEntry<K, T>(cache: Map<K, T>, key: K): T | undefined {
		const cached = cache.get(key);
		if (cached === undefined && !cache.has(key)) {
			return undefined;
		}
		cache.delete(key);
		cache.set(key, cached as T);
		return cached as T;
	}

	private _trimCache<K, T>(cache: Map<K, T>, maxSize: number): void {
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
		const entryCount = this._getBindingGroupCacheEntryCount();
		if (entryCount <= WEBGPU_BINDING_GROUP_CACHE_LIMIT) {
			return;
		}

		const candidates: Array<{
			hashKey: bigint;
			entry: CachedBindingGroupEntry;
		}> = [];
		for (const [hashKey, bucket] of this._bindingGroupCache.entries()) {
			for (const entry of bucket) {
				candidates.push({
					hashKey,
					entry,
				});
			}
		}
		candidates.sort((a, b) => {
			const frameDelta = a.entry.lastUsedFrame - b.entry.lastUsedFrame;
			if (frameDelta !== 0) {
				return frameDelta;
			}
			const touchDelta = a.entry.lastTouchedTick - b.entry.lastTouchedTick;
			if (touchDelta !== 0) {
				return touchDelta;
			}
			return a.entry.refCount - b.entry.refCount;
		});

		let remainingToEvict = entryCount - WEBGPU_BINDING_GROUP_CACHE_LIMIT;
		for (const candidate of candidates) {
			if (remainingToEvict <= 0) {
				break;
			}
			if (this._removeBindingGroupCacheEntry(candidate.hashKey, candidate.entry)) {
				remainingToEvict--;
			}
		}
	}

	private _trimRefCountedCache<
		K,
		T extends {
			refCount: number;
		},
	>(cache: Map<K, T>, maxSize: number): void {
		if (cache.size <= maxSize) {
			return;
		}
		const toEvict = cache.size - maxSize;
		let evicted = 0;
		for (const [key, entry] of cache.entries()) {
			if (entry.refCount > 0) {
				continue;
			}
			cache.delete(key);
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
		for (const [hashKey, bucket] of this._bindingGroupCache.entries()) {
			for (let i = bucket.length - 1; i >= 0; i--) {
				const entry = bucket[i];
				const frameAge = Math.max(0, this._frameSerial - entry.lastUsedFrame);
				const ttlBudget = WEBGPU_BINDING_GROUP_CACHE_TTL_FRAMES;
				if (frameAge > ttlBudget) {
					this._removeBindingGroupCacheEntry(hashKey, entry);
				}
			}
		}
	}

	private _getBindingGroupCacheEntryCount(): number {
		return this._bindingGroupCacheEntryCount;
	}

	private _removeBindingGroupCacheEntry(
		hashKey: bigint,
		target: CachedBindingGroupEntry,
	): boolean {
		const bucket = this._bindingGroupCache.get(hashKey);
		if (!bucket) {
			return false;
		}
		const index = bucket.indexOf(target);
		if (index < 0) {
			return false;
		}
		bucket.splice(index, 1);
		this._bindingGroupCacheEntryCount = Math.max(0, this._bindingGroupCacheEntryCount - 1);
		if (bucket.length <= 0) {
			this._bindingGroupCache.delete(hashKey);
		}
		return true;
	}

	private _prepareFramePassPlan(context: FrameContext): void {
		this._plannedPasses.clear();
		this._plannedPassOrder.clear();

		const hasParticleSystems = (context.scene.particleSystems?.length ?? 0) > 0;
		if (hasParticleSystems) {
			this._plannedPasses.add("particle-sim");
		}
		if (
			context.features.enableShadows &&
			(context.scene.shadowCasterPackets.length ||
				hasParticleShadowCasters(context.scene.particleSystems))
		) {
			this._plannedPasses.add("shadow");
		}
		if (context.features.enableReflection && context.scene.reflectivePackets.length) {
			this._plannedPasses.add("reflection");
		}
		this._plannedPasses.add("main-opaque");
		if (context.scene.transparentPackets.length > 0) {
			this._plannedPasses.add("main-transparent");
		}
		if (hasParticleSystems) {
			this._plannedPasses.add("particles");
		}
		if (context.features.enableSSAO) {
			this._plannedPasses.add("ssao");
		}
		if (context.features.enableSSGI) {
			this._plannedPasses.add("ssgi");
		}
		if (context.features.enableTAA) {
			this._plannedPasses.add("taa");
		}
		if (context.features.enableSSR) {
			this._plannedPasses.add("ssr");
		}
		if (context.features.enableVolumetric) {
			this._plannedPasses.add("volumetric");
		}
		if (isFogPostProcessEnabled(context.features)) {
			this._plannedPasses.add("fog");
		}
		if (context.features.enableMotionBlur) {
			this._plannedPasses.add("motion-blur");
		}
		if (context.features.enableDOF) {
			this._plannedPasses.add("dof");
		}
		if (context.features.enableBloom) {
			this._plannedPasses.add("bloom");
		}
		if (context.features.enableToneMapping !== false) {
			this._plannedPasses.add("tonemap");
		}
		if (context.features.enableColorFilter) {
			this._plannedPasses.add("color-filter");
		}
		if (context.features.enableFXAA) {
			this._plannedPasses.add("fxaa");
		}
		const interaction = context.transient.get(INTERACTION_TRANSIENT_STATE_KEY);
		if ((interaction?.selectedEntityIds?.length ?? 0) > 0) {
			this._plannedPasses.add("interaction-outline");
		}
		if (context.features.enableGamma) {
			this._plannedPasses.add("gamma");
		}
		this._validatePlannedPassGraph();
	}

	private _validatePassDependencies(pass: FramePass): void {
		if (this._plannedPasses.size > 0 && !this._plannedPasses.has(pass.stage)) {
			return;
		}
		const plannedIndex = this._plannedPassOrder.get(pass.stage);
		if (plannedIndex !== undefined) {
			const violated = Array.from(this._executedPasses).some((executedStage) => {
				const index = this._plannedPassOrder.get(executedStage);
				return index !== undefined && index > plannedIndex;
			});
			if (violated) {
				throw new Error(
					`WebGPU pass "${pass.stage}" execution order violates prevalidated pass plan.`,
				);
			}
		}
		const dependencies = this._resolvePassDependencies(pass.stage);
		if (!dependencies || dependencies.length <= 0) {
			return;
		}
		const missing = dependencies.filter(
			(dependency) =>
				this._plannedPasses.has(dependency) &&
				this._isDependencyApplicable(pass.stage, dependency) &&
				!this._executedPasses.has(dependency),
		);
		if (missing.length <= 0) {
			return;
		}
		throw new Error(
			`WebGPU pass "${pass.stage}" executed before dependencies: ${missing.join(", ")}`,
		);
	}

	private _validatePlannedPassGraph(): void {
		const visiting = new Set<FramePass["stage"]>();
		const visited = new Set<FramePass["stage"]>();
		const order: FramePass["stage"][] = [];

		const visit = (stage: FramePass["stage"]): void => {
			if (visited.has(stage)) {
				return;
			}
			if (visiting.has(stage)) {
				throw new Error(
					`WebGPU pass plan cycle detected at "${stage}" during _prepareFramePassPlan.`,
				);
			}
			visiting.add(stage);
			const dependencies = this._resolvePassDependencies(stage);
			for (const dependency of dependencies) {
				if (!this._plannedPasses.has(dependency)) {
					continue;
				}
				visit(dependency);
			}
			visiting.delete(stage);
			visited.add(stage);
			order.push(stage);
		};

		for (const stage of this._plannedPasses) {
			visit(stage);
		}
		for (let i = 0; i < order.length; i++) {
			this._plannedPassOrder.set(order[i], i);
		}
	}

	private _resolvePassDependencies(stage: FramePass["stage"]): FramePass["stage"][] {
		const dependencies = FRAME_PASS_DEPENDENCIES.get(stage);
		return dependencies ? Array.from(dependencies) : [];
	}

	private _isDependencyApplicable(
		stage: FramePass["stage"],
		dependency: FramePass["stage"],
	): boolean {
		const stageIndex = this._plannedPassOrder.get(stage);
		const dependencyIndex = this._plannedPassOrder.get(dependency);
		if (stageIndex === undefined || dependencyIndex === undefined) {
			return true;
		}
		if (dependencyIndex < stageIndex) {
			return true;
		}
		this._reportNonFatalError(
			"pass dependency order",
			`Ignoring stale dependency "${dependency}" for "${stage}".`,
		);
		return false;
	}

	private _markPassExecuted(stage: FramePass["stage"]): void {
		this._executedPasses.add(stage);
	}

	private _getCopyCommandEncoder(): GPUCommandEncoder {
		this._assertDeviceOperational("encode copy commands");
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
		if (!this.queue) {
			return;
		}
		this._runValidationScope("queue.submit.copyBatch", () => {
			this.queue.submit([commandBuffer]);
		});
	}

	private _scheduleCopyFlush(): void {
		if (!this._isFrameActive()) {
			this._submitPendingCopyCommands();
			return;
		}
		if (this._copyFlushScheduled) {
			return;
		}
		this._copyFlushScheduled = true;
		const scheduleMicrotask =
			typeof queueMicrotask === "function"
				? queueMicrotask
				: (callback: () => void) => {
						void Promise.resolve().then(callback);
					};
		scheduleMicrotask(() => {
			if (!this._copyFlushScheduled) {
				return;
			}
			this._submitPendingCopyCommands();
		});
	}

	private _isFrameActive(): boolean {
		return this._plannedPasses.size > 0 || this._executedPasses.size > 0;
	}

	private _selectCanvasDepthFormat(): TextureFormat {
		if (!this.device) {
			return TextureFormat.Depth24Plus;
		}
		const candidates: TextureFormat[] = [TextureFormat.Depth24Plus, TextureFormat.Depth32Float];
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
		const preferred = Math.max(1, Math.floor(this._preferredMSAASampleCount));
		return this._resolveSupportedMSAASampleCount(preferred);
	}

	private _onMSAASampleCountChanged(): void {
		this._submitPendingCopyCommands();
		this._renderPipelineCache.clear();
		this._pipelineBindGroupLayoutCache.clear();
		this._bindingGroupCache.clear();
		this._bindingGroupCacheEntryCount = 0;
		this._frameExecutor?.invalidateFrameTargets();
		this._resetCurrentCanvasTargets();
	}

	private _onShaderRuntimeChanged(): void {
		this._submitPendingCopyCommands();
		this._invalidateShaderDependentCaches();
		this._frameExecutor?.onShaderRuntimeChanged?.();
		this._resources?.onShaderRuntimeChanged?.();
		this._resetCurrentCanvasTargets();
	}

	private _invalidateShaderDependentCaches(): void {
		this._shaderModuleCache.clear();
		this._shaderCodeHashCache.clear();
		this._shaderModuleInFlight.clear();
		this._renderPipelineCache.clear();
		this._computePipelineCache.clear();
		this._bindingGroupCache.clear();
		this._bindingGroupCacheEntryCount = 0;
		this._pipelineBindGroupLayoutCache.clear();
		this._autoRenderPipelineLayoutCache.clear();
		this._autoComputePipelineLayoutCache.clear();
	}

	private _stripUtf8BomCharacters(code: string, label?: string): string {
		if (!code.includes("\uFEFF")) {
			return code;
		}
		const shaderLabel = label && label.length > 0 ? label : "unnamed";
		const key = `webgpu-shader-bom:${shaderLabel}`;
		Logger.warn(
			`[${key}] WebGPU shader source [${shaderLabel}] contained UTF-8 BOM characters; stripping before compilation.`,
			{ scope: "WebGPUBackend", onceKey: key },
		);
		return code.replace(/\uFEFF/g, "");
	}

	private async _processShaderSource(
		desc: ShaderModuleDesc,
	): Promise<ShaderBackendCompileResult> {
		const sanitizedCode = this._stripUtf8BomCharacters(desc.code, desc.label);
		const directiveSourcePath =
			desc.sourceMap?.segments[0]?.sourcePath ?? desc.label ?? "<webgpu-shader>";
		return this._shaderCompileStage.compileAsync({
			code: sanitizedCode,
			language: desc.language ?? "wgsl",
			stage: desc.stage ?? "unknown",
			entryPoint: desc.entryPoint,
			label: desc.label,
			sourceKind: desc.sourceKind ?? "unknown",
			sourceMap: desc.sourceMap ?? null,
			directiveSourcePath,
		});
	}

	private _createShaderModuleError(error: unknown, desc: ShaderModuleDesc): Error {
		if (error instanceof ShaderCompileError) {
			return error;
		}
		const compilerMessage: ShaderCompilerMessage = {
			type: "error",
			message: String(error),
		};
		return new ShaderCompileError({
			backend: "webgpu",
			language: desc.language ?? "wgsl",
			stage: desc.stage ?? "unknown",
			label: desc.label,
			sourceKind: desc.sourceKind ?? "unknown",
			variantKey: desc.variantKey,
			materialId: desc.materialId,
			code: desc.code,
			sourceMap: desc.sourceMap ?? null,
			messages: [compilerMessage],
			cause: error,
		});
	}

	private _reportShaderRuntimeDiagnostics(
		desc: ShaderModuleDesc,
		result: ShaderProcessResult,
	): void {
		const keyPrefix = desc.label && desc.label.length > 0 ? desc.label : "unnamed";
		for (const diagnostic of result.diagnostics) {
			const locationSuffix =
				diagnostic.sourcePath && typeof diagnostic.line === "number"
					? ` (${diagnostic.sourcePath}:${diagnostic.line}:${diagnostic.column ?? 1})`
					: "";
			const key =
				`webgpu-shader-runtime-${diagnostic.severity}` +
				`-${diagnostic.code}-${keyPrefix}` +
				`-${diagnostic.sourcePath ?? ""}-${diagnostic.line ?? ""}-${diagnostic.column ?? ""}`;
			Logger.warn(
				`[${key}] WebGPU shader runtime ${diagnostic.severity} [${keyPrefix}] ` +
					`${diagnostic.code}: ${diagnostic.message}${locationSuffix}`,
				{ scope: "WebGPUBackend", onceKey: key },
			);
		}
	}

	private _resolveSupportedMSAASampleCount(
		requested: number,
		probeFormats?: readonly GPUTextureFormat[],
	): number {
		const normalized = Math.max(1, Math.floor(requested));
		const maxColorAttachments = this.device?.limits?.maxColorAttachments ?? 0;
		const maxColorAttachmentBytesPerSample =
			this.device?.limits?.maxColorAttachmentBytesPerSample ?? 0;
		const formats = this._getMSAAProbeFormats(probeFormats);
		const formatsKey = formats.join(",");
		const cacheKey = [
			`device:${this._getCacheToken(this.device)}`,
			`requested:${normalized}`,
			`maxAttachments:${maxColorAttachments}`,
			`maxBytes:${maxColorAttachmentBytesPerSample}`,
			`formats:${formatsKey}`,
		].join("|");
		const cached = this._msaaSelectionCache.get(cacheKey);
		if (cached !== undefined) {
			return cached;
		}

		const candidates = Array.from(
			new Set([
				normalized,
				...WEBGPU_MSAA_SAMPLE_CANDIDATES.filter((sampleCount) => sampleCount <= normalized),
				1,
			]),
		).sort((left, right) => right - left);

		let selected = 1;
		for (const candidate of candidates) {
			if (this._isMSAASampleCountSupported(candidate, formats)) {
				selected = candidate;
				break;
			}
		}
		this._msaaSelectionCache.set(cacheKey, selected);
		return selected;
	}

	private _isMSAASampleCountSupported(
		sampleCount: number,
		probeFormats?: readonly GPUTextureFormat[],
	): boolean {
		if (!Number.isInteger(sampleCount) || sampleCount < 1) {
			return false;
		}
		if (sampleCount === 1) {
			return true;
		}
		if (!this.device || typeof this.device.createTexture !== "function") {
			return false;
		}
		const maxColorAttachments = this.device?.limits?.maxColorAttachments ?? 0;
		const maxColorAttachmentBytesPerSample =
			this.device?.limits?.maxColorAttachmentBytesPerSample ?? 0;
		if (
			maxColorAttachments < WEBGPU_MRT_COLOR_TARGET_COUNT ||
			maxColorAttachmentBytesPerSample < WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE
		) {
			return false;
		}
		for (const format of this._getMSAAProbeFormats(probeFormats)) {
			if (!this._probeSampleCountForFormat(sampleCount, format)) {
				return false;
			}
		}
		return true;
	}

	private _getMSAAProbeFormats(probeFormats?: readonly GPUTextureFormat[]): GPUTextureFormat[] {
		const formats = new Set<GPUTextureFormat>([
			this.canvasFormat,
			this.canvasDepthFormat as GPUTextureFormat,
			TextureFormat.RGBA16Float as GPUTextureFormat,
			TextureFormat.RGBA8Unorm as GPUTextureFormat,
			TextureFormat.Depth32Float as GPUTextureFormat,
		]);
		if (probeFormats) {
			for (const format of probeFormats) {
				formats.add(format);
			}
		}
		return Array.from(formats);
	}

	private _probeSampleCountForFormat(sampleCount: number, format: GPUTextureFormat): boolean {
		const device = this.device;
		if (!device) {
			return false;
		}
		try {
			const probeTexture = device.createTexture({
				size: [1, 1, 1],
				sampleCount,
				format,
				usage: GPUTextureUsage.RENDER_ATTACHMENT,
				label: `WebGPUMSAAProbe_${format}_${sampleCount}`,
			});
			probeTexture.destroy();
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
		const queue = this.queue;
		if (!queue) {
			return;
		}
		const queueWithTimestamp = queue as GPUQueue & {
			getTimestampPeriod?: () => number;
		};
		this._timestampPeriodNs =
			typeof queueWithTimestamp.getTimestampPeriod === "function"
				? queueWithTimestamp.getTimestampPeriod()
				: 1;
		const device = this.device;
		if (!device || typeof device.createQuerySet !== "function") {
			return;
		}
		if (
			typeof device.features?.has === "function" &&
			!device.features.has("timestamp-query" as GPUFeatureName)
		) {
			return;
		}
		try {
			this._timestampQuerySet = device.createQuerySet({
				type: "timestamp",
				count: WEBGPU_TIMESTAMP_QUERY_CAPACITY,
			});
			this._timestampResolveBuffer = device.createBuffer({
				label: "WebGPUTimestampResolveBuffer",
				size: WEBGPU_TIMESTAMP_QUERY_CAPACITY * BigUint64Array.BYTES_PER_ELEMENT,
				usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
			});
			this._timestampReadBuffer = device.createBuffer({
				label: "WebGPUTimestampReadBuffer",
				size: WEBGPU_TIMESTAMP_QUERY_CAPACITY * BigUint64Array.BYTES_PER_ELEMENT,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			});
			this._timestampSupported = true;
		} catch (error) {
			this._reportNonFatalError("init timestamp resources", error);
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
		if (!this.device) {
			return undefined;
		}
		const resolveEncoder = this.device.createCommandEncoder({
			label: "WebGPUTimestampResolveEncoder",
		});
		resolveEncoder.resolveQuerySet(
			this._timestampQuerySet,
			0,
			queryCount,
			this._timestampResolveBuffer,
			0,
		);
		resolveEncoder.copyBufferToBuffer(
			this._timestampResolveBuffer,
			0,
			this._timestampReadBuffer,
			0,
			queryCount * BigUint64Array.BYTES_PER_ELEMENT,
		);
		this._timestampQueryCursor = 0;
		this._timestampPairs = [];
		return {
			commandBuffer: resolveEncoder.finish(),
			queryCount,
			pairs,
		};
	}

	private _readTimestampResultsAsync(queryCount: number, pairs: TimestampPairEntry[]): void {
		if (this._timestampReadPending || !this._timestampReadBuffer || queryCount <= 0) {
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
					const durationMs = (Number(deltaTicks) * this._timestampPeriodNs) / 1_000_000;
					result.set(`${pair.label}#${i}`, durationMs);
				}
				this._timestampResults = result;
				this._timestampReadBuffer.unmap();
			})
			.catch((error) => {
				Logger.warn(`WebGPU timestamp readback failed: ${String(error)}`, {
					scope: "WebGPUBackend",
				});
				if (this._timestampReadBuffer) {
					try {
						this._timestampReadBuffer.unmap();
					} catch (unmapError) {
						this._reportNonFatalError("timestamp readback unmap", unmapError);
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
			} catch (error) {
				this._reportNonFatalError("timestamp read buffer destroy", error);
			}
			this._timestampReadBuffer = null;
		}
		if (this._timestampResolveBuffer) {
			try {
				this._timestampResolveBuffer.destroy();
			} catch (error) {
				this._reportNonFatalError("timestamp resolve buffer destroy", error);
			}
			this._timestampResolveBuffer = null;
		}
		if (this._timestampQuerySet) {
			try {
				this._timestampQuerySet.destroy();
			} catch (error) {
				this._reportNonFatalError("timestamp query set destroy", error);
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
		const state = (buffer as GPUBuffer & { mapState?: GPUBufferMapState }).mapState;
		return (state ?? "unmapped") !== "unmapped";
	}

	private _tryUnmapBuffer(buffer: GPUBuffer | null): void {
		if (!this._isBufferMapped(buffer) || !buffer) {
			return;
		}
		try {
			buffer.unmap();
		} catch (error) {
			this._reportNonFatalError("buffer.unmap", error);
		}
	}

	private _delayMs(milliseconds: number): Promise<void> {
		return new Promise((resolve) => {
			setTimeout(resolve, Math.max(0, Math.floor(milliseconds)));
		});
	}

	private _reportNonFatalError(scope: string, error: unknown): void {
		Logger.warn(`WebGPU backend ${scope} failed: ${String(error)}`, {
			scope: "WebGPUBackend",
		});
	}

	private _toUint8View(data: BufferSource): Uint8Array {
		if (data instanceof ArrayBuffer) {
			return new Uint8Array(data);
		}
		return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
	}

	private _configureContext(): void {
		if (!this.context || !this.canvas || !this.device) {
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

	private _mapBindingResource(resource: BindingResourceInput): GPUBindingResource {
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

		if (resource && typeof resource === "object") {
			const bufferBinding = resource as GPUBufferBinding;
			if (
				bufferBinding.buffer &&
				typeof (bufferBinding.buffer as GPUBuffer).destroy === "function"
			) {
				return bufferBinding;
			}
		}

		if (resource && typeof resource === "object") {
			const resourceWithHandle = resource as { _gpuResource?: unknown };
			const handle = resourceWithHandle._gpuResource;
			if (handle) {
				if (typeof (handle as GPUTexture).createView === "function") {
					return (handle as GPUTexture).createView();
				}
				if (typeof (handle as GPUBuffer).destroy === "function") {
					return { buffer: handle as GPUBuffer };
				}
				return handle as GPUBindingResource;
			}
			return resource as GPUBindingResource;
		}

		throw new Error(
			"Unsupported WebGPU binding resource: expected texture, buffer, sampler, or GPU-backed resource object.",
		);
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
		if (usage & BufferUsage.Indirect) {
			flags |= GPUBufferUsage.INDIRECT;
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

	private _toInternalCommandBuffer(command: ICommandBuffer): InternalCommandBuffer {
		const internal = command as
			| (Partial<InternalCommandBuffer> & {
					_backendCommandBuffer?: unknown;
					_gpuCommandBuffer?: unknown;
			  })
			| null;
		if (!internal || typeof internal !== "object") {
			throw new Error("Invalid command buffer for WebGPU submit().");
		}
		if (!internal._backendCommandBuffer && !internal._gpuCommandBuffer) {
			throw new Error("Invalid command buffer for WebGPU submit().");
		}
		if (!internal._backendCommandBuffer && internal._gpuCommandBuffer) {
			internal._backendCommandBuffer = internal._gpuCommandBuffer as GPUCommandBuffer;
		}
		if (internal._ownerToken !== this._commandBufferOwnerToken) {
			throw new Error("Command buffer does not belong to this WebGPU backend instance.");
		}
		if (internal._submitted) {
			throw new Error("WebGPU command buffer has already been submitted.");
		}
		return internal as InternalCommandBuffer;
	}

	private _resolveParticleDeltaTime(context: FrameContext): number {
		const value = context.transient.get(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY);
		if (typeof value !== "number" || !Number.isFinite(value)) {
			return 0;
		}
		return Math.max(0, value);
	}

	private _createPassHandlers(): Map<FramePass["stage"], WebGPUPassHandler> {
		return new Map<FramePass["stage"], WebGPUPassHandler>([
			["animation-sim", () => {}],
			[
				"particle-sim",
				(_pass, context) => {
					this._particleSimulator?.simulate(
						context,
						this._resolveParticleDeltaTime(context),
					);
					this._particleSimulator?.emitRenderBatches(context);
					this._resources?.updateParticleShadowVolumes?.(context);
				},
			],
		]);
	}
}
