/// <reference types="@webgpu/types" />
import {
	type ICommandBuffer,
	type ICommandEncoder,
} from "./ICommandEncoder";
import type {
	IRenderBackend,
	BackendCapabilities,
	RenderBackendDebugInfo,
	RenderBackendDeviceLostInfo,
	RenderBackendAttachContext,
	RenderBackendCompletedFrameCoverage,
	RenderBackendProfile,
	RenderSurfaceSize,
	WarmupOptions,
	WarmupReport,
} from "./IRenderBackend";
import type { RenderTargetReadbackOptions } from "./CustomRenderTargets";
import type { TextureReadbackResult } from "./IComputeRuntime";
import { type FrameAttachments, type FrameContext, type FramePass } from "../pipeline/types";
import type {
	NormalizedOcclusionCullingOptions,
	OcclusionCullingBackendAdapter,
} from "../pipeline/OcclusionCulling";
import {
	createRenderBackendExtensionRegistry,
	PROBE_CAPTURE_EXTENSION,
	RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
	RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT,
	WEBGPU_COMPUTE_EXTENSION,
	WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT,
} from "./BackendExtensions";
import { WebGPUErrorScopeHelper } from "./webgpu/WebGPUErrorScopeHelper";
import { WebGPUFrameExecutor } from "./webgpu/WebGPUFrameExecutor";
import type { WebGPUFrameHost } from "./webgpu/rendergraph/WebGPUFrameHost";
import { WebGPUPostProcessExecutor } from "./webgpu/WebGPUPostProcessExecutor";
import type { WebGPUPostProcessSessionPort } from "./webgpu/WebGPUPostProcessExecutor";
import { BackendPostProcessRuntime } from "../postprocess/BackendPostProcessRuntime";
import { WebGPUCommandScheduler } from "./webgpu/WebGPUCommandScheduler";
import { WebGPUCanvasTargetManager } from "./webgpu/WebGPUCanvasTargetManager";
import { WebGPUResourceManager } from "./webgpu/WebGPUResourceManager";
import { WebGPUShaderModuleCompiler } from "./webgpu/WebGPUShaderModuleCompiler";
import {
	WebGPUPipelineCache,
	type WebGPUPipelineCacheHost,
} from "./webgpu/WebGPUPipelineCache";
import {
	WebGPUBindingGroupCache,
	type WebGPUBindingGroupCacheHost,
} from "./webgpu/WebGPUBindingGroupCache";
import { WebGPUObjectIdentity } from "./webgpu/WebGPUObjectIdentity";
import {
	WebGPUMSAAController,
	type WebGPUMSAAControllerHost,
} from "./webgpu/WebGPUMSAAController";
import { WebGPUBackendPassDispatcher } from "./webgpu/WebGPUBackendPassDispatcher";
import { WebGPUWarmupCoordinator } from "./webgpu/WebGPUWarmupCoordinator";
import {
	WebGPUReflectionProbeCapturePass,
} from "./webgpu/WebGPUReflectionProbeCapturePass";
import type { ProbeWebGPUCaptureFaceRequest } from "../lights/runtime/ProbeCaptureRuntime";
import { WebGPURenderResources } from "./webgpu/WebGPURenderResources";
import type { WebGPUCommandSchedulerHost } from "./webgpu/WebGPUBackendContracts";
import {
	FramePassPlanValidator,
	type FramePassPlanValidatorState,
} from "../pipeline/FramePassPlanValidator";
import type { IParticleSimulator } from "../simulation/particles/IParticleSimulator";
import { WebGPUParticleSimulator } from "../simulation/particles/WebGPUParticleSimulator";
import {
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_TARGET_COUNT,
	WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT,
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
	WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLER_COUNT,
	WEBGPU_REQUIRED_FRAGMENT_STORAGE_BUFFER_COUNT,
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
	ShaderBackendCompileStage,
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	ShaderRuntime,
} from "../shaders/runtime";
import type { ShaderDirectiveCompileHook, ShaderRuntimeMode } from "../shaders/runtime";
import { ShaderSource } from "../shaders/ShaderSource";
import type { Texture } from "../core/Texture";
import {
	createWebGPUComputeFacade,
	invalidateWebGPUComputeFacade,
	type IWebGPUComputeFacade,
} from "./webgpu/ComputeFacade";
import { Logger } from "../foundation/Logger";

const DEVICE_RECOVERY_MAX_ATTEMPTS = 3;
const DEVICE_RECOVERY_BASE_DELAY_MS = 100;
const WEBGPU_DEBUG_INFO_UNINITIALIZED: RenderBackendDebugInfo = {
	backend: "webgpu",
	api: "webgpu",
	available: false,
	unavailableReason: "WebGPU backend has not been initialized.",
};
const WEBGPU_DEBUG_LIMIT_KEYS = [
	"maxTextureDimension2D",
	"maxTextureArrayLayers",
	"maxBindGroups",
	"maxBindingsPerBindGroup",
	"maxBufferSize",
	"maxStorageBufferBindingSize",
	"maxUniformBufferBindingSize",
	"maxSampledTexturesPerShaderStage",
	"maxSamplersPerShaderStage",
	"maxStorageBuffersPerShaderStage",
	"maxStorageTexturesPerShaderStage",
	"maxColorAttachments",
	"maxColorAttachmentBytesPerSample",
] as const;

export interface WebGPUBackendOptions {
	shaderMode?: ShaderRuntimeMode;
	directiveHook?: ShaderDirectiveCompileHook | null;
	msaaSampleCount?: number;
	enableEarlyZPrepass?: boolean;
	/**
	 * Enables WebGPU deferred opaque lighting when runtime limits allow it.
	 * Defaults to `true`; set to `false` to force the legacy MRT forward path.
	 */
	enableDeferredLighting?: boolean;
	/**
	 * Controls WebGPU internal frame graph validation diagnostics.
	 * Defaults to `"throw"` so tests and development builds fail on invalid
	 * internal resource declarations.
	 */
	frameGraphValidation?: "throw" | "warn";
	/**
	 * Enables WebGPU previous-frame Hi-Z occlusion culling support.
	 * Defaults to `true`; renderer features still opt in per frame.
	 */
	enableOcclusionCulling?: boolean;
}

export class WebGPUBackend implements IRenderBackend {
	private _postProcessExecutor: WebGPUPostProcessExecutor | null = null;
	private _postProcessRuntime: BackendPostProcessRuntime | null = null;
	private _postProcessSessionPort: WebGPUPostProcessSessionPort | null = null;
	public get postProcessRuntime(): BackendPostProcessRuntime {
		if (!this._postProcessRuntime) {
			throw new Error("WebGPU post-process runtime is not initialized.");
		}
		return this._postProcessRuntime;
	}
	private readonly _occlusionCullingExtensionApi: OcclusionCullingBackendAdapter = {
		getVisibilityProvider: (options: NormalizedOcclusionCullingOptions) =>
			this._frameExecutor?.getOcclusionVisibilityProvider(options) ?? null,
		resetOcclusionCulling: () => {
			this._frameExecutor?.resetOcclusionCulling();
		},
	};
	public readonly extensions;
	public readonly profile: RenderBackendProfile;

	private _attachContext: RenderBackendAttachContext | null = null;
	private _attached = false;
	private readonly _options: WebGPUBackendOptions;
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

	public canvasFormat: TextureFormat = TextureFormat.BGRA8Unorm;
	public canvasDepthFormat: TextureFormat = TextureFormat.Depth24Plus;
	public readonly shaderRuntime: ShaderRuntime;

	private readonly _canvasTargets = new WebGPUCanvasTargetManager();
	private _errorScopes: WebGPUErrorScopeHelper | null = null;
	private _resources: WebGPURenderResources | null = null;
	private _frameExecutor: WebGPUFrameExecutor | null = null;
	private _frameHost: WebGPUFrameHost | null = null;
	private _reflectionProbeCapturePass: WebGPUReflectionProbeCapturePass | null = null;
	private _particleSimulator: IParticleSimulator | null = null;
	private _deviceLost = false;
	private _deviceLostInfo: RenderBackendDeviceLostInfo | null = null;
	private _deviceLossPromise: Promise<GPUDeviceLostInfo> | null = null;
	private readonly _objectIdentity = new WebGPUObjectIdentity(() => {
		this._handleObjectIdentityRebase();
	});
	private _destroyRequested = false;
	private _deviceRecoveryNonce = 0;
	private _deviceRecoveryPromise: Promise<void> | null = null;
	private _frameSerial = 0;
	private _executedPasses = new Set<FramePass["stage"]>();
	private _plannedPasses = new Set<FramePass["stage"]>();
	private _plannedPassOrder = new Map<FramePass["stage"], number>();
	private _frameActive = false;
	private _pendingResize: {
		width: number;
		height: number;
	} | null = null;
	private _pendingShaderRuntimeInvalidation = false;
	private _debugInfo: RenderBackendDebugInfo = WEBGPU_DEBUG_INFO_UNINITIALIZED;

	private readonly _autoDisposeRegistry: FinalizationRegistry<string> | null =
		typeof FinalizationRegistry === "function"
			? new FinalizationRegistry<string>((label) => {
					const key = `webgpu-resource-gc:${label}`;
					Logger.warn(
						`[${key}] WebGPU resource "${label}" was garbage collected without explicit destroy().`,
						{ scope: "WebGPUBackend", onceKey: key },
					);
				})
			: null;
	private _warmupLogCompilationInfo = false;
	private readonly _msaaController: WebGPUMSAAController;
	private _enableEarlyZPrepass = true;
	private _enableDeferredLighting = true;
	private _enableOcclusionCulling = true;
	private _frameGraphValidationMode: "throw" | "warn" = "throw";
	private _completedFrameCoverage: RenderBackendCompletedFrameCoverage = "full-frame";
	private _shaderCompileStage: ShaderBackendCompileStage;
	private readonly _shaderModuleCompiler: WebGPUShaderModuleCompiler;
	private readonly _framePlanner = new FramePassPlanValidator("WebGPU");
	private readonly _commandScheduler: WebGPUCommandScheduler;
	private readonly _resourceManager: WebGPUResourceManager;
	private readonly _pipelineCache: WebGPUPipelineCache;
	private readonly _bindingGroupCache: WebGPUBindingGroupCache;
	private readonly _passDispatcher: WebGPUBackendPassDispatcher;
	private readonly _warmupCoordinator: WebGPUWarmupCoordinator;

	public constructor(options: WebGPUBackendOptions = {}) {
		this._options = options;
		if (Object.prototype.hasOwnProperty.call(options, "enableMSAA")) {
			throw new Error(
				"WebGPUBackendOptions.enableMSAA was removed; use msaaSampleCount: 1 to disable MSAA or msaaSampleCount: 4 to request 4x MSAA."
			);
		}
		const shaderMode = options.shaderMode ?? "strict";
		const thisRef = this;
		this._msaaController = new WebGPUMSAAController(
			this._createMSAAControllerHost(),
			options.msaaSampleCount
		);
		this._enableEarlyZPrepass = options.enableEarlyZPrepass !== false;
		this._enableDeferredLighting = options.enableDeferredLighting !== false;
		this._enableOcclusionCulling = options.enableOcclusionCulling !== false;
		this._frameGraphValidationMode = options.frameGraphValidation === "warn" ? "warn" : "throw";
		const capabilities: BackendCapabilities = {
			sh: true,
			shadows: true,
			reflection: true,
			environment: true,
			postProcess: true,
			clusteredLighting: true,
			oit: true,
			occlusionCulling: this._enableOcclusionCulling,
			customRenderTargets: true,
			customRenderPasses: true,
			renderTargetReadback: true,
		};
		this.profile = {
			id: "webgpu",
			capabilities,
			frameScheduling: "on-demand",
			shadow: {
				backendKey: "webgpu",
				supportsFilterModes: ["pcf", "vsm"],
				supportsDirectionalCSM: true,
				supportsSpotCSM: false,
				supportsPointCSM: false,
				maxDynamicShadowCost: 48,
				supportsPagedShadows: true,
				supportsPagedShadowRendering: true,
				maxPagedShadowPages: 2048,
				pagedShadowPageSizeRange: [64, 256],
			},
			lighting: { localizedProbeMode: "backend-local" },
		};
		this.shaderRuntime = new ShaderRuntime({
			mode: shaderMode,
		});
		this._shaderCompileStage = new ShaderBackendCompileStage({
			backend: "webgpu",
			runtime: this.shaderRuntime,
			profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
			hook: options.directiveHook ?? null,
			mode: shaderMode,
		});
		this._shaderModuleCompiler = new WebGPUShaderModuleCompiler(this._shaderCompileStage);
		this._pipelineCache = new WebGPUPipelineCache(this._createPipelineCacheHost());
		this._bindingGroupCache = new WebGPUBindingGroupCache(
			this._createBindingGroupCacheHost()
		);
		this._passDispatcher = new WebGPUBackendPassDispatcher({
			get frameExecutor() {
				return thisRef._frameExecutor;
			},
			get particleSimulator() {
				return thisRef._particleSimulator;
			},
			get postProcessRuntime() {
				return thisRef._postProcessRuntime;
			},
			get resources() {
				return thisRef._resources;
			},
		});
		this._warmupCoordinator = new WebGPUWarmupCoordinator({
			get profile() {
				return thisRef.profile;
			},
			get frameExecutor() {
				return thisRef._frameExecutor;
			},
			get resources() {
				return thisRef._resources;
			},
			get postProcessRuntime() {
				return thisRef._postProcessRuntime;
			},
			setWarmupLogCompilationInfo: (enabled) => {
				this._warmupLogCompilationInfo = enabled;
			},
		});
		this._commandScheduler = new WebGPUCommandScheduler(this._createCommandSchedulerHost());
		this._resourceManager = new WebGPUResourceManager(this._createResourceManagerHost());
		const computeFacade = createWebGPUComputeFacade(this);
		this.extensions = createRenderBackendExtensionRegistry([
			{
				id: RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
				insertionPoints: [
					RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT,
					WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT,
				],
				api: this._occlusionCullingExtensionApi,
			},
			{
				id: PROBE_CAPTURE_EXTENSION.id,
				insertionPoints: ["renderer:probe-capture"],
				api: {
					captureProbeFace: (request: ProbeWebGPUCaptureFaceRequest) =>
						this.captureProbeFace(request),
				},
			},
			{
				id: WEBGPU_COMPUTE_EXTENSION.id,
				insertionPoints: ["application:webgpu-compute"],
				api: computeFacade,
			},
		]);
		this.shaderRuntime.onDidChange(() => {
			this._onShaderRuntimeChanged();
		});
	}

	public attach(context: RenderBackendAttachContext): void {
		if (this._attached) {
			throw new Error("WebGPUBackend is already attached to a renderer.");
		}
		this._attachContext = context;
		this._attached = true;
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

	/**
	 * Returns whether deferred opaque lighting is allowed for WebGPU frames.
	 *
	 * @returns `true` when the backend may use deferred lighting if runtime
	 * limits and frame targets support it.
	 * @sideEffects None.
	 */
	public isDeferredLightingEnabled(): boolean {
		return this._enableDeferredLighting;
	}

	/**
	 * Returns whether the backend may run WebGPU occlusion culling work.
	 *
	 * @returns `true` when the runtime may expose a visibility provider and
	 * encode the internal occlusion-test node.
	 * @sideEffects None.
	 */
	public isOcclusionCullingEnabled(): boolean {
		return this._enableOcclusionCulling;
	}

	/**
	 * Returns the diagnostic mode for WebGPU internal frame graph validation.
	 *
	 * @returns `"throw"` for strict validation or `"warn"` for non-fatal logs.
	 * @sideEffects None.
	 */
	public getFrameGraphValidationMode(): "throw" | "warn" {
		return this._frameGraphValidationMode;
	}

	/**
	 * Returns the current WebGPU diagnostic snapshot.
	 *
	 * @returns Adapter identifiers, selected limits, and device features when
	 * initialized, otherwise an unavailable snapshot.
	 * @sideEffects None.
	 */
	public getDebugInfo(): RenderBackendDebugInfo {
		return this._debugInfo;
	}

	/**
	 * @internal WebGPU backend tests use this to observe delegated cache state
	 * without depending on concrete private `Map` fields.
	 */
	private getWebGPUCacheDebugStats() {
		return {
			pipeline: this._pipelineCache.getDebugStats(),
			bindingGroups: this._bindingGroupCache.getDebugStats(),
		};
	}

	/**
	 * @internal Test-only hook for binding-group hash collision coverage.
	 */
	private setBindingGroupHashOverrideForTesting(
		override: ((layoutId: number, signatures: readonly unknown[]) => bigint) | null,
	): void {
		this._bindingGroupCache.setHashOverrideForTesting(override as any);
	}

	public getAttachments(size: RenderSurfaceSize): FrameAttachments {
		const { width, height } = size;
		return {
			width,
			height,
		};
	}

	public async initialize(): Promise<void> {
		const canvas = this._requireAttachContext().surface.canvas;
		this._canvas = canvas;
		this._destroyRequested = false;

		if (!navigator.gpu) {
			throw new Error("WebGPU not supported on this browser.");
		}

		const adapter = await navigator.gpu.requestAdapter({
			powerPreference: "high-performance",
		});
		if (!adapter) {
			throw new Error("No appropriate GPUAdapter found.");
		}
		await ShaderSource.prepare("webgpu.utility.mipmapBlit.raw");

		let requestedDevice: GPUDevice;
		try {
			const requiredLimits: Record<string, number> = {};
			const requiredFeatures: GPUFeatureName[] = [];
			const adapterMaxTextureDimension2D = adapter.limits?.maxTextureDimension2D ?? 0;
			const adapterMaxSampledTexturesPerShaderStage =
				adapter.limits?.maxSampledTexturesPerShaderStage;
			const adapterMaxSamplersPerShaderStage = adapter.limits?.maxSamplersPerShaderStage;
			const adapterMaxStorageTexturesPerShaderStage =
				adapter.limits?.maxStorageTexturesPerShaderStage;
			const requiredSampledTexturesPerShaderStage =
				WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT;
			const requiredSamplersPerShaderStage = WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLER_COUNT;
			if ((adapter.limits?.maxColorAttachments ?? 0) >= WEBGPU_DEFERRED_COLOR_TARGET_COUNT) {
				requiredLimits.maxColorAttachments = WEBGPU_DEFERRED_COLOR_TARGET_COUNT;
			} else if (
				(adapter.limits?.maxColorAttachments ?? 0) >= WEBGPU_MRT_COLOR_TARGET_COUNT
			) {
				requiredLimits.maxColorAttachments = WEBGPU_MRT_COLOR_TARGET_COUNT;
			}
			if (
				(adapter.limits?.maxColorAttachmentBytesPerSample ?? 0) >=
				WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE
			) {
				requiredLimits.maxColorAttachmentBytesPerSample =
					WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE;
			} else if (
				(adapter.limits?.maxColorAttachmentBytesPerSample ?? 0) >=
				WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE
			) {
				requiredLimits.maxColorAttachmentBytesPerSample = WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE;
			}
			if (
				typeof adapterMaxStorageTexturesPerShaderStage === "number" &&
				adapterMaxStorageTexturesPerShaderStage >= WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT
			) {
				requiredLimits.maxStorageTexturesPerShaderStage =
					WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT;
			}
			if (adapterMaxTextureDimension2D > 0) {
				requiredLimits.maxTextureDimension2D = adapterMaxTextureDimension2D;
			}
			requiredLimits.maxSampledTexturesPerShaderStage = requiredSampledTexturesPerShaderStage;
			requiredLimits.maxSamplersPerShaderStage = requiredSamplersPerShaderStage;

			const adapterMaxStorageBuffersPerShaderStage =
				adapter.limits?.maxStorageBuffersPerShaderStage;
			const requiredStorageBuffersPerShaderStage =
				WEBGPU_REQUIRED_FRAGMENT_STORAGE_BUFFER_COUNT;
			if (
				typeof adapterMaxStorageBuffersPerShaderStage === "number" &&
				adapterMaxStorageBuffersPerShaderStage >= requiredStorageBuffersPerShaderStage
			) {
				requiredLimits.maxStorageBuffersPerShaderStage =
					adapterMaxStorageBuffersPerShaderStage;
			}

			if (
				typeof adapterMaxSampledTexturesPerShaderStage === "number" &&
				adapterMaxSampledTexturesPerShaderStage < requiredSampledTexturesPerShaderStage
			) {
				throw new Error(
					"WebGPU adapter maxSampledTexturesPerShaderStage " +
						`(${adapterMaxSampledTexturesPerShaderStage}) is below required ` +
						"WebGPU pipeline sampled texture count " +
						`(${requiredSampledTexturesPerShaderStage}).`,
				);
			}
			if (
				typeof adapterMaxSamplersPerShaderStage === "number" &&
				adapterMaxSamplersPerShaderStage < requiredSamplersPerShaderStage
			) {
				throw new Error(
					"WebGPU adapter maxSamplersPerShaderStage " +
						`(${adapterMaxSamplersPerShaderStage}) is below required ` +
						"scene pipeline sampler count " +
						`(${requiredSamplersPerShaderStage}).`,
				);
			}
			if (
				typeof adapterMaxStorageBuffersPerShaderStage === "number" &&
				adapterMaxStorageBuffersPerShaderStage < requiredStorageBuffersPerShaderStage
			) {
				throw new Error(
					"WebGPU adapter maxStorageBuffersPerShaderStage " +
						`(${adapterMaxStorageBuffersPerShaderStage}) is below required ` +
						"WebGPU pipeline storage buffer count " +
						`(${requiredStorageBuffersPerShaderStage}).`,
				);
			}
			if (
				typeof adapter.features?.has === "function" &&
				adapter.features.has("timestamp-query" as GPUFeatureName)
			) {
				requiredFeatures.push("timestamp-query" as GPUFeatureName);
			}

			if (
				typeof adapter.features?.has === "function" &&
				adapter.features.has("indirect-first-instance" as GPUFeatureName)
			) {
				requiredFeatures.push("indirect-first-instance" as GPUFeatureName);
			}

			requestedDevice = await adapter.requestDevice({
				requiredFeatures: requiredFeatures.length > 0 ? requiredFeatures : undefined,
				requiredLimits:
					Object.keys(requiredLimits).length > 0 ? (requiredLimits as any) : undefined,
			});
			const deviceMaxSampledTexturesPerShaderStage =
				requestedDevice.limits?.maxSampledTexturesPerShaderStage;
			const deviceMaxSamplersPerShaderStage =
				requestedDevice.limits?.maxSamplersPerShaderStage;
			const deviceMaxStorageBuffersPerShaderStage =
				requestedDevice.limits?.maxStorageBuffersPerShaderStage;

			if (
				typeof deviceMaxSampledTexturesPerShaderStage === "number" &&
				deviceMaxSampledTexturesPerShaderStage < requiredSampledTexturesPerShaderStage
			) {
				throw new Error(
					"Requested WebGPU device maxSampledTexturesPerShaderStage " +
						`(${deviceMaxSampledTexturesPerShaderStage}) is below required ` +
						"WebGPU pipeline sampled texture count " +
						`(${requiredSampledTexturesPerShaderStage}).`,
				);
			}
			if (
				typeof deviceMaxSamplersPerShaderStage === "number" &&
				deviceMaxSamplersPerShaderStage < requiredSamplersPerShaderStage
			) {
				throw new Error(
					"Requested WebGPU device maxSamplersPerShaderStage " +
						`(${deviceMaxSamplersPerShaderStage}) is below required ` +
						"scene pipeline sampler count " +
						`(${requiredSamplersPerShaderStage}).`,
				);
			}
			if (
				typeof deviceMaxStorageBuffersPerShaderStage === "number" &&
				deviceMaxStorageBuffersPerShaderStage < requiredStorageBuffersPerShaderStage
			) {
				throw new Error(
					"Requested WebGPU device maxStorageBuffersPerShaderStage " +
						`(${deviceMaxStorageBuffersPerShaderStage}) is below required ` +
						"WebGPU pipeline storage buffer count " +
						`(${requiredStorageBuffersPerShaderStage}).`,
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
		this._debugInfo = this._createDebugInfo(adapter, requestedDevice);
		this._deviceLossPromise = requestedDevice.lost.then((info) => {
			if (this.device !== requestedDevice) {
				return info;
			}
			this.onDeviceLost(info);
			this._requireAttachContext().events.emit({ type: "device-lost", info });
			return info;
		});

		try {
			this._errorScopes = new WebGPUErrorScopeHelper(requestedDevice);
			this.canvasDepthFormat = this._selectCanvasDepthFormat();
			this.canvasFormat = navigator.gpu.getPreferredCanvasFormat() as TextureFormat;
			this._msaaController.activateDevice();
			this._commandScheduler.initTimestampResources();
			this._context = context;
			this._configureContext();
			this._recreateDepthTexture();

			this._resources = new WebGPURenderResources(this, this._msaaController);
			await this._resources.init();
			this._frameHost = this._createFrameHost();
			this._postProcessExecutor = new WebGPUPostProcessExecutor(this._frameHost);
			this._postProcessRuntime = new BackendPostProcessRuntime({
				executor: this._postProcessExecutor,
				backend: this,
				warn: (key, message) =>
					Logger.warn(`[${key}] ${message}`, {
						scope: "WebGPUBackend",
						onceKey: key,
					}),
			});
			this._frameExecutor = new WebGPUFrameExecutor(
				this._frameHost,
				this._resources,
				this._msaaController
			);
			this._reflectionProbeCapturePass = new WebGPUReflectionProbeCapturePass(
				this,
				this._resources,
			);
			this._particleSimulator = new WebGPUParticleSimulator({
				backend: this,
				backendTag: this.profile.id,
				maxParticlesPerSystem: 300000,
			});
		} catch (error) {
			this._rollbackInitializationState();
			throw error;
		}
	}

	/**
	 * Marks WebGPU device resources as lost.
	 *
	 * @internal Backend lifecycle hook used by `GPUDevice.lost` handling and
	 * renderer recovery paths.
	 */
	public onDeviceLost(info?: RenderBackendDeviceLostInfo): void {
		this._handleDeviceLost(this._normalizeDeviceLostInfo(info));
	}

	public async restore(): Promise<void> {
		if (!this._canvas) {
			throw new Error("WebGPU backend cannot restore before a canvas has been initialized.");
		}

		const activeRecovery = this._deviceRecoveryPromise;
		if (activeRecovery) {
			await activeRecovery;
			if (this.device && this.queue && !this._deviceLost) {
				return;
			}
		}

		this._deviceRecoveryNonce++;
		this._deviceRecoveryPromise = null;
		if (!this._deviceLost && this.queue) {
			this._commandScheduler.submitPendingCopyCommands();
		}
		this._destroyPostProcessResourcesForReset();
		this._rollbackInitializationState();
		this._deviceLost = false;
		this._deviceLostInfo = null;
		await this.initialize();
	}

	public resize(size: RenderSurfaceSize): void {
		const { width, height } = size;
		if (!this.device || !this.context || !this.canvas) {
			return;
		}
		const { width: resolvedWidth, height: resolvedHeight } = this._resolveResizeDimensions(
			width,
			height,
		);
		if (this._isFrameActive()) {
			this._pendingResize = {
				width: resolvedWidth,
				height: resolvedHeight,
			};
			return;
		}

		this._applyResize(resolvedWidth, resolvedHeight);
	}

	private _resolveResizeDimensions(
		width: number,
		height: number,
	): { width: number; height: number } {
		const canvas = this.canvas;
		return {
			width: Number.isFinite(width) ? Math.max(0, Math.floor(width)) : (canvas?.width ?? 0),
			height: Number.isFinite(height)
				? Math.max(0, Math.floor(height))
				: (canvas?.height ?? 0),
		};
	}

	private _applyResize(
		resolvedWidth: number,
		resolvedHeight: number,
		options: { invalidateFrameTargets?: boolean } = {},
	): boolean {
		if (!this.device || !this.context || !this.canvas) {
			return false;
		}
		if (this.canvas.width !== resolvedWidth || this.canvas.height !== resolvedHeight) {
			this.canvas.width = resolvedWidth;
			this.canvas.height = resolvedHeight;
		}
		this._commandScheduler.submitPendingCopyCommands();
		this._configureContext();
		this._resetCurrentCanvasTargets();
		this._bindingGroupCache.clear();
		this._recreateDepthTexture();
		this._postProcessRuntime?.invalidateFrameSized();
		if (options.invalidateFrameTargets !== false) {
			this._frameExecutor?.invalidateFrameTargets();
		}
		return true;
	}

	public beginFrame(context: FrameContext): void {
		if (!this._resources || !this._frameExecutor) {
			throw new Error("WebGPU backend has not been initialized.");
		}

		this._frameActive = true;
		this._completedFrameCoverage = "full-frame";
		this._frameSerial++;
		this._commandScheduler.submitPendingCopyCommands();
		this._bindingGroupCache.evictStale();
		this._prepareFramePassPlan(context);
		this._executedPasses.clear();
		this._particleSimulator?.beginFrame(context);
		this._resources.beginFrameResourceLifecycle();
		this._frameExecutor.beginFrame(context);
		const portFactory = (
			this._frameExecutor as WebGPUFrameExecutor & {
				createPostProcessSessionPort?: () => WebGPUPostProcessSessionPort;
			}
		).createPostProcessSessionPort;
		const postProcessPort = portFactory?.call(this._frameExecutor) ?? null;
		if (postProcessPort) {
			this._postProcessExecutor?.bindSession(postProcessPort);
		}
		this._postProcessSessionPort = postProcessPort;
	}

	public executePass(pass: FramePass, context: FrameContext): Promise<void> | void {
		if (!this._frameExecutor) {
			throw new Error("WebGPU backend has not been initialized.");
		}

		this._validatePassDependencies(pass);
		const dispatched = this._passDispatcher.executePass(pass, context);
		const result =
			dispatched === null ? this._frameExecutor.executePass(pass, context) : dispatched;
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

	public readRenderTargetColor(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions,
	): Promise<TextureReadbackResult> {
		if (!this._frameExecutor) {
			return Promise.reject(new Error("WebGPU backend has not been initialized."));
		}
		return this._frameExecutor.readRenderTargetColor(id, attachmentIndex, options);
	}

	public async warmup(context: FrameContext, options: WarmupOptions = {}): Promise<WarmupReport> {
		return this._warmupCoordinator.warmup(context, options);
	}

	public async endFrame(): Promise<void> {
		const wasActive = this._frameActive;
		let frameError: unknown = null;
		try {
			await this._frameExecutor?.endFrame();
		} catch (error) {
			frameError = error;
		}
		try {
			if (wasActive) {
				this._particleSimulator?.endFrame();
			}
		} catch (error) {
			if (!frameError) {
				frameError = error;
			} else {
				this._reportNonFatalError("particle frame cleanup", error);
			}
		} finally {
			this._postProcessExecutor?.unbindSession(this._postProcessSessionPort ?? undefined);
			this._postProcessSessionPort = null;
			this._frameActive = false;
			this._clearFramePlannerState();
		}

		try {
			this._flushDeferredLifecycleChanges();
		} catch (error) {
			if (!frameError) {
				throw error;
			}
			this._reportNonFatalError("deferred lifecycle flush after failed frame", error);
		}
		if (frameError) {
			throw frameError;
		}
		this._postProcessRuntime?.commitFrame();
	}

	public async abortFrame(_error?: unknown): Promise<void> {
		const wasActive = this._frameActive;
		let abortError: unknown = null;
		try {
			await this._postProcessRuntime?.abortFrame(_error);
			this._frameExecutor?.abortFrame();
			if (wasActive) {
				this._particleSimulator?.endFrame();
			}
		} catch (error) {
			abortError = error;
		} finally {
			this._postProcessExecutor?.unbindSession(this._postProcessSessionPort ?? undefined);
			this._postProcessSessionPort = null;
			this._frameActive = false;
			this._clearFramePlannerState();
		}
		try {
			this._flushDeferredLifecycleChanges();
		} catch (error) {
			if (!abortError) {
				throw error;
			}
			this._reportNonFatalError("deferred lifecycle flush after failed abort", error);
		}
		if (abortError) {
			throw abortError;
		}
	}

	/** @internal Renderer frame-coordination coverage report. */
	public getCompletedFrameCoverage(): RenderBackendCompletedFrameCoverage {
		return this._completedFrameCoverage;
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

	/**
	 * Returns the scene target mode selected for the current WebGPU frame.
	 *
	 * @returns `"gbuffer"` for deferred opaque lighting, `"mrt"` for legacy
	 * MRT rendering, `"color"` for HDR color-only offscreen rendering, or
	 * `"single"` for direct canvas fallback.
	 * @sideEffects None.
	 */
	public getFrameSceneTargetMode(): "gbuffer" | "mrt" | "color" | "single" {
		return this._frameExecutor?.getSceneTargetModeForFrame() ?? "single";
	}

	public async captureProbeFace(
		request: ProbeWebGPUCaptureFaceRequest,
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
			this._commandScheduler.submitPendingCopyCommands();
		}
		this._destroyPostProcessResourcesForReset();
		this._rollbackInitializationState();
		invalidateWebGPUComputeFacade(this);
		this._deviceLost = false;
		this._deviceLostInfo = null;
		this._deviceLossPromise = null;
	}

	public createBuffer(desc: BufferDesc): IRenderBuffer {
		return this._resourceManager.createBuffer(desc);
	}

	public createTexture(desc: TextureDesc): IRenderTexture {
		return this._resourceManager.createTexture(desc);
	}

	public createSampler(desc: SamplerDesc): ISampler {
		return this._pipelineCache.createSampler(desc);
	}

	public async createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule> {
		return this._pipelineCache.createShaderModule(desc);
	}

	public async createPipeline(desc: PipelineDesc): Promise<IRenderPipeline> {
		return this._pipelineCache.createPipeline(desc);
	}

	public async createComputePipeline(desc: ComputePipelineDesc): Promise<IComputePipeline> {
		return this._pipelineCache.createComputePipeline(desc);
	}

	public createBindingGroup(desc: BindingGroupDesc): IBindingGroup {
		return this._bindingGroupCache.createBindingGroup(desc);
	}

	public createCommandEncoder(): ICommandEncoder {
		return this._commandScheduler.createCommandEncoder();
	}

	public writeBuffer(buffer: IRenderBuffer, data: BufferSource, offset: number = 0): void {
		this._resourceManager.writeBuffer(buffer, data, offset);
	}

	public writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		desc: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number },
	): void {
		this._resourceManager.writeTexture(texture, data, desc, size);
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
		this._commandScheduler.copyTextureToTexture(source, destination, copySize);
	}

	public submit(commands: ICommandBuffer[]): void {
		this._commandScheduler.submit(commands);
	}

	public getCanvasColorTexture(): IRenderTexture {
		if (!this.context || !this.canvas) {
			throw new Error("WebGPU not initialized");
		}
		return this._canvasTargets.getCanvasColorTexture(this.context, this.canvas);
	}

	public getCanvasDepthTexture(): IRenderTexture {
		return this._canvasTargets.getCanvasDepthTexture();
	}

	public createTextureView(
		texture: IRenderTexture,
		desc?: GPUTextureViewDescriptor,
	): GPUTextureView {
		return this._resourceManager.createTextureView(texture, desc);
	}

	public getCurrentColorView(): GPUTextureView {
		if (!this.context) {
			throw new Error("WebGPU canvas context is not initialized.");
		}

		return this._canvasTargets.getCurrentColorView(this.context);
	}

	public getCurrentDepthView(): GPUTextureView {
		return this._canvasTargets.getCurrentDepthView();
	}

	public getTimestampDurationsMs(): ReadonlyMap<string, number> {
		return this._commandScheduler.getTimestampDurationsMs();
	}

	public createPassTimestampWrites(label: string):
		| {
				querySet: GPUQuerySet;
				beginningOfPassWriteIndex: number;
				endOfPassWriteIndex: number;
		  }
		| undefined {
		return this._commandScheduler.createPassTimestampWrites(label);
	}

	private _createFrameHost(): WebGPUFrameHost {
		const backend: WebGPUBackend = this;
		const device = this.device;
		const queue = this.queue;
		if (!device || !queue) {
			throw new Error("WebGPU backend cannot create a frame host without a device.");
		}
		const assertActive = (operation: string): void => {
			if (backend.device !== device || backend.queue !== queue) {
				throw new Error(`WebGPU frame host is no longer active; cannot ${operation}.`);
			}
			backend._assertDeviceOperational(operation);
		};
		return {
			device,
			queue,
			canvasFormat: this.canvasFormat,
			canvasDepthFormat: this.canvasDepthFormat,
			computeFacade: this.getComputeFacade(),
			get postProcessRuntime() {
				return backend.postProcessRuntime;
			},
			enableEarlyZPrepass: this.isEarlyZPrepassEnabled(),
			enableDeferredLighting: this.isDeferredLightingEnabled(),
			frameGraphValidationMode: this.getFrameGraphValidationMode(),
			createBuffer: (desc) => {
				assertActive("create frame buffers");
				return backend.createBuffer(desc);
			},
			createTexture: (desc) => {
				assertActive("create frame textures");
				return backend.createTexture(desc);
			},
			createSampler: (desc) => {
				assertActive("create frame samplers");
				return backend.createSampler(desc);
			},
			createShaderModule: (desc) => {
				assertActive("create frame shader modules");
				return backend.createShaderModule(desc);
			},
			createPipeline: (desc) => {
				assertActive("create frame pipelines");
				return backend.createPipeline(desc);
			},
			createComputePipeline: (desc) => {
				assertActive("create frame compute pipelines");
				return backend.createComputePipeline(desc);
			},
			createBindingGroup: (desc) => {
				assertActive("create frame binding groups");
				return backend.createBindingGroup(desc);
			},
			createTextureView: (texture, desc) => {
				assertActive("create frame texture views");
				return backend.createTextureView(texture, desc);
			},
			createCommandEncoder: () => {
				assertActive("create frame command encoders");
				return backend.createCommandEncoder();
			},
			submit: (commands) => {
				assertActive("submit frame command buffers");
				backend.submit(commands);
			},
			writeBuffer: (buffer, data, offset) => {
				assertActive("write frame buffers");
				backend.writeBuffer(buffer, data, offset);
			},
			getCanvasColorTexture: () => {
				assertActive("resolve frame canvas color");
				return backend.getCanvasColorTexture();
			},
			getCanvasDepthTexture: () => {
				assertActive("resolve frame canvas depth");
				return backend.getCanvasDepthTexture();
			},
			assertDeviceOperational: assertActive,
		};
	}

	private _createMSAAControllerHost(): WebGPUMSAAControllerHost {
		const thisRef = this;
		return {
			get device() {
				return thisRef.device;
			},
			get canvasFormat() {
				return thisRef.canvasFormat;
			},
			get canvasDepthFormat() {
				return thisRef.canvasDepthFormat;
			},
			get objectIdentity() {
				return thisRef._objectIdentity;
			},
			onRuntimeFallback: () => this._onMSAARuntimeFallback(),
		};
	}

	private _createPipelineCacheHost(): WebGPUPipelineCacheHost {
		const thisRef = this;
		return {
			get device() {
				return thisRef.device;
			},
			get shaderModuleCompiler() {
				return thisRef._shaderModuleCompiler;
			},
			get warmupLogCompilationInfo() {
				return thisRef._warmupLogCompilationInfo;
			},
			get objectIdentity() {
				return thisRef._objectIdentity;
			},
			assertDeviceOperational: (operation) => {
				this._assertDeviceOperational(operation);
			},
			resolveSupportedMSAASampleCount: (requested, probeFormats) => {
				return this._msaaController.resolveSupportedSampleCount(requested, probeFormats);
			},
			createManagedDestroy: (target, options) => {
				return this._createManagedDestroy(target, options);
			},
			runValidationScopeAsync: (label, operation) => {
				return this._runValidationScopeAsync(label, operation);
			},
		};
	}

	private _createBindingGroupCacheHost(): WebGPUBindingGroupCacheHost {
		const thisRef = this;
		return {
			get device() {
				return thisRef.device;
			},
			get frameSerial() {
				return thisRef._frameSerial;
			},
			get objectIdentity() {
				return thisRef._objectIdentity;
			},
			assertDeviceOperational: (operation) => {
				this._assertDeviceOperational(operation);
			},
			createManagedDestroy: (target, options) => {
				return this._createManagedDestroy(target, options);
			},
			runValidationScope: (label, operation) => {
				return this._runValidationScope(label, operation);
			},
		};
	}

	private _createCommandSchedulerHost(): WebGPUCommandSchedulerHost {
		const thisRef = this;
		return {
			get device() {
				return thisRef.device;
			},
			get queue() {
				return thisRef.queue;
			},
			assertDeviceOperational: (operation) => {
				this._assertDeviceOperational(operation);
			},
			runValidationScope: (label, operation) => {
				return this._runValidationScope(label, operation);
			},
			reportNonFatalError: (scope, error) => {
				this._reportNonFatalError(scope, error);
			},
			onSubmittedCommandBuffers: () => {
				this._resetCurrentCanvasTargets();
			},
			isFrameActive: () => {
				return this._isFrameActive();
			},
			createPassTimestampWrites: (label) => {
				return this.createPassTimestampWrites(label);
			},
			getCurrentColorView: () => {
				return this.getCurrentColorView();
			},
			getCurrentDepthView: () => {
				return this.getCurrentDepthView();
			},
			getCanvasColorTexture: () => {
				return this.getCanvasColorTexture();
			},
		};
	}

	private _createResourceManagerHost() {
		const thisRef = this;
		return {
			get device() {
				return thisRef.device;
			},
			get queue() {
				return thisRef.queue;
			},
			assertDeviceOperational: (operation: string) => {
				this._assertDeviceOperational(operation);
			},
			mapBufferUsage: (usage: number) => {
				return this._mapBufferUsage(usage);
			},
			mapTextureUsage: (usage: number) => {
				return this._mapTextureUsage(usage);
			},
			resolveSupportedMSAASampleCount: (
				requested: number,
				probeFormats?: readonly GPUTextureFormat[],
			) => {
				return this._msaaController.resolveSupportedSampleCount(requested, probeFormats);
			},
			resolvePositiveInteger: (value: number, fallback: number) => {
				return this._resolvePositiveInteger(value, fallback);
			},
			toUint8View: (data: BufferSource) => {
				return this._toUint8View(data);
			},
			tryUnmapBuffer: (buffer: GPUBuffer | null) => {
				this._tryUnmapBuffer(buffer);
			},
			createManagedDestroy: (
				target: object,
				options: {
					label: string;
					dispose: () => void;
				},
			) => {
				return this._createManagedDestroy(target, options);
			},
			runValidationScope: <T>(label: string, operation: () => T): T => {
				return this._runValidationScope(label, operation);
			},
		};
	}

	private _runValidationScope<T>(label: string, operation: () => T): T {
		if (!this._errorScopes) {
			return operation();
		}
		return this._errorScopes.run("validation", label, operation);
	}

	private _runValidationScopeAsync<T>(label: string, operation: () => Promise<T>): Promise<T> {
		if (!this._errorScopes) {
			return operation();
		}
		return this._errorScopes.runAsync("validation", label, operation);
	}

	private _handleDeviceLost(info: RenderBackendDeviceLostInfo): void {
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
		this._destroyPostProcessResourcesForReset();
		this._rollbackInitializationState();
		if (this._destroyRequested || info.reason === "destroyed") {
			return;
		}
		this._scheduleDeviceRecovery(info);
	}

	private _scheduleDeviceRecovery(info: RenderBackendDeviceLostInfo): void {
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
		info: RenderBackendDeviceLostInfo,
	): Promise<void> {
		let lastError: unknown = null;
		for (let attempt = 1; attempt <= DEVICE_RECOVERY_MAX_ATTEMPTS; attempt++) {
			if (this._destroyRequested || nonce !== this._deviceRecoveryNonce) {
				return;
			}
			try {
				await this.initialize();
				if (this._destroyRequested || nonce !== this._deviceRecoveryNonce) {
					this._rollbackInitializationState();
					return;
				}
				Logger.warn(`WebGPU device recovery succeeded on attempt ${attempt}.`, {
					scope: "WebGPUBackend",
				});
				this._deviceLostInfo = null;
				this._requireAttachContext().events.emit({ type: "device-restored" });
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

	private _normalizeDeviceLostInfo(
		info?: RenderBackendDeviceLostInfo,
	): RenderBackendDeviceLostInfo {
		const reason =
			typeof info?.reason === "string" && info.reason.length > 0 ? info.reason : undefined;
		const message =
			typeof info?.message === "string" && info.message.length > 0
				? info.message
				: "Device loss was reported without a diagnostic message.";
		return reason ? { reason, message } : { message };
	}

	private _destroyPostProcessResourcesForReset(): void {
		this._postProcessRuntime?.destroy();
		this._postProcessRuntime = null;
	}

	private _rollbackInitializationState(): void {
		this._commandScheduler.reset();
		this._postProcessRuntime?.destroy();
		this._postProcessRuntime = null;
		this._postProcessExecutor?.unbindSession();
		this._postProcessExecutor = null;
		this._postProcessSessionPort = null;
		this._reflectionProbeCapturePass?.destroy();
		this._reflectionProbeCapturePass = null;
		this._frameExecutor?.destroy();
		this._frameExecutor = null;
		this._frameHost = null;
		this._resources?.destroy();
		this._resources = null;
		const particleSimulator = this._particleSimulator as
			| ({ destroy?: () => void } & IParticleSimulator)
			| null;
		particleSimulator?.destroy?.();
		this._particleSimulator = null;
		this._canvasTargets.release();
		this._resetCurrentCanvasTargets();
		this._errorScopes = null;
		this._pipelineCache.reset();
		this._bindingGroupCache.clear();
		this._objectIdentity.reset();
		this._frameSerial = 0;
		this._frameActive = false;
		this._pendingResize = null;
		this._pendingShaderRuntimeInvalidation = false;
		this._clearFramePlannerState();
		this._msaaController.resetDevice();
		this._debugInfo = WEBGPU_DEBUG_INFO_UNINITIALIZED;
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

	private _createDebugInfo(adapter: GPUAdapter, device: GPUDevice): RenderBackendDebugInfo {
		const adapterInfo = resolveWebGPUAdapterInfo(adapter, device);
		const raw = collectWebGPUAdapterRaw(adapterInfo);
		const deviceInfo =
			adapterInfo || Object.keys(raw).length > 0
				? {
						vendor: normalizeDebugString(adapterInfo?.vendor),
						architecture: normalizeDebugString(adapterInfo?.architecture),
						device: normalizeDebugString(adapterInfo?.device),
						description: normalizeDebugString(adapterInfo?.description),
						isFallbackAdapter:
							typeof adapterInfo?.isFallbackAdapter === "boolean"
								? adapterInfo.isFallbackAdapter
								: undefined,
						raw: Object.keys(raw).length > 0 ? raw : undefined,
					}
				: undefined;

		return {
			backend: "webgpu",
			api: "webgpu",
			available: true,
			device: deviceInfo,
			limits: collectWebGPULimits(adapter, device),
			features: collectWebGPUFeatures(device.features),
		};
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

	private _resetCurrentCanvasTargets(): void {
		this._canvasTargets.resetCurrentCanvasTargets();
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

	private _prepareFramePassPlan(context: FrameContext): void {
		this._framePlanner.preparePlan(context, this._getFramePlannerState());
	}

	private _validatePassDependencies(pass: FramePass): void {
		this._framePlanner.validatePassDependencies(pass, this._getFramePlannerState(), {
			reportNonFatalError: (scope, error) => this._reportNonFatalError(scope, error),
		});
	}

	private _markPassExecuted(stage: FramePass["stage"]): void {
		this._framePlanner.markPassExecuted(stage, this._getFramePlannerState());
	}

	private _clearFramePlannerState(): void {
		this._executedPasses.clear();
		this._plannedPasses.clear();
		this._plannedPassOrder.clear();
	}

	private _isFrameActive(): boolean {
		return this._frameActive || this._plannedPasses.size > 0 || this._executedPasses.size > 0;
	}

	private _getFramePlannerState(): FramePassPlanValidatorState {
		return {
			executedPasses: this._executedPasses,
			plannedPasses: this._plannedPasses,
			plannedPassOrder: this._plannedPassOrder,
		};
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

	private _onMSAARuntimeFallback(): void {
		this._pipelineCache.clearPipelineCaches();
		this._bindingGroupCache.clear();
		this._postProcessRuntime?.invalidateFrameSized();
		this._resetCurrentCanvasTargets();
	}

	private _onShaderRuntimeChanged(): void {
		if (this._isFrameActive()) {
			this._pendingShaderRuntimeInvalidation = true;
			return;
		}
		this._applyShaderRuntimeChanged();
	}

	private _applyShaderRuntimeChanged(): void {
		this._commandScheduler.submitPendingCopyCommands();
		this._invalidateShaderDependentCaches();
		this._postProcessRuntime?.destroy();
		this._frameExecutor?.onShaderRuntimeChanged?.();
		this._resources?.onShaderRuntimeChanged?.();
		this._resetCurrentCanvasTargets();
	}

	private _flushDeferredLifecycleChanges(): void {
		if (this._isFrameActive()) {
			return;
		}
		const applyShaderRuntimeInvalidation = this._pendingShaderRuntimeInvalidation;
		const pendingResize = this._pendingResize;
		this._pendingShaderRuntimeInvalidation = false;
		this._pendingResize = null;

		if (applyShaderRuntimeInvalidation) {
			this._applyShaderRuntimeChanged();
		}

		let needsFrameTargetInvalidation = false;
		if (pendingResize) {
			needsFrameTargetInvalidation =
				this._applyResize(pendingResize.width, pendingResize.height, {
					invalidateFrameTargets: false,
				}) || needsFrameTargetInvalidation;
		}
		if (needsFrameTargetInvalidation) {
			this._frameExecutor?.invalidateFrameTargets();
		}
	}

	private _invalidateShaderDependentCaches(): void {
		this._pipelineCache.invalidateShaderDependentCaches();
		this._bindingGroupCache.clear();
	}

	private _handleObjectIdentityRebase(): void {
		this._pipelineCache.clearPipelineCaches();
		this._bindingGroupCache.clear();
		this._msaaController.clearCapabilityCache();
		Logger.warn(
			"WebGPU object-id space rebased; related caches were cleared to avoid unbounded growth.",
			{ scope: "WebGPUBackend" },
		);
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
		this._canvasTargets.configureContext(this.context, this.device, this.canvasFormat);
	}

	private _recreateDepthTexture(): void {
		if (!this.device || !this.canvas) {
			return;
		}
		this._canvasTargets.recreateDepthTexture(this.canvas, this.canvasDepthFormat, (desc) =>
			this.createTexture(desc),
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

	private _requireAttachContext(): RenderBackendAttachContext {
		if (!this._attachContext) {
			throw new Error("WebGPUBackend.attach() must be called before initialize().");
		}
		return this._attachContext;
	}
}

type WebGPUAdapterInfoLike = Partial<GPUAdapterInfo> & {
	readonly isFallbackAdapter?: boolean;
};

function resolveWebGPUAdapterInfo(
	adapter: GPUAdapter,
	device: GPUDevice
): WebGPUAdapterInfoLike | null {
	const deviceInfo = (device as { adapterInfo?: WebGPUAdapterInfoLike })
		.adapterInfo;
	if (deviceInfo) {
		return deviceInfo;
	}
	return (adapter as { info?: WebGPUAdapterInfoLike }).info ?? null;
}

function collectWebGPUAdapterRaw(
	info: WebGPUAdapterInfoLike | null
): Record<string, string | number | boolean> {
	if (!info) {
		return {};
	}
	const raw: Record<string, string | number | boolean> = {};
	for (const key of [
		"vendor",
		"architecture",
		"device",
		"description",
		"isFallbackAdapter",
		"subgroupMinSize",
		"subgroupMaxSize",
	] as const) {
		const value = info[key];
		if (
			(typeof value === "string" && value.length > 0) ||
			typeof value === "number" ||
			typeof value === "boolean"
		) {
			raw[key] = value;
		}
	}
	return raw;
}

function collectWebGPULimits(
	adapter: GPUAdapter,
	device: GPUDevice
): Record<string, number> {
	const limits: Record<string, number> = {};
	for (const key of WEBGPU_DEBUG_LIMIT_KEYS) {
		const value =
			readNumericLimit(device.limits, key) ??
			readNumericLimit(adapter.limits, key);
		if (typeof value === "number") {
			limits[key] = value;
		}
	}
	return limits;
}

function readNumericLimit(limits: unknown, key: string): number | undefined {
	const value = (limits as Record<string, unknown> | null | undefined)?.[key];
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function collectWebGPUFeatures(features: unknown): readonly string[] {
	if (!features || typeof (features as Iterable<string>)[Symbol.iterator] !== "function") {
		return [];
	}
	try {
		return Array.from(features as Iterable<string>, (feature) => String(feature))
			.sort();
	} catch {
		return [];
	}
}

function normalizeDebugString(value: unknown): string | undefined {
	return typeof value === "string" && value.length > 0 ? value : undefined;
}
