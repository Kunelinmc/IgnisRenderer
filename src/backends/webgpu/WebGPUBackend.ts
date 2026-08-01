/// <reference types="@webgpu/types" />
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
} from "../IRenderBackend";
import type {
	RenderTargetReadbackOptions,
	RenderTargetReadbackResult,
} from "../../rendering/CustomRenderTargets";
import { type FrameAttachments, type FrameContext, type FramePass } from "../../pipeline/types";
import type {
	NormalizedOcclusionCullingOptions,
	OcclusionCullingBackendAdapter,
} from "../../pipeline/OcclusionCulling";
import {
	createRenderBackendExtensionRegistry,
	IBL_PREFILTER_EXECUTOR_EXTENSION,
	PROBE_CAPTURE_EXTENSION,
	RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
	RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT,
	WEBGPU_COMPUTE_EXTENSION,
	WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT,
} from "../BackendExtensions";
import { WebGPUErrorScopeHelper } from "./WebGPUErrorScopeHelper";
import { WebGPUFrameOrchestrator } from "./rendergraph/WebGPUFrameOrchestrator";
import type { WebGPUFrameHost } from "./rendergraph/WebGPUFrameHost";
import { WebGPUPostProcessExecutor } from "./WebGPUPostProcessExecutor";
import { WebGPUFrameTransaction } from "./WebGPUFrameTransaction";
import { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import { WebGPUCommandScheduler } from "./WebGPUCommandScheduler";
import { WebGPUCanvasTargetManager } from "./WebGPUCanvasTargetManager";
import { WebGPUResourceManager } from "./WebGPUResourceManager";
import { WebGPUShaderModuleCompiler } from "./WebGPUShaderModuleCompiler";
import { WebGPUPipelineCache, type WebGPUPipelineCacheHost } from "./WebGPUPipelineCache";
import {
	WebGPUBindingGroupCache,
	type WebGPUBindingGroupCacheHost,
} from "./WebGPUBindingGroupCache";
import { WebGPUObjectIdentity } from "./WebGPUObjectIdentity";
import { WebGPUMSAAController, type WebGPUMSAAControllerHost } from "./WebGPUMSAAController";
import { WebGPUBackendPassDispatcher } from "./WebGPUBackendPassDispatcher";
import { WebGPUWarmupCoordinator } from "./WebGPUWarmupCoordinator";
import { WebGPUReflectionProbeCapturePass } from "./WebGPUReflectionProbeCapturePass";
import type { ProbeCaptureFaceRequest } from "../../lights/runtime/ProbeCaptureRuntime";
import { WebGPUFrameServiceOwner } from "./WebGPUFrameServiceOwner";
import type { WebGPUCommandSchedulerHost } from "./WebGPUBackendContracts";
import {
	FramePassPlanValidator,
	type FramePassPlanValidatorState,
} from "../../pipeline/FramePassPlanValidator";
import type { IParticleSimulator } from "../../simulation/particles/IParticleSimulator";
import { WebGPUParticleSimulator } from "../../simulation/particles/WebGPUParticleSimulator";
import {
	type IRenderTexture,
	TextureFormat,
} from "../types";
import {
	ShaderBackendCompileStage,
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	ShaderRuntime,
} from "../../shaders/runtime";
import type { ShaderDirectiveCompileHook, ShaderRuntimeMode } from "../../shaders/runtime";
import { ShaderSource } from "../../shaders/ShaderSource";
import type { Texture } from "../../core/Texture";
import {
	createWebGPUComputeFacade,
	type IWebGPUComputeFacade,
	type WebGPUComputeFacadeHost,
} from "./ComputeFacade";
import { WebGPUIBLPrefilterExecutor } from "./WebGPUIBLPrefilterExecutor";
import {
	assertWebGPUMinimumLimits,
	createWebGPUDebugInfo,
	createWebGPURequiredLimits,
	selectSupportedWebGPUFeatures,
} from "./WebGPUDeviceCapabilities";
import { Logger } from "../../foundation/Logger";
import { WebGPUDisplayOutputManager } from "./WebGPUDisplayOutputManager";
import {
	DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	createSDRDisplayOutputState,
	displayOutputStatesEqual,
	type DisplayOutputOptions,
	type DisplayOutputState,
	type ResolvedDisplayOutputOptions,
} from "../../rendering/DisplayOutput";

const DEVICE_RECOVERY_MAX_ATTEMPTS = 3;
const DEVICE_RECOVERY_BASE_DELAY_MS = 100;
const WEBGPU_MAX_PARTICLES_PER_SYSTEM = 300_000;

const WEBGPU_DEBUG_INFO_UNINITIALIZED: RenderBackendDebugInfo = {
	backend: "webgpu",
	api: "webgpu",
	available: false,
	unavailableReason: "WebGPU backend has not been initialized.",
};

type WebGPUBackendState =
	| "detached"
	| "attached"
	| "initializing"
	| "ready"
	| "restoring"
	| "lost"
	| "destroyed";

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
	private _activeFrameTransaction: WebGPUFrameTransaction | null = null;
	private readonly _occlusionCullingExtensionApi: OcclusionCullingBackendAdapter = {
		getVisibilityProvider: (options: NormalizedOcclusionCullingOptions) =>
			this._frameOrchestrator?.getOcclusionVisibilityProvider(options) ?? null,
		resetOcclusionCulling: () => {
			this._frameOrchestrator?.resetOcclusionCulling();
		},
	};
	public readonly extensions;
	public readonly profile: RenderBackendProfile;

	private _attachContext: RenderBackendAttachContext | null = null;
	private _state: WebGPUBackendState = "detached";
	private _lifecycleEpoch = 0;
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

	public canvasFormat: TextureFormat = TextureFormat.BGRA8Unorm;
	public canvasDepthFormat: TextureFormat = TextureFormat.Depth24Plus;
	public readonly shaderRuntime: ShaderRuntime;

	private readonly _canvasTargets = new WebGPUCanvasTargetManager();
	private _errorScopes: WebGPUErrorScopeHelper | null = null;
	private _resources: WebGPUFrameServiceOwner | null = null;
	private _frameOrchestrator: WebGPUFrameOrchestrator | null = null;
	private _frameHost: WebGPUFrameHost | null = null;
	private _reflectionProbeCapturePass: WebGPUReflectionProbeCapturePass | null = null;
	private _particleSimulator: IParticleSimulator | null = null;
	private _deviceLostInfo: RenderBackendDeviceLostInfo | null = null;
	private readonly _objectIdentity = new WebGPUObjectIdentity(() => {
		this._handleObjectIdentityRebase();
	});
	private _deviceRecoveryPromise: Promise<void> | null = null;
	private _frameSerial = 0;
	private _executedPasses = new Set<FramePass["stage"]>();
	private _plannedPasses = new Set<FramePass["stage"]>();
	private _plannedPassOrder = new Map<FramePass["stage"], number>();
	private _pendingResize: {
		width: number;
		height: number;
	} | null = null;
	private readonly _displayOutput = new WebGPUDisplayOutputManager(
		DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	);
	private _displayOutputState = createSDRDisplayOutputState(
		DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	);
	private _canvasConfiguration: GPUCanvasConfiguration | null = null;
	private _preferredCanvasFormat: TextureFormat = TextureFormat.BGRA8Unorm;
	private _pendingDisplayOutput: {
		requested: ResolvedDisplayOutputOptions;
		resolve: Array<(state: DisplayOutputState) => void>;
		reject: Array<(error: unknown) => void>;
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
	private _enableEarlyZPrepass = true;
	private _enableDeferredLighting = true;
	private _enableOcclusionCulling = true;
	private _frameGraphValidationMode: "throw" | "warn" = "throw";
	private _completedFrameCoverage: RenderBackendCompletedFrameCoverage = "full-frame";
	private _shaderCompileStage: ShaderBackendCompileStage;

	private readonly _framePlanner = new FramePassPlanValidator("WebGPU");
	private readonly _msaaController: WebGPUMSAAController;
	private readonly _shaderModuleCompiler: WebGPUShaderModuleCompiler;
	private readonly _commandScheduler: WebGPUCommandScheduler;
	private readonly _resourceManager: WebGPUResourceManager;
	private readonly _pipelineCache: WebGPUPipelineCache;
	private readonly _bindingGroupCache: WebGPUBindingGroupCache;
	private readonly _computeFacade: IWebGPUComputeFacade;
	private readonly _iblPrefilterExecutor: WebGPUIBLPrefilterExecutor;
	private readonly _passDispatcher: WebGPUBackendPassDispatcher;
	private readonly _warmupCoordinator: WebGPUWarmupCoordinator;

	constructor(options: WebGPUBackendOptions = {}) {
		if (Object.prototype.hasOwnProperty.call(options, "enableMSAA")) {
			throw new Error(
				"WebGPUBackendOptions.enableMSAA was removed; use msaaSampleCount: 1 to disable MSAA or msaaSampleCount: 4 to request 4x MSAA.",
			);
		}
		const shaderMode = options.shaderMode ?? "strict";
		const thisRef = this;
		this._msaaController = new WebGPUMSAAController(
			this._createMSAAControllerHost(),
			options.msaaSampleCount,
		);
		this._enableEarlyZPrepass = options.enableEarlyZPrepass !== false;
		this._enableDeferredLighting = options.enableDeferredLighting !== false;
		this._enableOcclusionCulling = options.enableOcclusionCulling !== false;
		this._frameGraphValidationMode = options.frameGraphValidation === "warn" ? "warn" : "throw";
		const capabilities: BackendCapabilities = {
			displayHDR: true,
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
		this._bindingGroupCache = new WebGPUBindingGroupCache(this._createBindingGroupCacheHost());
		this._passDispatcher = new WebGPUBackendPassDispatcher({
			get frameOrchestrator() {
				return thisRef._frameOrchestrator;
			},
			get particleSimulator() {
				return thisRef._particleSimulator;
			},
			get postProcessRuntime() {
				return thisRef._postProcessRuntime;
			},
		});
		this._warmupCoordinator = new WebGPUWarmupCoordinator({
			get profile() {
				return thisRef.profile;
			},
			get frameOrchestrator() {
				return thisRef._frameOrchestrator;
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
		this._computeFacade = createWebGPUComputeFacade(this._createComputeFacadeHost());
		this._iblPrefilterExecutor = new WebGPUIBLPrefilterExecutor(
			this._computeFacade,
		);
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
					captureProbeFace: (request: ProbeCaptureFaceRequest) =>
						this._captureProbeFace(request),
				},
			},
			{
				id: WEBGPU_COMPUTE_EXTENSION.id,
				insertionPoints: ["application:webgpu-compute"],
				api: this._computeFacade,
			},
			{
				id: IBL_PREFILTER_EXECUTOR_EXTENSION.id,
				insertionPoints: ["application:ibl-prefilter"],
				api: this._iblPrefilterExecutor,
			},
		]);
		this.shaderRuntime.onDidChange(() => {
			this._onShaderRuntimeChanged();
		});
	}

	public attach(context: RenderBackendAttachContext): void {
		if (this._state !== "detached") {
			throw new Error("WebGPUBackend is already attached to a renderer.");
		}
		this._attachContext = context;
		const displayOutput = this._displayOutput.setRequested(
			context.surface.displayOutput,
		);
		this._displayOutputState = createSDRDisplayOutputState(
			displayOutput,
		);
		this._state = "attached";
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

	public getDisplayOutputState(): DisplayOutputState {
		return this._displayOutputState;
	}

	public setDisplayOutput(
		options: DisplayOutputOptions,
	): Promise<DisplayOutputState> {
		this._requireReady("setDisplayOutput");
		const requested = this._displayOutput.setRequested(options);
		if (this._isFrameActive()) {
			return new Promise<DisplayOutputState>((resolve, reject) => {
				if (this._pendingDisplayOutput) {
					this._pendingDisplayOutput.requested = requested;
					this._pendingDisplayOutput.resolve.push(resolve);
					this._pendingDisplayOutput.reject.push(reject);
				} else {
					this._pendingDisplayOutput = {
						requested,
						resolve: [resolve],
						reject: [reject],
					};
				}
			});
		}
		return Promise.resolve(this._applyDisplayOutput(requested, true));
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
		this._requireLifecycleState("initialize", ["attached"]);
		const epoch = ++this._lifecycleEpoch;
		this._state = "initializing";

		try {
			await this._initializeDeviceRuntime(epoch, "initializing");
			this._assertLifecycleOperation(epoch, "initializing", "complete initialization");
			this._state = "ready";
			this._displayOutput.observeDynamicRange(() => {
				this._refreshDynamicRangeOutput();
			});
		} catch (error) {
			if (this._isLifecycleOperationCurrent(epoch, "initializing")) {
				this._releaseDeviceRuntime();
				this._state = "attached";
			}
			throw error;
		}
	}

	private async _initializeDeviceRuntime(
		epoch: number,
		expectedState: "initializing" | "restoring",
	): Promise<void> {
		const canvas = this._requireAttachContext().surface.canvas;
		this._canvas = canvas;

		if (!navigator.gpu) {
			throw new Error("WebGPU not supported on this browser.");
		}

		const adapter = await navigator.gpu.requestAdapter({
			powerPreference: "high-performance",
		});
		this._assertLifecycleOperation(epoch, expectedState, "request a WebGPU adapter");
		if (!adapter) {
			throw new Error("No appropriate GPUAdapter found.");
		}
		await ShaderSource.prepare("webgpu.utility.mipmapBlit.raw");
		this._assertLifecycleOperation(epoch, expectedState, "prepare WebGPU shaders");

		let requestedDevice: GPUDevice | null = null;
		try {
			const requiredLimits = createWebGPURequiredLimits(adapter.limits);
			const requiredFeatures = selectSupportedWebGPUFeatures(adapter);
			requestedDevice = await adapter.requestDevice({
				requiredFeatures: requiredFeatures.length > 0 ? requiredFeatures : undefined,
				requiredLimits: requiredLimits as any,
			});
			if (!this._isLifecycleOperationCurrent(epoch, expectedState)) {
				requestedDevice.destroy();
				requestedDevice = null;
				this._assertLifecycleOperation(
					epoch,
					expectedState,
					"publish a WebGPU device",
				);
			}
			assertWebGPUMinimumLimits(requestedDevice.limits, "Requested WebGPU device");
		} catch (error) {
			requestedDevice?.destroy();
			throw new Error(`Failed to request WebGPU device: ${error}`);
		}

		let context: GPUCanvasContext | null = null;
		try {
			context = canvas.getContext("webgpu");
			if (!context) {
				throw new Error("Failed to acquire WebGPU canvas context.");
			}
			this._assertLifecycleOperation(
				epoch,
				expectedState,
				"acquire a WebGPU canvas context",
			);
		} catch (error) {
			requestedDevice.destroy();
			throw error;
		}

		this._deviceLostInfo = null;
		this._device = requestedDevice;
		this._queue = requestedDevice.queue;
		this._debugInfo = createWebGPUDebugInfo(adapter, requestedDevice);
		requestedDevice.lost.then((info) => {
			if (this._device !== requestedDevice) {
				return info;
			}
			this._handleDeviceLost(this._normalizeDeviceLostInfo(info));
			this._requireAttachContext().events.emit({ type: "device-lost", info });
			return info;
		});

		this._errorScopes = new WebGPUErrorScopeHelper(requestedDevice);
		this.canvasDepthFormat = this._selectCanvasDepthFormat();
		this._preferredCanvasFormat =
			navigator.gpu.getPreferredCanvasFormat() as TextureFormat;
		this._msaaController.activateDevice();
		this._commandScheduler.initTimestampResources();
		this._context = context;
		this._applyDisplayOutput(this._displayOutput.requested, false);
		this._recreateDepthTexture();
		this._frameHost = this._createFrameHost();
		this._resources = new WebGPUFrameServiceOwner(
			this._frameHost,
			this._resourceManager,
			this._computeFacade,
			this._msaaController,
		);
		await this._resources.init();
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
		this._frameOrchestrator = new WebGPUFrameOrchestrator(
			this._frameHost,
			this._resources,
			this._msaaController,
		);
		this._reflectionProbeCapturePass = new WebGPUReflectionProbeCapturePass(
			this._frameHost,
			this._resources,
		);
		this._particleSimulator = new WebGPUParticleSimulator({
			backend: this._computeFacade,
			backendTag: this.profile.id,
			maxParticlesPerSystem: WEBGPU_MAX_PARTICLES_PER_SYSTEM,
		});
		this._assertLifecycleOperation(
			epoch,
			expectedState,
			"complete WebGPU runtime initialization",
		);
	}

	public async restore(): Promise<void> {
		if (!this._canvas || this._state === "detached" || this._state === "attached") {
			throw new Error("WebGPU backend cannot restore before a canvas has been initialized.");
		}

		while (this._deviceRecoveryPromise) {
			const activeRecovery = this._deviceRecoveryPromise;
			await activeRecovery;
			if (this._state === "ready" && this._device && this._queue) {
				return;
			}
		}

		this._requireLifecycleState("restore", ["ready", "lost"]);
		const previousState = this._state;
		const epoch = ++this._lifecycleEpoch;
		this._state = "restoring";
		this._deviceRecoveryPromise = null;
		try {
			if (previousState === "ready" && this._queue) {
				this._commandScheduler.submitPendingCopyCommands();
			}
			this._releaseDeviceRuntime();
			await this._initializeDeviceRuntime(epoch, "restoring");
			this._assertLifecycleOperation(epoch, "restoring", "complete restoration");
			this._deviceLostInfo = null;
			this._state = "ready";
		} catch (error) {
			if (this._isLifecycleOperationCurrent(epoch, "restoring")) {
				this._releaseDeviceRuntime();
				this._state = "lost";
			}
			throw error;
		}
	}

	public resize(size: RenderSurfaceSize): void {
		const { width, height } = size;
		if (!this._device || !this.context || !this.canvas) {
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
		if (!this._device || !this.context || !this.canvas) {
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
		this._resources?.resetTemporalState?.();
		if (options.invalidateFrameTargets !== false) {
			this._frameOrchestrator?.invalidateFrameTargets();
		}
		return true;
	}

	public beginFrame(context: FrameContext): void {
		this._requireReady("beginFrame");
		if (this._activeFrameTransaction?.isOpen) {
			throw new Error(
				`WebGPUBackend.beginFrame() requires no active frame; current state is "${this._state}".`,
			);
		}
		if (!context || typeof context !== "object") {
			throw new Error("WebGPUBackend.beginFrame() requires a valid FrameContext.");
		}
		if (!this._resources || !this._frameOrchestrator) {
			throw new Error("WebGPU backend has not been initialized.");
		}

		this._completedFrameCoverage = "full-frame";
		this._frameSerial++;
		this._commandScheduler.submitPendingCopyCommands();
		this._bindingGroupCache.evictStale();
		this._prepareFramePassPlan(context);
		this._executedPasses.clear();
		const transaction = new WebGPUFrameTransaction(context, {
			orchestrator: this._frameOrchestrator,
			resources: this._resources,
			particleSimulator: this._particleSimulator,
			postProcessRuntime: this._postProcessRuntime,
			postProcessExecutor: this._postProcessExecutor,
			reportCleanupError: (scope, error) => this._reportNonFatalError(scope, error),
		});
		this._activeFrameTransaction = transaction;
		try {
			transaction.begin();
		} catch (error) {
			if (this._activeFrameTransaction === transaction) {
				this._activeFrameTransaction = null;
			}
			this._clearFramePlannerState();
			throw error;
		}
	}

	public executePass(pass: FramePass, context: FrameContext): Promise<void> | void {
		this._requireReady("executePass");
		const transaction = this._requireActiveFrame("executePass");
		transaction.assertRecordingContext(context);
		if (!this._frameOrchestrator) {
			throw new Error("WebGPU backend has not been initialized.");
		}

		this._validatePassDependencies(pass);
		if (pass.stage === "particle-sim") {
			this._frameOrchestrator.recordOpaqueGraphStage?.(
				pass.stage,
				"Particle simulation executes outside the logical frame graph.",
			);
		}
		const dispatched = this._passDispatcher.executePass(pass, context);
		const result =
			dispatched === null ? this._frameOrchestrator.executePass(pass, context) : dispatched;
		if (result && typeof (result as Promise<void>).then === "function") {
			return (result as Promise<void>).then(() => {
				this._markPassExecuted(pass.stage);
			});
		}
		this._markPassExecuted(pass.stage);
		return result;
	}

	public skipPass(pass: FramePass): void {
		this._requireActiveFrame("skipPass");
		this._markPassExecuted(pass.stage);
	}

	public readRenderTargetColor(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions,
	): Promise<RenderTargetReadbackResult> {
		if (!this._frameOrchestrator) {
			return Promise.reject(new Error("WebGPU backend has not been initialized."));
		}
		return this._frameOrchestrator.readRenderTargetColor(id, attachmentIndex, options);
	}

	public async warmup(context: FrameContext, options: WarmupOptions = {}): Promise<WarmupReport> {
		return this._warmupCoordinator.warmup(context, options);
	}

	public async endFrame(): Promise<void> {
		const transaction = this._requireActiveFrame("endFrame");
		let frameError: unknown = null;
		try {
			await transaction.commit();
		} catch (error) {
			frameError = error;
		} finally {
			if (this._activeFrameTransaction === transaction) {
				this._activeFrameTransaction = null;
			}
			this._clearFramePlannerState();
		}

		try {
			this._flushDeferredLifecycleChanges();
		} catch (error) {
			if (!frameError) {
				frameError = error;
			} else {
				this._reportNonFatalError("deferred lifecycle flush after failed frame", error);
			}
		}
		if (frameError) throw frameError;
	}

	public async abortFrame(_error?: unknown): Promise<void> {
		const transaction = this._activeFrameTransaction;
		try {
			await transaction?.abort(_error);
		} finally {
			if (this._activeFrameTransaction === transaction) {
				this._activeFrameTransaction = null;
			}
			this._clearFramePlannerState();
		}
		try {
			this._flushDeferredLifecycleChanges();
		} catch (error) {
			if (_error !== undefined) {
				this._reportNonFatalError("deferred lifecycle flush after failed abort", error);
				return;
			}
			throw error;
		}
	}

	/** @internal Renderer frame-coordination coverage report. */
	public getCompletedFrameCoverage(): RenderBackendCompletedFrameCoverage {
		return this._completedFrameCoverage;
	}

	private _resolveTextureForSlot(texture: Texture | null, slotIndex: number): IRenderTexture {
		this._assertDeviceOperational("resolve texture resources");
		if (!this._resources) {
			throw new Error(
				"WebGPU resources are not initialized; cannot resolve texture resources.",
			);
		}
		return this._resources.getTextureForSlot(texture, slotIndex);
	}

	private _registerExternalTexture(
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

	private _unregisterExternalTexture(texture: Texture): void {
		if (!this._resources) {
			return;
		}
		this._resources.unregisterExternalTexture(texture);
	}

	private async _captureProbeFace(
		request: ProbeCaptureFaceRequest,
	): Promise<Float32Array | null> {
		if (!this._reflectionProbeCapturePass) {
			return null;
		}
		return this._reflectionProbeCapturePass.captureFace(request);
	}

	public destroy(): void {
		if (this._state === "destroyed") {
			return;
		}
		++this._lifecycleEpoch;
		const wasReady = this._state === "ready";
		this._state = "destroyed";
		this._displayOutput.stopObservingDynamicRange();
		this._deviceRecoveryPromise = null;
		if (wasReady && this._queue) {
			try {
				this._commandScheduler.submitPendingCopyCommands();
			} catch (error) {
				this._reportNonFatalError("pending copy submission during destroy", error);
			}
		}
		this._releaseDeviceRuntime();
		this._computeFacade.destroy();
		this._deviceLostInfo = null;
	}

	private _createComputeFacadeHost(): WebGPUComputeFacadeHost {
		const backend = this;
		const assertReady = (operation: string): void => {
			backend._requireReady(`compute extension ${operation}`);
		};
		return {
			get device() {
				return backend._state === "ready" ? backend._device : null;
			},
			get queue() {
				return backend._state === "ready" ? backend._queue : null;
			},
			createSampler: (desc) => {
				assertReady("create samplers");
				return this._pipelineCache.createSampler(desc);
			},
			createShaderModule: (desc) => {
				assertReady("create shader modules");
				return this._pipelineCache.createShaderModule(desc);
			},
			createComputePipeline: (desc) => {
				assertReady("create compute pipelines");
				return this._pipelineCache.createComputePipeline(desc);
			},
			createBuffer: (desc) => {
				assertReady("create buffers");
				return this._resourceManager.createBuffer(desc);
			},
			createTexture: (desc) => {
				assertReady("create textures");
				return this._resourceManager.createTexture(desc);
			},
			createBindingGroup: (desc) => {
				assertReady("create binding groups");
				return this._bindingGroupCache.createBindingGroup(desc);
			},
			createBindGroupLayout: (desc) => {
				assertReady("create binding group layouts");
				return this._device!.createBindGroupLayout(desc);
			},
			createPipelineLayout: (desc) => {
				assertReady("create pipeline layouts");
				return this._device!.createPipelineLayout(desc);
			},
			createTextureView: (texture, desc) => {
				assertReady("create texture views");
				return this._resourceManager.createTextureView(texture, desc);
			},
			createCommandEncoder: () => {
				assertReady("create command encoders");
				return this._commandScheduler.createCommandEncoder();
			},
			submit: (commands) => {
				assertReady("submit command buffers");
				this._commandScheduler.submit(commands);
			},
			writeBuffer: (buffer, data, offset) => {
				assertReady("write buffers");
				this._resourceManager.writeBuffer(buffer, data, offset);
			},
			writeTexture: (texture, data, desc, size) => {
				assertReady("write textures");
				this._resourceManager.writeTexture(texture, data, desc, size);
			},
			resolveTextureForSlot: (texture, slotIndex) => {
				assertReady("resolve texture slots");
				return this._resolveTextureForSlot(texture, slotIndex);
			},
			registerExternalTexture: (texture, resource, uploadedVersion, mipLevelCount) => {
				assertReady("register external textures");
				this._registerExternalTexture(texture, resource, uploadedVersion, mipLevelCount);
			},
			unregisterExternalTexture: (texture) => {
				if (this._state === "destroyed") {
					return;
				}
				assertReady("unregister external textures");
				this._unregisterExternalTexture(texture);
			},
		};
	}

	private _createFrameHost(): WebGPUFrameHost {
		const backend: WebGPUBackend = this;
		const device = this._device;
		const queue = this._queue;
		if (!device || !queue) {
			throw new Error("WebGPU backend cannot create a frame host without a device.");
		}
		const assertActive = (operation: string): void => {
			if (backend._device !== device || backend._queue !== queue) {
				throw new Error(`WebGPU frame host is no longer active; cannot ${operation}.`);
			}
			backend._assertDeviceOperational(operation);
		};
		return {
			device,
			queue,
			get canvasFormat() {
				return backend.canvasFormat;
			},
			get displayOutputState() {
				return backend._displayOutputState;
			},
			canvasDepthFormat: this.canvasDepthFormat,
			computeFacade: this._computeFacade,
			get postProcessRuntime() {
				const runtime = backend._postProcessRuntime;
				if (!runtime) {
					throw new Error("WebGPU post-process runtime is not initialized.");
				}
				return runtime;
			},
			enableEarlyZPrepass: this._enableEarlyZPrepass,
			enableDeferredLighting: this._enableDeferredLighting,
			frameGraphValidationMode: this._frameGraphValidationMode,
			get shaderRuntime() {
				return backend.shaderRuntime;
			},
			getShaderDirectiveCacheTag: () => backend._shaderCompileStage.getCacheFingerprintTag(),
			createBuffer: (desc) => {
				assertActive("create frame buffers");
				return backend._resourceManager.createBuffer(desc);
			},
			createTexture: (desc) => {
				assertActive("create frame textures");
				return backend._resourceManager.createTexture(desc);
			},
			createSampler: (desc) => {
				assertActive("create frame samplers");
				return backend._pipelineCache.createSampler(desc);
			},
			createShaderModule: (desc) => {
				assertActive("create frame shader modules");
				return backend._pipelineCache.createShaderModule(desc);
			},
			createPipeline: (desc) => {
				assertActive("create frame pipelines");
				return backend._pipelineCache.createPipeline(desc);
			},
			createComputePipeline: (desc) => {
				assertActive("create frame compute pipelines");
				return backend._pipelineCache.createComputePipeline(desc);
			},
			createBindingGroup: (desc) => {
				assertActive("create frame binding groups");
				return backend._bindingGroupCache.createBindingGroup(desc);
			},
			createTextureView: (texture, desc) => {
				assertActive("create frame texture views");
				return backend._resourceManager.createTextureView(texture, desc);
			},
			createCommandEncoder: () => {
				assertActive("create frame command encoders");
				return backend._commandScheduler.createCommandEncoder();
			},
			submit: (commands) => {
				assertActive("submit frame command buffers");
				backend._commandScheduler.submit(commands);
			},
			writeBuffer: (buffer, data, offset) => {
				assertActive("write frame buffers");
				backend._resourceManager.writeBuffer(buffer, data, offset);
			},
			getCanvasColorTexture: () => {
				assertActive("resolve frame canvas color");
				if (!backend.context || !backend.canvas) {
					throw new Error("WebGPU not initialized");
				}
				return backend._canvasTargets.getCanvasColorTexture(
					backend.context,
					backend.canvas,
				);
			},
			getCanvasDepthTexture: () => {
				assertActive("resolve frame canvas depth");
				return backend._canvasTargets.getCanvasDepthTexture();
			},
			assertDeviceOperational: assertActive,
		};
	}

	private _createMSAAControllerHost(): WebGPUMSAAControllerHost {
		const thisRef = this;
		return {
			get device() {
				return thisRef._device;
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
				return thisRef._device;
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
				return thisRef._device;
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
				return thisRef._device;
			},
			get queue() {
				return thisRef._queue;
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
				return this._commandScheduler.createPassTimestampWrites(label);
			},
			getCurrentColorView: () => {
				if (!this._context) {
					throw new Error("WebGPU canvas context is not initialized.");
				}
				return this._canvasTargets.getCurrentColorView(this._context);
			},
			getCurrentDepthView: () => {
				return this._canvasTargets.getCurrentDepthView();
			},
			getCanvasColorTexture: () => {
				if (!this.context || !this.canvas) {
					throw new Error("WebGPU not initialized");
				}
				return this._canvasTargets.getCanvasColorTexture(this.context, this.canvas);
			},
		};
	}

	private _createResourceManagerHost() {
		const thisRef = this;
		return {
			get device() {
				return thisRef._device;
			},
			get queue() {
				return thisRef._queue;
			},
			assertDeviceOperational: (operation: string) => {
				this._assertDeviceOperational(operation);
			},
			resolveSupportedMSAASampleCount: (
				requested: number,
				probeFormats?: readonly GPUTextureFormat[],
			) => {
				return this._msaaController.resolveSupportedSampleCount(requested, probeFormats);
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
		if (this._state === "destroyed" || this._state === "lost") {
			return;
		}
		++this._lifecycleEpoch;
		this._deviceRecoveryPromise = null;
		this._state = "lost";
		this._deviceLostInfo = info;
		const reason =
			typeof info.reason === "string" && info.reason.length > 0 ? ` (${info.reason})` : "";
		Logger.error(`WebGPU device was lost${reason}: ${info.message}`, {
			scope: "WebGPUBackend",
		});
		this._releaseDeviceRuntime();
		if (info.reason === "destroyed") {
			return;
		}
		this._scheduleDeviceRecovery(info);
	}

	private _scheduleDeviceRecovery(info: RenderBackendDeviceLostInfo): void {
		if (this._deviceRecoveryPromise || this._state === "destroyed") {
			return;
		}
		const canvas = this.canvas;
		if (!canvas) {
			Logger.error("WebGPU device recovery skipped: backend is missing canvas.", {
				scope: "WebGPUBackend",
			});
			return;
		}
		const epoch = ++this._lifecycleEpoch;
		this._state = "restoring";
		this._deviceRecoveryPromise = this._recoverDeviceAfterLoss(epoch, info).finally(
			() => {
				if (this._lifecycleEpoch === epoch) {
					this._deviceRecoveryPromise = null;
				}
			},
		);
	}

	private async _recoverDeviceAfterLoss(
		epoch: number,
		info: RenderBackendDeviceLostInfo,
	): Promise<void> {
		let lastError: unknown = null;
		for (let attempt = 1; attempt <= DEVICE_RECOVERY_MAX_ATTEMPTS; attempt++) {
			if (!this._isLifecycleOperationCurrent(epoch, "restoring")) {
				return;
			}
			try {
				await this._initializeDeviceRuntime(epoch, "restoring");
				this._assertLifecycleOperation(
					epoch,
					"restoring",
					"complete automatic device recovery",
				);
				Logger.warn(`WebGPU device recovery succeeded on attempt ${attempt}.`, {
					scope: "WebGPUBackend",
				});
				this._deviceLostInfo = null;
				this._state = "ready";
				this._requireAttachContext().events.emit({ type: "device-restored" });
				return;
			} catch (error) {
				if (!this._isLifecycleOperationCurrent(epoch, "restoring")) {
					return;
				}
				this._releaseDeviceRuntime();
				lastError = error;
				this._reportNonFatalError(`device recovery attempt ${attempt}`, error);
				if (attempt < DEVICE_RECOVERY_MAX_ATTEMPTS) {
					await this._delayMs(DEVICE_RECOVERY_BASE_DELAY_MS * attempt);
				}
			}
		}
		if (this._isLifecycleOperationCurrent(epoch, "restoring")) {
			this._state = "lost";
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

	private _releaseDeviceRuntime(): void {
		const cleanup = (scope: string, operation: () => void): void => {
			try {
				operation();
			} catch (error) {
				this._reportNonFatalError(`runtime cleanup ${scope}`, error);
			}
		};
		const activeTransaction = this._activeFrameTransaction;
		this._activeFrameTransaction = null;
		if (activeTransaction) {
			cleanup("active frame transaction", () => activeTransaction.invalidate());
		}

		const postProcessRuntime = this._postProcessRuntime;
		this._postProcessRuntime = null;
		if (postProcessRuntime) {
			cleanup("post-process runtime", () => postProcessRuntime.destroy());
		}
		const postProcessExecutor = this._postProcessExecutor;
		this._postProcessExecutor = null;
		if (postProcessExecutor) {
			cleanup("post-process session", () => postProcessExecutor.unbindSession());
		}
		const reflectionProbeCapturePass = this._reflectionProbeCapturePass;
		this._reflectionProbeCapturePass = null;
		if (reflectionProbeCapturePass) {
			cleanup("reflection probe capture", () => reflectionProbeCapturePass.destroy());
		}
		const frameOrchestrator = this._frameOrchestrator;
		this._frameOrchestrator = null;
		this._frameHost = null;
		if (frameOrchestrator) {
			cleanup("frame orchestrator", () => frameOrchestrator.destroy());
		}
		const resources = this._resources;
		this._resources = null;
		if (resources) {
			cleanup("frame resources", () => resources.destroy());
		}
		const particleSimulator = this._particleSimulator as
			| ({ destroy?: () => void } & IParticleSimulator)
			| null;
		this._particleSimulator = null;
		if (particleSimulator?.destroy) {
			cleanup("particle simulator", () => particleSimulator.destroy!());
		}
		cleanup("canvas targets", () => this._canvasTargets.release());
		this._resetCurrentCanvasTargets();
		this._errorScopes = null;
		cleanup("command scheduler", () => this._commandScheduler.reset());
		cleanup("pipeline cache", () => this._pipelineCache.reset());
		cleanup("binding-group cache", () => this._bindingGroupCache.clear());
		cleanup("object identity", () => this._objectIdentity.reset());
		cleanup("MSAA controller", () => this._msaaController.resetDevice());
		this._frameSerial = 0;
		this._pendingResize = null;
		this._pendingShaderRuntimeInvalidation = false;
		const pendingDisplayOutput = this._pendingDisplayOutput;
		this._pendingDisplayOutput = null;
		for (const reject of pendingDisplayOutput?.reject ?? []) {
			reject(new Error("WebGPU backend was released before display output changed."));
		}
		this._canvasConfiguration = null;
		this._clearFramePlannerState();
		this._debugInfo = WEBGPU_DEBUG_INFO_UNINITIALIZED;
		if (this.context) {
			const context = this.context;
			this._context = null;
			cleanup("context unconfigure", () => context.unconfigure());
		}
		if (this._device) {
			const device = this._device;
			this._device = null;
			cleanup("device destroy", () => device.destroy());
		}
		this._queue = null;
	}

	private _assertDeviceOperational(operation: string): void {
		if (this._state === "lost") {
			const reason =
				typeof this._deviceLostInfo?.reason === "string" &&
				this._deviceLostInfo.reason.length > 0
					? ` (${this._deviceLostInfo.reason})`
					: "";
			const message = this._deviceLostInfo?.message ?? "unknown cause";
			throw new Error(`WebGPU device is lost${reason}; cannot ${operation}: ${message}`);
		}
		if (this._state === "destroyed") {
			throw new Error(`WebGPU backend is destroyed; cannot ${operation}.`);
		}
		if (
			this._state !== "initializing" &&
			this._state !== "restoring" &&
			this._state !== "ready"
		) {
			throw new Error(
				`WebGPU backend lifecycle state is "${this._state}"; cannot ${operation}.`,
			);
		}
		if (!this._device || !this._queue) {
			throw new Error(`WebGPU backend is not initialized; cannot ${operation}.`);
		}
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
		return this._activeFrameTransaction?.isOpen === true;
	}

	private _getFramePlannerState(): FramePassPlanValidatorState {
		return {
			executedPasses: this._executedPasses,
			plannedPasses: this._plannedPasses,
			plannedPassOrder: this._plannedPassOrder,
		};
	}

	private _selectCanvasDepthFormat(): TextureFormat {
		if (!this._device) {
			return TextureFormat.Depth24Plus;
		}
		const candidates: TextureFormat[] = [TextureFormat.Depth24Plus, TextureFormat.Depth32Float];
		for (const candidate of candidates) {
			try {
				const probe = this._device.createTexture({
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
		this._frameOrchestrator?.onShaderRuntimeChanged();
		this._resources?.onShaderRuntimeChanged?.();
		this._resetCurrentCanvasTargets();
	}

	private _flushDeferredLifecycleChanges(): void {
		if (this._isFrameActive()) {
			return;
		}
		const applyShaderRuntimeInvalidation = this._pendingShaderRuntimeInvalidation;
		const pendingResize = this._pendingResize;
		const pendingDisplayOutput = this._pendingDisplayOutput;
		this._pendingShaderRuntimeInvalidation = false;
		this._pendingResize = null;
		this._pendingDisplayOutput = null;

		if (applyShaderRuntimeInvalidation) {
			this._applyShaderRuntimeChanged();
		}

		if (pendingDisplayOutput) {
			try {
				const state = this._applyDisplayOutput(
					pendingDisplayOutput.requested,
					true,
				);
				for (const resolve of pendingDisplayOutput.resolve) resolve(state);
			} catch (error) {
				for (const reject of pendingDisplayOutput.reject) reject(error);
			}
		}

		let needsFrameTargetInvalidation = false;
		if (pendingResize) {
			needsFrameTargetInvalidation =
				this._applyResize(pendingResize.width, pendingResize.height, {
					invalidateFrameTargets: false,
				}) || needsFrameTargetInvalidation;
		}
		if (needsFrameTargetInvalidation) {
			this._frameOrchestrator?.invalidateFrameTargets();
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

	private _configureContext(): void {
		if (!this.context || !this.canvas || !this._device) {
			return;
		}
		this._canvasConfiguration ??= {
			device: this._device,
			format: this.canvasFormat as GPUTextureFormat,
			alphaMode: "premultiplied",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
		};
		this._canvasTargets.configureContext(
			this.context,
			this._canvasConfiguration,
			this.canvasFormat,
		);
	}

	private _applyDisplayOutput(
		requested: ResolvedDisplayOutputOptions,
		emitChange: boolean,
	): DisplayOutputState {
		if (!this.context || !this._device) {
			this._displayOutput.setRequested(requested);
			this._displayOutputState = createSDRDisplayOutputState(requested);
			return this._displayOutputState;
		}
		const previous = this._displayOutputState;
		this._displayOutput.setRequested(requested);
		const resolved = this._displayOutput.configure(
			this.context,
			this._device,
			this._preferredCanvasFormat,
		);
		const formatChanged = this.canvasFormat !== resolved.format;
		this.canvasFormat = resolved.format;
		this._canvasConfiguration = resolved.canvas;
		this._displayOutputState = resolved.state;
		this._configureContext();

		if (formatChanged) {
			this._pipelineCache.clearPipelineCaches();
			this._bindingGroupCache.clear();
			this._msaaController.clearCapabilityCache();
			this._frameOrchestrator?.onDisplayOutputChanged();
		} else {
			this._frameOrchestrator?.invalidatePostProcessBindings();
		}
		this._resetCurrentCanvasTargets();

		if (resolved.state.fallbackReason) {
			const configurationFailed =
				resolved.state.fallbackReason ===
				"hdr-context-configuration-failed";
			const key = configurationFailed ?
				"display-hdr-configuration-failed" : "display-hdr-unavailable";
			if (requested.mode === "hdr" || configurationFailed) {
				Logger.warn(
					`[${key}] WebGPU Display HDR fell back to SDR ` +
					`(${resolved.state.fallbackReason}).`,
					{ scope: "WebGPUBackend", onceKey: key },
				);
			}
		}
		if (emitChange && !displayOutputStatesEqual(previous, resolved.state)) {
			this._requireAttachContext().events.emit({
				type: "display-output-change",
				previous,
				current: resolved.state,
			});
		}
		return resolved.state;
	}

	private _refreshDynamicRangeOutput(): void {
		if (this._state !== "ready") return;
		const requested = this._displayOutput.requested;
		if (requested.mode === "sdr") return;
		if (this._isFrameActive()) {
			if (this._pendingDisplayOutput) {
				this._pendingDisplayOutput.requested = requested;
			} else {
				this._pendingDisplayOutput = {
					requested,
					resolve: [],
					reject: [],
				};
			}
			return;
		}
		this._applyDisplayOutput(requested, true);
	}

	private _recreateDepthTexture(): void {
		if (!this._device || !this.canvas) {
			return;
		}
		this._canvasTargets.recreateDepthTexture(this.canvas, this.canvasDepthFormat, (desc) =>
			this._resourceManager.createTexture(desc),
		);
	}

	private _requireLifecycleState(
		operation: string,
		allowedStates: readonly WebGPUBackendState[],
	): void {
		if (allowedStates.includes(this._state)) {
			return;
		}
		throw new Error(
			`WebGPUBackend.${operation}() is invalid in lifecycle state "${this._state}".`,
		);
	}

	private _isLifecycleOperationCurrent(
		epoch: number,
		expectedState: "initializing" | "restoring",
	): boolean {
		return this._lifecycleEpoch === epoch && this._state === expectedState;
	}

	private _assertLifecycleOperation(
		epoch: number,
		expectedState: "initializing" | "restoring",
		operation: string,
	): void {
		if (this._isLifecycleOperationCurrent(epoch, expectedState)) {
			return;
		}
		throw new Error(
			`WebGPU lifecycle changed while attempting to ${operation}; current state is "${this._state}".`,
		);
	}

	private _requireReady(operation: string): void {
		if (this._state === "ready" && this._device && this._queue) {
			return;
		}
		throw new Error(
			`WebGPUBackend.${operation}() requires lifecycle state "ready"; current state is "${this._state}".`,
		);
	}

	private _requireActiveFrame(operation: string): WebGPUFrameTransaction {
		const transaction = this._activeFrameTransaction;
		if (transaction?.isOpen) {
			return transaction;
		}
		throw new Error(
			`WebGPUBackend.${operation}() requires an active frame; current lifecycle state is "${this._state}".`,
		);
	}

	private _requireAttachContext(): RenderBackendAttachContext {
		if (!this._attachContext) {
			throw new Error("WebGPUBackend.attach() must be called before initialize().");
		}
		return this._attachContext;
	}
}
