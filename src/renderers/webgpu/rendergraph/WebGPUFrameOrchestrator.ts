import type { DrawPacket, FrameContext, FramePass } from "../../../pipeline/types";
import type {
	LogicalGBufferBridge,
	PostProcessPassExecutionContextRequest,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../../postprocess";
import type { ICommandEncoder } from "../../ICommandEncoder";
import type { IRenderTexture } from "../../types";
import type { WebGPUBackend } from "../../WebGPUBackend";
import type { WebGPUMSAAContext } from "../WebGPUMSAAController";
import type {
	WebGPUPreparedFrameResources,
	WebGPURenderResources,
} from "../WebGPURenderResources";
import { resolveWebGPUComputeFacade } from "../ComputeFacade";
import { WebGPUHiZBuilder } from "../WebGPUHiZBuilder";
import { BackendPostProcessRuntime } from "../../../postprocess/BackendPostProcessRuntime";
import { WebGPUPostProcessExecutor } from "../WebGPUPostProcessExecutor";

import {
	isWebGPUPostProcessContextMetadata,
	type WebGPUFrameTargets,
	type WebGPUPostProcessContextMetadata,
} from "../WebGPUPostProcessContracts";
import {
	WebGPUPostProcessRuntime,
} from "../WebGPUPostProcessRuntime";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../../pipeline/WarmupPlanner";
import { toShaderCompileError } from "../../../pipeline/WarmupPlanner";
import { createWarmupYieldController } from "../../../pipeline/WarmupScheduler";
import type { ShaderCompileError } from "../../../shaders/runtime";
import type { WarmupOptions } from "../../IRenderBackend";
import { Logger } from "../../../foundation/Logger";
import { materialUsesTransmission } from "../../../materials/transparency";
import {
	WebGPUPlanarReflectionPass,
	type WebGPUPlanarReflectionMSAATargets,
} from "../WebGPUPlanarReflectionPass";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import {
	WebGPUFrameTargetManager,
	type WebGPUFrameTargetEnsureResult,
	type WebGPUFrameMSAATargets,
} from "./WebGPUFrameTargetManager";
import {
	WebGPUFrameConfigurationResolver,
	type WebGPUFrameConfiguration,
	type WebGPUFrameDiagnostic,
} from "./WebGPUFrameConfigurationResolver";
import { WebGPUFrameGraphPlanner } from "./WebGPUFrameGraphPlanner";
import { WebGPUFrameGraphCompiler } from "./WebGPUFrameGraphCompiler";
import { WebGPUPostProcessBridge } from "./WebGPUPostProcessBridge";
import { WebGPUOITPass } from "./WebGPUOITPass";
import { WebGPUDeferredLightingPass } from "./WebGPUDeferredLightingPass";
import { WebGPUDeferredDecalPass } from "./WebGPUDeferredDecalPass";
import { WebGPUDirtyRectResolver } from "./WebGPUDirtyRectResolver";
import { WebGPUDepthDirtyClearPass } from "./WebGPUDepthDirtyClearPass";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";
import {
	WebGPUScenePassRecorder,
	type WebGPUDeferredOpaqueFrameState,
} from "./WebGPUScenePassRecorder";
import {
	WebGPUOcclusionCullingRuntime,
} from "../WebGPUOcclusionCullingRuntime";
import type { WebGPUPagedShadowFrameRequest } from "../WebGPUPagedShadowRuntime";
import {
	normalizeOcclusionCullingOptions,
	type NormalizedOcclusionCullingOptions,
	type OcclusionVisibilityProvider,
} from "../../../pipeline/OcclusionCulling";
import type {
	WebGPUCompiledFrameGraphStage,
	WebGPUFrameGraphDebugState,
	WebGPUFrameGraphNode,
	WebGPUFrameGraphNodeKind,
	WebGPUFrameGraphValidationMode,
} from "./types";
import { WebGPUPresentPass } from "./WebGPUPresentPass";
import { WebGPUCustomRenderTargetRuntime } from "./WebGPUCustomRenderTargetRuntime";
import type { RenderTargetReadbackOptions } from "../../CustomRenderTargets";
import type { TextureReadbackResult } from "../../IComputeRuntime";

const WEBGPU_DEFERRED_RUNTIME_FALLBACK_KEY = "webgpu-deferred-runtime-fallback";
const WEBGPU_MSAA_RUNTIME_FALLBACK_KEY = "webgpu-msaa-runtime-fallback-1x";
const WEBGPU_OIT_DISABLED_MRT_KEY = "webgpu-oit-disabled-mrt-unavailable";
const WEBGPU_OIT_DISABLED_MSAA_KEY = "webgpu-oit-disabled-msaa";
const WEBGPU_OIT_DISABLED_RUNTIME_KEY = "webgpu-oit-disabled-runtime";

interface WebGPUFrameScope {
	readonly context: FrameContext;
	readonly configuration: WebGPUFrameConfiguration;
	encoder: ICommandEncoder | null;
	resources: WebGPUPreparedFrameResources | null;
	presented: boolean;
	motionHistoryWriteTarget: IRenderTexture | null;
	deferredOpaqueFrameState: WebGPUDeferredOpaqueFrameState | null;
	hiZStatus: "unavailable" | "pending" | "ready" | "failed";
	hiZBuildCount: number;
}

export interface WebGPUFrameOrchestratorOptions {
	readonly enableEarlyZPrepass: boolean;
	readonly enableDeferredLighting: boolean;
	readonly frameGraphValidationMode: WebGPUFrameGraphValidationMode;
	readonly getFrameExecutor: () => import("../WebGPUFrameExecutor").WebGPUFrameExecutor | null;
}

export class WebGPUFrameOrchestrator {
	private _backend: WebGPUBackend;
	private _resources: WebGPURenderResources;
	private _msaa: WebGPUMSAAContext;
	private _frame: WebGPUFrameScope | null = null;
	private _lastConfiguration: WebGPUFrameConfiguration | null = null;
	private _postRuntime: WebGPUPostProcessRuntime;
	private _postBridge: WebGPUPostProcessBridge;
	private _fallbackPostProcessRuntime?: BackendPostProcessRuntime;
	private _pendingFrameTargetInvalidation = false;
	private _pendingShaderRuntimeInvalidation = false;
	private _enableEarlyZPrepass = true;
	private _enableDeferredLighting = true;
	private readonly _frameGraphValidationMode: WebGPUFrameGraphValidationMode;
	private readonly _getFrameExecutor: WebGPUFrameOrchestratorOptions["getFrameExecutor"];
	private readonly _configurationResolver = new WebGPUFrameConfigurationResolver();
	private readonly _dirtyRectResolver = new WebGPUDirtyRectResolver();
	private _recordingContext: WebGPUFrameGraphRecordingContext;
	private _depthDirtyClearPass: WebGPUDepthDirtyClearPass;
	private _planarReflectionPass: WebGPUPlanarReflectionPass;
	private _presentPass: WebGPUPresentPass;
	private _customRenderTargets: WebGPUCustomRenderTargetRuntime;
	private _frameTargetManager: WebGPUFrameTargetManager;
	private _oitPass: WebGPUOITPass;
	private _deferredLightingPass: WebGPUDeferredLightingPass;
	private _deferredDecalPass: WebGPUDeferredDecalPass;
	private _scenePassRecorder: WebGPUScenePassRecorder;
	private _occlusionRuntime: WebGPUOcclusionCullingRuntime;
	private _hiZBuilder: WebGPUHiZBuilder;
	private readonly _graphPlanner = new WebGPUFrameGraphPlanner();
	private readonly _graphCompiler = new WebGPUFrameGraphCompiler();
	private readonly _nodeExecutors: Map<
		WebGPUFrameGraphNodeKind,
		(node: WebGPUFrameGraphNode, context: FrameContext) => Promise<void>
	>;
	private _lastPlannedGraphNodes: WebGPUFrameGraphNode[] = [];
	private _lastCompiledGraphStages: WebGPUCompiledFrameGraphStage[] = [];
	private _lastExecutedGraphNodeIds: string[] = [];

	constructor(
		backend: WebGPUBackend,
		resources: WebGPURenderResources,
		msaa: WebGPUMSAAContext,
		options: WebGPUFrameOrchestratorOptions,
	) {
		this._backend = backend;
		this._resources = resources;
		this._msaa = msaa;
		this._enableEarlyZPrepass = options.enableEarlyZPrepass;
		this._enableDeferredLighting = options.enableDeferredLighting;
		this._frameGraphValidationMode = options.frameGraphValidationMode;
		this._getFrameExecutor = options.getFrameExecutor;
		const computeFacade = resolveWebGPUComputeFacade(backend);
		this._hiZBuilder = new WebGPUHiZBuilder(computeFacade);
		this._postRuntime = new WebGPUPostProcessRuntime(
			computeFacade,
			(key, message) =>
				Logger.warn(`[${key}] ${message}`, {
					scope: "WebGPUFrameExecutor",
					onceKey: key,
				}),
			resources.sceneFrameLayout,
			this._hiZBuilder,
		);
		this._postBridge = new WebGPUPostProcessBridge(backend, this._postRuntime, {
			getEncoder: () => this._encoder,
			getFrameTargets: () => this._frameTargets,
			isHiZReady: () => this._hiZStatus === "ready",
			requireFrameResources: () => this._requireFrameResources(),
			presentToCanvas: (source) => this._presentToCanvas(source),
			warmupPresent: () => this._ensurePresentResources(),
			setMotionHistoryWriteTarget: (texture) => {
				this._motionHistoryWriteTarget = texture;
			},
		});
		this._planarReflectionPass = new WebGPUPlanarReflectionPass(backend, resources);
		this._presentPass = new WebGPUPresentPass(backend);
		this._customRenderTargets = new WebGPUCustomRenderTargetRuntime(backend);
		this._frameTargetManager = new WebGPUFrameTargetManager(backend);
		this._recordingContext = {
			getEncoder: () => this._encoder,
			getFrameTargets: () => this._frameTargets,
			getMSAATargets: () => this._msaaTargets,
			getTargetWidth: () => this._targetWidth,
			getTargetHeight: () => this._targetHeight,
			getTargetMSAASampleCount: () => this._targetMSAASampleCount,
			getSceneTargetMode: () => this._targetSceneTargetMode,
			isMRTEnabled: () => this._mrtEnabled,
			isEarlyZPrepassEnabled: () => this._enableEarlyZPrepass,
			requireFrameResources: () => this._requireFrameResources(),
			isIncrementalPartial: (context) =>
				this._dirtyRectResolver.isIncrementalPartial(context),
			resolveDirtyRects: (context, width, height) =>
				this._dirtyRectResolver.resolveDirtyRects(context, width, height),
			selectPacketsForRect: (context, packets, rect) =>
				this._dirtyRectResolver.selectPacketsForRect(context, packets, rect),
			selectTransparentSubsetForRect: (context, packets, rect) =>
				this._dirtyRectResolver.selectTransparentSubsetForRect(context, packets, rect),
		};
		this._depthDirtyClearPass = new WebGPUDepthDirtyClearPass(backend);
		this._deferredLightingPass = new WebGPUDeferredLightingPass(backend, resources, {
			recordingContext: this._recordingContext,
		});
		this._deferredDecalPass = new WebGPUDeferredDecalPass(backend, resources, {
			recordingContext: this._recordingContext,
		});
		this._scenePassRecorder = new WebGPUScenePassRecorder(
			backend,
			resources,
			this._recordingContext,
			this._depthDirtyClearPass,
			{
				getGBufferWriteBinding: () => this._deferredLightingPass.getGBufferWriteBinding(),
			},
		);
		this._occlusionRuntime = new WebGPUOcclusionCullingRuntime(backend);
		this._oitPass = new WebGPUOITPass(backend, resources, {
			recordingContext: this._recordingContext,
			recordLegacyMainPass: (context, packets, clear, earlyZ) =>
				this._scenePassRecorder.recordLegacyMainPass(context, packets, clear, earlyZ),
			drawTransmissionFallback: (context, packets) =>
				this._scenePassRecorder.drawTransmissionPackets(context, packets),
			warnDisabled: (key, message) => this._warnOITDisabled(key, message),
		});
		this._nodeExecutors = this._createNodeExecutors();
	}

	private get _encoder(): ICommandEncoder | null {
		return this._frame?.encoder ?? null;
	}

	private set _encoder(value: ICommandEncoder | null) {
		if (!this._frame) {
			if (value !== null) {
				throw new Error("WebGPUFrameOrchestrator cannot assign an encoder outside an active frame.");
			}
			return;
		}
		this._frame.encoder = value;
	}

	private get _frameContext(): FrameContext | null {
		return this._frame?.context ?? null;
	}

	private get _frameResources(): WebGPUPreparedFrameResources | null {
		return this._frame?.resources ?? null;
	}

	private set _frameResources(value: WebGPUPreparedFrameResources | null) {
		if (this._frame) this._frame.resources = value;
	}

	private get _hasPresentedInFrame(): boolean {
		return this._frame?.presented ?? false;
	}

	private set _hasPresentedInFrame(value: boolean) {
		if (this._frame) this._frame.presented = value;
	}

	private get _mrtEnabled(): boolean {
		return this._frame?.configuration.mrtSupported ?? this._lastConfiguration?.mrtSupported ?? true;
	}

	private get _deferredEnabled(): boolean {
		return this._frame?.configuration.deferredActive ?? false;
	}

	private get _oitActive(): boolean {
		return this._frame?.configuration.oitActive ?? false;
	}

	private get _motionHistoryWriteTarget(): IRenderTexture | null {
		return this._frame?.motionHistoryWriteTarget ?? null;
	}

	private set _motionHistoryWriteTarget(value: IRenderTexture | null) {
		if (this._frame) this._frame.motionHistoryWriteTarget = value;
	}

	private get _deferredOpaqueFrameState(): WebGPUDeferredOpaqueFrameState | null {
		return this._frame?.deferredOpaqueFrameState ?? null;
	}

	private set _deferredOpaqueFrameState(value: WebGPUDeferredOpaqueFrameState | null) {
		if (this._frame) this._frame.deferredOpaqueFrameState = value;
	}

	private get _hiZStatus(): "unavailable" | "pending" | "ready" | "failed" {
		return this._frame?.hiZStatus ?? "unavailable";
	}

	private set _hiZStatus(value: "unavailable" | "pending" | "ready" | "failed") {
		if (this._frame) this._frame.hiZStatus = value;
	}

	private get _hiZBuildCount(): number {
		return this._frame?.hiZBuildCount ?? 0;
	}

	private set _hiZBuildCount(value: number) {
		if (this._frame) this._frame.hiZBuildCount = value;
	}

	private get _frameTargets(): WebGPUFrameTargets | null {
		return this._frameTargetManager.frameTargets;
	}

	private get _msaaTargets(): WebGPUFrameMSAATargets | null {
		return this._frameTargetManager.msaaTargets;
	}

	private get _targetWidth(): number {
		return this._frameTargetManager.targetWidth;
	}

	private get _targetHeight(): number {
		return this._frameTargetManager.targetHeight;
	}

	private get _targetMSAASampleCount(): number {
		return this._frameTargetManager.targetMSAASampleCount;
	}

	private get _targetSceneTargetMode(): WebGPUSceneTargetMode {
		return this._frameTargetManager.targetSceneTargetMode;
	}

	public beginFrame(context: FrameContext): void {
		this._frame = null;
		this._oitPass.resetFrameState();
		this._postBridge.clearPendingFrameState();
		this._lastPlannedGraphNodes = [];
		this._lastCompiledGraphStages = [];
		this._lastExecutedGraphNodeIds = [];
		this._graphCompiler.beginFrame([]);
		this._occlusionRuntime.beginFrame(context);
		const targetWidth = this._resolveAttachmentDimension(context.attachments.width);
		const targetHeight = this._resolveAttachmentDimension(context.attachments.height);

		if (targetWidth <= 0 || targetHeight <= 0) {
			this._destroyFrameTargets();
			return;
		}

		const encoder = this._backend.createCommandEncoder();
		this._customRenderTargets.sync(context);
		const configuration = this._configureFrameTargets(context, encoder, targetWidth, targetHeight);
		this._lastConfiguration = configuration;
		this._frame = {
			context,
			configuration,
			encoder,
			resources: null,
			presented: false,
			motionHistoryWriteTarget: null,
			deferredOpaqueFrameState: null,
			hiZStatus: this._frameTargets?.hiZ ? "pending" : "unavailable",
			hiZBuildCount: 0,
		};
		this._graphCompiler.beginFrame(this._collectInitialGraphResources());
		this.prepareFrameResources(context);
	}

	private _configureFrameTargets(
		context: FrameContext,
		encoder: ICommandEncoder,
		width: number,
		height: number,
	): WebGPUFrameConfiguration {
		let forceDeferredFallback = false;
		let forceForwardMrt = false;
		for (let attempts = 0; attempts < 3; attempts++) {
			const configuration = this._resolveFrameConfiguration(
				context,
				encoder,
				forceDeferredFallback,
				forceForwardMrt,
			);
			this._emitConfigurationDiagnostics(configuration.diagnostics);
			if (!configuration.targetRequirements) {
				this._destroyFrameTargets();
				return configuration;
			}
			const result = this._frameTargetManager.ensureFrameTargets({
				width,
				height,
				sampleCount: this._msaa.sampleCount,
				requirements: configuration.targetRequirements,
			});
			if (result.status === "ready") return configuration;
			if (result.status === "retry-legacy-mrt") {
				forceDeferredFallback = true;
				forceForwardMrt = true;
				this._warnFrameTargetRetry(WEBGPU_DEFERRED_RUNTIME_FALLBACK_KEY, result);
				continue;
			}
			if (!this._msaa.fallbackToSingleSample()) {
				throw result.error;
			}
			this._warnFrameTargetRetry(WEBGPU_MSAA_RUNTIME_FALLBACK_KEY, result);
		}
		throw new Error("WebGPU frame target allocation did not converge after fallback.");
	}

	private _resolveFrameConfiguration(
		context: FrameContext,
		encoder: ICommandEncoder,
		forceDeferredFallback: boolean,
		forceForwardMrt: boolean,
	): WebGPUFrameConfiguration {
		const device = this._backend.device;
		return this._configurationResolver.resolve(context, {
			maxColorAttachments: device?.limits?.maxColorAttachments ?? 8,
			maxColorAttachmentBytesPerSample:
				device?.limits?.maxColorAttachmentBytesPerSample ?? 32,
			maxStorageTexturesPerShaderStage:
				device?.limits?.maxStorageTexturesPerShaderStage ?? 4,
		}, {
			enableEarlyZPrepass: this._enableEarlyZPrepass,
			enableDeferredLighting: this._enableDeferredLighting,
			sampleCount: this._msaa.sampleCount,
			supportsInFrameTextureCopy: typeof encoder.copyTextureToTexture === "function",
			forceDeferredFallback,
			forceForwardMrt,
			particleOpaquePackets: this._buildParticleMeshDrawPackets(context, {
				includeOpaque: true,
				includeTransparent: false,
			}),
			particleTransparentPackets: this._buildParticleMeshDrawPackets(context, {
				includeOpaque: false,
				includeTransparent: true,
			}),
		});
	}

	private _emitConfigurationDiagnostics(diagnostics: readonly WebGPUFrameDiagnostic[]): void {
		for (const diagnostic of diagnostics) {
			Logger.warn(`[${diagnostic.code}] ${diagnostic.message}`, {
				scope: "WebGPUFrameOrchestrator",
				onceKey: diagnostic.code,
			});
		}
	}

	private _warnFrameTargetRetry(
		key: string,
		result: Exclude<WebGPUFrameTargetEnsureResult, { status: "ready" }>,
	): void {
		const message = key === WEBGPU_DEFERRED_RUNTIME_FALLBACK_KEY
			? "WebGPU deferred frame target allocation failed; retrying with legacy MRT forward path."
			: `WebGPU ${this._msaa.sampleCount}x MSAA target allocation failed; retrying at 1x.`;
		Logger.warn(`[${key}] ${message} ${String(result.error)}`, {
			scope: "WebGPUFrameOrchestrator",
			onceKey: key,
		});
	}

	/**
	 * Prepares scoped WebGPU frame resources for the active frame.
	 *
	 * @param context Current frame context.
	 * @returns Prepared main-frame resources, or `null` when the frame was
	 * rejected before an encoder was created.
	 * @sideEffects Writes frame uniforms and clustered lighting buffers.
	 */
	public prepareFrameResources(context: FrameContext): WebGPUPreparedFrameResources | null {
		if (!this._frameContext || !this._encoder) {
			this._frameResources = null;
			return null;
		}
		this._frameResources = this._resources.prepareFrame(context, {
			scopeKey: "main",
			sceneTargetMode: this.getSceneTargetModeForFrame(),
		});
		return this._frameResources;
	}

	/**
	 * Returns the resources prepared for the active main frame.
	 *
	 * @returns Main-frame resources, or `null` before preparation/after submit.
	 * @sideEffects None.
	 */
	public getPreparedFrameResources(): WebGPUPreparedFrameResources | null {
		return this._frameResources;
	}

	public createPostProcessResource(
		desc: PostProcessResourceDescriptor,
	): PostProcessResourceHandle {
		return this._postBridge.createResource(desc);
	}

	public destroyPostProcessResource(handle: PostProcessResourceHandle): void {
		this._postBridge.destroyResource(handle);
	}

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return this._postBridge.createGBufferBridge(context);
	}

	public getSceneTargetModeForFrame(): WebGPUSceneTargetMode {
		if (!this._mrtEnabled || !this._frameTargets) {
			return "single";
		}
		return this._targetSceneTargetMode;
	}

	public getOcclusionVisibilityProvider(
		options: NormalizedOcclusionCullingOptions,
	): OcclusionVisibilityProvider {
		return this._occlusionRuntime.getVisibilityProvider(options);
	}

	public resetOcclusionCulling(): void {
		this._occlusionRuntime.resetVisibility();
	}

	public getDebugState(): WebGPUFrameGraphDebugState {
		return {
			active: this._hasActiveFrameState(),
			sceneTargetMode: this.getSceneTargetModeForFrame(),
			deferredActive: this._deferredEnabled,
			oitActive: this._oitActive,
			targetWidth: this._targetWidth,
			targetHeight: this._targetHeight,
			texturePoolOwnerCount: this._frameTargetManager.texturePoolOwnerCount,
			frameTargets: this._frameTargets,
			msaaTargets: this._msaaTargets,
			motionHistoryWriteTarget: this._motionHistoryWriteTarget,
			pendingFrameTargetInvalidation: this._pendingFrameTargetInvalidation,
			pendingShaderRuntimeInvalidation: this._pendingShaderRuntimeInvalidation,
			hiZ: {
				allocated: !!this._frameTargets?.hiZ,
				status: this._hiZStatus,
				width: this._frameTargets?.hiZ?.width ?? 0,
				height: this._frameTargets?.hiZ?.height ?? 0,
				mipLevelCount: this._frameTargets?.hiZ ?
					Math.floor(Math.log2(Math.max(this._frameTargets.hiZ.width, this._frameTargets.hiZ.height))) + 1
				: 0,
				buildCount: this._hiZBuildCount,
			},
			lastPlannedNodeIds: this._lastPlannedGraphNodes.map((node) => node.id),
			lastExecutedNodeIds: this._lastExecutedGraphNodeIds.slice(),
			compiledStages: this._lastCompiledGraphStages.slice(),
			graphResources: this._graphCompiler.getResourceDebugState(),
			graphBarriers: this._graphCompiler.getBarriers(),
			graphDiagnostics: this._graphCompiler.getDiagnostics(),
			targetManager: this._frameTargetManager.getDebugState(),
		};
	}

	private _requireFrameResources(): WebGPUPreparedFrameResources {
		if (!this._frameResources) {
			throw new Error("WebGPUFrameExecutor requires prepared main-frame resources.");
		}
		return this._frameResources;
	}

	public getPassExecutionContext(request: PostProcessPassExecutionContextRequest): unknown {
		return this._postBridge.getPassExecutionContext(request);
	}

	/**
	 * Applies backend-owned output recorded by a completed logical pass.
	 *
	 * @param request Logical pass request that just completed.
	 * @param result Pass execution result.
	 * @returns Nothing.
	 * @sideEffects May replace the active scene color target after validating
	 * the published texture belongs to this frame's WebGPU post-process targets.
	 */
	public completePostProcessPass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult,
	): void {
		this._postBridge.completePass(request, result);
	}

	/**
	 * Force frame targets to be rebuilt on the next beginFrame().
	 * Call on canvas resize so the post-process pipeline picks up
	 * the new dimensions.
	 */
	public invalidateFrameTargets(): void {
		if (this._hasActiveFrameState()) {
			this._pendingFrameTargetInvalidation = true;
			return;
		}
		this._invalidateFrameTargetsNow();
	}

	private _invalidateFrameTargetsNow(): void {
		this._destroyFrameTargets();
		this._postRuntime.invalidateBindings();
		this._hiZBuilder.invalidateBindings();
		this._occlusionRuntime.invalidateFrameResources();
		this._planarReflectionPass.destroy();
	}

	public invalidatePostProcessBindings(): void {
		this._postRuntime.invalidateBindings();
	}

	public onShaderRuntimeChanged(): void {
		if (this._hasActiveFrameState()) {
			this._pendingShaderRuntimeInvalidation = true;
			return;
		}
		this._applyShaderRuntimeChangedNow();
	}

	private _applyShaderRuntimeChangedNow(): void {
		this._presentPass.onShaderRuntimeChanged();
		this._oitPass.onShaderRuntimeChanged();
		this._destroyDeferredBindings();
		this._depthDirtyClearPass.onShaderRuntimeChanged();
		this._postRuntime.onShaderRuntimeChanged();
		this._hiZBuilder.invalidateShaderResources();
		this._occlusionRuntime.onShaderRuntimeChanged();
		this._planarReflectionPass.destroy();
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions = {},
	): Promise<WarmupPhaseCounters> {
		let total = 1;
		let compiled = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];
		const yieldController = createWarmupYieldController(options);
		try {
			await this._ensurePresentResources();
			compiled++;
		} catch (error) {
			failed++;
			errors.push(toShaderCompileError(error, "webgpu", "WebGPUPresentWarmup"));
		}
		await yieldController.yieldIfNeeded();

		const postRuntime =
			this._backend.postProcessRuntime ?? this._getFallbackPostProcessRuntime();
		const warmupGraph = postRuntime.compileWarmupGraph(context);
		const hints = new Set<string>();
		if (plan.includePostProcess) {
			for (const passId of plan.postProcessPasses) {
				const compiledPass = warmupGraph.passes.find((p) => p.id === passId);
				const implementation = compiledPass?.implementation;
				for (const hint of implementation?.metadata?.warmupHints ?? []) {
					hints.add(hint);
				}
			}
		}
		if (hints.size > 0) {
			total += hints.size;
			const postWarmup = await this._postRuntime.warmupHints(Array.from(hints));
			compiled += postWarmup.compiled;
			failed += postWarmup.failed;
			if (postWarmup.errors.length > 0) {
				errors.push(...postWarmup.errors);
			}
			await yieldController.yieldIfNeeded();
		}

		const warmedPassImplementations = new Set<string>();
		for (const passId of plan.postProcessPasses) {
			if (warmedPassImplementations.has(passId)) {
				continue;
			}
			const compiledPass = warmupGraph.passes.find((p) => p.id === passId);
			const implementation = compiledPass?.implementation;
			if (typeof implementation?.warmup !== "function") {
				continue;
			}
			warmedPassImplementations.add(passId);
			total++;
			try {
				const warmupContext = this._getPassWarmupExecutionContext(implementation);
				await implementation.warmup(warmupContext, {
					frameContext: context,
					postProcess: context.postProcess,
					backend: "webgpu",
					context: warmupContext,
					options: compiledPass?.options,
				});
				compiled++;
			} catch (error) {
				failed++;
				errors.push(toShaderCompileError(error, "webgpu", `WebGPUPostWarmup:${passId}`));
			}
			await yieldController.yieldIfNeeded();
		}

		return {
			phase: "webgpu-frame",
			total,
			compiled,
			skipped: Math.max(0, total - compiled - failed),
			failed,
			errors,
		};
	}

	private _getPassWarmupExecutionContext(implementation: PostProcessPassImplementation): unknown {
		const metadata = implementation.metadata?.context;
		if (!isWebGPUPostProcessContextMetadata(metadata)) {
			return undefined;
		}
		return this._postBridge.getPassWarmupExecutionContext(metadata);
	}

	private _getFallbackPostProcessRuntime(): BackendPostProcessRuntime {
		if (!this._fallbackPostProcessRuntime) {
			this._fallbackPostProcessRuntime = new BackendPostProcessRuntime({
					executor: new WebGPUPostProcessExecutor({
					getFrameExecutor: () => this._getFrameExecutor(),
					assertDeviceOperational: () => {},
				}),
				backend: this._backend,
				warn: () => {},
			});
		}
		return this._fallbackPostProcessRuntime;
	}

	/**
	 * Release all GPU resources held by this executor.
	 */
	public destroy(): void {
		this._destroyFrameTargets();
		this._destroyTexturePools();
		this._postRuntime.destroy();
		this._occlusionRuntime.destroy();
		this._hiZBuilder.destroy();
		this._planarReflectionPass.destroy();
		this._presentPass.destroy();
		this._customRenderTargets.destroy();
		this._oitPass.destroy();
		this._depthDirtyClearPass.destroy();
		this._pendingFrameTargetInvalidation = false;
		this._pendingShaderRuntimeInvalidation = false;
		this._clearActiveFrameState(false);
	}

	public async executePass(pass: FramePass, context: FrameContext): Promise<void> {
		if (!this._encoder) return;

		if (this._customRenderTargets.hasPass(pass, context)) {
			await this._customRenderTargets.executePass(pass, context, this._encoder);
			return;
		}

		const plan = this._graphPlanner.planStage(pass, context, {
			deferredActive: this._deferredEnabled,
			oitActive: this._oitActive,
			sceneTargetMode: this.getSceneTargetModeForFrame(),
			hasFrameTargets: !!this._frameTargets,
			hasMSAATargets: !!this._msaaTargets,
			needsTransmissionTargets: !!this._frameTargets?.transmissionSceneColorCopy,
			needsPlanarReflectionMask: !!this._frameTargets?.planarReflectionMask,
			needsOcclusionTest: this._frame?.configuration.needsOcclusionTest === true,
			needsHiZBuild: this._frame?.configuration.needsHiZBuild === true && this._hiZStatus === "pending",
		});
		if (plan.nodes.length === 0) {
			const key = `webgpu-pass-unsupported-${pass.stage}`;
			Logger.warn(
				`[${key}] WebGPU backend does not support pass "${pass.stage}" yet; skipping`,
				{ scope: "WebGPUFrameOrchestrator", onceKey: key },
			);
			return;
		}
		const compiled = this._graphCompiler.compileStage(plan);
		this._handleGraphDiagnostics(compiled);
		this._lastCompiledGraphStages = this._graphCompiler.getCompiledStages().slice();
		this._lastPlannedGraphNodes = [...plan.nodes];
		for (const node of plan.nodes) {
			await this._executeGraphNode(node, context);
			this._lastExecutedGraphNodeIds.push(node.id);
		}
	}

	private async _executeGraphNode(
		node: WebGPUFrameGraphNode,
		context: FrameContext,
	): Promise<void> {
		const executor = this._nodeExecutors.get(node.kind);
		if (!executor) {
			throw new Error(`WebGPU frame graph node kind "${node.kind}" has no executor.`);
		}
		await executor(node, context);
	}

	private _createNodeExecutors(): Map<
		WebGPUFrameGraphNodeKind,
		(node: WebGPUFrameGraphNode, context: FrameContext) => Promise<void>
	> {
		return new Map([
			[
				"shadow",
				async (_node, context) => {
					await this._resources.renderShadows(context, this._encoder ?? undefined);
				},
			],
			[
				"paged-shadow-page-mark",
				async (_node, context) => {
					const request = this._createPagedShadowRequest(context);
					this._resources.preparePagedShadowFrame(request);
					await this._resources.recordPagedShadowPageMarkPass(request);
				},
			],
			[
				"paged-shadow-page-allocate",
				async (_node, context) => {
					await this._resources.recordPagedShadowPageAllocationPass(
						this._createPagedShadowRequest(context),
					);
				},
			],
			[
				"paged-shadow-page-table-copy",
				async (_node, context) => {
					await this._resources.recordPagedShadowPageTableCopyPass(
						this._createPagedShadowRequest(context),
					);
				},
			],
			[
				"paged-shadow-depth",
				async (_node, context) => {
					await this._resources.recordPagedShadowDepthPass(
						this._createPagedShadowRequest(context),
					);
				},
			],
			[
				"paged-shadow-feedback",
				async (_node, context) => {
					await this._resources.recordPagedShadowFeedbackPass(
						this._createPagedShadowRequest(context),
					);
				},
			],
			[
				"planar-reflection-capture",
				async (_node, context) => {
					await this._recordPlanarReflectionPass(context);
				},
			],
			[
				"opaque-scene",
				async (_node, context) => {
					this._deferredOpaqueFrameState = await this._scenePassRecorder.recordOpaque(
						context,
						this._deferredEnabled,
					);
					if (!this._deferredOpaqueFrameState) {
						await this._recordPlanarReflectionComposite(context);
					}
				},
			],
			[
				"deferred-decal",
				async (_node, context) => {
					await this._recordDeferredDecalNode(context);
				},
			],
			[
				"deferred-lighting",
				async (_node, context) => {
					await this._recordDeferredLightingNode(context);
				},
			],
			[
				"hiz-build",
				async (_node, context) => {
					await this._recordHiZBuildNode(context);
				},
			],
			[
				"occlusion-test",
				async (_node, context) => {
					await this._recordOcclusionTestNode(context);
				},
			],
			[
				"oit-transparent",
				async (_node, context) => {
					await this._recordOITTransparentPass(context);
				},
			],
			[
				"transparent-scene",
				async (_node, context) => {
					await this._recordTransparentScenePass(context);
				},
			],
			[
				"oit-particles",
				async (_node, context) => {
					await this._recordOITParticlePass(context);
				},
			],
			[
				"particles",
				async (_node, context) => {
					await this._scenePassRecorder.recordParticlePass(context);
				},
			],
		]);
	}

	private _createPagedShadowRequest(context: FrameContext): WebGPUPagedShadowFrameRequest {
		const frameTargets = this._frameTargets;
		return {
			context,
			encoder: this._encoder,
			renderSets: context.shadowMaps,
			shadowCasterPackets: context.scene.shadowCasterPackets,
			shadowTransmitterPackets: context.scene.shadowTransmitterPackets,
			feedbackDepthTexture: frameTargets?.depth ?? null,
			feedbackMotionDepthTexture: frameTargets?.gMotionDepth ?? null,
		};
	}

	public async endFrame(): Promise<void> {
		if (!this._encoder) {
			this._clearActiveFrameState();
			return;
		}

		const encoder = this._encoder;
		const width = this._targetWidth;
		const height = this._targetHeight;
		const motionSource =
			this._mrtEnabled && this._motionHistoryWriteTarget
				? this._frameTargets?.gMotionDepth
				: null;
		const motionTarget = this._mrtEnabled ? this._motionHistoryWriteTarget : null;

		try {
			if (this._mrtEnabled && this._frameTargets && !this._hasPresentedInFrame) {
				await this._presentToCanvas(this._frameTargets.sceneColor);
			}

			this._backend.submit([encoder.finish()]);
			this._customRenderTargets.markFrameCommitted();
			this._occlusionRuntime.scheduleQueuedReadbacks();
			if (motionSource && motionTarget && width > 0 && height > 0) {
				this._backend.copyTextureToTexture(
					{ texture: motionSource },
					{ texture: motionTarget },
					{ width, height, depthOrArrayLayers: 1 },
				);
			}
		} finally {
			this._clearActiveFrameState();
		}
	}

	public abortFrame(): void {
		this._customRenderTargets.markFrameAborted();
		this._clearActiveFrameState();
	}

	public readRenderTargetColor(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions,
	): Promise<TextureReadbackResult> {
		return this._customRenderTargets.readColor(id, attachmentIndex, options);
	}

	private _collectInitialGraphResources(): string[] {
		const resources = new Set<string>(["canvas:scene-color-main", "canvas:depth"]);
		const targets = this._frameTargets;
		if (targets) {
			resources.add("frame:scene-color-main");
			resources.add("frame:depth");
			if (targets.postPing) resources.add("post:ping");
			if (targets.postPong) resources.add("post:pong");
			if (targets.hiZ) resources.add("frame:hiz");
			if (targets.gAlbedoAlpha) resources.add("gbuffer:albedo-alpha");
			if (targets.gNormalRoughMetal) {
				resources.add("gbuffer:normal-rough-metal");
			}
			if (targets.gEmissiveOcclusion) {
				resources.add("gbuffer:emissive-occlusion");
			}
			if (targets.gMotionDepth) resources.add("gbuffer:motion-depth");
			if (targets.gSpecular) resources.add("gbuffer:specular");
			if (targets.gCoatSheen) resources.add("gbuffer:coat-sheen");
			if (targets.gSheenReflectance) {
				resources.add("gbuffer:sheen-reflectance");
			}
			if (targets.gMaterialExt0) resources.add("gbuffer:material-ext0");
			if (targets.gMaterialExt1) resources.add("gbuffer:material-ext1");
			if (targets.gMaterialExt2) resources.add("gbuffer:material-ext2");
			if (targets.gMaterialExt3) resources.add("gbuffer:material-ext3");
			if (targets.oitAccum) resources.add("oit:accum");
			if (targets.oitReveal) resources.add("oit:reveal");
			if (targets.oitSceneColorCopy) resources.add("oit:scene-color-copy");
			if (targets.transmissionSceneColorCopy) {
				resources.add("transmission:scene-color-copy");
			}
			if (targets.transmissionLighting) {
				resources.add("transmission:lighting");
			}
			if (targets.gTransmissionSurface0) {
				resources.add("transmission:surface0");
			}
			if (targets.gTransmissionSurface1) {
				resources.add("transmission:surface1");
			}
			if (targets.gTransmissionSurface2) {
				resources.add("transmission:surface2");
			}
			if (targets.transmissionDepth) resources.add("transmission:depth");
			if (targets.planarReflectionMask) {
				resources.add("planar-reflection:mask");
			}
		}
		if (this._msaaTargets) {
			resources.add("msaa:scene-color-main");
			resources.add("msaa:depth");
			if (this._msaaTargets.gAlbedoAlpha) {
				resources.add("msaa:gbuffer:albedo-alpha");
			}
			if (this._msaaTargets.gNormalRoughMetal) {
				resources.add("msaa:gbuffer:normal-rough-metal");
			}
			if (this._msaaTargets.gEmissiveOcclusion) {
				resources.add("msaa:gbuffer:emissive-occlusion");
			}
			if (this._msaaTargets.gMotionDepth) {
				resources.add("msaa:gbuffer:motion-depth");
			}
			if (this._msaaTargets.planarReflectionMask) {
				resources.add("msaa:planar-reflection:mask");
			}
		}
		return Array.from(resources);
	}

	private _handleGraphDiagnostics(compiled: WebGPUCompiledFrameGraphStage): void {
		const errors = compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error");
		if (errors.length <= 0) {
			return;
		}
		const message =
			`WebGPU internal frame graph validation failed for ` +
			`stage "${compiled.pass.stage}": ` +
			errors.map((diagnostic) => diagnostic.message).join(" ");
		if (this._frameGraphValidationMode === "throw") {
			throw new Error(message);
		}
		Logger.warn(`[webgpu-frame-graph-validation] ${message}`, {
			scope: "WebGPUFrameOrchestrator",
			onceKey: `webgpu-frame-graph-validation:${compiled.pass.stage}`,
		});
	}

	private _warnOITDisabled(key: string, message: string): void {
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGPUFrameExecutor",
			onceKey: key,
		});
	}

	private _destroyFrameTargets(): void {
		this._frameTargetManager.destroyFrameTargets();
		this._presentPass.invalidateBindings();
		this._oitPass.invalidateBindings();
		this._destroyDeferredBindings();
		this._oitPass.resetFrameState();
		this._motionHistoryWriteTarget = null;
		this._postBridge.clearPendingFrameState();
		this._deferredOpaqueFrameState = null;
		this._occlusionRuntime.invalidateFrameResources();
	}

	private _clearActiveFrameState(flushPendingLifecycle = true): void {
		this._frame = null;
		this._oitPass.resetFrameState();
		if (flushPendingLifecycle) {
			this._flushPendingLifecycleInvalidations();
		}
	}

	private _flushPendingLifecycleInvalidations(): void {
		const applyShaderRuntimeInvalidation = this._pendingShaderRuntimeInvalidation;
		const applyFrameTargetInvalidation = this._pendingFrameTargetInvalidation;
		this._pendingShaderRuntimeInvalidation = false;
		this._pendingFrameTargetInvalidation = false;
		if (applyShaderRuntimeInvalidation) {
			this._applyShaderRuntimeChangedNow();
		}
		if (applyFrameTargetInvalidation) {
			this._invalidateFrameTargetsNow();
		}
	}

	private _hasActiveFrameState(): boolean {
		return (
			this._encoder !== null || this._frameContext !== null || this._frameResources !== null
		);
	}

	private _resolveAttachmentDimension(value: number): number {
		if (!Number.isFinite(value)) {
			return 0;
		}
		return Math.max(0, Math.floor(value));
	}

	private _destroyTexturePools(): void {
		this._frameTargetManager.destroyTexturePools();
	}

	private _destroyDeferredBindings(): void {
		this._deferredLightingPass.destroyBindings();
		this._deferredDecalPass.destroyBindings();
	}

	private async _ensurePresentResources(): Promise<void> {
		await this._presentPass.warmup();
	}

	private async _presentToCanvas(source: IRenderTexture): Promise<void> {
		if (!this._encoder) return;
		await this._presentPass.present({
			encoder: this._encoder,
			frameContext: this._frameContext,
			source,
			resolveDirtyRects: (context, width, height) =>
				this._recordingContext.resolveDirtyRects(context, width, height),
		});
		this._hasPresentedInFrame = true;
	}

	private async _recordTransparentScenePass(context: FrameContext): Promise<void> {
		const transparentPackets = [
			...context.scene.transparentPackets,
			...this._buildParticleMeshDrawPackets(context, {
				includeOpaque: false,
				includeTransparent: true,
			}),
		];
		if (!this._frameTargets?.transmissionSceneColorCopy) {
			await this._scenePassRecorder.recordMainPass(context, transparentPackets, false, false);
			return;
		}
		const opaqueTransparentPackets = transparentPackets.filter(
			(packet) => !materialUsesTransmission(packet.material),
		);
		const transmissionPackets = transparentPackets.filter((packet) =>
			materialUsesTransmission(packet.material),
		);
		if (opaqueTransparentPackets.length > 0) {
			await this._scenePassRecorder.recordMainPass(
				context,
				opaqueTransparentPackets,
				false,
				false,
			);
		}
		await this._scenePassRecorder.drawTransmissionPackets(context, transmissionPackets);
	}

	private async _recordOITTransparentPass(context: FrameContext): Promise<void> {
		if (!this._encoder) {
			return;
		}
		const transparentPackets = [
			...context.scene.transparentPackets,
			...this._buildParticleMeshDrawPackets(context, {
				includeOpaque: false,
				includeTransparent: true,
			}),
		];
		if (!this._mrtEnabled || !this._frameTargets) {
			await this._scenePassRecorder.recordMainPass(context, transparentPackets, false, false);
			return;
		}
		await this._oitPass.recordTransparentPass(context, transparentPackets);
	}

	private async _recordOITParticlePass(context: FrameContext): Promise<void> {
		if (!this._encoder) {
			return;
		}
		if (!this._mrtEnabled || !this._frameTargets?.oitAccum || !this._frameTargets.oitReveal) {
			await this._scenePassRecorder.recordParticlePass(context);
			return;
		}
		await this._oitPass.recordParticlePass(context);
	}

	private _buildParticleMeshDrawPackets(
		context: FrameContext,
		options: {
			includeOpaque?: boolean;
			includeTransparent?: boolean;
		},
	): DrawPacket[] {
		const resources = this._resources as WebGPURenderResources & {
			buildParticleMeshDrawPackets?: (
				context: FrameContext,
				options: {
					includeOpaque?: boolean;
					includeTransparent?: boolean;
				},
			) => DrawPacket[];
		};
		return resources.buildParticleMeshDrawPackets?.(context, options) ?? [];
	}

	private async _recordPlanarReflectionPass(context: FrameContext): Promise<void> {
		if (!this._encoder) {
			return;
		}
		this._submitCurrentFrameEncoder();
		await this._planarReflectionPass.capture(context);
		this._encoder = this._backend.createCommandEncoder();
	}

	private async _recordPlanarReflectionComposite(context: FrameContext): Promise<void> {
		if (!this._encoder || !this._mrtEnabled || !this._frameTargets) {
			return;
		}
		this._clearPlanarReflectionMask();
		await this._planarReflectionPass.composite({
			encoder: this._encoder,
			context,
			frameResources: this._requireFrameResources(),
			frameTargets: this._frameTargets,
			msaaTargets: this._msaaTargets as WebGPUPlanarReflectionMSAATargets | null,
		});
	}

	private _clearPlanarReflectionMask(): void {
		const mask = this._frameTargets?.planarReflectionMask;
		if (!this._encoder || !mask) {
			return;
		}
		this._encoder.beginRenderPass({
			label: "WebGPUPlanarReflectionMaskClear",
			colorAttachments: [
				{
					view: mask,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
		});
		this._encoder.endRenderPass();
	}

	private _submitCurrentFrameEncoder(): void {
		if (!this._encoder) {
			return;
		}
		const encoder = this._encoder;
		this._backend.submit([encoder.finish()]);
		this._encoder = null;
	}

	private async _recordDeferredDecalNode(context: FrameContext): Promise<void> {
		if (!this._deferredOpaqueFrameState?.lightingEnabled) {
			return;
		}
		await this._deferredDecalPass.recordDecalPass(context);
	}

	private async _recordDeferredLightingNode(context: FrameContext): Promise<void> {
		const state = this._deferredOpaqueFrameState;
		if (!state) {
			return;
		}
		try {
			if (state.lightingEnabled) {
				await this._recordDeferredLightingPass(context, state.clearSceneColor);
			}
			if (state.fallbackPackets.length > 0) {
				await this._scenePassRecorder.recordMainPass(
					context,
					state.fallbackPackets,
					false,
					false,
				);
			}
			await this._recordPlanarReflectionComposite(context);
		} finally {
			this._deferredOpaqueFrameState = null;
		}
	}

	private async _recordDeferredLightingPass(
		context: FrameContext,
		clearSceneColor: boolean,
	): Promise<void> {
		await this._deferredLightingPass.recordLightingPass(context, clearSceneColor);
	}

	private async _recordHiZBuildNode(_context: FrameContext): Promise<void> {
		if (!this._encoder || !this._frameTargets?.gMotionDepth || !this._frameTargets.hiZ) {
			this._hiZStatus = "unavailable";
			return;
		}
		try {
			await this._hiZBuilder.build({
				encoder: this._encoder,
				depth: this._frameTargets.gMotionDepth,
				hiZ: this._frameTargets.hiZ,
			});
			this._hiZStatus = "ready";
			this._hiZBuildCount++;
		} catch (error) {
			this._hiZStatus = "failed";
			Logger.warn(
				`[webgpu-hiz-build-failed] Shared WebGPU Hi-Z build failed; dependent effects will be skipped. ${String(error)}`,
				{ scope: "WebGPUFrameExecutor", onceKey: "webgpu-hiz-build-failed" },
			);
		}
	}

	private async _recordOcclusionTestNode(context: FrameContext): Promise<void> {
		if (!this._encoder || this._hiZStatus !== "ready" || !this._frameTargets?.hiZ) {
			return;
		}
		await this._occlusionRuntime.recordVisibilityPass({
			context,
			encoder: this._encoder,
			hiZ: this._frameTargets.hiZ,
			options: normalizeOcclusionCullingOptions(context.features.occlusionCullingOptions),
		});
	}
}
