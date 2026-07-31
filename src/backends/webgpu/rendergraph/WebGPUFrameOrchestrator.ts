import type { FramePreparationRequirements } from "../../../pipeline/FrameRequirements";
import type { DrawPacket, FrameContext, FramePass } from "../../../pipeline/types";
import type {
	LogicalGBufferBridge,
	PostProcessPassExecutionContextRequest,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessPassCompletion,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../../postprocess";
import type { ICommandEncoder } from "../../ICommandEncoder";
import type { IRenderTexture } from "../../types";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import type { WebGPUMSAAContext } from "../WebGPUMSAAController";
import type {
	WebGPUFrameResourceScope,
	WebGPUPreparedFrameResources,
} from "../WebGPUResourceContracts";
import type { WebGPUFrameServiceOwner } from "../WebGPUFrameServiceOwner";
import { WebGPUHiZBuilder } from "../WebGPUHiZBuilder";
import type {
	BackendPostProcessRuntime,
	PostProcessExecutionPlan,
	PostProcessRenderGraphFrame,
} from "../../../postprocess/BackendPostProcessRuntime";
import type { PostProcessPlan } from "../../../postprocess/PostProcessPlanner";
import type { PostProcessColorDomain } from "../../../postprocess/PostProcessPass";
import type {
	RenderGraphDiagnostic,
	RenderGraphResourceDescriptor,
} from "../../../rendergraph/types";
import { createWebGPUPostProcessGraphComposition } from "./WebGPUPostProcessGraphAdapter";

import {
	type WebGPUFrameTargets,
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
import {
	WebGPUFrameNodeExecutorRegistry,
} from "./WebGPUFrameNodeExecutorRegistry";
import {
	WebGPUDeferredNodeRuntime,
	WebGPUPresentationNodeRuntime,
	WebGPUPostProcessNodeRuntime,
	WebGPUReflectionNodeRuntime,
	WebGPUSceneNodeRuntime,
	WebGPUShadowNodeRuntime,
	WebGPUVisibilityNodeRuntime,
	type WebGPUFrameNodeRuntime,
} from "./WebGPUFrameNodeRuntimes";
import { WebGPUFrameSession } from "./WebGPUFrameSession";
import { WebGPUFrameCommitter, type WebGPUFrameCommitDebugState } from "./WebGPUFrameCommitter";
import { WebGPUFrameFeatureAnalyzer, type WebGPUFrameFeatureAnalysis } from "./WebGPUFrameFeatureAnalyzer";
import {
	collectActiveWebGPUFrameGraphResources,
	collectWebGPUFrameGraphResourceCatalog,
	WEBGPU_FRAME_GRAPH_RESOURCES,
} from "./WebGPUFrameGraphResourceCatalog";
import { WebGPUPostProcessBridge } from "./WebGPUPostProcessBridge";
import { WebGPUTransparencyRuntime } from "./WebGPUTransparencyRuntime";
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
	WebGPUFrameGraphFramePlan,
	WebGPUFrameGraphDebugState,
	WebGPUFrameGraphNode,
	WebGPUFrameGraphPlannerState,
	WebGPUFrameGraphStagePlan,
	WebGPUFrameGraphValidationMode,
} from "./types";
import { WebGPUPresentPass } from "./WebGPUPresentPass";
import { WebGPUCustomRenderTargetRuntime } from "./WebGPUCustomRenderTargetRuntime";
import type {
	RenderTargetReadbackOptions,
	RenderTargetReadbackResult,
} from "../../../rendering/CustomRenderTargets";
import type { WebGPUPostProcessSessionPort } from "../WebGPUPostProcessExecutor";
import { SINGLE_SAMPLE_WEBGPU_MSAA_CONTEXT } from "../WebGPUMSAAController";

const WEBGPU_DEFERRED_RUNTIME_FALLBACK_KEY = "webgpu-deferred-runtime-fallback";
const WEBGPU_MSAA_RUNTIME_FALLBACK_KEY = "webgpu-msaa-runtime-fallback-1x";
const WEBGPU_OIT_DISABLED_MRT_KEY = "webgpu-oit-disabled-mrt-unavailable";
const WEBGPU_OIT_DISABLED_MSAA_KEY = "webgpu-oit-disabled-msaa";
const WEBGPU_OIT_DISABLED_RUNTIME_KEY = "webgpu-oit-disabled-runtime";

export interface WebGPUFrameOrchestratorOptions {
	readonly enableEarlyZPrepass: boolean;
	readonly enableDeferredLighting: boolean;
	readonly frameGraphValidationMode: WebGPUFrameGraphValidationMode;
}

export class WebGPUFrameOrchestrator {
	private _host: WebGPUFrameHost;
	private _resources: WebGPUFrameServiceOwner;
	private readonly _mainFrameScope: WebGPUFrameResourceScope;
	private _msaa: WebGPUMSAAContext;
	private _session: WebGPUFrameSession | null = null;
	private _lastConfiguration: WebGPUFrameConfiguration | null = null;
	private _postRuntime: WebGPUPostProcessRuntime;
	private readonly _postProcessRuntime: BackendPostProcessRuntime;
	private _postBridge: WebGPUPostProcessBridge;
	private _pendingFrameTargetInvalidation = false;
	private _pendingShaderRuntimeInvalidation = false;
	private _enableEarlyZPrepass = true;
	private _enableDeferredLighting = true;
	private readonly _frameGraphValidationMode: WebGPUFrameGraphValidationMode;
	private readonly _configurationResolver = new WebGPUFrameConfigurationResolver();
	private readonly _featureAnalyzer = new WebGPUFrameFeatureAnalyzer();
	private readonly _dirtyRectResolver = new WebGPUDirtyRectResolver();
	private _recordingContext: WebGPUFrameGraphRecordingContext;
	private _depthDirtyClearPass: WebGPUDepthDirtyClearPass;
	private _planarReflectionPass: WebGPUPlanarReflectionPass;
	private _presentPass: WebGPUPresentPass;
	private _customRenderTargets: WebGPUCustomRenderTargetRuntime;
	private _frameTargetManager: WebGPUFrameTargetManager;
	private _transparencyRuntime: WebGPUTransparencyRuntime;
	private _deferredLightingPass: WebGPUDeferredLightingPass;
	private _deferredDecalPass: WebGPUDeferredDecalPass;
	private _scenePassRecorder: WebGPUScenePassRecorder;
	private _occlusionRuntime: WebGPUOcclusionCullingRuntime;
	private _hiZBuilder: WebGPUHiZBuilder;
	private readonly _graphPlanner = new WebGPUFrameGraphPlanner();
	private readonly _graphCompiler = new WebGPUFrameGraphCompiler();
	private readonly _nodeExecutors: WebGPUFrameNodeExecutorRegistry;
	private readonly _nodeRuntimes: readonly WebGPUFrameNodeRuntime[];
	private _lastPlannedGraphNodes: WebGPUFrameGraphNode[] = [];
	private _lastCompiledGraphStages: WebGPUCompiledFrameGraphStage[] = [];
	private _lastExecutedGraphNodeIds: string[] = [];
	private _lastCommitDebugState: WebGPUFrameCommitDebugState | null = null;
	private _wholeFrameGraphCompiled = false;
	private _postProcessGraphFrame: PostProcessRenderGraphFrame | null = null;
	private _postProcessOutputColor: string = WEBGPU_FRAME_GRAPH_RESOURCES.frameColor;
	private _postProcessOutputColorDomain: PostProcessColorDomain =
		"scene-linear-hdr";
	private readonly _graphPhysicalResources = new Map<string, IRenderTexture>();

	constructor(
		host: WebGPUFrameHost,
		resources: WebGPUFrameServiceOwner,
		msaa: WebGPUMSAAContext = SINGLE_SAMPLE_WEBGPU_MSAA_CONTEXT,
		options: WebGPUFrameOrchestratorOptions = {
			enableEarlyZPrepass: host.enableEarlyZPrepass,
			enableDeferredLighting: host.enableDeferredLighting,
			frameGraphValidationMode: host.frameGraphValidationMode,
		},
	) {
		this._host = host;
		this._resources = resources;
		this._mainFrameScope = resources.createFrameScope();
		this._msaa = msaa;
		this._enableEarlyZPrepass = options.enableEarlyZPrepass;
		this._enableDeferredLighting = options.enableDeferredLighting;
		this._frameGraphValidationMode = options.frameGraphValidationMode;
		this._postProcessRuntime = host.postProcessRuntime;
		const computeFacade = host.computeFacade;
		this._hiZBuilder = new WebGPUHiZBuilder(computeFacade);
		this._postRuntime = new WebGPUPostProcessRuntime(
			computeFacade,
			(key, message) =>
				Logger.warn(`[${key}] ${message}`, {
					scope: "WebGPUFrameOrchestrator",
					onceKey: key,
				}),
			resources.sceneFrameLayout,
			this._hiZBuilder,
			() => this._host.displayOutputState,
		);
		this._postBridge = new WebGPUPostProcessBridge(host, this._postRuntime, {
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
		this._planarReflectionPass = new WebGPUPlanarReflectionPass(host, resources);
		this._presentPass = new WebGPUPresentPass(host);
		this._customRenderTargets = new WebGPUCustomRenderTargetRuntime(host);
		this._frameTargetManager = new WebGPUFrameTargetManager(host);
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
		this._depthDirtyClearPass = new WebGPUDepthDirtyClearPass(host);
		this._deferredLightingPass = new WebGPUDeferredLightingPass(host, resources, {
			recordingContext: this._recordingContext,
		});
		this._deferredDecalPass = new WebGPUDeferredDecalPass(host, resources, {
			recordingContext: this._recordingContext,
		});
		this._scenePassRecorder = new WebGPUScenePassRecorder(
			host,
			resources,
			this._recordingContext,
			this._depthDirtyClearPass,
			{
				getGBufferWriteBinding: () => this._deferredLightingPass.getGBufferWriteBinding(),
			},
		);
		this._occlusionRuntime = new WebGPUOcclusionCullingRuntime(host);
		this._transparencyRuntime = new WebGPUTransparencyRuntime(
			host,
			resources,
			this._recordingContext,
			this._scenePassRecorder,
			{
				warnOnce: (code, message, cause) => this._warnFrameDiagnostic(code, message, cause),
			},
		);
		this._nodeRuntimes = this._createNodeRuntimes();
		this._nodeExecutors = WebGPUFrameNodeExecutorRegistry.fromRuntimes(this._nodeRuntimes);
	}

	private get _encoder(): ICommandEncoder | null {
		return this._session?.encoder ?? null;
	}

	private set _encoder(value: ICommandEncoder | null) {
		if (!this._session) {
			if (value !== null) {
				throw new Error(
					"WebGPUFrameOrchestrator cannot assign an encoder outside an active frame.",
				);
			}
			return;
		}
		this._session.encoder = value;
	}

	private get _frameContext(): FrameContext | null {
		return this._session?.context ?? null;
	}

	private get _frameResources(): WebGPUPreparedFrameResources | null {
		return this._session?.resources ?? null;
	}

	private set _frameResources(value: WebGPUPreparedFrameResources | null) {
		if (this._session) this._session.resources = value;
	}

	private get _hasPresentedInFrame(): boolean {
		return this._session?.presented ?? false;
	}

	private set _hasPresentedInFrame(value: boolean) {
		if (this._session) this._session.presented = value;
	}

	private get _mrtEnabled(): boolean {
		return (
			this._session?.configuration?.mrtSupported ??
			this._lastConfiguration?.mrtSupported ??
			true
		);
	}

	private get _deferredEnabled(): boolean {
		return this._session?.configuration?.deferredActive ?? false;
	}

	private get _oitActive(): boolean {
		return this._session?.configuration?.oitActive ?? false;
	}

	private get _motionHistoryWriteTarget(): IRenderTexture | null {
		return this._session?.motionHistoryWriteTarget ?? null;
	}

	private set _motionHistoryWriteTarget(value: IRenderTexture | null) {
		if (this._session) this._session.motionHistoryWriteTarget = value;
	}

	private get _deferredOpaqueFrameState(): WebGPUDeferredOpaqueFrameState | null {
		return this._session?.deferredOpaqueFrameState ?? null;
	}

	private set _deferredOpaqueFrameState(value: WebGPUDeferredOpaqueFrameState | null) {
		if (this._session) this._session.deferredOpaqueFrameState = value;
	}

	private get _hiZStatus(): "unavailable" | "pending" | "ready" | "failed" {
		return this._session?.hiZStatus ?? "unavailable";
	}

	private set _hiZStatus(value: "unavailable" | "pending" | "ready" | "failed") {
		if (this._session) this._session.hiZStatus = value;
	}

	private get _hiZBuildCount(): number {
		return this._session?.hiZBuildCount ?? 0;
	}

	private set _hiZBuildCount(value: number) {
		if (this._session) this._session.hiZBuildCount = value;
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
		this._postProcessGraphFrame = null;
		this._postProcessOutputColor = WEBGPU_FRAME_GRAPH_RESOURCES.frameColor;
		this._postProcessOutputColorDomain = "scene-linear-hdr";
		if (this._session) {
			throw new Error("WebGPUFrameOrchestrator already has an active frame session.");
		}
		for (const runtime of this._nodeRuntimes) runtime.beginFrame?.(context);
		this._postBridge.clearPendingFrameState();
		this._lastPlannedGraphNodes = [];
		this._lastCompiledGraphStages = [];
		this._lastExecutedGraphNodeIds = [];
		this._wholeFrameGraphCompiled = false;
		this._occlusionRuntime.beginFrame(context);
		const targetWidth = this._resolveAttachmentDimension(context.attachments.width);
		const targetHeight = this._resolveAttachmentDimension(context.attachments.height);

		if (targetWidth <= 0 || targetHeight <= 0) {
			this._destroyFrameTargets();
			this._session = WebGPUFrameSession.createSkipped(context);
			if (context.framePlan) {
				this._compileWholeFrameGraph(context);
			} else {
				this._graphCompiler.beginFrame([]);
			}
			return;
		}

		const postProcessDeclarations = this._postProcessRuntime.describeFrame(context);
		const encoder = this._host.createCommandEncoder();
		this._customRenderTargets.sync(context);
		const analysis = this._featureAnalyzer.analyze(context, {
			particleOpaquePackets: this._buildParticleMeshDrawPackets(context, {
				includeOpaque: true,
				includeTransparent: false,
			}),
			particleTransparentPackets: this._buildParticleMeshDrawPackets(context, {
				includeOpaque: false,
				includeTransparent: true,
			}),
			postProcessPasses: postProcessDeclarations.passes,
		});
		const configuration = this._configureFrameTargets(
			context,
			analysis,
			encoder,
			targetWidth,
			targetHeight,
		);
		this._lastConfiguration = configuration;
		this._session = WebGPUFrameSession.createRecording({
			context,
			configuration,
			encoder,
			hiZStatus: this._frameTargets?.hiZ ? "pending" : "unavailable",
			analysis,
			committer: new WebGPUFrameCommitter(this._host),
		});
		try {
			this._postProcessGraphFrame = this._postProcessRuntime.buildRenderGraphFrame(
				context,
				postProcessDeclarations,
			);
			if (context.framePlan) {
				this._compileWholeFrameGraph(context);
			} else {
				this._graphCompiler.beginFrame(this._collectInitialGraphResources());
			}
		} catch (error) {
			this._graphCompiler.abort(error);
			this._clearActiveSession();
			throw error;
		}
		this.prepareFrameResources(
			context,
			this._postProcessGraphFrame.graph.frameRequirements,
		);
	}

	private _configureFrameTargets(
		context: FrameContext,
		analysis: WebGPUFrameFeatureAnalysis,
		encoder: ICommandEncoder,
		width: number,
		height: number,
	): WebGPUFrameConfiguration {
		let forceDeferredFallback = false;
		let forceForwardMrt = false;
		for (let attempts = 0; attempts < 3; attempts++) {
			const configuration = this._resolveFrameConfiguration(
				context,
				analysis,
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
		analysis: WebGPUFrameFeatureAnalysis,
		encoder: ICommandEncoder,
		forceDeferredFallback: boolean,
		forceForwardMrt: boolean,
	): WebGPUFrameConfiguration {
		const device = this._host.device;
		return this._configurationResolver.resolve(
			analysis,
			{
				maxColorAttachments: device?.limits?.maxColorAttachments ?? 8,
				maxColorAttachmentBytesPerSample:
					device?.limits?.maxColorAttachmentBytesPerSample ?? 32,
				maxStorageTexturesPerShaderStage:
					device?.limits?.maxStorageTexturesPerShaderStage ?? 4,
			},
			{
				enableEarlyZPrepass: this._enableEarlyZPrepass,
				enableDeferredLighting: this._enableDeferredLighting,
				sampleCount: this._msaa.sampleCount,
				supportsInFrameTextureCopy: typeof encoder.copyTextureToTexture === "function",
				forceDeferredFallback,
				forceForwardMrt,
			},
		);
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
		const message =
			key === WEBGPU_DEFERRED_RUNTIME_FALLBACK_KEY
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
	public prepareFrameResources(
		context: FrameContext,
		frameRequirements: FramePreparationRequirements,
	): WebGPUPreparedFrameResources | null {
		if (!this._frameContext || !this._encoder) {
			this._frameResources = null;
			return null;
		}
		this._frameResources = this._mainFrameScope.prepare(context, {
			sceneTargetMode: this.getSceneTargetModeForFrame(),
			frameRequirements,
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

	/** @internal Updates main-scope particle shadow volumes after simulation. */
	public updateParticleShadowVolumes(context: FrameContext): void {
		if (this._frameResources) {
			this._mainFrameScope.updateParticleShadowVolumes(context);
		}
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
			active: this._session !== null,
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
				mipLevelCount: this._frameTargets?.hiZ
					? Math.floor(
							Math.log2(
								Math.max(
									this._frameTargets.hiZ.width,
									this._frameTargets.hiZ.height,
								),
							),
						) + 1
					: 0,
				buildCount: this._hiZBuildCount,
			},
			lastPlannedNodeIds: this._lastPlannedGraphNodes.map((node) => node.id),
			lastExecutedNodeIds: this._lastExecutedGraphNodeIds.slice(),
			compiledStages: this._lastCompiledGraphStages.slice(),
			compiledGraph: this._graphCompiler.getCompiledFrame()?.graph ?? null,
			graphResources: this._graphCompiler.getResourceDebugState(),
			graphBarriers: this._graphCompiler.getBarriers(),
			graphDiagnostics: this._graphCompiler.getDiagnostics(),
			graphAnalysis: this._graphCompiler.getGraphAnalysis(),
			targetManager: this._frameTargetManager.getDebugState(),
			commit: this._session?.committer?.getDebugState() ?? this._lastCommitDebugState,
			postProcess: this._postProcessRuntime.getDebugState(),
		};
	}

	private _requireFrameResources(): WebGPUPreparedFrameResources {
		if (!this._frameResources) {
			throw new Error("WebGPUFrameOrchestrator requires prepared main-frame resources.");
		}
		return this._frameResources;
	}

	public createPassExecutionContext(request: PostProcessPassExecutionContextRequest): unknown {
		return this._postBridge.createPassExecutionContext(request);
	}

	public createPostProcessSessionPort(): WebGPUPostProcessSessionPort {
		return {
			createGBufferBridge: (context) => this.createGBufferBridge(context),
			createPassExecutionContext: (request) => this.createPassExecutionContext(request),
			completePass: (request, result) => this.completePostProcessPass(request, result),
			isGraphResourceAvailable: (resourceId) =>
				this._isPostProcessSharedResourceAvailable(resourceId),
			invalidateResourceBindings: () => this.invalidatePostProcessBindings(),
		};
	}

	private _isPostProcessSharedResourceAvailable(resourceId: string): boolean {
		const targets = this._frameTargets;
		switch (resourceId) {
			case "backend:frame-hiz": return !!targets?.hiZ;
			case "backend:transmission-scene-color":
				return !!targets?.transmissionSceneColorCopy;
			case "backend:transmission-lighting":
				return !!targets?.transmissionLighting;
			case "backend:transmission-surface-1":
				return !!targets?.gTransmissionSurface1;
			case "backend:transmission-surface-2":
				return !!targets?.gTransmissionSurface2;
			case "backend:planar-reflection-mask":
				return !!targets?.planarReflectionMask;
			default: return false;
		}
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
	): PostProcessPassCompletion {
		return this._postBridge.completePass(request, result);
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
	}

	public invalidatePostProcessBindings(): void {
		this._postRuntime.invalidateBindings();
	}

	public onDisplayOutputChanged(): void {
		this._presentPass.onShaderRuntimeChanged();
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
		for (const runtime of this._nodeRuntimes) runtime.onShaderRuntimeChanged?.();
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions = {},
		postProcessPlan?: PostProcessPlan,
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

		const postRuntime = this._postProcessRuntime;
		const warmupGraph = postProcessPlan ?? (plan.includePostProcess ?
			postRuntime.planWarmup(context) : null);
		const warmedPassImplementations = new Set<string>();
		for (const passId of plan.postProcessPasses) {
			if (warmedPassImplementations.has(passId)) {
				continue;
			}
			const compiledPass = warmupGraph?.passes.find((p) => p.id === passId);
			const implementation = compiledPass?.implementation;
			if (typeof implementation?.warmup !== "function") {
				continue;
			}
			warmedPassImplementations.add(passId);
			total++;
			try {
				const warmupContext = this._postBridge.getPassWarmupExecutionContext(
					compiledPass.id,
					compiledPass.declaration,
				);
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

	/**
	 * Release all GPU resources held by this executor.
	 */
	public destroy(): void {
		this._destroyFrameTargets();
		this._destroyTexturePools();
		for (const runtime of this._nodeRuntimes) runtime.destroy();
		this._customRenderTargets.destroy();
		this._mainFrameScope.destroy();
		this._pendingFrameTargetInvalidation = false;
		this._pendingShaderRuntimeInvalidation = false;
		this._clearActiveSession(false);
	}

	public async executePass(pass: FramePass, context: FrameContext): Promise<void> {
		const session = this._session;
		if (!session) {
			throw new Error("WebGPUFrameOrchestrator has no active frame session.");
		}
		session.assertContext(context);
		if (session.state === "skipped") {
			return;
		}
		if (session.state !== "recording") {
			throw new Error(
				`WebGPU frame session cannot execute passes in state "${session.state}".`,
			);
		}
		if (!session.encoder) {
			throw new Error("WebGPU recording frame session has no command encoder.");
		}

		if (this._customRenderTargets.hasPass(pass, context)) {
			if (!this._wholeFrameGraphCompiled) {
				this._graphCompiler.recordOpaqueStage(
					pass.stage,
					`Custom render target pass "${pass.stage}" executes outside the logical graph.`,
				);
			}
			await this._customRenderTargets.executePass(pass, context, session.encoder);
			this._recordPrecompiledStageExecution(pass.stage);
			return;
		}

		if (this._wholeFrameGraphCompiled) {
			const compiled = this._findCompiledStage(pass.stage);
			if (pass.stage === "postprocess") {
				await this._executePostProcessStage(compiled);
				return;
			}
			if (!compiled || compiled.nodes.length === 0) {
				this._warnUnsupportedPass(pass);
				return;
			}
			for (const node of compiled.nodes) {
				await this._nodeExecutors.execute(node, session);
				this._lastExecutedGraphNodeIds.push(node.id);
			}
			return;
		}

		const plan = this._graphPlanner.planStage(pass, context, this._createPlannerState());
		if (plan.nodes.length === 0) {
			this._warnUnsupportedPass(pass);
			return;
		}
		const compiled = this._graphCompiler.compileStage(plan);
		this._handleGraphDiagnostics(compiled);
		this._lastCompiledGraphStages.push(compiled);
		this._lastPlannedGraphNodes.push(...plan.nodes);
		for (const node of plan.nodes) {
			await this._nodeExecutors.execute(node, session);
			this._lastExecutedGraphNodeIds.push(node.id);
		}
	}

	private _createNodeRuntimes(): readonly WebGPUFrameNodeRuntime[] {
		const scene = new WebGPUSceneNodeRuntime(
			"scene",
			{
				"frame-setup": async () => {},
				"opaque-external": async () => {},
				"opaque-scene": async (_node, session) => {
					this._deferredOpaqueFrameState = await this._scenePassRecorder.recordOpaque(
						session.context,
						this._deferredEnabled,
					);
				},
			},
			{
				destroy: () => this._depthDirtyClearPass.destroy(),
				onShaderRuntimeChanged: () => this._depthDirtyClearPass.onShaderRuntimeChanged(),
			},
		);
		const shadow = new WebGPUShadowNodeRuntime("shadow", {
			shadow: async (_node, session) => {
				await this._resources.renderShadows(session.context, this._encoder ?? undefined);
			},
			"paged-shadow-page-mark": async (_node, session) => {
				const request = this._createPagedShadowRequest(session.context);
				this._resources.preparePagedShadowFrame(request);
				await this._resources.recordPagedShadowPageMarkPass(request);
			},
			"paged-shadow-page-allocate": async (_node, session) => {
				await this._resources.recordPagedShadowPageAllocationPass(
					this._createPagedShadowRequest(session.context),
				);
			},
			"paged-shadow-page-table-copy": async (_node, session) => {
				await this._resources.recordPagedShadowPageTableCopyPass(
					this._createPagedShadowRequest(session.context),
				);
			},
			"paged-shadow-depth": async (_node, session) => {
				await this._resources.recordPagedShadowDepthPass(
					this._createPagedShadowRequest(session.context),
				);
			},
			"paged-shadow-feedback": async (_node, session) => {
				await this._resources.recordPagedShadowFeedbackPass(
					this._createPagedShadowRequest(session.context),
				);
			},
		});
		const reflection = new WebGPUReflectionNodeRuntime(
			"reflection",
			{
				"planar-reflection-capture": async (_node, session) => {
					await this._recordPlanarReflectionPass(session.context);
				},
				"planar-reflection-composite": async (_node, session) => {
					await this._recordPlanarReflectionComposite(session.context);
				},
			},
			{
				destroy: () => this._planarReflectionPass.destroy(),
				invalidateFrameResources: () => this._planarReflectionPass.destroy(),
				onShaderRuntimeChanged: () => this._planarReflectionPass.destroy(),
			},
		);
		const deferred = new WebGPUDeferredNodeRuntime(
			"deferred",
			{
				"deferred-decal": async (_node, session) => {
					await this._recordDeferredDecalNode(session.context);
				},
				"deferred-lighting": async (_node, session) => {
					await this._recordDeferredLightingNode(session.context);
				},
			},
			{
				invalidateFrameResources: () => this._destroyDeferredBindings(),
				onShaderRuntimeChanged: () => this._destroyDeferredBindings(),
			},
		);
		const visibility = new WebGPUVisibilityNodeRuntime(
			"visibility",
			{
				"hiz-build": async (_node, session) => {
					await this._recordHiZBuildNode(session.context);
				},
				"occlusion-test": async (_node, session) => {
					await this._recordOcclusionTestNode(session.context);
				},
			},
			{
				invalidateFrameResources: () => this._occlusionRuntime.invalidateFrameResources(),
				onShaderRuntimeChanged: () => {
					this._hiZBuilder.invalidateShaderResources();
					this._occlusionRuntime.onShaderRuntimeChanged();
				},
				destroy: () => {
					this._occlusionRuntime.destroy();
					this._hiZBuilder.destroy();
				},
			},
		);
		const postProcess = new WebGPUPostProcessNodeRuntime(
			"post-process",
			{
				"post-process-pass": async () => {
					throw new Error(
						"WebGPU post-process pass nodes require the stage transaction coordinator.",
					);
				},
			},
			{
				invalidateFrameResources: () => this._postRuntime.invalidateBindings(),
				onShaderRuntimeChanged: () => {
					this._postRuntime.onShaderRuntimeChanged();
					this._postProcessRuntime.invalidateImplementations();
				},
				destroy: () => this._postRuntime.destroy(),
			},
		);
		const presentation = new WebGPUPresentationNodeRuntime(
			"presentation",
			{
				presentation: async (_node, session) => {
					if (!session.presented && this._frameTargets) {
						await this._presentToCanvas(this._frameTargets.sceneColor);
					}
				},
			},
			{
				invalidateFrameResources: () => this._presentPass.invalidateBindings(),
				onShaderRuntimeChanged: () => this._presentPass.onShaderRuntimeChanged(),
				destroy: () => this._presentPass.destroy(),
			},
		);
		return [
			scene,
			shadow,
			deferred,
			this._transparencyRuntime,
			reflection,
			visibility,
			postProcess,
			presentation,
		];
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
		const session = this._session;
		if (!session) {
			throw new Error("WebGPUFrameOrchestrator has no active frame session.");
		}
		if (session.state === "skipped") {
			this._graphCompiler.seal();
			this._clearActiveSession();
			return;
		}
		session.beginCommit();
		if (this._wholeFrameGraphCompiled) {
			const compiled = this._findCompiledStage("webgpu-present");
			for (const node of compiled?.nodes ?? []) {
				await this._nodeExecutors.execute(node, session);
				this._lastExecutedGraphNodeIds.push(node.id);
			}
		} else {
			const finalization = this._graphPlanner.planFinalization(
				{ stage: "postprocess", executor: "backend", enabled: true, dependsOn: [] },
				this._createPlannerState(),
			);
			if (finalization.nodes.length > 0) {
				const compiled = this._graphCompiler.compileStage(finalization);
				this._handleGraphDiagnostics(compiled);
				this._lastCompiledGraphStages.push(compiled);
				this._lastPlannedGraphNodes.push(...finalization.nodes);
				for (const node of finalization.nodes) {
					await this._nodeExecutors.execute(node, session);
					this._lastExecutedGraphNodeIds.push(node.id);
				}
			}
		}
		this._graphCompiler.seal();
		const encoder = session.encoder;
		const committer = session.committer;
		if (!encoder || !committer) {
			this._clearActiveSession();
			throw new Error("WebGPU committing frame session has no encoder or committer.");
		}

		const width = this._targetWidth;
		const height = this._targetHeight;
		const motionSource =
			this._mrtEnabled && this._motionHistoryWriteTarget
				? this._frameTargets?.gMotionDepth
				: null;
		const motionTarget = this._mrtEnabled ? this._motionHistoryWriteTarget : null;

		try {
			if (motionSource && motionTarget && width > 0 && height > 0) {
				encoder.copyTextureToTexture?.(
					{ texture: motionSource },
					{ texture: motionTarget },
					{ width, height, depthOrArrayLayers: 1 },
				);
			}
			committer.enqueueEncoder("main:final", encoder);
			session.encoder = null;
			await committer.commit(() => {
				this._customRenderTargets.markFrameCommitted();
				this._occlusionRuntime.scheduleQueuedReadbacks();
			});
		} finally {
			this._lastCommitDebugState = committer.getDebugState();
			this._clearActiveSession();
		}
	}

	public abortFrame(error?: unknown): void {
		this._session?.committer?.abort();
		this._lastCommitDebugState = this._session?.committer?.getDebugState() ?? null;
		this._customRenderTargets.markFrameAborted();
		this._graphCompiler.abort(error);
		this._clearActiveSession();
	}

	/** @internal Completes graph analysis after all backend transaction commits. */
	public commitGraphAnalysis(): void {
		this._graphCompiler.commit();
	}

	/** @internal Aborts graph analysis without changing native frame ownership. */
	public abortGraphAnalysis(error?: unknown): void {
		this._graphCompiler.abort(error);
	}

	/** @internal Records a backend pass that bypasses logical resource analysis. */
	public recordOpaqueGraphStage(stage: string, message: string): void {
		if (this._wholeFrameGraphCompiled) return;
		this._graphCompiler.recordOpaqueStage(stage, message);
	}

	public readRenderTargetColor(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions,
	): Promise<RenderTargetReadbackResult> {
		return this._customRenderTargets.readColor(id, attachmentIndex, options);
	}

	private _collectInitialGraphResources() {
		return collectActiveWebGPUFrameGraphResources(this._frameTargets, this._msaaTargets);
	}

	private _compileWholeFrameGraph(context: FrameContext): void {
		const includeShadowResources = context.framePlan?.backendPasses.some(
			(pass) => pass.enabled && pass.stage === "shadow",
		) === true;
		const catalog = collectWebGPUFrameGraphResourceCatalog(
			this._frameTargets,
			this._msaaTargets,
			Math.max(1, this._targetWidth),
			Math.max(1, this._targetHeight),
			this._targetMSAASampleCount,
			this._graphPhysicalResources,
			includeShadowResources,
		);
		const stages: WebGPUFrameGraphStagePlan[] = [];
		const postProcessImportResources: RenderGraphResourceDescriptor[] = [];
		const shadowDiagnostics: RenderGraphDiagnostic[] = [];
		const setupPass: FramePass = {
			stage: "webgpu-setup",
			executor: "backend",
			enabled: true,
			dependsOn: [],
		};
		stages.push({
			pass: setupPass,
			nodes: [
				{
					id: "webgpu-setup:frame-setup",
					stage: setupPass.stage,
					kind: "frame-setup",
					label: "WebGPUFrameSetup",
					domain: "cpu",
					retention: "always",
				},
			],
		});

		let hasOpaqueStage = false;
		let lastStage = setupPass.stage;
		for (const pass of context.framePlan?.backendPasses ?? []) {
			if (!pass.enabled) continue;
			let stagePlan: WebGPUFrameGraphStagePlan;
			const custom = this._customRenderTargets.hasPass(pass, context);
			if (pass.stage === "particle-sim" || custom) {
				const reason = custom ? "custom render target" : "particle simulation";
				stagePlan = {
					pass,
					nodes: [
						{
							id: `${pass.stage}:opaque-external`,
							stage: pass.stage,
							kind: "opaque-external",
							label: `WebGPUOpaque:${pass.stage}`,
							domain: "cpu",
							retention: "always",
							opaque: true,
						},
					],
				};
				hasOpaqueStage = true;
				shadowDiagnostics.push({
					phase: "compile",
					enforcement: "shadow",
					severity: "warning",
					code: "opaque-stage-effects",
					stage: pass.stage,
					message: `WebGPU ${reason} stage "${pass.stage}" has undeclared resource effects.`,
				});
			} else if (pass.stage === "postprocess") {
				const frame = this._postProcessGraphFrame;
				if (frame && frame.graph.passes.length > 0) {
					const composition = createWebGPUPostProcessGraphComposition(frame);
					postProcessImportResources.push(...composition.importResources);
					this._postProcessOutputColor = composition.outputColor;
					this._postProcessOutputColorDomain =
						frame.graph.outputColorDomain;
					stagePlan = {
						pass,
						nodes: [],
						composition: {
							namespace: "postprocess",
							definition: composition.definition,
							inputs: composition.inputs,
						},
					};
				} else {
					stagePlan = { pass, nodes: [] };
				}
			} else {
				stagePlan = this._graphPlanner.planStage(pass, context, this._createPlannerState());
				if (stagePlan.nodes.length === 0) this._warnUnsupportedPass(pass);
			}
			stages.push(stagePlan);
			if (stagePlan.nodes.length > 0 || stagePlan.composition) lastStage = pass.stage;
		}

		const presentationPass: FramePass = {
			stage: "webgpu-present",
			executor: "backend",
			enabled: true,
			dependsOn: [lastStage],
		};
		stages.push(
			this._graphPlanner.planFinalization(
				presentationPass,
				this._createPlannerState(),
				this._postProcessOutputColor,
			),
		);
		const framePlan: WebGPUFrameGraphFramePlan = {
			resources: [...catalog.resources, ...postProcessImportResources],
			bindings: catalog.bindings,
			stages,
			exports: [
				{ name: "presented-color", resource: WEBGPU_FRAME_GRAPH_RESOURCES.canvasColor },
			],
			completeness: hasOpaqueStage ? "opaque" : "complete",
			shadowDiagnostics,
		};
		const compiled = this._graphCompiler.compileFrame(framePlan);
		this._handleWholeFrameGraphDiagnostics(compiled.graph.diagnostics);
		this._lastCompiledGraphStages = compiled.stages.slice();
		this._lastPlannedGraphNodes = compiled.stages.flatMap((stage) => [...stage.nodes]);
		this._lastExecutedGraphNodeIds.push("webgpu-setup:frame-setup");
		this._wholeFrameGraphCompiled = true;
	}

	private async _executePostProcessStage(
		compiled: WebGPUCompiledFrameGraphStage | undefined,
	): Promise<void> {
		const graphFrame = this._postProcessGraphFrame;
		const nodes = (compiled?.nodes ?? []).filter((node) => !!node.postProcess);
		if (!graphFrame || nodes.length === 0) return;
		const plan: PostProcessExecutionPlan = {
			graph: graphFrame.graph,
			outputColor: this._postProcessOutputColor,
			nodes: nodes.map((node) => ({
				...node.postProcess!,
				nodeId: node.id,
			})),
		};
		const frame = await this._postProcessRuntime.beginGraphFrame(plan);
		if (!frame) return;
		let executedColorDomain = plan.graph.initialColorDomain;
		try {
			for (const node of plan.nodes) {
				const result = await this._postProcessRuntime.executeGraphPass(frame, node.passId);
				this._lastExecutedGraphNodeIds.push(node.nodeId);
				if (result.ran === false && node.plannedOutputColor) {
					this._graphCompiler.recordSkippedNode(
						node.nodeId,
						node.plannedOutputColor,
						this._postProcessRuntime.resolveGraphColor(frame, node.plannedOutputColor),
					);
				}
				if (result.ran !== false) {
					const plannedPass = plan.graph.passes.find(
						(pass) => pass.id === node.passId,
					);
					if (plannedPass?.pass.colorContract?.input === executedColorDomain) {
						executedColorDomain = plannedPass.pass.colorContract.output;
					}
				}
			}
			this._postProcessOutputColorDomain = executedColorDomain;
			await this._postProcessRuntime.endGraphFrame(frame);
		} catch (error) {
			await this._postProcessRuntime.abortFrame(error);
			throw error;
		}
	}

	private _createPlannerState(): WebGPUFrameGraphPlannerState {
		const session = this._session;
		return {
			deferredActive: this._deferredEnabled,
			oitActive: this._oitActive,
			sceneTargetMode: this.getSceneTargetModeForFrame(),
			hasFrameTargets: !!this._frameTargets,
			hasMSAATargets: !!this._msaaTargets,
			needsTransmissionTargets: !!this._frameTargets?.transmissionSceneColorCopy,
			needsPlanarReflectionMask: !!this._frameTargets?.planarReflectionMask,
			needsOcclusionTest: session?.configuration?.needsOcclusionTest === true,
			needsHiZBuild:
				session?.configuration?.needsHiZBuild === true && this._hiZStatus === "pending",
			needsPlanarReflectionComposite: session?.analysis?.needsPlanarReflection === true,
			hasOITMeshContributors: (session?.analysis?.transparency.oitPackets.length ?? 0) > 0,
			hasTransmissionPackets:
				(session?.analysis?.transparency.transmissionPackets.length ?? 0) > 0,
			hasAlphaBillboardParticles:
				session?.analysis?.transparency.hasAlphaBillboardParticles === true,
			hasAdditiveBillboardParticles:
				session?.analysis?.transparency.hasAdditiveBillboardParticles === true,
		};
	}

	private _findCompiledStage(stage: string): WebGPUCompiledFrameGraphStage | undefined {
		return this._graphCompiler
			.getCompiledStages()
			.find((compiled) => compiled.pass.stage === stage);
	}

	private _recordPrecompiledStageExecution(stage: string): void {
		if (!this._wholeFrameGraphCompiled) return;
		for (const node of this._findCompiledStage(stage)?.nodes ?? []) {
			this._lastExecutedGraphNodeIds.push(node.id);
		}
	}

	private _warnUnsupportedPass(pass: FramePass): void {
		const key = `webgpu-pass-unsupported-${pass.stage}`;
		Logger.warn(`[${key}] WebGPU backend does not support pass "${pass.stage}" yet; skipping`, {
			scope: "WebGPUFrameOrchestrator",
			onceKey: key,
		});
	}

	private _handleWholeFrameGraphDiagnostics(diagnostics: readonly RenderGraphDiagnostic[]): void {
		const errors = diagnostics.filter(
			(diagnostic) =>
				diagnostic.enforcement === "enforced" && diagnostic.severity === "error",
		);
		if (errors.length <= 0) return;
		const message = `WebGPU internal whole-frame graph validation failed: ${errors
			.map((diagnostic) => diagnostic.message)
			.join(" ")}`;
		if (this._frameGraphValidationMode === "throw") throw new Error(message);
		Logger.warn(`[webgpu-frame-graph-validation] ${message}`, {
			scope: "WebGPUFrameOrchestrator",
			onceKey: "webgpu-frame-graph-validation:whole-frame",
		});
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

	private _warnFrameDiagnostic(code: string, message: string, cause?: unknown): void {
		Logger.warn(`[${code}] ${message}${cause ? ` ${String(cause)}` : ""}`, {
			scope: "WebGPUFrameOrchestrator",
			onceKey: code,
		});
	}

	private _destroyFrameTargets(): void {
		for (const runtime of this._nodeRuntimes) runtime.invalidateFrameResources?.();
		this._graphPhysicalResources.clear();
		this._frameTargetManager.destroyFrameTargets();
		this._motionHistoryWriteTarget = null;
		this._postBridge.clearPendingFrameState();
		this._deferredOpaqueFrameState = null;
		this._occlusionRuntime.invalidateFrameResources();
	}

	private _clearActiveSession(flushPendingLifecycle = true): void {
		this._session = null;
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
		return this._session !== null;
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
			colorDomain: this._postProcessOutputColorDomain,
			resolveDirtyRects: (context, width, height) =>
				this._recordingContext.resolveDirtyRects(context, width, height),
		});
		this._hasPresentedInFrame = true;
	}

	private _buildParticleMeshDrawPackets(
		context: FrameContext,
		options: {
			includeOpaque?: boolean;
			includeTransparent?: boolean;
		},
	): DrawPacket[] {
		const resources = this._resources as WebGPUFrameServiceOwner & {
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
		const session = this._session;
		if (!this._encoder || !session?.committer) {
			return;
		}
		session.committer.enqueueEncoder("main:before-reflection", this._encoder);
		this._encoder = null;
		await this._planarReflectionPass.capture(context, (label, encoder) => {
			session.committer!.enqueueEncoder(label, encoder);
		});
		this._encoder = this._host.createCommandEncoder();
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
				`[webgpu-hiz-build-failed] Shared WebGPU Hi-Z build failed; ` +
					`dependent effects will be skipped. ${String(error)}`,
				{ scope: "WebGPUFrameOrchestrator", onceKey: "webgpu-hiz-build-failed" },
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
