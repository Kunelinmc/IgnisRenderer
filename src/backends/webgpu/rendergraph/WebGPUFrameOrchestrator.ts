import type { FramePreparationRequirements } from "../../../pipeline/FrameRequirements";
import type { DrawPacket, FrameContext, FramePass } from "../../../pipeline/types";
import type {
	FramePacketProvider,
	PreparedFramePacketSet,
} from "../../../pipeline/FramePacketContributorRegistry";
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
import { TextureFormat, type IRenderTexture } from "../../types";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import type { WebGPUSampleCountResolver } from "../WebGPUSampleCountResolver";
import type {
	WebGPUFrameResourceScope,
	WebGPUParticleBillboardRenderer,
	WebGPUPreparedFrameResources,
	WebGPUShadowRenderProvider,
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
import { renderGraphResourceId } from "../../../rendergraph/types";
import { createWebGPUPostProcessGraphComposition } from "./WebGPUPostProcessGraphAdapter";

import {
	type WebGPUFrameTargets,
} from "../WebGPUFrameTargetContracts";
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
	type WebGPUFrameSamplePlan,
} from "./WebGPUFrameConfigurationResolver";
import { WebGPUFrameGraphPlanner } from "./WebGPUFrameGraphPlanner";
import { WebGPUFrameGraphCompiler } from "./WebGPUFrameGraphCompiler";
import {
	WebGPUFrameNodeExecutorRegistry,
} from "./WebGPUFrameNodeExecutorRegistry";
import {
	WebGPUDeferredNodeRuntime,
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
import { WebGPUCustomRenderTargetRuntime } from "./WebGPUCustomRenderTargetRuntime";
import { getWebGPUPostProcessSharedResourceDescriptor } from "./WebGPUPostProcessSharedResourceCatalog";
import { WebGPUPresentationRuntime } from "./WebGPUPresentationRuntime";
import type {
	RenderTargetReadbackOptions,
	RenderTargetReadbackResult,
} from "../../../rendering/CustomRenderTargets";
import type { WebGPUPostProcessSessionPort } from "../WebGPUPostProcessExecutor";

const WEBGPU_DEFERRED_RUNTIME_FALLBACK_KEY = "webgpu-deferred-runtime-fallback";
const WEBGPU_MAIN_TARGET_SAMPLE_COUNT_RUNTIME_FALLBACK_KEY =
	"webgpu-scene-sample-count-runtime-fallback-1x";
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
	private readonly _shadowRenderer: WebGPUShadowRenderProvider;
	private readonly _framePacketProvider: FramePacketProvider;
	private readonly _mainFrameScope: WebGPUFrameResourceScope;
	private readonly _sampleCountResolver: WebGPUSampleCountResolver;
	private readonly _requestedSampleCount: number;
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
	private _presentationRuntime: WebGPUPresentationRuntime;
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
	private _postProcessOutputColorDomain: PostProcessColorDomain = "scene-linear-hdr";
	private readonly _graphPhysicalResources = new Map<string, IRenderTexture>();

	constructor(
		host: WebGPUFrameHost,
		frameServiceOwner: WebGPUFrameServiceOwner,
		framePacketProvider: FramePacketProvider,
		particleRenderer: WebGPUParticleBillboardRenderer,
		sampleCountResolver: WebGPUSampleCountResolver,
		requestedSampleCount: number,
		options: WebGPUFrameOrchestratorOptions = {
			enableEarlyZPrepass: host.enableEarlyZPrepass,
			enableDeferredLighting: host.enableDeferredLighting,
			frameGraphValidationMode: host.frameGraphValidationMode,
		},
	) {
		this._host = host;
		this._shadowRenderer = frameServiceOwner;
		this._framePacketProvider = framePacketProvider;
		this._mainFrameScope = frameServiceOwner.createFrameScope();
		this._sampleCountResolver = sampleCountResolver;
		this._requestedSampleCount = requestedSampleCount;
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
			frameServiceOwner.sceneFrameLayout,
			this._hiZBuilder,
			() => this._host.displayOutputState,
		);
		this._planarReflectionPass = new WebGPUPlanarReflectionPass(
			host,
			frameServiceOwner,
			framePacketProvider,
		);
		this._customRenderTargets = new WebGPUCustomRenderTargetRuntime(host, sampleCountResolver);
		this._frameTargetManager = new WebGPUFrameTargetManager(host);
		this._recordingContext = {
			getEncoder: () => this._encoder,
			getFrameTargets: () => this._frameTargets,
			getMSAATargets: () => this._msaaTargets,
			getTargetWidth: () => this._targetWidth,
			getTargetHeight: () => this._targetHeight,
			getSampleCount: () => this._targetSampleCount,
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
		this._presentationRuntime = new WebGPUPresentationRuntime(host, {
			recording: this._recordingContext,
			getOutputColorDomain: () => this._postProcessOutputColorDomain,
		});
		this._postBridge = new WebGPUPostProcessBridge(host, this._postRuntime, {
			getEncoder: () => this._encoder,
			getFrameTargets: () => this._frameTargets,
			isHiZReady: () => this._hiZStatus === "ready",
			requireFrameResources: () => this._requireFrameResources(),
			presentToCanvas: (source) =>
				this._presentationRuntime.present(source, this._requireActiveSession()),
			warmupPresent: () => this._presentationRuntime.warmup(),
			setMotionHistoryWriteTarget: (texture) => {
				this._motionHistoryWriteTarget = texture;
			},
		});
		this._depthDirtyClearPass = new WebGPUDepthDirtyClearPass(host);
		this._deferredLightingPass = new WebGPUDeferredLightingPass(host, frameServiceOwner, {
			recordingContext: this._recordingContext,
		});
		this._deferredDecalPass = new WebGPUDeferredDecalPass(host, frameServiceOwner, {
			recordingContext: this._recordingContext,
		});
		this._scenePassRecorder = new WebGPUScenePassRecorder(
			host,
			frameServiceOwner,
			particleRenderer,
			this._recordingContext,
			this._depthDirtyClearPass,
			{
				getGBufferWriteBinding: () => this._deferredLightingPass.getGBufferWriteBinding(),
				getDeferredGBufferLayout: () =>
					this._session?.configuration?.deferredGBufferLayout ?? "extended",
				preflightDeferredFrame: async (context) => {
					await this._deferredLightingPass.preflight();
					await this._deferredDecalPass.preflight(context);
				},
			},
		);
		this._occlusionRuntime = new WebGPUOcclusionCullingRuntime(host);
		this._transparencyRuntime = new WebGPUTransparencyRuntime(
			host,
			frameServiceOwner,
			particleRenderer,
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

	private get _targetSampleCount(): number {
		return this._frameTargetManager.targetSampleCount;
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
		this._session = WebGPUFrameSession.createPreparing(context);
		if (this._requiresParticleSimulation(context)) {
			return;
		}
		this._sealFrame(context, targetWidth, targetHeight);
	}

	/** @internal Seals deferred frame preparation after particle simulation. */
	public sealParticleSimulation(context: FrameContext): void {
		const session = this._session;
		if (!session) {
			throw new Error("WebGPUFrameOrchestrator has no active frame session.");
		}
		session.assertContext(context);
		if (session.state === "skipped" || session.state === "recording") {
			return;
		}
		if (session.state !== "preparing") {
			throw new Error(`WebGPU frame session cannot seal from state "${session.state}".`);
		}
		this._sealFrame(
			context,
			this._resolveAttachmentDimension(context.attachments.width),
			this._resolveAttachmentDimension(context.attachments.height),
		);
		this.updateParticleShadowVolumes(context);
	}

	private _sealFrame(context: FrameContext, targetWidth: number, targetHeight: number): void {
		try {
			const framePackets = this._framePacketProvider.prepare(context, "main");
			const postProcessDeclarations = this._postProcessRuntime.describeFrame(context);
			const encoder = this._host.createCommandEncoder();
			this._customRenderTargets.sync(context);
			const analysis = this._featureAnalyzer.analyze(context, {
				framePackets,
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
				framePackets,
				committer: new WebGPUFrameCommitter(this._host),
			});
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
		this.prepareFrameResources(context, this._postProcessGraphFrame.graph.frameRequirements);
	}

	private _requiresParticleSimulation(context: FrameContext): boolean {
		return (
			context.framePlan?.backendPasses.some(
				(pass) => pass.stage === "particle-sim" && pass.enabled,
			) === true
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
		for (let attempts = 0; attempts < 8; attempts++) {
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
				sampleCount: configuration.samplePlan.sampleCount,
				requirements: configuration.targetRequirements,
			});
			if (result.status === "ready") return configuration;
			if (result.status === "retry-legacy-mrt") {
				forceDeferredFallback = true;
				forceForwardMrt = true;
				this._warnFrameTargetRetry(WEBGPU_DEFERRED_RUNTIME_FALLBACK_KEY, result);
				continue;
			}
			const failedSampleCount = configuration.samplePlan.sampleCount;
			if (
				!this._sampleCountResolver.fallbackToSingleSample(
					configuration.samplePlan.selectionSignature,
				)
			) {
				throw result.error;
			}
			this._host.onMainTargetSampleCountRuntimeFallback();
			this._warnFrameTargetRetry(
				WEBGPU_MAIN_TARGET_SAMPLE_COUNT_RUNTIME_FALLBACK_KEY,
				result,
				failedSampleCount,
			);
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
		const capabilities = {
			maxColorAttachments: device?.limits?.maxColorAttachments ?? 8,
			maxColorAttachmentBytesPerSample:
				device?.limits?.maxColorAttachmentBytesPerSample ?? 32,
			maxStorageTexturesPerShaderStage: device?.limits?.maxStorageTexturesPerShaderStage ?? 4,
		};
		let samplePlan: WebGPUFrameSamplePlan = {
			requestedSampleCount: this._requestedSampleCount,
			sampleCount: this._requestedSampleCount,
			selectionSignature: "main-scene:pending",
			runtimeFallbackActive: false,
		};
		const visited = new Set<string>();
		for (let attempts = 0; attempts < 6; attempts++) {
			const configuration = this._configurationResolver.resolve(analysis, capabilities, {
				enableEarlyZPrepass: this._enableEarlyZPrepass,
				enableDeferredLighting: this._enableDeferredLighting,
				samplePlan,
				supportsInFrameTextureCopy: typeof encoder.copyTextureToTexture === "function",
				forceDeferredFallback,
				forceForwardMrt,
			});
			const formats = this._getSampleCountProbeFormats(configuration);
			const nextPlan =
				formats.length === 0
					? {
							requestedSampleCount: this._requestedSampleCount,
							sampleCount: 1,
							selectionSignature: "main-scene:single",
							runtimeFallbackActive: false,
						}
					: this._toFrameSamplePlan(
							this._sampleCountResolver.resolveDomainSampleCount(
								"main-scene",
								this._requestedSampleCount,
								formats,
							),
							samplePlan.sampleCount,
						);
			if (
				nextPlan.sampleCount === samplePlan.sampleCount &&
				nextPlan.selectionSignature === samplePlan.selectionSignature
			) {
				return configuration;
			}
			const stateKey = `${nextPlan.sampleCount}|${nextPlan.selectionSignature}`;
			if (visited.has(stateKey)) {
				throw new Error("WebGPU scene sample-count planning did not converge.");
			}
			visited.add(stateKey);
			samplePlan = nextPlan;
		}
		throw new Error("WebGPU scene sample-count planning exceeded its retry limit.");
	}

	private _toFrameSamplePlan(
		selection: ReturnType<WebGPUSampleCountResolver["resolveDomainSampleCount"]>,
		upperBound: number,
	): WebGPUFrameSamplePlan {
		return {
			requestedSampleCount: selection.requestedSampleCount,
			sampleCount: Math.min(upperBound, selection.sampleCount),
			selectionSignature: selection.signature,
			runtimeFallbackActive: selection.runtimeFallbackActive,
		};
	}

	private _getSampleCountProbeFormats(
		configuration: WebGPUFrameConfiguration,
	): GPUTextureFormat[] {
		const requirements = configuration.targetRequirements;
		if (!requirements) {
			return [];
		}
		const formats = new Set<GPUTextureFormat>([
			TextureFormat.RGBA16Float as GPUTextureFormat,
			TextureFormat.Depth32Float as GPUTextureFormat,
		]);
		if (requirements.sceneTargetMode === "mrt" || requirements.sceneTargetMode === "gbuffer") {
			formats.add(TextureFormat.RGBA8Unorm as GPUTextureFormat);
		}
		if (requirements.needsPlanarReflectionMask) {
			formats.add(TextureFormat.R8Unorm as GPUTextureFormat);
		}
		return Array.from(formats);
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
		failedSampleCount?: number,
	): void {
		const message =
			key === WEBGPU_DEFERRED_RUNTIME_FALLBACK_KEY
				? "WebGPU deferred frame target allocation failed; retrying with legacy MRT forward path."
				: `WebGPU ${failedSampleCount ?? 1}x scene sample-count target allocation failed; retrying at 1x.`;
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
			framePackets: this._requireFramePackets(),
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

	private _requireActiveSession(): WebGPUFrameSession {
		if (!this._session) {
			throw new Error("WebGPU frame session is unavailable.");
		}
		return this._session;
	}

	private _requireFramePackets(): PreparedFramePacketSet {
		return this._requireSessionFramePackets(this._requireActiveSession());
	}

	private _requireSessionFramePackets(session: WebGPUFrameSession): PreparedFramePacketSet {
		if (!session.framePackets) {
			throw new Error("WebGPU frame session has no prepared frame packets.");
		}
		return session.framePackets;
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
		return (
			getWebGPUPostProcessSharedResourceDescriptor(resourceId)?.isAllocated(
				this._frameTargets,
			) ?? false
		);
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
		this._presentationRuntime.onShaderRuntimeChanged();
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
			await this._presentationRuntime.warmup();
			compiled++;
		} catch (error) {
			failed++;
			errors.push(toShaderCompileError(error, "webgpu", "WebGPUPresentWarmup"));
		}
		await yieldController.yieldIfNeeded();

		const postRuntime = this._postProcessRuntime;
		const warmupGraph =
			postProcessPlan ?? (plan.includePostProcess ? postRuntime.planWarmup(context) : null);
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
						this._requireSessionFramePackets(session),
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
				await this._shadowRenderer.renderShadows(
					session.context,
					this._requireSessionFramePackets(session),
					this._encoder ?? undefined,
				);
			},
			"paged-shadow-page-mark": async (_node, session) => {
				const request = this._createPagedShadowRequest(session.context);
				this._shadowRenderer.preparePagedShadowFrame(request);
				await this._shadowRenderer.recordPagedShadowPageMarkPass(request);
			},
			"paged-shadow-page-allocate": async (_node, session) => {
				await this._shadowRenderer.recordPagedShadowPageAllocationPass(
					this._createPagedShadowRequest(session.context),
				);
			},
			"paged-shadow-page-table-copy": async (_node, session) => {
				await this._shadowRenderer.recordPagedShadowPageTableCopyPass(
					this._createPagedShadowRequest(session.context),
				);
			},
			"paged-shadow-depth": async (_node, session) => {
				await this._shadowRenderer.recordPagedShadowDepthPass(
					this._createPagedShadowRequest(session.context),
				);
			},
			"paged-shadow-feedback": async (_node, session) => {
				await this._shadowRenderer.recordPagedShadowFeedbackPass(
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
		return [
			scene,
			shadow,
			deferred,
			this._transparencyRuntime,
			reflection,
			visibility,
			postProcess,
			this._presentationRuntime,
		];
	}

	private _createPagedShadowRequest(context: FrameContext): WebGPUPagedShadowFrameRequest {
		const frameTargets = this._frameTargets;
		const framePackets = this._requireFramePackets();
		return {
			context,
			encoder: this._encoder,
			renderSets: context.shadowMaps,
			shadowCasterPackets: framePackets.shadowCasters.slice(),
			shadowTransmitterPackets: framePackets.shadowTransmitters.slice(),
			feedbackDepthTexture: frameTargets?.depth ?? null,
			feedbackMotionDepthTexture: frameTargets?.gMotionDepth ?? null,
		};
	}

	public async endFrame(postSubmit?: () => void | Promise<void>): Promise<void> {
		const session = this._session;
		if (!session) {
			throw new Error("WebGPUFrameOrchestrator has no active frame session.");
		}
		if (session.state === "skipped") {
			this._graphCompiler.seal();
			try {
				await postSubmit?.();
			} finally {
				this._clearActiveSession();
			}
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
			await committer.commit(async () => {
				this._occlusionRuntime.scheduleQueuedReadbacks();
				await postSubmit?.();
			});
		} finally {
			this._lastCommitDebugState = committer.getDebugState();
			this._clearActiveSession();
		}
	}

	/** @internal Discards active native frame recording without logical publication. */
	public abortRecording(_error?: unknown): void {
		this._session?.committer?.abort();
		this._lastCommitDebugState = this._session?.committer?.getDebugState() ?? null;
		this._clearActiveSession();
	}

	/** @internal Publishes logical frame state after every post-submit commit succeeds. */
	public commitFrameState(): void {
		this._customRenderTargets.markFrameCommitted();
		this._graphCompiler.commit();
	}

	/** @internal Aborts unpublished logical frame state. */
	public abortFrameState(error?: unknown): void {
		this._customRenderTargets.markFrameAborted();
		this._graphCompiler.abort(error);
	}

	/** @internal Compatibility wrapper for direct frame-runtime tests. */
	public abortFrame(error?: unknown): void {
		this.abortRecording(error);
		this.abortFrameState(error);
	}

	/** @internal Compatibility alias; new code must use `commitFrameState()`. */
	public commitGraphAnalysis(): void {
		this.commitFrameState();
	}

	/** @internal Compatibility alias; new code must use `abortFrameState()`. */
	public abortGraphAnalysis(error?: unknown): void {
		this.abortFrameState(error);
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
		const includeShadowResources =
			context.framePlan?.backendPasses.some(
				(pass) => pass.enabled && pass.stage === "shadow",
			) === true;
		const catalog = collectWebGPUFrameGraphResourceCatalog(
			this._frameTargets,
			this._msaaTargets,
			Math.max(1, this._targetWidth),
			Math.max(1, this._targetHeight),
			this._targetSampleCount,
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
					this._postProcessOutputColorDomain = frame.graph.outputColorDomain;
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
				{
					name: "presented-color",
					resource: renderGraphResourceId(WEBGPU_FRAME_GRAPH_RESOURCES.canvasColor),
				},
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
					const plannedPass = plan.graph.passes.find((pass) => pass.id === node.passId);
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
			deferredGBufferLayout:
				session?.configuration?.deferredGBufferLayout ?? "extended",
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
			sampleCount: this._lastConfiguration?.samplePlan.sampleCount ?? 1,
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
