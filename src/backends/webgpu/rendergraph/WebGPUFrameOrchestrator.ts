import type { FramePreparationRequirements } from "../../../pipeline/FrameRequirements";
import type { DrawPacket, FrameContext, FramePass } from "../../../pipeline/types";
import type {
	FramePacketProvider,
	PreparedFramePacketSet,
} from "../../../pipeline/FramePacketContributorRegistry";
import type { ICommandEncoder } from "../../ICommandEncoder";
import { TextureFormat, type IRenderTexture } from "../../types";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import type { WebGPUSampleCountResolver } from "../WebGPUSampleCountResolver";
import type {
	WebGPUFrameResourceScope,
	WebGPUPreparedFrameResources,
} from "../WebGPUResourceContracts";
import type { PostProcessPlan } from "../../../postprocess/PostProcessPlanner";
import type {
	RenderGraphDiagnostic,
	RenderGraphResourceDescriptor,
} from "../../../rendergraph/types";
import { renderGraphResourceId } from "../../../rendergraph/types";

import {
	type WebGPUFrameTargets,
} from "../WebGPUFrameTargetContracts";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../../pipeline/WarmupPlanner";
import type { WarmupOptions } from "../../IRenderBackend";
import { Logger } from "../../../foundation/Logger";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import {
	WebGPUFrameTargetManager,
	type WebGPUFrameTargetEnsureResult,
	type WebGPUFrameMSAATargets,
} from "./WebGPUFrameTargetManager";
import {
	type WebGPUFrameConfiguration,
	type WebGPUFrameDiagnostic,
	type WebGPUFrameSamplePlan,
} from "./WebGPUFrameConfigurationResolver";
import { WebGPUFrameGraphCompiler } from "./WebGPUFrameGraphCompiler";
import { WebGPUFrameGraphModuleRegistry } from "./WebGPUFrameGraphModuleRegistry";
import type { WebGPUFrameModuleStateStore } from "./WebGPUFrameGraphModule";
import { WebGPUFrameSession } from "./WebGPUFrameSession";
import { WebGPUFrameCommitter, type WebGPUFrameCommitDebugState } from "./WebGPUFrameCommitter";
import {
	WEBGPU_REFLECTION_FEATURE_ANALYSIS,
	WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS,
} from "./WebGPUFrameModuleStateKeys";
import {
	collectActiveWebGPUFrameGraphResources,
	collectWebGPUFrameGraphResourceCatalog,
	WEBGPU_FRAME_GRAPH_RESOURCES,
} from "./WebGPUFrameGraphResourceCatalog";
import { WebGPUDirtyRectResolver } from "./WebGPUDirtyRectResolver";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";
import type {
	WebGPUCompiledFrameGraphStage,
	WebGPUFrameGraphFramePlan,
	WebGPUFrameGraphDebugState,
	WebGPUFrameGraphNode,
	WebGPUFrameGraphPlannerState,
	WebGPUFrameGraphStagePlan,
	WebGPUFrameGraphValidationMode,
} from "./types";
import type {
	WebGPUFrameRuntimeComposition,
	WebGPUFrameRuntimeCapabilities,
	WebGPUFrameRuntimeCompositionFactory,
} from "./WebGPUFrameRuntimeComposition";

const WEBGPU_DEFERRED_RUNTIME_FALLBACK_KEY = "webgpu-deferred-runtime-fallback";
const WEBGPU_MAIN_TARGET_SAMPLE_COUNT_RUNTIME_FALLBACK_KEY =
	"webgpu-scene-sample-count-runtime-fallback-1x";

export interface WebGPUFrameOrchestratorOptions {
	readonly enableEarlyZPrepass: boolean;
	readonly enableDeferredLighting: boolean;
	readonly frameGraphValidationMode: WebGPUFrameGraphValidationMode;
}

export class WebGPUFrameOrchestrator {
	public readonly runtimeCapabilities: WebGPUFrameRuntimeCapabilities;
	private _host: WebGPUFrameHost;
	private readonly _framePacketProvider: FramePacketProvider;
	private readonly _mainFrameScope: WebGPUFrameResourceScope;
	private readonly _sampleCountResolver: WebGPUSampleCountResolver;
	private readonly _requestedSampleCount: number;
	private _session: WebGPUFrameSession | null = null;
	private _lastConfiguration: WebGPUFrameConfiguration | null = null;
	private _pendingFrameTargetInvalidation = false;
	private _pendingShaderRuntimeInvalidation = false;
	private _enableEarlyZPrepass = true;
	private _enableDeferredLighting = true;
	private readonly _frameGraphValidationMode: WebGPUFrameGraphValidationMode;
	private readonly _configurationModule: WebGPUFrameRuntimeComposition["configuration"];
	private readonly _dirtyRectResolver = new WebGPUDirtyRectResolver();
	private _recordingContext: WebGPUFrameGraphRecordingContext;
	private _frameTargetManager: WebGPUFrameTargetManager;
	private readonly _graphCompiler = new WebGPUFrameGraphCompiler();
	private readonly _frameModules: WebGPUFrameGraphModuleRegistry;
	private _lastPlannedGraphNodes: WebGPUFrameGraphNode[] = [];
	private _lastCompiledGraphStages: WebGPUCompiledFrameGraphStage[] = [];
	private _lastExecutedGraphNodeIds: string[] = [];
	private _lastCommitDebugState: WebGPUFrameCommitDebugState | null = null;
	private _wholeFrameGraphCompiled = false;
	private readonly _graphPhysicalResources = new Map<string, IRenderTexture>();

	constructor(
		host: WebGPUFrameHost,
		mainFrameScope: WebGPUFrameResourceScope,
		framePacketProvider: FramePacketProvider,
		sampleCountResolver: WebGPUSampleCountResolver,
		requestedSampleCount: number,
		createRuntimeComposition: WebGPUFrameRuntimeCompositionFactory,
		options: WebGPUFrameOrchestratorOptions = {
			enableEarlyZPrepass: host.enableEarlyZPrepass,
			enableDeferredLighting: host.enableDeferredLighting,
			frameGraphValidationMode: host.frameGraphValidationMode,
		},
	) {
		this._host = host;
		this._framePacketProvider = framePacketProvider;
		this._mainFrameScope = mainFrameScope;
		this._sampleCountResolver = sampleCountResolver;
		this._requestedSampleCount = requestedSampleCount;
		this._enableEarlyZPrepass = options.enableEarlyZPrepass;
		this._enableDeferredLighting = options.enableDeferredLighting;
		this._frameGraphValidationMode = options.frameGraphValidationMode;
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
		const composition = createRuntimeComposition({
			recording: this._recordingContext,
			getSession: () => this._session,
			requireSession: () => this._requireActiveSession(),
			requireFrameResources: () => this._requireFrameResources(),
			warnOnce: (code, message, cause) =>
				this._warnFrameDiagnostic(code, message, cause),
		});
		this._frameModules = composition.modules;
		this.runtimeCapabilities = composition;
		this._configurationModule = composition.configuration;
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
		if (this._session) {
			throw new Error("WebGPUFrameOrchestrator already has an active frame session.");
		}
		this._frameModules.beginFrame(context);
		this._lastPlannedGraphNodes = [];
		this._lastCompiledGraphStages = [];
		this._lastExecutedGraphNodeIds = [];
		this._wholeFrameGraphCompiled = false;
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
		let frameRequirements: FramePreparationRequirements | null = null;
		try {
			const framePackets = this._framePacketProvider.prepare(context, "main");
			const postProcessDeclarations = this.runtimeCapabilities.postProcess.describeFrame(context);
			const encoder = this._host.createCommandEncoder();
			this.runtimeCapabilities.customRenderTargets.sync(context);
			const moduleState = this._frameModules.analyze({
				context,
				framePackets,
				postProcessPasses: postProcessDeclarations.passes,
			});
			const configuration = this._configureFrameTargets(
				context,
				framePackets,
				moduleState,
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
				framePackets,
				committer: new WebGPUFrameCommitter(this._host),
				moduleState,
			});
			const postProcessGraphFrame = this.runtimeCapabilities.postProcess.buildGraphFrame(
				context,
				postProcessDeclarations,
			);
			frameRequirements = postProcessGraphFrame.graph.frameRequirements;
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
		this.prepareFrameResources(context, frameRequirements!);
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
		framePackets: PreparedFramePacketSet,
		moduleState: WebGPUFrameModuleStateStore,
		encoder: ICommandEncoder,
		width: number,
		height: number,
	): WebGPUFrameConfiguration {
		let forceDeferredFallback = false;
		let forceForwardMrt = false;
		for (let attempts = 0; attempts < 8; attempts++) {
			const configuration = this._resolveFrameConfiguration(
				context,
				framePackets,
				moduleState,
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
		framePackets: PreparedFramePacketSet,
		moduleState: WebGPUFrameModuleStateStore,
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
		const contributions = this._frameModules.collectConfigurationContributions({
			context,
			state: moduleState,
		});
		for (let attempts = 0; attempts < 6; attempts++) {
			const configuration = this._configurationModule.resolve(
				framePackets,
				contributions,
				capabilities,
				{
					enableEarlyZPrepass: this._enableEarlyZPrepass,
					enableDeferredLighting: this._enableDeferredLighting,
					samplePlan,
					supportsInFrameTextureCopy:
						typeof encoder.copyTextureToTexture === "function",
					forceDeferredFallback,
					forceForwardMrt,
				},
			);
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

	public getSceneTargetModeForFrame(): WebGPUSceneTargetMode {
		if (!this._mrtEnabled || !this._frameTargets) {
			return "single";
		}
		return this._targetSceneTargetMode;
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
			postProcess: this.runtimeCapabilities.postProcess.getDebugState(),
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

	public onDisplayOutputChanged(): void {
		this._frameModules.onDisplayOutputChanged();
	}

	public onShaderRuntimeChanged(): void {
		if (this._hasActiveFrameState()) {
			this._pendingShaderRuntimeInvalidation = true;
			return;
		}
		this._applyShaderRuntimeChangedNow();
	}

	private _applyShaderRuntimeChangedNow(): void {
		this._frameModules.onShaderRuntimeChanged();
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions = {},
		postProcessPlan?: PostProcessPlan,
	): Promise<WarmupPhaseCounters> {
		return this.runtimeCapabilities.postProcess.warmup(
			context,
			plan,
			options,
			postProcessPlan,
		);
	}

	/**
	 * Release all GPU resources held by this executor.
	 */
	public destroy(): void {
		this._destroyFrameTargets();
		this._destroyTexturePools();
		this._frameModules.destroy();
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

		if (this.runtimeCapabilities.customRenderTargets.hasPass(pass, context)) {
			if (!this._wholeFrameGraphCompiled) {
				this._graphCompiler.recordOpaqueStage(
					pass.stage,
					`Custom render target pass "${pass.stage}" executes outside the logical graph.`,
				);
			}
			await this.runtimeCapabilities.customRenderTargets.executePass(
				pass,
				context,
				session.encoder,
			);
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
				await this._frameModules.execute(node, session);
				this._lastExecutedGraphNodeIds.push(node.id);
			}
			return;
		}

		const plan = this._planStage(pass, context);
		if (plan.nodes.length === 0) {
			this._warnUnsupportedPass(pass);
			return;
		}
		const compiled = this._graphCompiler.compileStage(plan);
		this._handleGraphDiagnostics(compiled);
		this._lastCompiledGraphStages.push(compiled);
		this._lastPlannedGraphNodes.push(...plan.nodes);
		for (const node of plan.nodes) {
			await this._frameModules.execute(node, session);
			this._lastExecutedGraphNodeIds.push(node.id);
		}
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
				await this._frameModules.execute(node, session);
				this._lastExecutedGraphNodeIds.push(node.id);
			}
		} else {
			const finalization = this._planStage(
				{ stage: "postprocess", executor: "backend", enabled: true, dependsOn: [] },
				session.context,
				true,
			);
			if (finalization.nodes.length > 0) {
				const compiled = this._graphCompiler.compileStage(finalization);
				this._handleGraphDiagnostics(compiled);
				this._lastCompiledGraphStages.push(compiled);
				this._lastPlannedGraphNodes.push(...finalization.nodes);
				for (const node of finalization.nodes) {
					await this._frameModules.execute(node, session);
					this._lastExecutedGraphNodeIds.push(node.id);
				}
			}
		}
		this._graphCompiler.seal();
		await this._frameModules.finalizeRecording(session);
		const encoder = session.encoder;
		const committer = session.committer;
		if (!encoder || !committer) {
			this._clearActiveSession();
			throw new Error("WebGPU committing frame session has no encoder or committer.");
		}

		try {
			committer.enqueueEncoder("main:final", encoder);
			session.encoder = null;
			await committer.commit(async () => {
				await this._frameModules.afterSubmit(session);
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
		this._frameModules.commitFrameState();
		this._graphCompiler.commit();
	}

	/** @internal Aborts unpublished logical frame state. */
	public abortFrameState(error?: unknown): void {
		this._frameModules.abortFrameState(error);
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
		stages.push(this._planStage(setupPass, context));

		let hasOpaqueStage = false;
		let lastStage = setupPass.stage;
		for (const pass of context.framePlan?.backendPasses ?? []) {
			if (!pass.enabled) continue;
			let stagePlan: WebGPUFrameGraphStagePlan;
			const custom = this.runtimeCapabilities.customRenderTargets.hasPass(pass, context);
			if (pass.stage === "particle-sim" || custom) {
				const reason = custom ? "custom render target" : "particle simulation";
				stagePlan = this._planStage(
					pass,
					context,
					false,
					undefined,
					custom ? this.runtimeCapabilities.customRenderTargets.id : "scene",
				);
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
				stagePlan = this._planStage(pass, context);
				postProcessImportResources.push(
					...this.runtimeCapabilities.postProcess.getImportResources(),
				);
			} else {
				stagePlan = this._planStage(pass, context);
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
		stages.push(this._planStage(
			presentationPass,
			context,
			true,
			this.runtimeCapabilities.postProcess.outputColor,
		));
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
		await this.runtimeCapabilities.postProcess.executeStage(
			compiled,
			this._graphCompiler,
			(nodeId) => this._lastExecutedGraphNodeIds.push(nodeId),
		);
	}

	private _planStage(
		pass: FramePass,
		context: FrameContext,
		finalization = false,
		finalColorResource?: string,
		exclusiveModuleId?: string,
	): WebGPUFrameGraphStagePlan {
		const session = this._requireActiveSession();
		return this._frameModules.planStage({
			pass,
			context,
			state: this._createPlannerState(),
			moduleState: session.moduleState,
			finalization,
			finalColorResource,
			exclusiveModuleId,
		});
	}

	private _createPlannerState(): WebGPUFrameGraphPlannerState {
		const session = this._session;
		const reflection = session?.moduleState.get(WEBGPU_REFLECTION_FEATURE_ANALYSIS);
		const transparency = session?.moduleState.get(WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS);
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
			needsPlanarReflectionComposite: reflection?.needsPlanarReflection === true,
			hasOITMeshContributors: (transparency?.oitPackets.length ?? 0) > 0,
			hasTransmissionPackets:
				(transparency?.transmissionPackets.length ?? 0) > 0,
			hasAlphaBillboardParticles:
				transparency?.hasAlphaBillboardParticles === true,
			hasAdditiveBillboardParticles:
				transparency?.hasAdditiveBillboardParticles === true,
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
		this._frameModules.invalidateFrameResources();
		this._graphPhysicalResources.clear();
		this._frameTargetManager.destroyFrameTargets();
		this._motionHistoryWriteTarget = null;
		if (this._session) this._session.deferredOpaqueFrameState = null;
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
}
