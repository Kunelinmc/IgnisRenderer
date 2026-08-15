import type { DrawPacket, FrameContext, FramePass } from "../../../pipeline/types";
import type {
	FramePacketProvider,
} from "../../../pipeline/FramePacketContributorRegistry";
import type { ICommandEncoder } from "../../ICommandEncoder";
import { TextureFormat, type IRenderTexture } from "../../types";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import type { WebGPUSampleCountResolver } from "../WebGPUSampleCountResolver";
import type {
	WebGPUFrameResourceScope,
} from "../WebGPUResourceContracts";
import type {
	RenderGraphDiagnostic,
	RenderGraphResourceDescriptor,
} from "../../../rendergraph/types";
import { renderGraphResourceId } from "../../../rendergraph/types";

import { Logger } from "../../../foundation/Logger";
import {
	WebGPUFrameTargetManager,
	type WebGPUFrameTargetEnsureResult,
} from "./WebGPUFrameTargetManager";
import {
	type WebGPUFrameConfiguration,
	type WebGPUFrameDiagnostic,
	type WebGPUFrameSamplePlan,
} from "./WebGPUFrameConfiguration";
import { WebGPUFrameGraphCompiler } from "./WebGPUFrameGraphCompiler";
import { WebGPUFrameGraphModuleRegistry } from "./WebGPUFrameGraphModuleRegistry";
import type { WebGPUFrameMessageSnapshot } from "./WebGPUFrameMessage";
import {
	WebGPUFrameSession,
	type WebGPUFrameSessionState,
	type WebGPURecordingFrameSession,
} from "./WebGPUFrameSession";
import { WebGPUFrameCommandStream } from "./WebGPUFrameCommandStream";
import type { WebGPUFrameDiagnosticsObserver } from "./WebGPUFrameDiagnostics";
import {
	WEBGPU_FRAME_CONFIGURATION_MESSAGE,
	WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE,
	WEBGPU_FRAME_CONTEXT_MESSAGE,
	WEBGPU_FRAME_PACKETS_MESSAGE,
} from "./WebGPUFrameMessages";
import {
	collectActiveWebGPUFrameGraphResources,
	collectWebGPUFrameGraphResourceCatalog,
	WEBGPU_FRAME_GRAPH_RESOURCES,
} from "./WebGPUFrameGraphResourceCatalog";
import { WebGPUDirtyRectResolver } from "./WebGPUDirtyRectResolver";
import type {
	WebGPUCompiledFrameGraphStage,
	WebGPUFrameGraphFramePlan,
	WebGPUFrameResourceAllocationSnapshot,
	WebGPUFrameGraphStagePlan,
	WebGPUFrameGraphValidationMode,
} from "./types";

const WEBGPU_DEFERRED_RUNTIME_FALLBACK_KEY = "webgpu-deferred-runtime-fallback";
const WEBGPU_MAIN_TARGET_SAMPLE_COUNT_RUNTIME_FALLBACK_KEY =
	"webgpu-scene-sample-count-runtime-fallback-1x";

export interface WebGPUFrameOrchestratorOptions {
	readonly enableEarlyZPrepass: boolean;
	readonly enableDeferredLighting: boolean;
	readonly frameGraphValidationMode: WebGPUFrameGraphValidationMode;
	readonly diagnosticsObserver?: WebGPUFrameDiagnosticsObserver;
}

export class WebGPUFrameOrchestrator {
	private _host: WebGPUFrameHost;
	private readonly _framePacketProvider: FramePacketProvider;
	private readonly _mainFrameScope: WebGPUFrameResourceScope;
	private readonly _sampleCountResolver: WebGPUSampleCountResolver;
	private readonly _requestedSampleCount: number;
	private _session: WebGPUFrameSession | null = null;
	private _pendingFrameTargetInvalidation = false;
	private _enableEarlyZPrepass = true;
	private _enableDeferredLighting = true;
	private readonly _frameGraphValidationMode: WebGPUFrameGraphValidationMode;
	private readonly _diagnosticsObserver: WebGPUFrameDiagnosticsObserver | null;
	private readonly _dirtyRectResolver = new WebGPUDirtyRectResolver();
	private _frameTargetManager: WebGPUFrameTargetManager;
	private readonly _graphCompiler = new WebGPUFrameGraphCompiler();
	private readonly _frameModules: WebGPUFrameGraphModuleRegistry;
	private _wholeFrameGraphCompiled = false;
	private readonly _graphPhysicalResources = new Map<string, IRenderTexture>();

	constructor(
		host: WebGPUFrameHost,
		mainFrameScope: WebGPUFrameResourceScope,
		framePacketProvider: FramePacketProvider,
		sampleCountResolver: WebGPUSampleCountResolver,
		requestedSampleCount: number,
		frameModules: WebGPUFrameGraphModuleRegistry,
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
		this._diagnosticsObserver = options.diagnosticsObserver ?? null;
		this._frameTargetManager = new WebGPUFrameTargetManager(host);
		this._frameModules = frameModules;
	}

	public async beginFrame(context: FrameContext): Promise<void> {
		if (this._session) {
			throw new Error("WebGPUFrameOrchestrator already has an active frame session.");
		}
		this._frameModules.beginFrame(context);
		this._wholeFrameGraphCompiled = false;
		const targetWidth = this._resolveAttachmentDimension(context.attachments.width);
		const targetHeight = this._resolveAttachmentDimension(context.attachments.height);

		if (targetWidth <= 0 || targetHeight <= 0) {
			this._destroyFrameTargets();
			const skipped = WebGPUFrameSession.createSkipped(context);
			this._transitionSession(skipped);
			if (context.framePlan) {
				await this._compileWholeFrameGraph(
					context,
					skipped.messages,
					null,
				);
			} else {
				this._graphCompiler.beginFrame([]);
			}
			return;
		}
		this._transitionSession(WebGPUFrameSession.createPreparing(context));
		if (this._requiresParticleSimulation(context)) {
			return;
		}
		await this._sealFrame(context, targetWidth, targetHeight);
	}

	/** @internal Seals deferred frame preparation after particle simulation. */
	public async sealParticleSimulation(context: FrameContext): Promise<void> {
		const session = this._session;
		if (!session) {
			throw new Error("WebGPUFrameOrchestrator has no active frame session.");
		}
		WebGPUFrameSession.assertContext(session, context);
		if (session.state === "skipped" || session.state === "recording") {
			return;
		}
		if (session.state !== "preparing") {
			throw new Error(`WebGPU frame session cannot seal from state "${session.state}".`);
		}
		await this._sealFrame(
			context,
			this._resolveAttachmentDimension(context.attachments.width),
			this._resolveAttachmentDimension(context.attachments.height),
		);
		if (this._getRecordingSession()) {
			this._mainFrameScope.updateParticleShadowVolumes(context);
		}
	}

	private async _sealFrame(
		context: FrameContext,
		targetWidth: number,
		targetHeight: number,
	): Promise<void> {
		let commands: WebGPUFrameCommandStream | null = null;
		try {
			const framePackets = this._framePacketProvider.prepare(context, "main");
			commands = new WebGPUFrameCommandStream(this._host);
			const encoder = commands.requireEncoder();
			this._frameModules.syncFrame(context);
			const analysisMessages = await this._frameModules.dispatchMessages("analysis", {
				seeds: [
					{ descriptor: WEBGPU_FRAME_CONTEXT_MESSAGE, value: context },
					{ descriptor: WEBGPU_FRAME_PACKETS_MESSAGE, value: framePackets },
					...this._frameModules.createAnalysisSeeds(context),
				],
			});
			const configured = await this._configureFrameTargets(
				context,
				analysisMessages,
				encoder,
				targetWidth,
				targetHeight,
			);
			const configuration = configured.configuration;
			const targets = this._frameTargetManager.getTargetView(targetWidth, targetHeight);
			this._diagnosticsObserver?.onTargetsConfigured?.(
				this._frameTargetManager.getDebugState(),
			);
			const frameRequirements = this._frameModules.sealFrame(context);
			if (context.framePlan) {
				await this._compileWholeFrameGraph(
					context,
					configured.messages,
					configuration,
				);
			} else {
				this._graphCompiler.beginFrame(this._collectInitialGraphResources());
			}
			const resources = this._mainFrameScope.prepare(context, {
				sceneTargetMode: configuration.mrtSupported
					? targets.sceneTargetMode
					: "single",
				framePackets,
				frameRequirements,
			});
			const recording = WebGPUFrameSession.createRecording({
				context,
				configuration,
				resources,
				framePackets,
				messages: configured.messages,
				targets,
				commands,
				earlyZPrepassEnabled: this._enableEarlyZPrepass,
				dirtyRects: this._dirtyRectResolver,
			});
			this._transitionSession(recording);
			this._frameModules.activateFrame(recording);
		} catch (error) {
			commands?.abort();
			if (commands) {
				this._diagnosticsObserver?.onCommitSettled?.(commands.getDebugState());
			}
			this._graphCompiler.abort(error);
			this._closeSession();
			throw error;
		}
	}

	private _requiresParticleSimulation(context: FrameContext): boolean {
		return (
			context.framePlan?.backendPasses.some(
				(pass) => pass.stage === "particle-sim" && pass.enabled,
			) === true
		);
	}

	private async _configureFrameTargets(
		context: FrameContext,
		analysisMessages: WebGPUFrameMessageSnapshot,
		encoder: ICommandEncoder,
		width: number,
		height: number,
	): Promise<{
		readonly configuration: WebGPUFrameConfiguration;
		readonly messages: WebGPUFrameMessageSnapshot;
	}> {
		let forceDeferredFallback = false;
		let forceForwardMrt = false;
		for (let attempts = 0; attempts < 8; attempts++) {
			const configured = await this._resolveFrameConfiguration(
				context,
				analysisMessages,
				encoder,
				forceDeferredFallback,
				forceForwardMrt,
			);
			const configuration = configured.configuration;
			this._emitConfigurationDiagnostics(configuration.diagnostics);
			if (!configuration.targetRequirements) {
				this._destroyFrameTargets();
				return configured;
			}
			const result = this._frameTargetManager.ensureFrameTargets({
				width,
				height,
				sampleCount: configuration.samplePlan.sampleCount,
				requirements: configuration.targetRequirements,
			});
			if (result.status === "ready") return configured;
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

	private async _resolveFrameConfiguration(
		context: FrameContext,
		analysisMessages: WebGPUFrameMessageSnapshot,
		encoder: ICommandEncoder,
		forceDeferredFallback: boolean,
		forceForwardMrt: boolean,
	): Promise<{
		readonly configuration: WebGPUFrameConfiguration;
		readonly messages: WebGPUFrameMessageSnapshot;
	}> {
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
			const messages = await this._frameModules.dispatchMessages("configuration", {
				prior: analysisMessages,
				seeds: [{
					descriptor: WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE,
					value: {
						context,
						capabilities,
						options: {
					enableEarlyZPrepass: this._enableEarlyZPrepass,
					enableDeferredLighting: this._enableDeferredLighting,
					samplePlan,
					supportsInFrameTextureCopy:
						typeof encoder.copyTextureToTexture === "function",
					forceDeferredFallback,
					forceForwardMrt,
						},
					},
				}],
			});
			const configuration = messages.get(WEBGPU_FRAME_CONFIGURATION_MESSAGE);
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
				return { configuration, messages };
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

	private _getRecordingSession(): WebGPURecordingFrameSession | null {
		return this._session?.state === "recording" ? this._session : null;
	}

	private _requireRecordingSession(): WebGPURecordingFrameSession {
		const session = this._getRecordingSession();
		if (!session) {
			throw new Error("WebGPU recording frame session is unavailable.");
		}
		return session;
	}

	/**
	 * Force frame targets to be rebuilt on the next beginFrame().
	 * Call on canvas resize so the post-process pipeline picks up
	 * the new dimensions.
	 */
	public invalidateFrameTargets(): void {
		if (this._session) {
			this._pendingFrameTargetInvalidation = true;
			return;
		}
		this._invalidateFrameTargetsNow();
	}

	private _invalidateFrameTargetsNow(): void {
		this._destroyFrameTargets();
	}

	/**
	 * Release all GPU resources held by this executor.
	 */
	public destroy(): void {
		this.abortRecording();
		this._destroyFrameTargets();
		this._destroyTexturePools();
		this._mainFrameScope.destroy();
		this._pendingFrameTargetInvalidation = false;
	}

	public async executePass(pass: FramePass, context: FrameContext): Promise<void> {
		const session = this._session;
		if (!session) {
			throw new Error("WebGPUFrameOrchestrator has no active frame session.");
		}
		WebGPUFrameSession.assertContext(session, context);
		if (session.state === "skipped") {
			return;
		}
		if (session.state !== "recording") {
			throw new Error(
				`WebGPU frame session cannot execute passes in state "${session.state}".`,
			);
		}
		if (!session.commands.encoder) {
			throw new Error("WebGPU recording frame session has no command encoder.");
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
				this._diagnosticsObserver?.onNodeExecuted?.(node.id);
			}
			return;
		}

		const plan = await this._planStage(
			pass,
			context,
			session.messages,
			session.configuration,
		);
		if (plan.nodes.length === 0) {
			this._warnUnsupportedPass(pass);
			return;
		}
		const compiled = this._graphCompiler.compileStage(plan);
		this._handleGraphDiagnostics(compiled);
		this._diagnosticsObserver?.onGraphCompiled?.(null, [compiled]);
		for (const node of plan.nodes) {
			await this._frameModules.execute(node, session);
			this._diagnosticsObserver?.onNodeExecuted?.(node.id);
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
				this._closeSession();
			}
			return;
		}
		if (session.state !== "recording") {
			throw new Error(
				`WebGPU frame session cannot end from state "${session.state}".`,
			);
		}
		if (this._wholeFrameGraphCompiled) {
			const compiled = this._findCompiledStage("webgpu-present");
			for (const node of compiled?.nodes ?? []) {
				await this._frameModules.execute(node, session);
				this._diagnosticsObserver?.onNodeExecuted?.(node.id);
			}
		} else {
			const finalization = await this._planStage(
				{ stage: "postprocess", executor: "backend", enabled: true, dependsOn: [] },
				session.context,
				session.messages,
				session.configuration,
				true,
			);
			if (finalization.nodes.length > 0) {
				const compiled = this._graphCompiler.compileStage(finalization);
				this._handleGraphDiagnostics(compiled);
				this._diagnosticsObserver?.onGraphCompiled?.(null, [compiled]);
				for (const node of finalization.nodes) {
					await this._frameModules.execute(node, session);
					this._diagnosticsObserver?.onNodeExecuted?.(node.id);
				}
			}
		}
		this._graphCompiler.seal();
		await this._frameModules.finalizeRecording(session);
		const committing = WebGPUFrameSession.beginCommit(session);
		this._transitionSession(committing);

		try {
			await committing.commands.commit("main:final", async () => {
				await this._frameModules.afterSubmit(committing);
				await postSubmit?.();
			});
		} finally {
			this._diagnosticsObserver?.onCommitSettled?.(
				committing.commands.getDebugState(),
			);
			this._closeSession();
		}
	}

	/** @internal Discards active native frame recording without logical publication. */
	public abortRecording(_error?: unknown): void {
		const session = this._session;
		if (session?.state === "recording" || session?.state === "committing") {
			session.commands.abort();
			this._diagnosticsObserver?.onCommitSettled?.(
				session.commands.getDebugState(),
			);
		}
		this._closeSession();
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

	/** @internal Records a backend pass that bypasses logical resource analysis. */
	public recordOpaqueGraphStage(stage: string, message: string): void {
		if (this._wholeFrameGraphCompiled) return;
		this._graphCompiler.recordOpaqueStage(stage, message);
	}

	private _collectInitialGraphResources() {
		return collectActiveWebGPUFrameGraphResources(
			this._frameTargetManager.frameTargets,
			this._frameTargetManager.msaaTargets,
		);
	}

	private async _compileWholeFrameGraph(
		context: FrameContext,
		messages: WebGPUFrameMessageSnapshot,
		configuration: WebGPUFrameConfiguration | null,
	): Promise<void> {
		const includeShadowResources =
			context.framePlan?.backendPasses.some(
				(pass) => pass.enabled && pass.stage === "shadow",
			) === true;
		const catalog = collectWebGPUFrameGraphResourceCatalog(
			this._frameTargetManager.frameTargets,
			this._frameTargetManager.msaaTargets,
			Math.max(1, this._frameTargetManager.targetWidth),
			Math.max(1, this._frameTargetManager.targetHeight),
			this._frameTargetManager.targetSampleCount,
			this._graphPhysicalResources,
			includeShadowResources,
		);
		const stages: WebGPUFrameGraphStagePlan[] = [];
		const graphImportResources: RenderGraphResourceDescriptor[] = [];
		const shadowDiagnostics: RenderGraphDiagnostic[] = [];
		const setupPass: FramePass = {
			stage: "webgpu-setup",
			executor: "backend",
			enabled: true,
			dependsOn: [],
		};
		stages.push(await this._planStage(setupPass, context, messages, configuration));

		let hasOpaqueStage = false;
		let lastStage = setupPass.stage;
		for (const pass of context.framePlan?.backendPasses ?? []) {
			if (!pass.enabled) continue;
			const stagePlan = await this._planStage(
				pass,
				context,
				messages,
				configuration,
			);
			graphImportResources.push(...(stagePlan.imports ?? []));
			if (stagePlan.nodes.some((node) => node.opaque === true)) {
				hasOpaqueStage = true;
				shadowDiagnostics.push({
					phase: "compile",
					enforcement: "shadow",
					severity: "warning",
					code: "opaque-stage-effects",
					stage: pass.stage,
					message: `WebGPU stage "${pass.stage}" has undeclared resource effects.`,
				});
			}
			if (
				stagePlan.nodes.length === 0 &&
				!stagePlan.composition &&
				pass.stage !== "postprocess"
			) this._warnUnsupportedPass(pass);
			stages.push(stagePlan);
			if (stagePlan.nodes.length > 0 || stagePlan.composition) lastStage = pass.stage;
		}

		const presentationPass: FramePass = {
			stage: "webgpu-present",
			executor: "backend",
			enabled: true,
			dependsOn: [lastStage],
		};
		stages.push(await this._planStage(
			presentationPass,
			context,
			messages,
			configuration,
			true,
		));
		const framePlan: WebGPUFrameGraphFramePlan = {
			resources: [...catalog.resources, ...graphImportResources],
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
		this._diagnosticsObserver?.onGraphCompiled?.(compiled, compiled.stages);
		this._diagnosticsObserver?.onNodeExecuted?.("webgpu-setup:scene:frame-setup");
		this._wholeFrameGraphCompiled = true;
	}

	private async _executePostProcessStage(
		compiled: WebGPUCompiledFrameGraphStage | undefined,
	): Promise<void> {
		await this._frameModules.executeComposedStage(
			compiled,
			this._graphCompiler,
			(nodeId) => this._diagnosticsObserver?.onNodeExecuted?.(nodeId),
		);
	}

	private _planStage(
		pass: FramePass,
		context: FrameContext,
		messages: WebGPUFrameMessageSnapshot,
		configuration: WebGPUFrameConfiguration | null,
		finalization = false,
		finalColorResource?: string,
	): Promise<WebGPUFrameGraphStagePlan> {
		return this._frameModules.planStage({
			pass,
			context,
			state: this._createPlannerState(configuration),
			finalization,
			finalColorResource,
		}, messages);
	}

	private _createPlannerState(
		configuration: WebGPUFrameConfiguration | null,
	): WebGPUFrameResourceAllocationSnapshot {
		return {
			deferredActive: configuration?.deferredActive ?? false,
			oitActive: configuration?.oitActive ?? false,
			sceneTargetMode: configuration?.mrtSupported
				? this._frameTargetManager.targetSceneTargetMode
				: "single",
			deferredGBufferLayout:
				configuration?.deferredGBufferLayout ?? "extended",
			hasFrameTargets: !!this._frameTargetManager.frameTargets,
			hasMSAATargets: !!this._frameTargetManager.msaaTargets,
			needsTransmissionTargets:
				!!this._frameTargetManager.frameTargets?.transmissionSceneColorCopy,
			needsPlanarReflectionMask:
				!!this._frameTargetManager.frameTargets?.planarReflectionMask,
			needsOcclusionTest: configuration?.needsOcclusionTest === true,
			needsHiZBuild:
				configuration?.needsHiZBuild === true &&
				!!this._frameTargetManager.frameTargets?.hiZ,
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
			this._diagnosticsObserver?.onNodeExecuted?.(node.id);
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

	private _destroyFrameTargets(): void {
		this._frameModules.invalidateFrameResources();
		this._graphPhysicalResources.clear();
		this._frameTargetManager.destroyFrameTargets();
		this._diagnosticsObserver?.onTargetsConfigured?.(
			this._frameTargetManager.getDebugState(),
		);
	}

	private _transitionSession(next: WebGPUFrameSession): void {
		const previous = this._session?.state ?? null;
		this._session = next;
		this._emitSessionTransition(previous, next.state);
	}

	private _closeSession(flushPendingLifecycle = true): void {
		const previous = this._session?.state ?? null;
		this._frameModules.closeFrame();
		this._session = null;
		this._emitSessionTransition(previous, null);
		if (flushPendingLifecycle) {
			this._flushPendingLifecycleInvalidations();
		}
	}

	private _emitSessionTransition(
		previous: WebGPUFrameSessionState | null,
		next: WebGPUFrameSessionState | null,
	): void {
		const observer = this._diagnosticsObserver?.onSessionTransition;
		if (!observer) return;
		observer.call(this._diagnosticsObserver, Object.freeze({ previous, next }));
	}

	private _flushPendingLifecycleInvalidations(): void {
		const applyFrameTargetInvalidation = this._pendingFrameTargetInvalidation;
		this._pendingFrameTargetInvalidation = false;
		if (applyFrameTargetInvalidation) {
			this._invalidateFrameTargetsNow();
		}
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
