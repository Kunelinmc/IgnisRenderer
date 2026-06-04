import type {
	DrawPacket,
	FrameContext,
	FramePass,
} from "../../../pipeline/types";
import type {
	LogicalGBufferBridge,
	PostProcessPass,
	PostProcessPassExecutionContextRequest,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
	ResolvedPostProcessPass,
} from "../../../postprocess";
import {
	GAMMA_PASS_ID,
	SCREEN_SPACE_REFLECTIONS_PASS_ID,
	resolvePostProcessExecutionOrder,
} from "../../../postprocess";
import type { ICommandEncoder } from "../../ICommandEncoder";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	TextureFormat,
	TextureUsage,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from "../../types";
import type { WebGPUBackend } from "../../WebGPUBackend";
import type {
	WebGPUPreparedFrameResources,
	WebGPURenderResources,
} from "../WebGPURenderResources";
import { resolveWebGPUComputeFacade } from "../ComputeFacade";
import { loadWebGPUUtilityShaderComposite } from "../../../shaders/webgpu/shaderSource";
import {
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_TARGET_COUNT,
	WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT,
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
} from "../constants";
import {
	isWebGPUPostProcessContextMetadata,
	type WebGPUFrameTargets,
	type WebGPUPostProcessContextMetadata,
	type WebGPUPostProcessFrameTargets,
} from "../WebGPUPostProcessContracts";
import {
	WebGPUPostProcessRuntime,
} from "../WebGPUPostProcessRuntime";
import {
	getDefaultWebGPUDrawBindings,
	submitWebGPUDraws,
} from "../WebGPUDrawSubmission";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../../pipeline/WarmupPlanner";
import {
	WARMUP_POST_PROCESS_DESCRIPTORS_TRANSIENT_KEY,
	toShaderCompileError,
} from "../../../pipeline/WarmupPlanner";
import type { ShaderCompileError } from "../../../shaders/runtime";
import { Logger } from "../../../foundation/Logger";
import { materialUsesTransmission } from "../../../materials/transparency";
import { ParticleBlendMode } from "../../../particles";
import { materialSupportsWebGPUDeferredLighting } from "../material";
import {
	WebGPUPlanarReflectionPass,
	type WebGPUPlanarReflectionMSAATargets,
} from "../WebGPUPlanarReflectionPass";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import {
	WebGPUFrameTargetManager,
	type WebGPUFrameMSAATargets,
	type WebGPUFrameTargetRequirements,
} from "./WebGPUFrameTargetManager";
import { WebGPUFrameGraphPlanner } from "./WebGPUFrameGraphPlanner";
import { WebGPUFrameGraphCompiler } from "./WebGPUFrameGraphCompiler";
import { WebGPUPostProcessBridge } from "./WebGPUPostProcessBridge";
import { WebGPUOITPass } from "./WebGPUOITPass";
import { WebGPUDeferredLightingPass } from "./WebGPUDeferredLightingPass";
import { WebGPUDeferredDecalPass } from "./WebGPUDeferredDecalPass";
import type {
	WebGPUCompiledFrameGraphStage,
	WebGPUFrameGraphDebugState,
	WebGPUFrameGraphNode,
	WebGPUFrameGraphNodeKind,
	WebGPUFrameGraphValidationMode,
} from "./types";
import { WebGPUPresentPass } from "./WebGPUPresentPass";

const WEBGPU_OIT_DISABLED_MRT_KEY = "webgpu-oit-disabled-mrt-unavailable";
const WEBGPU_OIT_DISABLED_MSAA_KEY = "webgpu-oit-disabled-msaa";
const WEBGPU_OIT_DISABLED_RUNTIME_KEY = "webgpu-oit-disabled-runtime";

export class WebGPUFrameGraphRuntime {
	private _backend: WebGPUBackend;
	private _resources: WebGPURenderResources;
	private _encoder: ICommandEncoder | null = null;
	private _frameContext: FrameContext | null = null;
	private _frameResources: WebGPUPreparedFrameResources | null = null;
	private _hasPresentedInFrame = false;
	private _mrtEnabled = true;
	private _mrtSupportChecked = false;
	private _deferredEnabled = false;
	private _postRuntime: WebGPUPostProcessRuntime;
	private _postBridge: WebGPUPostProcessBridge;
	private _presentShaderModule: IShaderModule | null = null;
	private _presentPipeline: IRenderPipeline | null = null;
	private _presentSampler: ISampler | null = null;
	private _presentParamsBuffer: IRenderBuffer | null = null;
	private _presentBinding: IBindingGroup | null = null;
	private _presentBindingSource: IRenderTexture | null = null;
	private _oitActive = false;
	private _motionHistoryWriteTarget: IRenderTexture | null = null;
	private _pendingPostProcessColorTarget: IRenderTexture | null = null;
	private _depthDirtyClearShaderModule: IShaderModule | null = null;
	private _depthDirtyClearPipelines = new Map<string, IRenderPipeline>();
	private _pendingFrameTargetInvalidation = false;
	private _pendingShaderRuntimeInvalidation = false;
	private _enableEarlyZPrepass = true;
	private _enableDeferredLighting = true;
	private _planarReflectionPass: WebGPUPlanarReflectionPass;
	private _presentPass: WebGPUPresentPass;
	private _frameTargetManager: WebGPUFrameTargetManager;
	private _oitPass: WebGPUOITPass;
	private _deferredLightingPass: WebGPUDeferredLightingPass;
	private _deferredDecalPass: WebGPUDeferredDecalPass;
	private readonly _graphPlanner = new WebGPUFrameGraphPlanner();
	private readonly _graphCompiler = new WebGPUFrameGraphCompiler();
	private readonly _nodeExecutors: Map<
		WebGPUFrameGraphNodeKind,
		(node: WebGPUFrameGraphNode, context: FrameContext) => Promise<void>
	>;
	private _lastPlannedGraphNodes: WebGPUFrameGraphNode[] = [];
	private _lastCompiledGraphStages: WebGPUCompiledFrameGraphStage[] = [];
	private _lastExecutedGraphNodeIds: string[] = [];
	private _frameGraphValidationMode: WebGPUFrameGraphValidationMode = "throw";

	constructor(backend: WebGPUBackend, resources: WebGPURenderResources) {
		this._backend = backend;
		this._resources = resources;
		const backendOptions = this._backend as {
			isEarlyZPrepassEnabled?: () => boolean;
			enableEarlyZPrepass?: boolean;
			isDeferredLightingEnabled?: () => boolean;
			enableDeferredLighting?: boolean;
			getFrameGraphValidationMode?: () => WebGPUFrameGraphValidationMode;
			frameGraphValidation?: WebGPUFrameGraphValidationMode;
		};
		const earlyZGetter = backendOptions.isEarlyZPrepassEnabled;
		this._enableEarlyZPrepass =
			typeof earlyZGetter === "function" ?
				earlyZGetter.call(this._backend)
			:	backendOptions.enableEarlyZPrepass !== false;
		const deferredLightingGetter = backendOptions.isDeferredLightingEnabled;
		this._enableDeferredLighting =
			typeof deferredLightingGetter === "function" ?
				deferredLightingGetter.call(this._backend)
			:	backendOptions.enableDeferredLighting !== false;
		const validationGetter = backendOptions.getFrameGraphValidationMode;
		this._frameGraphValidationMode =
			typeof validationGetter === "function" ?
				validationGetter.call(this._backend)
			:	backendOptions.frameGraphValidation ?? "throw";
		const computeFacade = resolveWebGPUComputeFacade(backend);
		this._postRuntime = new WebGPUPostProcessRuntime(
			computeFacade,
			(key, message) =>
				Logger.warn(`[${key}] ${message}`, {
					scope: "WebGPUFrameExecutor",
					onceKey: key,
				}),
			resources.sceneFrameLayout
		);
		this._postBridge = new WebGPUPostProcessBridge(
			backend,
			this._postRuntime,
			{
				getEncoder: () => this._encoder,
				getFrameTargets: () => this._frameTargets,
				requireFrameResources: () => this._requireFrameResources(),
				presentToCanvas: (source, applyGamma) =>
					this._presentToCanvas(source, applyGamma),
				warmupPresent: () => this._ensurePresentResources(),
				setMotionHistoryWriteTarget: (texture) => {
					this._motionHistoryWriteTarget = texture;
				},
			}
		);
		this._planarReflectionPass = new WebGPUPlanarReflectionPass(
			backend,
			resources
		);
		this._presentPass = new WebGPUPresentPass(backend);
		this._frameTargetManager = new WebGPUFrameTargetManager(backend, {
			resolveMSAASampleCount: () => this._resolveMSAASampleCount(),
			configureDeferredLightingSupport: () =>
				this._configureDeferredLightingSupport(),
			frameHasDeferredLightingWork: (context) =>
				this._frameHasDeferredLightingWork(context),
			getFrameContext: () => this._frameContext,
			isDeferredEnabled: () => this._deferredEnabled,
			setDeferredEnabled: (enabled) => {
				this._deferredEnabled = enabled;
			},
		});
		this._oitPass = new WebGPUOITPass(backend, resources, {
			getEncoder: () => this._encoder,
			getFrameTargets: () => this._frameTargets,
			getMSAATargets: () => this._msaaTargets,
			getTargetWidth: () => this._targetWidth,
			getTargetHeight: () => this._targetHeight,
			getSceneTargetMode: () => this._targetSceneTargetMode,
			requireFrameResources: () => this._requireFrameResources(),
			resolveDirtyRects: (context, width, height) =>
				this._resolveDirtyRects(context, width, height),
			resolveTransparentSubsetForRect: (context, packets, rect) =>
				this._resolveTransparentSubsetForRect(context, packets, rect),
			recordLegacyMainPass: (context, packets, clear, earlyZ) =>
				this._recordLegacyMainPass(context, packets, clear, earlyZ),
			drawTransmissionFallback: (context, packets) =>
				this._drawTransmissionPackets(context, packets),
			warnDisabled: (key, message) => this._warnOITDisabled(key, message),
		});
		this._deferredLightingPass = new WebGPUDeferredLightingPass(
			backend,
			resources,
			{
				getEncoder: () => this._encoder,
				getFrameTargets: () => this._frameTargets,
				requireFrameResources: () => this._requireFrameResources(),
				resolveDirtyRects: (context, width, height) =>
					this._resolveDirtyRects(context, width, height),
			}
		);
		this._deferredDecalPass = new WebGPUDeferredDecalPass(
			backend,
			resources,
			{
				getEncoder: () => this._encoder,
				getFrameTargets: () => this._frameTargets,
				requireFrameResources: () => this._requireFrameResources(),
				resolveDirtyRects: (context, width, height) =>
					this._resolveDirtyRects(context, width, height),
			}
		);
		this._nodeExecutors = this._createNodeExecutors();
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
		this._frameContext = context;
		this._hasPresentedInFrame = false;
		this._oitActive = false;
		this._oitPass.resetFrameState();
		this._motionHistoryWriteTarget = null;
		this._pendingPostProcessColorTarget = null;
		this._postBridge.clearPendingFrameState();
		this._lastPlannedGraphNodes = [];
		this._lastCompiledGraphStages = [];
		this._lastExecutedGraphNodeIds = [];
		this._graphCompiler.beginFrame([]);
		const targetWidth = this._resolveAttachmentDimension(
			context.attachments.width
		);
		const targetHeight = this._resolveAttachmentDimension(
			context.attachments.height
		);

		if (targetWidth <= 0 || targetHeight <= 0) {
			this._destroyFrameTargets();
			this._frameContext = null;
			this._frameResources = null;
			this._encoder = null;
			return;
		}

		this._encoder = this._backend.createCommandEncoder();

		this._ensureMRTSupport();
		this._configureDeferredLightingSupport();
		const targetRequirements = this._resolveFrameTargetRequirements(context);
		this._deferredEnabled =
			targetRequirements?.sceneTargetMode === "gbuffer";
		if (this._mrtEnabled && targetRequirements) {
			this._ensureFrameTargets(targetWidth, targetHeight, targetRequirements);
		} else {
			this._destroyFrameTargets();
		}
		this._configureOIT(context);
		this._graphCompiler.beginFrame(this._collectInitialGraphResources());
		this.prepareFrameResources(context);
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
		context: FrameContext
	): WebGPUPreparedFrameResources | null {
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
		desc: PostProcessResourceDescriptor
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
			pendingShaderRuntimeInvalidation:
				this._pendingShaderRuntimeInvalidation,
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
			throw new Error(
				"WebGPUFrameExecutor requires prepared main-frame resources."
			);
		}
		return this._frameResources;
	}

	public getPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
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
		result: PostProcessPassResult
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
		this._destroyManagedResource(this._depthDirtyClearShaderModule);
		for (const pipeline of this._depthDirtyClearPipelines.values()) {
			this._destroyManagedResource(pipeline);
		}
		this._depthDirtyClearShaderModule = null;
		this._depthDirtyClearPipelines.clear();
		this._postRuntime.onShaderRuntimeChanged();
		this._planarReflectionPass.destroy();
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan
	): Promise<WarmupPhaseCounters> {
		let total = 1;
		let compiled = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];
		this._ensureMRTSupport();
		this._configureDeferredLightingSupport();

		try {
			await this._ensurePresentResources();
			compiled++;
		} catch (error) {
			failed++;
			errors.push(toShaderCompileError(error, "webgpu", "WebGPUPresentWarmup"));
		}

		const descriptorById = this._getWarmupPostProcessDescriptorMap(context);
		const hints = new Set<string>();
		if (plan.includePostProcess) {
			for (const passId of plan.postProcessPasses) {
				const implementation = descriptorById
					.get(passId)
					?.getImplementation("webgpu");
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
		}

		const warmedPassImplementations = new Set<string>();
		for (const passId of plan.postProcessPasses) {
			if (warmedPassImplementations.has(passId)) {
				continue;
			}
			const implementation = descriptorById
				.get(passId)
				?.getImplementation("webgpu");
			if (typeof implementation?.warmup !== "function") {
				continue;
			}
			warmedPassImplementations.add(passId);
			total++;
			try {
				const warmupContext =
					this._getPassWarmupExecutionContext(implementation);
				await implementation.warmup(warmupContext, {
					frameContext: context,
					postProcess: context.postProcess,
					backend: "webgpu",
					context: warmupContext,
					options:
						context.postProcess.getOptions(passId) ??
						descriptorById.get(passId)?.normalizeOptions({
							frameContext: context,
							postProcess: context.postProcess,
							backend: "webgpu",
						}),
				});
				compiled++;
			} catch (error) {
				failed++;
				errors.push(
					toShaderCompileError(
						error,
						"webgpu",
						`WebGPUPostWarmup:${passId}`
					)
				);
			}
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

	private _getWarmupPostProcessDescriptorMap(
		context: FrameContext
	): Map<string, PostProcessPass> {
		const descriptors =
			context.transient?.get(WARMUP_POST_PROCESS_DESCRIPTORS_TRANSIENT_KEY) ??
			context.postProcess.getEnabledPasses().map((pass) => pass.pass);
		return new Map(descriptors.map((pass) => [pass.id, pass]));
	}

	private _getPassWarmupExecutionContext(
		implementation: PostProcessPassImplementation
	): unknown {
		const metadata = implementation.metadata?.context;
		if (!isWebGPUPostProcessContextMetadata(metadata)) {
			return undefined;
		}
		return this._postBridge.getPassWarmupExecutionContext(metadata);
	}

	/**
	 * Release all GPU resources held by this executor.
	 */
	public destroy(): void {
		this._destroyFrameTargets();
		this._destroyTexturePools();
		this._postRuntime.destroy();
		this._planarReflectionPass.destroy();
		this._presentPass.destroy();
		this._oitPass.destroy();
		this._destroyManagedResource(this._depthDirtyClearShaderModule);
		for (const pipeline of this._depthDirtyClearPipelines.values()) {
			this._destroyManagedResource(pipeline);
		}
		this._depthDirtyClearShaderModule = null;
		this._depthDirtyClearPipelines.clear();
		this._pendingFrameTargetInvalidation = false;
		this._pendingShaderRuntimeInvalidation = false;
		this._clearActiveFrameState(false);
	}

	public async executePass(
		pass: FramePass,
		context: FrameContext
	): Promise<void> {
		if (!this._encoder) return;

		const plan = this._graphPlanner.planStage(pass, context, {
			deferredActive: this._deferredEnabled,
			oitActive: this._oitActive,
			sceneTargetMode: this.getSceneTargetModeForFrame(),
			hasFrameTargets: !!this._frameTargets,
			hasMSAATargets: !!this._msaaTargets,
			needsPlanarReflectionMask: !!this._frameTargets?.planarReflectionMask,
		});
		if (plan.nodes.length === 0) {
			const key = `webgpu-pass-unsupported-${pass.stage}`;
			Logger.warn(
				`[${key}] WebGPU backend does not support pass "${pass.stage}" yet; skipping`,
				{ scope: "WebGPUFrameGraphRuntime", onceKey: key }
			);
			return;
		}
		const compiled = this._graphCompiler.compileStage(plan);
		this._handleGraphDiagnostics(compiled);
		this._lastCompiledGraphStages = this._graphCompiler
			.getCompiledStages()
			.slice();
		this._lastPlannedGraphNodes = [...plan.nodes];
		for (const node of plan.nodes) {
			await this._executeGraphNode(node, context);
			this._lastExecutedGraphNodeIds.push(node.id);
		}
	}

	private async _executeGraphNode(
		node: WebGPUFrameGraphNode,
		context: FrameContext
	): Promise<void> {
		const executor = this._nodeExecutors.get(node.kind);
		if (!executor) {
			throw new Error(
				`WebGPU frame graph node kind "${node.kind}" has no executor.`
			);
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
					await this._resources.renderShadows(
						context,
						this._encoder ?? undefined
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
					await this._recordOpaquePass(context);
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
					await this._recordMainPass(
						context,
						context.scene.transparentPackets,
						false,
						false
					);
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
					await this._recordParticlePass(context);
				},
			],
		]);
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
			this._mrtEnabled && this._motionHistoryWriteTarget ?
				this._frameTargets?.gMotionDepth
			:	null;
		const motionTarget =
			this._mrtEnabled ? this._motionHistoryWriteTarget : null;

		try {
			if (this._mrtEnabled && this._frameTargets && !this._hasPresentedInFrame) {
				await this._presentToCanvas(
					this._frameTargets.sceneColor,
					this._frameContext?.postProcess.isEnabled("gamma") !== false
				);
			}

			this._backend.submit([encoder.finish()]);
			if (motionSource && motionTarget && width > 0 && height > 0) {
				this._backend.copyTextureToTexture(
					{ texture: motionSource },
					{ texture: motionTarget },
					{ width, height, depthOrArrayLayers: 1 }
				);
			}
		} finally {
			this._clearActiveFrameState();
		}
	}

	public abortFrame(): void {
		this._clearActiveFrameState();
	}

	private _collectInitialGraphResources(): string[] {
		const resources = new Set<string>([
			"canvas:scene-color-main",
			"canvas:depth",
		]);
		const targets = this._frameTargets;
		if (targets) {
			resources.add("frame:scene-color-main");
			resources.add("frame:depth");
			if (targets.postPing) resources.add("post:ping");
			if (targets.postPong) resources.add("post:pong");
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

	private _handleGraphDiagnostics(
		compiled: WebGPUCompiledFrameGraphStage
	): void {
		const errors = compiled.diagnostics.filter(
			(diagnostic) => diagnostic.severity === "error"
		);
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
			scope: "WebGPUFrameGraphRuntime",
			onceKey: `webgpu-frame-graph-validation:${compiled.pass.stage}`,
		});
	}

	private _ensureMRTSupport(): void {
		if (this._mrtSupportChecked) return;
		this._mrtSupportChecked = true;

		const maxColorAttachments =
			this._backend.device?.limits?.maxColorAttachments ?? 8;
		const maxColorAttachmentBytesPerSample =
			this._backend.device?.limits?.maxColorAttachmentBytesPerSample ?? 32;

		if (
			maxColorAttachments >= WEBGPU_MRT_COLOR_TARGET_COUNT &&
			maxColorAttachmentBytesPerSample >= WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE
		) {
			return;
		}

		this._mrtEnabled = false;
		if (maxColorAttachments < WEBGPU_MRT_COLOR_TARGET_COUNT) {
			const key = "webgpu-mrt-disabled-attachments";
			Logger.warn(
				`[${key}] WebGPU device maxColorAttachments is ${maxColorAttachments}, requires ${WEBGPU_MRT_COLOR_TARGET_COUNT}; disabling MRT/GBuffer post-process pipeline`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
		if (maxColorAttachmentBytesPerSample < WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE) {
			const key = "webgpu-mrt-disabled-bytes";
			Logger.warn(
				`[${key}] WebGPU device maxColorAttachmentBytesPerSample is ${maxColorAttachmentBytesPerSample}, requires ${WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE}; disabling MRT/GBuffer post-process pipeline`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
	}

	private _configureDeferredLightingSupport(): void {
		if (!this._enableDeferredLighting) {
			this._deferredEnabled = false;
			return;
		}
		if (!this._mrtEnabled) {
			this._deferredEnabled = false;
			const key = "webgpu-deferred-disabled-mrt";
			Logger.warn(
				`[${key}] WebGPU deferred lighting requires MRT scene targets; using the non-deferred fallback path.`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
			return;
		}

		const sampleCount = this._resolveMSAASampleCount();
		const maxColorAttachments =
			this._backend.device?.limits?.maxColorAttachments ?? 8;
		const maxColorAttachmentBytesPerSample =
			this._backend.device?.limits?.maxColorAttachmentBytesPerSample ?? 32;
		const maxStorageTexturesPerShaderStage =
			this._backend.device?.limits?.maxStorageTexturesPerShaderStage ?? 4;

		const supportsDeferred =
			sampleCount === 1 &&
			maxColorAttachments >= WEBGPU_DEFERRED_COLOR_TARGET_COUNT &&
			maxColorAttachmentBytesPerSample >=
				WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE &&
			maxStorageTexturesPerShaderStage >=
				WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT;

		this._deferredEnabled = supportsDeferred;
		if (supportsDeferred) {
			return;
		}

		if (sampleCount !== 1) {
			const key = "webgpu-deferred-disabled-msaa";
			Logger.warn(
				`[${key}] WebGPU deferred lighting requires sampleCount=1; using legacy MRT forward path for ${sampleCount}x MSAA.`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
		if (maxColorAttachments < WEBGPU_DEFERRED_COLOR_TARGET_COUNT) {
			const key = "webgpu-deferred-disabled-attachments";
			Logger.warn(
				`[${key}] WebGPU device maxColorAttachments is ${maxColorAttachments}, requires ${WEBGPU_DEFERRED_COLOR_TARGET_COUNT}; using legacy MRT forward path.`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
		if (
			maxColorAttachmentBytesPerSample <
			WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE
		) {
			const key = "webgpu-deferred-disabled-bytes";
			Logger.warn(
				`[${key}] WebGPU device maxColorAttachmentBytesPerSample is ${maxColorAttachmentBytesPerSample}, requires ${WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE}; using legacy MRT forward path.`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
		if (
			maxStorageTexturesPerShaderStage <
			WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT
		) {
			const key = "webgpu-deferred-disabled-storage-textures";
			Logger.warn(
				`[${key}] WebGPU device maxStorageTexturesPerShaderStage is ${maxStorageTexturesPerShaderStage}, requires ${WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT}; using legacy MRT forward path.`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
	}

	private _resolveFrameTargetRequirements(
		context: FrameContext
	): WebGPUFrameTargetRequirements | null {
		if (!this._mrtEnabled) {
			return null;
		}
		const postProcessPasses = resolvePostProcessExecutionOrder(
			context.postProcess,
			{
				backend: "webgpu",
				frameContext: context,
			}
		);
		const needsPostProcessTargets = postProcessPasses.some(
			(resolved) => resolved.id !== GAMMA_PASS_ID
		);
		const needsPostProcessGBuffer = this._postProcessNeedsGBuffer(
			context,
			postProcessPasses
		);
		const needsPlanarReflection =
			context.features.enableReflection &&
			context.scene.reflectivePackets.length > 0;
		const needsPlanarReflectionMask =
			needsPlanarReflection ||
			postProcessPasses.some(
				(resolved) => resolved.id === SCREEN_SPACE_REFLECTIONS_PASS_ID
			);
		const msaaSampleCount = this._resolveMSAASampleCount();
		const needsOITTargets =
			msaaSampleCount <= 1 &&
			context.features.enableOIT === true &&
			this._frameHasOITWork(context);
		const enableDeferred =
			this._deferredEnabled && this._frameHasDeferredLightingWork(context);
		if (
			!enableDeferred &&
			!needsPostProcessTargets &&
			!needsPlanarReflection &&
			!needsOITTargets
		) {
			return null;
		}
		const sceneTargetMode: Exclude<WebGPUSceneTargetMode, "single"> =
			enableDeferred ? "gbuffer"
			: needsPostProcessGBuffer ? "mrt"
			: "color";
		return {
			sceneTargetMode,
			needsPostProcessTargets,
			needsOITTargets,
			needsPlanarReflectionMask,
		};
	}

	private _postProcessNeedsGBuffer(
		context: FrameContext,
		passes: readonly ResolvedPostProcessPass[]
	): boolean {
		for (const resolved of passes) {
			if (!resolved.pass.builtIn) {
				return true;
			}
			const requirements = resolved.pass.getRequirements({
				frameContext: context,
				postProcess: context.postProcess,
				backend: "webgpu",
				options: resolved.options,
			});
			if ((requirements.gBuffer?.length ?? 0) > 0) {
				return true;
			}
		}
		return false;
	}

	private _frameHasDeferredLightingWork(context: FrameContext): boolean {
		return context.scene.opaquePackets.some((packet) =>
			materialSupportsWebGPUDeferredLighting(packet.material)
		);
	}

	private _frameHasOITWork(context: FrameContext): boolean {
		return (
			context.scene.transparentPackets.length > 0 ||
			(context.scene.particleSystems?.length ?? 0) > 0
		);
	}

	private _configureOIT(context: FrameContext): void {
		if (context.features.enableOIT !== true) {
			this._oitActive = false;
			return;
		}
		if (!this._frameHasOITWork(context)) {
			this._oitActive = false;
			return;
		}
		const sampleCount = this._resolveMSAASampleCount();
		if (sampleCount > 1) {
			this._warnOITDisabled(
				WEBGPU_OIT_DISABLED_MSAA_KEY,
				"WebGPU OIT v1 only supports sampleCount=1; falling back to legacy transparent rendering."
			);
			this._oitActive = false;
			return;
		}
		if (!this._mrtEnabled || !this._frameTargets) {
			this._warnOITDisabled(
				WEBGPU_OIT_DISABLED_MRT_KEY,
				"WebGPU OIT requires MRT scene targets; falling back to legacy transparent rendering."
			);
			this._oitActive = false;
			return;
		}
		if (
			!this._frameTargets.oitAccum ||
			!this._frameTargets.oitReveal ||
			!this._frameTargets.oitSceneColorCopy
		) {
			this._warnOITDisabled(
				WEBGPU_OIT_DISABLED_RUNTIME_KEY,
				"WebGPU OIT runtime targets are unavailable; falling back to legacy transparent rendering."
			);
			this._oitActive = false;
			return;
		}
		if (typeof this._encoder?.copyTextureToTexture !== "function") {
			this._warnOITDisabled(
				WEBGPU_OIT_DISABLED_RUNTIME_KEY,
				"WebGPU OIT requires in-frame texture-copy support; falling back to legacy transparent rendering."
			);
			this._oitActive = false;
			return;
		}
		this._oitActive = true;
	}

	private _warnOITDisabled(key: string, message: string): void {
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGPUFrameExecutor",
			onceKey: key,
		});
	}

	private _ensureFrameTargets(
		width: number,
		height: number,
		requirementsOrDeferred: WebGPUFrameTargetRequirements | boolean
	): void {
		this._frameTargetManager.ensureFrameTargets(
			width,
			height,
			requirementsOrDeferred
		);
	}

	private _resolveMSAASampleCount(): number {
		const getter = (this._backend as { getMSAASampleCount?: () => number })
			.getMSAASampleCount;
		if (typeof getter !== "function") {
			return 1;
		}
		const sampleCount = getter.call(this._backend);
		if (!Number.isFinite(sampleCount)) {
			return 1;
		}
		return Math.max(1, Math.floor(sampleCount));
	}

	private _destroyFrameTargets(): void {
		this._frameTargetManager.destroyFrameTargets();
		this._presentPass.invalidateBindings();
		this._oitPass.invalidateBindings();
		this._destroyDeferredBindings();
		this._oitActive = false;
		this._oitPass.resetFrameState();
		this._motionHistoryWriteTarget = null;
		this._pendingPostProcessColorTarget = null;
		this._postBridge.clearPendingFrameState();
	}

	private _clearActiveFrameState(flushPendingLifecycle = true): void {
		this._encoder = null;
		this._frameContext = null;
		this._frameResources = null;
		this._motionHistoryWriteTarget = null;
		this._pendingPostProcessColorTarget = null;
		this._hasPresentedInFrame = false;
		this._oitActive = false;
		this._oitPass.resetFrameState();
		if (flushPendingLifecycle) {
			this._flushPendingLifecycleInvalidations();
		}
	}

	private _flushPendingLifecycleInvalidations(): void {
		const applyShaderRuntimeInvalidation =
			this._pendingShaderRuntimeInvalidation;
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
			this._encoder !== null ||
			this._frameContext !== null ||
			this._frameResources !== null
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

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}

	private _destroyDeferredBindings(): void {
		this._deferredLightingPass.destroyBindings();
		this._deferredDecalPass.destroyBindings();
	}

	private _destroyManagedResource(resource: unknown): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(resource);
		}
	}

	private _isIncrementalPartial(context: FrameContext | null): boolean {
		if (!context?.incremental) {
			return false;
		}
		return (
			context.incremental.enabled &&
			!context.incremental.forceFullFrame &&
			context.incremental.dirtyRects.length > 0
		);
	}

	private _resolveDirtyRects(
		context: FrameContext | null,
		targetWidth: number,
		targetHeight: number
	): Array<{ x: number; y: number; width: number; height: number }> {
		const width = Math.max(1, Math.floor(targetWidth));
		const height = Math.max(1, Math.floor(targetHeight));
		if (!context) {
			return [{
				x: 0,
				y: 0,
				width,
				height,
			}];
		}
		if (!this._isIncrementalPartial(context)) {
			return [{
				x: 0,
				y: 0,
				width,
				height,
			}];
		}
		const sourceWidth = Math.max(1, Math.floor(context.attachments.width));
		const sourceHeight = Math.max(1, Math.floor(context.attachments.height));
		const scaleX = width / sourceWidth;
		const scaleY = height / sourceHeight;
		const resolved: Array<{
			x: number;
			y: number;
			width: number;
			height: number;
		}> = [];
		for (const rect of context.incremental.dirtyRects) {
			const minX = Math.max(0, Math.floor(rect.x * scaleX));
			const minY = Math.max(0, Math.floor(rect.y * scaleY));
			const maxX = Math.min(
				width,
				Math.ceil((rect.x + rect.width) * scaleX)
			);
			const maxY = Math.min(
				height,
				Math.ceil((rect.y + rect.height) * scaleY)
			);
			const rectWidth = maxX - minX;
			const rectHeight = maxY - minY;
			if (rectWidth <= 0 || rectHeight <= 0) {
				continue;
			}
			resolved.push({
				x: minX,
				y: minY,
				width: rectWidth,
				height: rectHeight,
			});
		}
		return resolved;
	}

	private _resolvePacketsForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: { x: number; y: number; width: number; height: number }
	): DrawPacket[] {
		const spatialIndex = context.scene.spatialIndex;
		if (!spatialIndex) {
			return packets;
		}
		if (packets === context.scene.opaquePackets) {
			return spatialIndex.queryOpaquePackets(rect);
		}
		if (packets === context.scene.transparentPackets) {
			return spatialIndex.queryTransparentPackets(rect);
		}
		return packets;
	}

	private _resolveTransparentSubsetForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: { x: number; y: number; width: number; height: number }
	): DrawPacket[] {
		const spatialIndex = context.scene.spatialIndex;
		if (!spatialIndex) {
			return packets;
		}
		const rectPackets = spatialIndex.queryTransparentPackets(rect);
		if (packets === context.scene.transparentPackets) {
			return rectPackets;
		}
		if (packets.length <= 0 || rectPackets.length <= 0) {
			return [];
		}
		const packetSet = new Set(packets);
		return rectPackets.filter((packet) => packetSet.has(packet));
	}

	private async _clearDepthForDirtyRects(
		depthAttachment: IRenderTexture,
		depthFormat: TextureFormat,
		sampleCount: number,
		dirtyRects: Array<{ x: number; y: number; width: number; height: number }>
	): Promise<boolean> {
		if (!this._encoder || dirtyRects.length === 0) {
			return false;
		}
		try {
			const pipeline = await this._getDepthDirtyClearPipeline(
				depthFormat,
				sampleCount
			);
			this._encoder.beginRenderPass({
				label: "WebGPUDepthDirtyClear",
				colorAttachments: [],
				depthStencilAttachment: {
					view: depthAttachment,
					depthLoadOp: "load",
					depthStoreOp: "store",
					depthClearValue: 1,
				},
			});
			this._encoder.setPipeline(pipeline);
			for (const rect of dirtyRects) {
				this._encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
				this._encoder.draw(3);
			}
			this._encoder.endRenderPass();
			return true;
		} catch (error) {
			const key = "webgpu-depth-partial-reuse-fallback";
			Logger.warn(
				`[${key}] WebGPU partial depth reuse unavailable; falling back to full depth clear. ${String(error)}`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
			return false;
		}
	}

	private async _getDepthDirtyClearPipeline(
		depthFormat: TextureFormat,
		sampleCount: number
	): Promise<IRenderPipeline> {
		const resolvedSampleCount = Math.max(1, Math.floor(sampleCount || 1));
		const cacheKey = `${depthFormat}|${resolvedSampleCount}`;
		const cached = this._depthDirtyClearPipelines.get(cacheKey);
		if (cached) {
			return cached;
		}

		if (!this._depthDirtyClearShaderModule) {
			const composite =
				await loadWebGPUUtilityShaderComposite("depthDirtyClear");
			this._depthDirtyClearShaderModule = await this._backend.createShaderModule({
				label: "WebGPUDepthDirtyClearShader",
				code: composite.code,
				sourceMap: composite.sourceMap,
				language: "wgsl",
				stage: "unknown",
				sourceKind: "postprocess",
			});
		}

		const pipeline = this._backend.createPipeline({
			label: `WebGPUDepthDirtyClearPipeline_${cacheKey}`,
			vertex: {
				module: this._depthDirtyClearShaderModule,
				entryPoint: "vsMain",
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: "none",
				frontFace: "ccw",
			},
			depthStencil: {
				format: depthFormat,
				depthWriteEnabled: true,
				depthCompare: "always",
			},
			sampleCount: resolvedSampleCount,
		} as any);
		this._depthDirtyClearPipelines.set(cacheKey, pipeline);
		return pipeline;
	}

	private async _ensurePresentResources(): Promise<void> {
		await this._presentPass.warmup();
	}

	private async _presentToCanvas(
		source: IRenderTexture,
		applyGamma: boolean
	): Promise<void> {
		if (!this._encoder) return;
		await this._presentPass.present({
			encoder: this._encoder,
			frameContext: this._frameContext,
			source,
			applyGamma,
			resolveDirtyRects: (context, width, height) =>
				this._resolveDirtyRects(context, width, height),
		});
		this._hasPresentedInFrame = true;
	}

	private async _drawTransmissionPackets(
		context: FrameContext,
		packets: DrawPacket[]
	): Promise<void> {
		if (!this._encoder || packets.length <= 0) {
			return;
		}
		const frameResources = this._requireFrameResources();
		if (!this._mrtEnabled || !this._frameTargets) {
			await this._recordLegacyMainPass(context, packets, false, false);
			return;
		}
		const msaaTargets = this._msaaTargets;
		if (this._targetSceneTargetMode === "color") {
			const sceneColorAttachment =
				msaaTargets?.sceneColorMain ?? this._frameTargets.sceneColorMain;
			const depthAttachment = msaaTargets?.depth ?? this._frameTargets.depth;
			this._encoder.beginRenderPass({
				label: "WebGPUTransmissionColor",
				colorAttachments: [
					{
						view: sceneColorAttachment,
						resolveTarget:
							msaaTargets ? this._frameTargets.sceneColorMain : undefined,
						loadOp: "load",
						storeOp: "store",
					},
				],
				depthStencilAttachment: {
					view: depthAttachment,
					depthLoadOp: "load",
					depthStoreOp: "store",
				},
			});
			const dirtyRects = this._resolveDirtyRects(
				context,
				sceneColorAttachment.width,
				sceneColorAttachment.height
			);
			await submitWebGPUDraws({
				encoder: this._encoder,
				resources: this._resources,
				frameResources,
				packets,
				dirtyRects,
				selectPacketsForRect: (candidatePackets, rect) =>
					this._resolveTransparentSubsetForRect(
						context,
						candidatePackets,
						rect
					),
				resolveDrawOptions: () => ({
					sceneTargetMode: "color",
					transparentPipelineMode: "transmission",
				}),
			});
			this._encoder.endRenderPass();
			return;
		}
		const sceneColorAttachment =
			msaaTargets?.sceneColorMain ?? this._frameTargets.sceneColorMain;
		const gAlbedoAttachment =
			msaaTargets?.gAlbedoAlpha ?? this._frameTargets.gAlbedoAlpha;
		const gNormalAttachment =
			msaaTargets?.gNormalRoughMetal ?? this._frameTargets.gNormalRoughMetal;
		const gEmissiveAttachment =
			msaaTargets?.gEmissiveOcclusion ?? this._frameTargets.gEmissiveOcclusion;
		const gMotionAttachment =
			msaaTargets?.gMotionDepth ?? this._frameTargets.gMotionDepth;
		const depthAttachment = msaaTargets?.depth ?? this._frameTargets.depth;
		this._encoder.beginRenderPass({
			label: "WebGPUTransmissionMRT",
			colorAttachments: [
				{
					view: sceneColorAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.sceneColorMain : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gAlbedoAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gAlbedoAlpha : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gNormalAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gNormalRoughMetal : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gEmissiveAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gEmissiveOcclusion : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gMotionAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gMotionDepth : undefined,
					loadOp: "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});
		const dirtyRects = this._resolveDirtyRects(
			context,
			this._frameTargets.oitAccum.width,
			this._frameTargets.oitAccum.height
		);
		await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolveTransparentSubsetForRect(context, candidatePackets, rect),
			resolveDrawOptions: () => ({
				sceneTargetMode: "mrt",
				transparentPipelineMode: "transmission",
			}),
		});
		this._encoder.endRenderPass();
	}

	private async _recordOITTransparentPass(context: FrameContext): Promise<void> {
		if (!this._encoder) {
			return;
		}
		if (!this._mrtEnabled || !this._frameTargets) {
			await this._recordMainPass(
				context,
				context.scene.transparentPackets,
				false,
				false
			);
			return;
		}
		await this._oitPass.recordTransparentPass(context);
	}

	private async _recordOITParticlePass(context: FrameContext): Promise<void> {
		if (!this._encoder) {
			return;
		}
		if (
			!this._mrtEnabled ||
			!this._frameTargets?.oitAccum ||
			!this._frameTargets.oitReveal
		) {
			await this._recordParticlePass(context);
			return;
		}
		await this._oitPass.recordParticlePass(context);
	}

	private _getGBufferWriteBinding(): IBindingGroup {
		return this._deferredLightingPass.getGBufferWriteBinding();
	}

	private _getGBufferReadBinding(): IBindingGroup {
		return this._deferredLightingPass.getGBufferReadBinding();
	}

	private async _recordPlanarReflectionPass(
		context: FrameContext
	): Promise<void> {
		if (!this._encoder) {
			return;
		}
		this._submitCurrentFrameEncoder();
		await this._planarReflectionPass.capture(context);
		this._encoder = this._backend.createCommandEncoder();
	}

	private async _recordPlanarReflectionComposite(
		context: FrameContext
	): Promise<void> {
		if (!this._encoder || !this._mrtEnabled || !this._frameTargets) {
			return;
		}
		this._clearPlanarReflectionMask();
		await this._planarReflectionPass.composite({
			encoder: this._encoder,
			context,
			frameResources: this._requireFrameResources(),
			frameTargets: this._frameTargets,
			msaaTargets:
				this._msaaTargets as WebGPUPlanarReflectionMSAATargets | null,
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

	private async _recordOpaquePass(context: FrameContext): Promise<void> {
		if (!this._deferredEnabled || !this._mrtEnabled || !this._frameTargets) {
			await this._recordMainPass(
				context,
				context.scene.opaquePackets,
				true,
				true
			);
			await this._recordPlanarReflectionComposite(context);
			return;
		}

		const deferredPackets: DrawPacket[] = [];
		const fallbackPackets: DrawPacket[] = [];
		for (const packet of context.scene.opaquePackets) {
			if (materialSupportsWebGPUDeferredLighting(packet.material)) {
				deferredPackets.push(packet);
			} else {
				fallbackPackets.push(packet);
			}
		}

		if (deferredPackets.length <= 0 && fallbackPackets.length > 0) {
			await this._recordMainPass(context, fallbackPackets, true, true);
			await this._recordPlanarReflectionComposite(context);
			return;
		}

		await this._recordDeferredOpaquePass(
			context,
			deferredPackets,
			true,
			true
		);
		if (fallbackPackets.length > 0) {
			await this._recordMainPass(context, fallbackPackets, false, false);
		}
		await this._recordPlanarReflectionComposite(context);
	}

	private async _recordDeferredOpaquePass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean
	): Promise<void> {
		if (!this._encoder || !this._frameTargets) {
			return;
		}
		if (
			!this._frameTargets.gSpecular ||
			!this._frameTargets.gCoatSheen ||
			!this._frameTargets.gSheenReflectance ||
			!this._frameTargets.gMaterialExt0 ||
			!this._frameTargets.gMaterialExt1 ||
			!this._frameTargets.gMaterialExt2 ||
			!this._frameTargets.gMaterialExt3
		) {
			await this._recordMainPass(context, packets, clearAttachments, true);
			return;
		}

		const frameResources = this._requireFrameResources();
		await this._resources.buildClusteredLighting(this._encoder, frameResources);
		const incrementalPartial = this._isIncrementalPartial(context);
		const sceneColorAttachment = this._frameTargets.sceneColorMain;
		const depthAttachment = this._frameTargets.depth;
		const dirtyRects = this._resolveDirtyRects(
			context,
			sceneColorAttachment.width,
			sceneColorAttachment.height
		);
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._clearDepthForDirtyRects(
				depthAttachment,
				TextureFormat.Depth32Float,
				1,
				dirtyRects
			);
		}

		let environmentDrawn = false;
		if (shouldClearAttachments) {
			const environmentResources =
				await this._resources.getEnvironmentResources(
					frameResources,
					"gbuffer"
				);
			if (environmentResources) {
				this._encoder.beginRenderPass({
					label: "WebGPUEnvironmentDeferred",
					colorAttachments: [
						{
							view: sceneColorAttachment,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "clear",
							storeOp: "store",
						},
					],
					depthStencilAttachment: {
						view: depthAttachment,
						depthClearValue: 1,
						depthLoadOp: "clear",
						depthStoreOp: "store",
					},
				});
				this._encoder.setPipeline(environmentResources.pipeline);
				this._encoder.setBindingGroup(0, environmentResources.frameBinding);
				this._encoder.draw(3);
				this._encoder.endRenderPass();
				environmentDrawn = true;
			}
		}

		const shouldRunEarlyZ =
			allowEarlyZPrepass &&
			this._enableEarlyZPrepass &&
			packets.length > 0;
		const earlyZPacketIds =
			shouldRunEarlyZ ?
				await this._recordEarlyZPrepass(
					context,
					packets,
					dirtyRects,
					"gbuffer",
					depthAttachment,
					this._resolveMRTMainDepthLoadOp(
						depthPartialReuseApplied,
						incrementalPartial,
						shouldClearAttachments,
						environmentDrawn,
						false
					)
				)
			:	new Set<string>();
		const earlyZExecuted = earlyZPacketIds.size > 0;
		const gbufferWriteBinding = this._getGBufferWriteBinding();

		this._encoder.beginRenderPass({
			label:
				shouldClearAttachments ?
					"WebGPUGBuffer_Clear"
				:	"WebGPUGBuffer_Load",
			colorAttachments: [
				{
					view: this._frameTargets.gAlbedoAlpha,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gNormalRoughMetal,
					clearValue: { r: 0.5, g: 0.5, b: 1, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gEmissiveOcclusion,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gMotionDepth,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gSpecular,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gCoatSheen,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gSheenReflectance,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp: this._resolveMRTMainDepthLoadOp(
					depthPartialReuseApplied,
					incrementalPartial,
					shouldClearAttachments,
					environmentDrawn,
					earlyZExecuted
				),
				depthStoreOp: "store",
			},
		});

		await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolvePacketsForRect(context, candidatePackets, rect),
			resolveDrawOptions: (packet) => ({
				sceneTargetMode: "gbuffer",
				drawMode:
					earlyZExecuted && earlyZPacketIds.has(packet.id) ?
						"early-z-color"
					:	"default",
			}),
			resolveBindings: (draw) => [
				...getDefaultWebGPUDrawBindings(draw),
				{ slot: 3, group: gbufferWriteBinding },
			],
		});

		this._encoder.endRenderPass();
		await this._deferredDecalPass.recordDecalPass(context);
		await this._recordDeferredLightingPass(
			context,
			shouldClearAttachments && !environmentDrawn
		);
	}

	private async _recordDeferredLightingPass(
		context: FrameContext,
		clearSceneColor: boolean
	): Promise<void> {
		await this._deferredLightingPass.recordLightingPass(
			context,
			clearSceneColor
		);
	}

	private async _recordMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean
	): Promise<void> {
		if (!this._encoder) return;
		const frameResources = this._requireFrameResources();
		await this._resources.buildClusteredLighting(this._encoder, frameResources);
		const incrementalPartial = this._isIncrementalPartial(context);
		if (!this._mrtEnabled || !this._frameTargets) {
			await this._recordLegacyMainPass(
				context,
				packets,
				clearAttachments,
				allowEarlyZPrepass
			);
			return;
		}
		if (this._targetSceneTargetMode === "color") {
			await this._recordColorMainPass(
				context,
				packets,
				clearAttachments,
				allowEarlyZPrepass,
				frameResources
			);
			return;
		}
		const msaaTargets = this._msaaTargets;
		const sceneColorAttachment =
			msaaTargets?.sceneColorMain ?? this._frameTargets.sceneColorMain;
		const gAlbedoAttachment =
			msaaTargets?.gAlbedoAlpha ?? this._frameTargets.gAlbedoAlpha;
		const gNormalAttachment =
			msaaTargets?.gNormalRoughMetal ?? this._frameTargets.gNormalRoughMetal;
		const gEmissiveAttachment =
			msaaTargets?.gEmissiveOcclusion ?? this._frameTargets.gEmissiveOcclusion;
		const gMotionAttachment =
			msaaTargets?.gMotionDepth ?? this._frameTargets.gMotionDepth;
		const depthAttachment = msaaTargets?.depth ?? this._frameTargets.depth;
		const dirtyRects = this._resolveDirtyRects(
			context,
			sceneColorAttachment.width,
			sceneColorAttachment.height
		);
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._clearDepthForDirtyRects(
				depthAttachment,
				TextureFormat.Depth32Float,
				msaaTargets ? this._targetMSAASampleCount : 1,
				dirtyRects
			);
		}

		let environmentDrawn = false;
		if (shouldClearAttachments) {
			const environmentResources =
				await this._resources.getEnvironmentResources(frameResources, "mrt");
			if (environmentResources) {
				this._encoder.beginRenderPass({
					label: "WebGPUEnvironmentMRT",
					colorAttachments: [
						{
							view: sceneColorAttachment,
							resolveTarget:
								msaaTargets ? this._frameTargets.sceneColorMain : undefined,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "clear",
							storeOp: "store",
						},
					],
					depthStencilAttachment: {
						view: depthAttachment,
						depthClearValue: 1,
						depthLoadOp: "clear",
						depthStoreOp: "store",
					},
				});
				this._encoder.setPipeline(environmentResources.pipeline);
				this._encoder.setBindingGroup(0, environmentResources.frameBinding);
				this._encoder.draw(3);
				this._encoder.endRenderPass();
				environmentDrawn = true;
			}
		}
		const shouldRunEarlyZ =
			allowEarlyZPrepass &&
			this._enableEarlyZPrepass &&
			packets.length > 0;
		const earlyZPacketIds =
			shouldRunEarlyZ ?
				await this._recordEarlyZPrepass(
					context,
					packets,
					dirtyRects,
					"mrt",
					depthAttachment,
					this._resolveMRTMainDepthLoadOp(
						depthPartialReuseApplied,
						incrementalPartial,
						shouldClearAttachments,
						environmentDrawn,
						false
					)
				)
			:	new Set<string>();
		const earlyZExecuted = earlyZPacketIds.size > 0;

		this._encoder.beginRenderPass({
			label:
				shouldClearAttachments ? "WebGPUMainMRT_Clear" : "WebGPUMainMRT_Load",
			colorAttachments: [
				{
					view: sceneColorAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.sceneColorMain : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments && !environmentDrawn ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gAlbedoAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gAlbedoAlpha : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gNormalAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gNormalRoughMetal : undefined,
					clearValue: { r: 0.5, g: 0.5, b: 1, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gEmissiveAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gEmissiveOcclusion : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gMotionAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gMotionDepth : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp: this._resolveMRTMainDepthLoadOp(
					depthPartialReuseApplied,
					incrementalPartial,
					shouldClearAttachments,
					environmentDrawn,
					earlyZExecuted
				),
				depthStoreOp: "store",
			},
		});

		await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolvePacketsForRect(context, candidatePackets, rect),
			resolveDrawOptions: (packet) => ({
				sceneTargetMode: "mrt",
				drawMode:
					earlyZExecuted && earlyZPacketIds.has(packet.id) ?
						"early-z-color"
					:	"default",
			}),
		});

		this._encoder.endRenderPass();
	}

	private async _recordColorMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean,
		frameResources: WebGPUPreparedFrameResources
	): Promise<void> {
		if (!this._encoder || !this._frameTargets) {
			return;
		}
		const msaaTargets = this._msaaTargets;
		const sceneColorAttachment =
			msaaTargets?.sceneColorMain ?? this._frameTargets.sceneColorMain;
		const depthAttachment = msaaTargets?.depth ?? this._frameTargets.depth;
		const incrementalPartial = this._isIncrementalPartial(context);
		const dirtyRects = this._resolveDirtyRects(
			context,
			sceneColorAttachment.width,
			sceneColorAttachment.height
		);
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._clearDepthForDirtyRects(
				depthAttachment,
				TextureFormat.Depth32Float,
				msaaTargets ? this._targetMSAASampleCount : 1,
				dirtyRects
			);
		}

		let environmentDrawn = false;
		if (shouldClearAttachments) {
			const environmentResources =
				await this._resources.getEnvironmentResources(frameResources, "color");
			if (environmentResources) {
				this._encoder.beginRenderPass({
					label: "WebGPUEnvironmentColor",
					colorAttachments: [
						{
							view: sceneColorAttachment,
							resolveTarget:
								msaaTargets ? this._frameTargets.sceneColorMain : undefined,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "clear",
							storeOp: "store",
						},
					],
					depthStencilAttachment: {
						view: depthAttachment,
						depthClearValue: 1,
						depthLoadOp: "clear",
						depthStoreOp: "store",
					},
				});
				this._encoder.setPipeline(environmentResources.pipeline);
				this._encoder.setBindingGroup(0, environmentResources.frameBinding);
				this._encoder.draw(3);
				this._encoder.endRenderPass();
				environmentDrawn = true;
			}
		}

		const shouldRunEarlyZ =
			allowEarlyZPrepass &&
			this._enableEarlyZPrepass &&
			packets.length > 0;
		const earlyZPacketIds =
			shouldRunEarlyZ ?
				await this._recordEarlyZPrepass(
					context,
					packets,
					dirtyRects,
					"color",
					depthAttachment,
					this._resolveMRTMainDepthLoadOp(
						depthPartialReuseApplied,
						incrementalPartial,
						shouldClearAttachments,
						environmentDrawn,
						false
					)
				)
			:	new Set<string>();
		const earlyZExecuted = earlyZPacketIds.size > 0;

		this._encoder.beginRenderPass({
			label:
				shouldClearAttachments ?
					"WebGPUMainColor_Clear"
				:	"WebGPUMainColor_Load",
			colorAttachments: [
				{
					view: sceneColorAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.sceneColorMain : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments && !environmentDrawn ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp: this._resolveMRTMainDepthLoadOp(
					depthPartialReuseApplied,
					incrementalPartial,
					shouldClearAttachments,
					environmentDrawn,
					earlyZExecuted
				),
				depthStoreOp: "store",
			},
		});

		await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolvePacketsForRect(context, candidatePackets, rect),
			resolveDrawOptions: (packet) => ({
				sceneTargetMode: "color",
				drawMode:
					earlyZExecuted && earlyZPacketIds.has(packet.id) ?
						"early-z-color"
					:	"default",
			}),
		});

		this._encoder.endRenderPass();
	}

	private async _recordLegacyMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean
	): Promise<void> {
		if (!this._encoder) return;
		const frameResources = this._requireFrameResources();
		await this._resources.buildClusteredLighting(this._encoder, frameResources);
		const incrementalPartial = this._isIncrementalPartial(context);
		const colorTexture = this._backend.getCanvasColorTexture();
		const depthTexture = this._backend.getCanvasDepthTexture();
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		const dirtyRects = this._resolveDirtyRects(
			context,
			colorTexture.width,
			colorTexture.height
		);
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._clearDepthForDirtyRects(
				depthTexture,
				this._backend.canvasDepthFormat,
				1,
				dirtyRects
			);
		}
		const shouldRunEarlyZ =
			allowEarlyZPrepass &&
			this._enableEarlyZPrepass &&
			packets.length > 0;
		const earlyZPacketIds =
			shouldRunEarlyZ ?
				await this._recordEarlyZPrepass(
					context,
					packets,
					dirtyRects,
					"single",
					depthTexture,
					this._resolveLegacyMainDepthLoadOp(
						depthPartialReuseApplied,
						incrementalPartial,
						shouldClearAttachments,
						false
					)
				)
			:	new Set<string>();
		const earlyZExecuted = earlyZPacketIds.size > 0;

		this._encoder.beginRenderPass({
			colorAttachments: [
				{
					view: colorTexture,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthTexture,
				depthClearValue: 1,
				depthLoadOp: this._resolveLegacyMainDepthLoadOp(
					depthPartialReuseApplied,
					incrementalPartial,
					shouldClearAttachments,
					earlyZExecuted
				),
				depthStoreOp: "store",
			},
		});

		if (shouldClearAttachments) {
			const environmentResources =
				await this._resources.getEnvironmentResources(frameResources, "single");
			if (environmentResources) {
				this._encoder.setPipeline(environmentResources.pipeline);
				this._encoder.setBindingGroup(0, environmentResources.frameBinding);
				this._encoder.draw(3);
			}
		}

		await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolvePacketsForRect(context, candidatePackets, rect),
			resolveDrawOptions: (packet) => ({
				sceneTargetMode: "single",
				drawMode:
					earlyZExecuted && earlyZPacketIds.has(packet.id) ?
						"early-z-color"
					:	"default",
			}),
		});

		this._encoder.endRenderPass();
	}

	private async _recordEarlyZPrepass(
		context: FrameContext,
		packets: DrawPacket[],
		dirtyRects: Array<{ x: number; y: number; width: number; height: number }>,
		sceneTargetMode: WebGPUSceneTargetMode,
		depthAttachment: IRenderTexture,
		depthLoadOp: "clear" | "load"
	): Promise<Set<string>> {
		const prepassedPacketIds = new Set<string>();
		if (!this._encoder || packets.length <= 0) {
			return prepassedPacketIds;
		}
		this._encoder.beginRenderPass({
			label:
				sceneTargetMode === "gbuffer" ? "WebGPUEarlyZPrepassGBuffer"
				: sceneTargetMode === "mrt" ?
					"WebGPUEarlyZPrepassMRT"
				: sceneTargetMode === "color" ?
					"WebGPUEarlyZPrepassColor"
				:	"WebGPUEarlyZPrepassSingle",
			colorAttachments: [],
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp,
				depthStoreOp: "store",
			},
		});

		const submission = await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources: this._requireFrameResources(),
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolvePacketsForRect(context, candidatePackets, rect),
			resolveDrawOptions: () => ({
				sceneTargetMode,
				drawMode: "early-z-prepass",
			}),
		});

		this._encoder.endRenderPass();
		return submission.submittedPacketIds;
	}

	private _resolveMRTMainDepthLoadOp(
		depthPartialReuseApplied: boolean,
		incrementalPartial: boolean,
		shouldClearAttachments: boolean,
		environmentDrawn: boolean,
		earlyZExecuted: boolean
	): "load" | "clear" {
		if (earlyZExecuted || depthPartialReuseApplied) {
			return "load";
		}
		return incrementalPartial || (shouldClearAttachments && !environmentDrawn) ?
				"clear"
			:	"load";
	}

	private _resolveLegacyMainDepthLoadOp(
		depthPartialReuseApplied: boolean,
		incrementalPartial: boolean,
		shouldClearAttachments: boolean,
		earlyZExecuted: boolean
	): "load" | "clear" {
		if (earlyZExecuted || depthPartialReuseApplied) {
			return "load";
		}
		return incrementalPartial || shouldClearAttachments ? "clear" : "load";
	}

	private async _recordParticlePass(context: FrameContext): Promise<void> {
		if (!this._encoder) return;
		const frameResources = this._requireFrameResources();

		if (this._mrtEnabled && this._frameTargets) {
			const msaaTargets = this._msaaTargets;
			const sceneTargetMode =
				this._targetSceneTargetMode === "color" ? "color" : "mrt";
			await this._resources.renderParticles(
				this._encoder,
				context,
				{
					label: "WebGPUParticlesMRT",
					colorAttachments: [
						{
							view:
								msaaTargets?.sceneColorMain ??
								this._frameTargets.sceneColorMain,
							resolveTarget:
								msaaTargets ? this._frameTargets.sceneColorMain : undefined,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "load",
							storeOp: "store",
						},
				],
					depth: msaaTargets?.depth ?? this._frameTargets.depth,
				},
				frameResources,
				sceneTargetMode,
				{
					pipelineMode: "legacy",
				}
			);
			return;
		}

		await this._resources.renderParticles(
			this._encoder,
			context,
			{
				label: "WebGPUParticlesSingle",
				colorAttachments: [
					{
						view: this._backend.getCanvasColorTexture(),
						clearValue: { r: 0, g: 0, b: 0, a: 1 },
						loadOp: "load",
						storeOp: "store",
					},
				],
				depth: this._backend.getCanvasDepthTexture(),
			},
			frameResources,
			"single",
			{
				pipelineMode: "legacy",
			}
		);
	}
}
