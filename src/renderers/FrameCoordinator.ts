import { Camera } from "../cameras/Camera";
import {
	LightProbe,
	LightType,
	ReflectionProbe,
	type ShadowCastingLight,
} from "../lights";
import { Matrix4 } from "../maths/Matrix4";
import { SH } from "../maths/SH";
import { Vector3 } from "../maths/Vector3";
import { sRGBToLinear } from "../maths/Common";
import type { ShadowRenderSet } from "../lights/shadows/ShadowMapping";
import { PBR_AMBIENT_FALLBACK_LINEAR } from "../lights/constants";
import { Scene } from "../core/Scene";
import { Logger } from "../foundation/Logger";
import { CSGMeshInstance } from "../meshes/CSGMeshInstance";
import { LODMeshInstance } from "../meshes/LODMeshInstance";
import { resolveFeatureState } from "../pipeline/FeatureResolver";
import {
	PostProcessPassRegistry,
	type ResolvedPostProcessState,
} from "../postprocess";
import type {
	CustomRenderPassRegistry,
	RenderTargetRegistry,
} from "./CustomRenderTargets";
import { AnimationRuntime } from "../simulation/animation/AnimationRuntime";
import { PreparedSceneBuilder } from "../pipeline/PreparedSceneBuilder";
import {
	PreparedSceneCache,
	type PreparedSceneCacheBuildResult,
} from "../pipeline/PreparedSceneCache";
import { ProbeCaptureRuntime } from "../lights/runtime/ProbeCaptureRuntime";
import type { RendererStageDefinition } from "../pipeline/RendererStageGraph";
import { isLocalizedLightProbe } from "../lights/runtime/lightProbeRuntime";
import {
	createTransientStore,
	type FrameContext,
	type FramePassStage,
	type TransientStore,
} from "../pipeline/types";
import { RenderPipelineRegistry } from "../pipeline/RenderPipelineRegistry";
import type {
	RendererFeatures,
	FrameTransientContributor,
} from "./Renderer";
import type { AnimationSystem } from "../animation/AnimationSystem";
import type { PhysicsSystem } from "../physics";
import type { SHCoefficients } from "../maths/types";
import {
	buildDirtyTileCoverage,
	createIncrementalTileCoverage,
	IncrementalFramePlanner,
	makeFullScreenRect,
	renderDirtyReasonToMask,
	type IncrementalFrameContext,
	type DirtyTileCoverage,
	type IncrementalFrameStats,
	type IncrementalFrameStatus,
	type IncrementalRenderingOptions,
} from "../pipeline/incremental";
import type { IRenderBackend, WarmupOptions, WarmupReport } from "./IRenderBackend";
import {
	PROBE_CAPTURE_EXTENSION,
	type RenderBackendExtensionKey,
} from "./BackendExtensions";
import { RendererOcclusionCullingController } from "./RendererOcclusionCullingController";

const _tmpRendererCameraWorldPosition = { x: 0, y: 0, z: 0 };

type RenderSceneFrame = ReturnType<typeof PreparedSceneBuilder.build>;
type RenderSceneFeatureState = ReturnType<typeof resolveFeatureState>;
type RendererStageExecutor = (
	delegate: FrameCoordinatorDelegate,
	state: RenderSceneFrameState
) => void | Promise<void>;

export interface FrameCoordinatorDelegate {
	readonly canvas: HTMLCanvasElement;
	readonly scene: Scene;
	readonly camera: Camera;
	readonly animationSystem: AnimationSystem;
	readonly physicsSystem: PhysicsSystem | null;
	readonly pipeline: RenderPipelineRegistry;
	readonly postProcess: PostProcessPassRegistry;
	readonly renderTargets: RenderTargetRegistry;
	readonly renderPasses: CustomRenderPassRegistry;
	readonly features: RendererFeatures;
	readonly incrementalOptions: IncrementalRenderingOptions;
	readonly animationAutoRender: boolean;
	readonly frameTransientContributors: Set<FrameTransientContributor>;

	setSHCoefficients(coeffs: SHCoefficients): void;
	setSHAmbientCoefficients(coeffs: SHCoefficients): void;
	setShadowMaps(shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>): void;
	setLastIncrementalFrameStats(stats: IncrementalFrameStats | null): void;

	getBackendExtension<T>(key: RenderBackendExtensionKey<T>): T | null;
	emitPostAnimation(now: number, deltaTime: number, transient: TransientStore): void;
	refreshReflectionProbeCaches(): void;
	warn(key: string, message: string): void;
}

interface RenderSceneFrameState {
	now: number;
	deltaTimeSeconds: number;
	frameDirtyReasonMask: number;
	transient: TransientStore;
	resolved: RenderSceneFeatureState;
	resolvedPostProcess: ResolvedPostProcessState;
	frame: RenderSceneFrame | null;
	context: FrameContext | null;
	frameStarted: boolean;
	emittedPostAnimation: boolean;
	stageOrder: RendererStageDefinition[];
	stageIndexById: Map<string, number>;
	hasAnimationStage: boolean;
	initialFullFrameRect: ReturnType<typeof makeFullScreenRect>;
	initialFullFrameTiles: DirtyTileCoverage;
	incrementalFrameContext: IncrementalFrameContext;
	incrementalStartStageIndex: number;
}

export class FrameCoordinator {
	private readonly _backend: IRenderBackend;
	private readonly _preparedSceneCache = new PreparedSceneCache();
	private readonly _probeCaptureRuntime = new ProbeCaptureRuntime();
	private readonly _occlusionCullingController: RendererOcclusionCullingController;
	private readonly _animationRuntime = new AnimationRuntime();
	private readonly _stageExecutors: Map<string, RendererStageExecutor>;

	constructor(backend: IRenderBackend) {
		this._backend = backend;
		this._occlusionCullingController = new RendererOcclusionCullingController(
			backend.extensions,
		);
		this._stageExecutors = this._createStageExecutors();
	}

	public reset(): void {
		this._preparedSceneCache.reset();
		this._occlusionCullingController.reset();
	}

	public async warmup(
		delegate: FrameCoordinatorDelegate,
		context: FrameContext,
		options: WarmupOptions,
	): Promise<WarmupReport> {
		if (!this._backend.warmup) {
			const startedAt = Date.now();
			return {
				backend: this._backend.profile.id,
				startedAt,
				finishedAt: startedAt,
				durationMs: 0,
				total: 0,
				compiled: 0,
				skipped: 0,
				failed: 0,
				phases: [],
				errors: [],
			};
		}
		return this._backend.warmup(context, options);
	}

	public async executeFrame(
		delegate: FrameCoordinatorDelegate,
		now: number,
		deltaTime: number,
		frameDirtyReasonMask: number,
		hasParticleSystems: boolean,
		hasActiveAnimations: boolean,
		deltaTimeSeconds: number,
		transient: TransientStore,
	): Promise<IncrementalFrameStatus> {
		const state = this._createRenderSceneFrameState(delegate, {
			now,
			deltaTimeSeconds,
			frameDirtyReasonMask,
			transient,
			hasActiveAnimations,
			hasParticleSystems,
		});

		try {
			for (const stage of state.stageOrder) {
				await this._executeRenderStage(delegate, stage.id, state, deltaTime);
			}

			if (state.frameStarted) {
				await this._backend.endFrame();
				state.frameStarted = false;
			}
			const status = this._createIncrementalFrameStatus(state.incrementalFrameContext);
			delegate.setLastIncrementalFrameStats(status.plan);
			return status;
		} catch (error) {
			await this._abortFailedFrame(error, state.frameStarted);
			throw error;
		}
	}

	/**
	 * Creates the no-op incremental status returned by an on-demand clean frame.
	 *
	 * @internal Owned by `Renderer.renderFrame()`.
	 */
	public createCleanIncrementalFrameStatus(
		delegate: FrameCoordinatorDelegate,
	): IncrementalFrameStatus {
		const { fullFrameTiles } = this._createFullFrameCoverage(delegate);
		const plan: IncrementalFrameStats = {
			enabled: delegate.incrementalOptions.enabled,
			reasonMask: 0,
			forceFullFrame: false,
			temporalHistoryReset: false,
			firstPass: null,
			postProcessStartPass: null,
			dirtyRectCount: 0,
			dirtyTileCount: 0,
			dirtyTileSize: fullFrameTiles.tileSize,
			dirtyTileColumns: fullFrameTiles.tileColumns,
			dirtyTileRows: fullFrameTiles.tileRows,
			dirtyAreaRatio: 0,
			dirtyRects: [],
			dirtyTiles: [],
		};
		const coverage = createIncrementalTileCoverage(
			fullFrameTiles.tileSize,
			fullFrameTiles.tileColumns,
			fullFrameTiles.tileRows,
			[],
			"unchanged",
		);
		return {
			plan,
			plannedCoverage: coverage,
			finalOutputCoverage: coverage,
		};
	}

	private _createRenderSceneFrameState(
		delegate: FrameCoordinatorDelegate,
		options: {
			now: number;
			deltaTimeSeconds: number;
			frameDirtyReasonMask: number;
			transient: TransientStore;
			hasActiveAnimations: boolean;
			hasParticleSystems: boolean;
		},
	): RenderSceneFrameState {
		const resolved = resolveFeatureState(
			delegate.features,
			this._backend.profile.capabilities,
			this._backend.profile.id,
		);
		const resolvedPostProcess = delegate.postProcess.createSnapshot(this._backend.profile.id);
		const { fullFrameRect: initialFullFrameRect, fullFrameTiles: initialFullFrameTiles } =
			this._createFullFrameCoverage(delegate);
		const incrementalFrameContext = this._createInitialIncrementalFrameContext(
			options.frameDirtyReasonMask,
			initialFullFrameRect,
			initialFullFrameTiles,
		);
		const stageOrder = delegate.pipeline.getExecutionOrder(
			{
				hasActiveAnimations: options.hasActiveAnimations && delegate.animationAutoRender,
				hasParticleSystems: options.hasParticleSystems,
			},
			(key, message) => delegate.warn(key, message),
		);
		const stageIndexById = new Map<string, number>();
		for (let index = 0; index < stageOrder.length; index++) {
			stageIndexById.set(stageOrder[index].id, index);
		}

		return {
			now: options.now,
			deltaTimeSeconds: options.deltaTimeSeconds,
			frameDirtyReasonMask: options.frameDirtyReasonMask,
			transient: options.transient,
			resolved,
			resolvedPostProcess,
			frame: null,
			context: null,
			frameStarted: false,
			emittedPostAnimation: false,
			stageOrder,
			stageIndexById,
			hasAnimationStage: stageOrder.some((stage) => stage.id === "animation-sim"),
			initialFullFrameRect,
			initialFullFrameTiles,
			incrementalFrameContext,
			incrementalStartStageIndex: -1,
		};
	}

	private async _executeRenderStage(
		delegate: FrameCoordinatorDelegate,
		stageId: string,
		state: RenderSceneFrameState,
		deltaTime: number,
	): Promise<void> {
		const executor = this._stageExecutors.get(stageId);
		if (executor) {
			await executor(delegate, state);
			return;
		}
		const stageKind = delegate.pipeline.getStageKind(stageId as FramePassStage);
		if (stageKind === "backend-pass") {
			await this._executeFramePassStage(delegate, stageId, state);
			return;
		}
		if (stageKind === "renderer") {
			const key = `renderer-stage-executor-missing-${stageId}`;
			delegate.warn(
				key,
				`Renderer pipeline stage "${stageId}" has no internal executor; skipping stage`,
			);
			return;
		}
		const key = `renderer-stage-kind-missing-${stageId}`;
		delegate.warn(
			key,
			`Renderer pipeline stage "${stageId}" is not registered; skipping stage`,
		);
	}

	private _createStageExecutors(): Map<string, RendererStageExecutor> {
		return new Map<string, RendererStageExecutor>([
			[
				"feature-resolution",
				(delegate, state) => {
					state.resolved = resolveFeatureState(
						delegate.features,
						this._backend.profile.capabilities,
						this._backend.profile.id,
					);
					state.resolvedPostProcess = delegate.postProcess.createSnapshot(
						this._backend.profile.id,
					);
					for (const warning of [
						...state.resolved.warnings,
						...state.resolvedPostProcess.getWarnings(),
					]) {
						delegate.warn(warning.key, warning.message);
					}
				},
			],
			["sync-in", (delegate, state) => this._executeSyncInStage(delegate, state)],
			["animation-sim", (delegate, state) => this._executeAnimationStage(delegate, state)],
			["physics-sim", (delegate, state) => this._executePhysicsStage(delegate, state)],
			["transform-update", (delegate) => this._executeTransformUpdateStage(delegate)],
			["lod-resolve", (delegate) => this._resolveLODMeshes(delegate)],
			["csg-resolve", (delegate) => this._resolveCSGMeshes(delegate)],
			[
				"prepared-scene-build",
				(delegate, state) => this._executePreparedSceneBuildStage(delegate, state),
			],
			["probe-capture", (delegate, state) => this._executeProbeCaptureStage(delegate, state)],
			["sync-out", (delegate) => delegate.scene.syncECSToNode()],
		]);
	}

	private _executeSyncInStage(
		delegate: FrameCoordinatorDelegate,
		state: RenderSceneFrameState,
	): void {
		delegate.scene.syncNodeToECS();
		if (!state.hasAnimationStage && !state.emittedPostAnimation) {
			this._emitPostAnimation(delegate, state, state.deltaTimeSeconds * 1000);
		}
	}

	private _executeAnimationStage(
		delegate: FrameCoordinatorDelegate,
		state: RenderSceneFrameState,
	): void {
		const dt = state.deltaTimeSeconds * 1000;
		this._animationRuntime.update(
			delegate.animationSystem,
			state.deltaTimeSeconds,
			state.transient,
			delegate.scene,
		);
		if (!state.emittedPostAnimation) {
			this._emitPostAnimation(delegate, state, dt);
		}
	}

	private async _executePhysicsStage(
		delegate: FrameCoordinatorDelegate,
		state: RenderSceneFrameState,
	): Promise<void> {
		if (delegate.physicsSystem) {
			await delegate.physicsSystem.stepAsync(state.deltaTimeSeconds);
		}
	}

	private _executeTransformUpdateStage(delegate: FrameCoordinatorDelegate): void {
		delegate.scene.updateWorldMatrices();
		this._assertCameraInScene(delegate.scene, delegate.camera, "renderScene");
		delegate.camera.updateMatrices();
		delegate.refreshReflectionProbeCaches();
	}

	private _resolveLODMeshes(delegate: FrameCoordinatorDelegate): void {
		const cameraWorldPosition = delegate.camera.getWorldPosition(
			_tmpRendererCameraWorldPosition,
		);
		const meshInstances = delegate.scene.getMeshInstances();
		for (const meshInstance of meshInstances) {
			if (!(meshInstance instanceof LODMeshInstance)) continue;
			meshInstance.updateLODForCamera(cameraWorldPosition, {
				notifyScene: false,
			});
		}
	}

	private async _resolveCSGMeshes(delegate: FrameCoordinatorDelegate): Promise<void> {
		const meshInstances = delegate.scene.getMeshInstances();
		for (const meshInstance of meshInstances) {
			if (!(meshInstance instanceof CSGMeshInstance)) continue;
			if (!meshInstance.isCSGDirty) continue;
			if (meshInstance.physicsSync === "auto" && !meshInstance.physicsSystem) {
				meshInstance.physicsSystem = delegate.physicsSystem;
			}

			const flushResult = meshInstance.flushCSG();
			const result = flushResult instanceof Promise ? await flushResult : flushResult;
			for (const diagnostic of result.diagnostics) {
				if (diagnostic.severity === "info") continue;
				const key =
					`csg-diagnostic-${meshInstance.id}-` +
					`${diagnostic.code}-${diagnostic.message}`;
				Logger.warn(`[${key}] [CSG:${diagnostic.code}] ${diagnostic.message}`, {
					scope: "Renderer",
				});
			}
		}
	}

	private async _executePreparedSceneBuildStage(
		delegate: FrameCoordinatorDelegate,
		state: RenderSceneFrameState,
	): Promise<void> {
		if (delegate.features.enableSH) {
			this.updateSH(delegate);
		}
		// Build a baseline shadow map — PreparedSceneBuilder only maps shadow-
		// caster geometry, it does not depend on actual shadow map content.
		// Use an empty map here to avoid the destructive side effect of
		// ShadowManager.buildFrameState (which unbinds lights not in the
		// given light list). The real shadow maps are built later in
		// _createFrameContext below.
		const preparedResult = this._preparedSceneCache.build({
			source: {
				scene: delegate.scene,
				camera: delegate.camera,
				shadowMaps: new Map(),
				hasActiveAnimations: delegate.animationSystem.hasActiveActions(),
			},
			viewportWidth: delegate.canvas.width,
			viewportHeight: delegate.canvas.height,
			features: state.resolved,
			postProcess: state.resolvedPostProcess,
			incrementalOptions: delegate.incrementalOptions,
			occlusionVisibilityProvider: this._occlusionCullingController.getVisibilityProvider(
				state.resolved,
			),
			occlusionCullingOptions: state.resolved.occlusionCullingOptions,
		});
		state.frame = preparedResult.frame;
		const incrementalPlan = IncrementalFramePlanner.plan({
			enabled: delegate.incrementalOptions.enabled,
			reasonMask: state.frameDirtyReasonMask,
			features: state.resolved,
			postProcess: state.resolvedPostProcess,
			registry: delegate.pipeline.incremental,
		});
		state.incrementalFrameContext = this._buildIncrementalFrameContext(
			delegate,
			incrementalPlan,
			preparedResult,
			state.initialFullFrameRect,
			state.initialFullFrameTiles,
		);
		state.incrementalStartStageIndex =
			state.incrementalFrameContext.enabled &&
			!state.incrementalFrameContext.forceFullFrame &&
			state.incrementalFrameContext.firstPass
				? (state.stageIndexById.get(state.incrementalFrameContext.firstPass) ?? -1)
				: -1;
		const context = this._createFrameContext(
			delegate,
			state.frame,
			state.resolved,
			state.resolvedPostProcess,
			state.transient,
			state.incrementalFrameContext,
		);
		state.context = {
			...context,
			framePlan: delegate.pipeline.createFramePlan({
				stageOrder: state.stageOrder,
				frame: state.frame,
				features: state.resolved,
				postProcess: state.resolvedPostProcess,
				transient: state.transient,
				backendType: this._backend.profile.id,
				backendCapabilities: this._backend.profile.capabilities,
				incremental: state.incrementalFrameContext,
				frameContext: context,
				incrementalStartStageIndex: state.incrementalStartStageIndex,
			}),
		};
		state.frameStarted = true;
		await this._backend.beginFrame(state.context);
	}

	private async _executeProbeCaptureStage(
		delegate: FrameCoordinatorDelegate,
		state: RenderSceneFrameState,
	): Promise<void> {
		const cameraWorldPosition = delegate.camera.getWorldPosition(
			_tmpRendererCameraWorldPosition,
		);
		const captureRuntime = (delegate as any)._probeCaptureRuntime || this._probeCaptureRuntime;
		await captureRuntime.execute({
			scene: delegate.scene,
			nowMs: state.now,
			frameDirtyReasonMask: state.frameDirtyReasonMask,
			frameContext: state.context,
			cameraWorldPosition,
			webgpuCaptureSource: delegate.getBackendExtension(PROBE_CAPTURE_EXTENSION),
		});
	}

	private async _executeFramePassStage(
		delegate: FrameCoordinatorDelegate,
		stageId: string,
		state: RenderSceneFrameState,
	): Promise<void> {
		if (!state.context || !state.frame) return;
		if (!delegate.pipeline.isFramePassStage(stageId as FramePassStage)) return;

		const pass =
			state.context.framePlan?.backendPasses.find(
				(candidate) => candidate.stage === stageId,
			) ?? delegate.pipeline.createFramePass(stageId as FramePassStage);
		if (!pass.enabled) {
			this._backend.skipPass?.(pass);
			return;
		}
		await this._backend.executePass(pass, state.context);
	}

	private _emitPostAnimation(
		delegate: FrameCoordinatorDelegate,
		state: RenderSceneFrameState,
		deltaTime: number,
	): void {
		delegate.emitPostAnimation(state.now, deltaTime, state.transient);
		state.emittedPostAnimation = true;
	}

	private _createIncrementalFrameStatus(
		incrementalFrameContext: IncrementalFrameContext,
	): IncrementalFrameStatus {
		const plan: IncrementalFrameStats = {
			enabled: incrementalFrameContext.enabled,
			reasonMask: incrementalFrameContext.reasonMask,
			forceFullFrame: incrementalFrameContext.forceFullFrame,
			temporalHistoryReset: incrementalFrameContext.temporalHistoryReset,
			firstPass: incrementalFrameContext.firstPass,
			postProcessStartPass: incrementalFrameContext.postProcessStartPass,
			dirtyRectCount: incrementalFrameContext.dirtyRects.length,
			dirtyTileCount: incrementalFrameContext.dirtyTiles.length,
			dirtyTileSize: incrementalFrameContext.dirtyTileSize,
			dirtyTileColumns: incrementalFrameContext.dirtyTileColumns,
			dirtyTileRows: incrementalFrameContext.dirtyTileRows,
			dirtyAreaRatio: incrementalFrameContext.dirtyAreaRatio,
			dirtyRects: incrementalFrameContext.dirtyRects.map((rect) => ({
				x: rect.x,
				y: rect.y,
				width: rect.width,
				height: rect.height,
			})),
			dirtyTiles: incrementalFrameContext.dirtyTiles.slice(),
		};
		const plannedCoverage = createIncrementalTileCoverage(
			incrementalFrameContext.dirtyTileSize,
			incrementalFrameContext.dirtyTileColumns,
			incrementalFrameContext.dirtyTileRows,
			incrementalFrameContext.dirtyTiles,
			incrementalFrameContext.forceFullFrame ||
			!incrementalFrameContext.enabled ||
			incrementalFrameContext.temporalHistoryReset ?
				"full"
			: incrementalFrameContext.dirtyTiles.length === 0 ?
				"unchanged"
			: "partial",
		);
		const finalOutputCoverage =
			this._backend.getCompletedFrameCoverage?.() === "dirty-tiles" &&
			plannedCoverage.mode === "partial" ?
				plannedCoverage
			: createIncrementalTileCoverage(
					incrementalFrameContext.dirtyTileSize,
					incrementalFrameContext.dirtyTileColumns,
					incrementalFrameContext.dirtyTileRows,
					[],
					plannedCoverage.mode === "unchanged" ? "unchanged" : "full",
				);
		return { plan, plannedCoverage, finalOutputCoverage };
	}

	private _createFullFrameCoverage(delegate: FrameCoordinatorDelegate): {
		fullFrameRect: ReturnType<typeof makeFullScreenRect>;
		fullFrameTiles: DirtyTileCoverage;
	} {
		const fullFrameRect = makeFullScreenRect(delegate.canvas.width, delegate.canvas.height);
		const fullFrameTiles = buildDirtyTileCoverage(
			[fullFrameRect],
			fullFrameRect.width,
			fullFrameRect.height,
			delegate.incrementalOptions.dirtyTileSize,
		);
		return {
			fullFrameRect,
			fullFrameTiles,
		};
	}

	private _createInitialIncrementalFrameContext(
		reasonMask: number,
		fullFrameRect: ReturnType<typeof makeFullScreenRect>,
		fullFrameTiles: DirtyTileCoverage,
	): IncrementalFrameContext {
		return {
			enabled: true,
			forceFullFrame: true,
			dirtyRects: [fullFrameRect],
			dirtyTileSize: fullFrameTiles.tileSize,
			dirtyTileColumns: fullFrameTiles.tileColumns,
			dirtyTileRows: fullFrameTiles.tileRows,
			dirtyTiles: fullFrameTiles.dirtyTiles.slice(),
			dirtyAreaRatio: 1,
			firstPass: null,
			postProcessStartPass: null,
			reasonMask,
			temporalHistoryReset: true,
		};
	}

	public createWarmupContext(
		delegate: FrameCoordinatorDelegate,
		frame: RenderSceneFrame,
		resolved: RenderSceneFeatureState,
		postProcess: ResolvedPostProcessState,
		transient: TransientStore,
		incremental: IncrementalFrameContext,
	): FrameContext {
		return this._createFrameContext(
			delegate,
			frame,
			resolved,
			postProcess,
			transient,
			incremental,
		);
	}

	private _createFrameContext(
		delegate: FrameCoordinatorDelegate,
		frame: RenderSceneFrame,
		resolved: RenderSceneFeatureState,
		postProcess: ResolvedPostProcessState,
		transient: TransientStore,
		incremental: IncrementalFrameContext,
	): FrameContext {
		const cameraPosition = delegate.camera.getWorldPosition(_tmpRendererCameraWorldPosition);
		const shadowFrameState = delegate.scene.shadows.buildFrameState({
			lights: frame.lights,
			enableShadows: resolved.enableShadows,
			cameraPosition,
			backendCapabilities: this._backend.profile.shadow,
		});
		delegate.setShadowMaps(shadowFrameState.shadowMaps);
		frame.shadowMaps = shadowFrameState.shadowMaps;

		const attachments = this._backend.getAttachments({
			width: delegate.canvas.width,
			height: delegate.canvas.height,
		});

		// Build SH if not done
		const shCoeffs =
			delegate.scene.version === 0 ? SH.empty() : this.updateSH(delegate).shCoeffs;
		const shAmbientCoeffs =
			delegate.scene.version === 0 ? SH.empty() : this.updateSH(delegate).shAmbientCoeffs;

		return {
			backendProfile: this._backend.profile,
			viewCamera: delegate.camera,
			attachments,
			features: resolved,
			postProcess,
			renderTargets: delegate.renderTargets.createSnapshot(),
			customRenderPasses: delegate.renderPasses.createSnapshot(),
			shadowMaps: shadowFrameState.shadowMaps,
			scene: frame,
			shCoeffs,
			shAmbientCoeffs,
			worldMatrix: delegate.features.worldMatrix || Matrix4.identity(),
			incremental,
			transient,
		};
	}

	private _buildIncrementalFrameContext(
		delegate: FrameCoordinatorDelegate,
		plan: ReturnType<typeof IncrementalFramePlanner.plan>,
		prepared: PreparedSceneCacheBuildResult,
		initialFullFrameRect: ReturnType<typeof makeFullScreenRect>,
		initialFullFrameTiles: DirtyTileCoverage,
	): IncrementalFrameContext {
		const enabled = delegate.incrementalOptions.enabled;
		const forceFullFrame = plan.forceFullFrame || prepared.forceFullFrame;
		const fullFrameRect = initialFullFrameRect;
		const fullFrameTiles = initialFullFrameTiles;
		let dirtyRects = enabled && !forceFullFrame ? prepared.dirtyRects.slice() : [fullFrameRect];
		let dirtyTiles =
			enabled && !forceFullFrame
				? prepared.dirtyTiles.slice()
				: fullFrameTiles.dirtyTiles.slice();
		let dirtyTileSize =
			enabled && !forceFullFrame ? prepared.dirtyTileSize : fullFrameTiles.tileSize;
		let dirtyTileColumns =
			enabled && !forceFullFrame ? prepared.dirtyTileColumns : fullFrameTiles.tileColumns;
		let dirtyTileRows =
			enabled && !forceFullFrame ? prepared.dirtyTileRows : fullFrameTiles.tileRows;
		let dirtyAreaRatio = enabled && !forceFullFrame ? prepared.dirtyAreaRatio : 1;
		if (enabled && !forceFullFrame && dirtyRects.length === 0 && plan.firstPass) {
			dirtyRects = [fullFrameRect];
			dirtyTiles = fullFrameTiles.dirtyTiles.slice();
			dirtyTileSize = fullFrameTiles.tileSize;
			dirtyTileColumns = fullFrameTiles.tileColumns;
			dirtyTileRows = fullFrameTiles.tileRows;
			dirtyAreaRatio = 1;
		}
		return {
			enabled,
			forceFullFrame,
			dirtyRects,
			dirtyTileSize,
			dirtyTileColumns,
			dirtyTileRows,
			dirtyTiles,
			dirtyAreaRatio,
			firstPass: forceFullFrame ? null : plan.firstPass,
			postProcessStartPass: forceFullFrame ? null : plan.postProcessStartPass,
			reasonMask: plan.reasonMask,
			temporalHistoryReset: plan.temporalHistoryReset || forceFullFrame,
		};
	}

	public updateSH(delegate: FrameCoordinatorDelegate): {
		shCoeffs: SHCoefficients;
		shAmbientCoeffs: SHCoefficients;
	} {
		let ambientProbeSH: SHCoefficients = SH.empty();
		let ambientR = 0;
		let ambientG = 0;
		let ambientB = 0;
		let hasAmbient = false;
		for (const light of delegate.scene.getLights()) {
			if (light.type === LightType.Ambient) {
				const color = light.color || { r: 255, g: 255, b: 255 };
				const intensity = light.intensity ?? 1;
				ambientR += sRGBToLinear(color.r / 255) * 255 * intensity;
				ambientG += sRGBToLinear(color.g / 255) * 255 * intensity;
				ambientB += sRGBToLinear(color.b / 255) * 255 * intensity;
				hasAmbient = true;
				continue;
			}

			if (light.type === LightType.LightProbe) {
				const probe = light as LightProbe;
				if (
					this._backend.profile.lighting.localizedProbeMode === "backend-local" &&
					isLocalizedLightProbe(probe)
				) {
					continue;
				}
				const probeSH = probe.sh;
				const coeffCount = Math.min(ambientProbeSH.length, probeSH.length);
				for (let i = 0; i < coeffCount; i++) {
					ambientProbeSH[i].r += probeSH[i].r;
					ambientProbeSH[i].g += probeSH[i].g;
					ambientProbeSH[i].b += probeSH[i].b;
				}
			}
		}

		if (
			!hasAmbient &&
			ambientProbeSH[0].r === 0 &&
			ambientProbeSH[0].g === 0 &&
			ambientProbeSH[0].b === 0
		) {
			const fallbackLinear = PBR_AMBIENT_FALLBACK_LINEAR * 255;
			ambientR = fallbackLinear;
			ambientG = fallbackLinear;
			ambientB = fallbackLinear;
		}

		ambientProbeSH[0].r += ambientR / Math.PI / 0.282095;
		ambientProbeSH[0].g += ambientG / Math.PI / 0.282095;
		ambientProbeSH[0].b += ambientB / Math.PI / 0.282095;

		const shAmbientCoeffs = ambientProbeSH.map((coefficient) => ({
			r: coefficient.r,
			g: coefficient.g,
			b: coefficient.b,
		})) as SHCoefficients;

		let totalSH: SHCoefficients = shAmbientCoeffs.map((coefficient) => ({
			r: coefficient.r,
			g: coefficient.g,
			b: coefficient.b,
		})) as SHCoefficients;

		for (const light of delegate.scene.getLights()) {
			if (light.type !== LightType.Directional) continue;

			const worldDirection = light.getWorldLightDirection();
			const direction = Vector3.normalize({
				x: -worldDirection.x,
				y: -worldDirection.y,
				z: -worldDirection.z,
			});
			const intensity = light.intensity ?? 1;
			const lightSH = SH.projectDirectionalLight(direction, {
				r: light.color.r * intensity,
				g: light.color.g * intensity,
				b: light.color.b * intensity,
			});
			totalSH = SH.addCoeffs(totalSH, lightSH);
		}

		delegate.setSHCoefficients(totalSH);
		delegate.setSHAmbientCoefficients(shAmbientCoeffs);

		return {
			shCoeffs: totalSH,
			shAmbientCoeffs,
		};
	}

	private _assertCameraInScene(
		scene: Scene,
		camera: Camera,
		caller: "setScene" | "setCamera" | "renderScene",
	): void {
		if (scene.contains(camera)) return;
		throw new Error(
			`Renderer.${caller} requires the active camera to belong to the active scene graph`,
		);
	}

	private async _abortFailedFrame(error: unknown, frameStarted: boolean): Promise<void> {
		if (!frameStarted) {
			return;
		}
		try {
			await this._backend.abortFrame(error);
		} catch (abortError) {
			Logger.warn(
				`[renderer-backend-abort-failed] Failed to abort backend frame state: ${String(abortError)}`,
				{ scope: "Renderer" },
			);
		}
	}
}
