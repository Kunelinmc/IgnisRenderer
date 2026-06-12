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
import {
	PBR_AMBIENT_FALLBACK_LINEAR,
} from "../lights/constants";
import { EventEmitter } from "../core/EventEmitter";
import { Scene } from "../core/Scene";
import { Texture } from "../core/Texture";
import { Logger, type LoggerStatic } from "../foundation/Logger";
import { CSGMeshInstance } from "../meshes/CSGMeshInstance";
import { LODMeshInstance } from "../meshes/LODMeshInstance";
import { resolveFeatureState } from "../pipeline/FeatureResolver";
import {
	GammaPass,
	PostProcessPassRegistry,
	ToneMappingPass,
	type ResolvedPostProcessState,
} from "../postprocess";
import { AnimationSimulationStage } from "../pipeline/AnimationSimulationStage";
import { PreparedSceneBuilder } from "../pipeline/PreparedSceneBuilder";
import {
	PreparedSceneCache,
	type PreparedSceneCacheBuildResult,
} from "../pipeline/PreparedSceneCache";
import {
	ProbeCaptureRuntime,
	type ProbeWebGPUCaptureSource,
} from "../lights/runtime/ProbeCaptureRuntime";
import type { RendererStageDefinition } from "../pipeline/RendererStageGraph";
import { RenderPipelineRegistry } from "../pipeline/RenderPipelineRegistry";
import { createDefaultPipelineStages } from "../pipeline/defaultPipeline";
import { isLocalizedLightProbe } from "../lights/runtime/lightProbeRuntime";
import {
	ANIMATION_SIM_DELTA_TIME_MS_KEY,
	createTransientStore,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
} from "../pipeline/types";
import { getWarmupStartDelay } from "../pipeline/WarmupScheduler";
import { AnimationSystem } from "../animation/AnimationSystem";
import type { PhysicsSystem } from "../physics";
import type { SHCoefficients } from "../maths/types";
import type { IShadowBackendCapabilities } from "../lights/shadows";
import {
	buildDirtyTileCoverage,
	DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
	IncrementalFramePlanner,
	makeFullScreenRect,
	mergeIncrementalRenderingOptions,
	renderDirtyReasonToMask,
	type IncrementalFrameContext,
	type DirtyTileCoverage,
	type IncrementalFrameStats,
	type IncrementalRenderingOptions,
	type RenderDirtyReason,
} from "../pipeline/incremental";
import type {
	FrameContext,
	FramePassStage,
	RendererFeatureFlags,
	RendererFeatureResolvedOptions,
	TransientStore,
} from "../pipeline/types";
import type {
	IRenderBackend,
	RenderBackendDeviceLostInfo,
	RendererBackendResourceEvent,
	WarmupOptions,
	WarmupReport,
} from "./IRenderBackend";
import { RendererOcclusionCullingController } from "./RendererOcclusionCullingController";

export type {
	IncrementalFrameStats,
	IncrementalRenderingOptions,
	RenderDirtyReason,
} from "../pipeline/incremental";

export interface RendererEvents {
	tick: [{ now: number; deltaTime: number }];
	framestart: [{ now: number; deltaTime: number }];
	devicelost: [
		{
			backend: IRenderBackend["type"];
			info?: RenderBackendDeviceLostInfo;
		},
	];
	backendresourceevent: [RendererBackendResourceEvent];
	postanimation: [
		{
			now: number;
			deltaTime: number;
			scene: Scene;
			transient: TransientStore;
		},
	];
	frameend: [{ now: number; deltaTime: number }];
	[key: string]: any[];
}

export interface FrameTransientContributorContext {
	now: number;
	deltaTime: number;
	scene: Scene;
	camera: Camera;
	transient: TransientStore;
}

export type FrameTransientContributor = (
	context: FrameTransientContributorContext
) => void;

export type RendererFeatures = RendererFeatureFlags &
	RendererFeatureResolvedOptions & {
		worldMatrix: Matrix4;
	};

const _tmpRendererCameraWorldPosition = { x: 0, y: 0, z: 0 };

type RenderSceneFrame = ReturnType<typeof PreparedSceneBuilder.build>;
type RenderSceneFeatureState = ReturnType<typeof resolveFeatureState>;
type RendererStageExecutor = (
	state: RenderSceneFrameState
) => void | Promise<void>;

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

export class Renderer extends EventEmitter<RendererEvents> {
	public readonly backend: IRenderBackend;
	public readonly animationSystem: AnimationSystem;
	public readonly features: RendererFeatures;
	public readonly pipeline: RenderPipelineRegistry;
	public readonly postProcess: PostProcessPassRegistry;
	public animationAutoRender: boolean;

	public readonly logger: Pick<LoggerStatic, "warn">;
	private _canvas: HTMLCanvasElement;
	private _shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>;
	private _shCoeffs: SHCoefficients;
	private _shAmbientCoeffs: SHCoefficients;
	private _scene: Scene;
	private _camera: Camera;
	private _lastTime: number;
	private _deviceScaleFactor: number;
	private _deltaTime: number;
	private _frameDirty: boolean;
	private _animationStage: AnimationSimulationStage;
	private _physicsSystem: PhysicsSystem | null;
	private _frameTransientContributors: Set<FrameTransientContributor>;
	private _incrementalOptions: IncrementalRenderingOptions;
	private _lastIncrementalFrameStats: IncrementalFrameStats | null;
	private _preparedSceneCache: PreparedSceneCache;
	private _probeCaptureRuntime: ProbeCaptureRuntime;
	private _pendingDirtyReasonMask: number;
	private _lastKnownSceneVersion: number;
	private _occlusionCullingController: RendererOcclusionCullingController;
	private _stageExecutors: Map<string, RendererStageExecutor>;

	constructor(
		backend: IRenderBackend,
		canvas: HTMLCanvasElement,
		camera: Camera | null = null
	) {
		super();
		this.backend = backend;
		this.animationSystem = new AnimationSystem();
		this.pipeline = new RenderPipelineRegistry({
			stages: createDefaultPipelineStages(),
		});
		this._stageExecutors = this._createStageExecutors();
		this.logger = Logger;
		this.postProcess = new PostProcessPassRegistry();
		this._occlusionCullingController =
			new RendererOcclusionCullingController(this.backend);
		this.postProcess.on("change", (change) => {
			if (change.reason === "register") {
				const pass = this.postProcess.getPass(change.passId);
				if (pass) {
					this.pipeline.registerPostProcessPass(pass);
				}
			} else if (change.reason === "unregister" && !change.builtIn) {
				this.pipeline.unregisterPostProcessPass(change.passId);
			}
			if (this._scene) {
				this._markFrameDirty("postfx");
			} else {
				this._frameDirty = true;
				this._pendingDirtyReasonMask |= renderDirtyReasonToMask("postfx");
			}
		});
		this.postProcess.registerPass(new ToneMappingPass({ enabled: true }));
		this.postProcess.registerPass(new GammaPass({ enabled: true }));
		this._canvas = canvas;
		this._deviceScaleFactor = window.devicePixelRatio || 1;
		this._deltaTime = 0;
		this._frameDirty = true;
		this.animationAutoRender = true;
		this._animationStage = new AnimationSimulationStage(this.animationSystem);
		this._physicsSystem = null;
		this._frameTransientContributors = new Set();
		this._incrementalOptions = { ...DEFAULT_INCREMENTAL_RENDERING_OPTIONS };
		this._lastIncrementalFrameStats = null;
		this._preparedSceneCache = new PreparedSceneCache();
		this._probeCaptureRuntime = new ProbeCaptureRuntime();
		this._pendingDirtyReasonMask = renderDirtyReasonToMask("unknown");
		this._lastKnownSceneVersion = 0;

		this.features = {
			enableLighting: true,
			enableSH: false,
			enableShadows: true,
			enableReflection: true,
			enableEnvironment: true,
			enableOIT: false,
			enableClusteredLighting: false,
			clusteredLightingOptions: {},
			enableOcclusionCulling: false,
			occlusionCullingOptions: {},
			worldMatrix: Matrix4.identity(),
		};

		this._shadowMaps = new Map();
		this._shCoeffs = SH.empty();
		this._shAmbientCoeffs = SH.empty();
		this._scene = new Scene();
		this._lastKnownSceneVersion = this._scene.version;
		this._camera = camera || new Camera();

		// Only add to the default internal scene if the camera doesn't already have a parent.
		// This prevents the constructor from "stealing" a camera that the user has already
		// placed in their own scene graph.
		if (!this._camera.parent) {
			this._scene.add(this._camera);
		}

		this._lastTime = 0;

		if (!camera) {
			this._camera.position.set(0, 200, 200);
			this._camera.fov = 60;
		}

		this._camera.aspectRatio = this._getSafeAspectRatio(
			this.canvas.width,
			this.canvas.height
		);
		this._camera.updateMatrices();
		this.backend.setRenderer?.(this);
	}

	public async init(): Promise<void> {
		await this.backend.init(this.canvas);
		this.resizeCanvas();
		requestAnimationFrame((time) => this.renderScene(time));
	}

	/**
	 * Forwards device/context loss notification to the active backend.
	 *
	 * @internal Backend lifecycle bridge. Applications should subscribe to
	 * `devicelost` instead of calling this method directly.
	 */
	public onDeviceLost(
		info?: RenderBackendDeviceLostInfo
	): void | Promise<void> {
		const result = this.backend.onDeviceLost?.(info);
		this._preparedSceneCache.reset();
		this._occlusionCullingController.reset();
		this._markFrameDirty("unknown");
		this.emit("devicelost", {
			backend: this.backend.type,
			info,
		});
		return result;
	}

	/**
	 * Rebuilds the active backend after device/context loss.
	 */
	public async restore(): Promise<void> {
		if (this.backend.restore) {
			await this.backend.restore(this.canvas);
		} else {
			await this.backend.init(this.canvas);
		}
		this._preparedSceneCache.reset();
		this._occlusionCullingController.reset();
		this.resizeCanvas();
	}

	/**
	 * Handles backend resource lifetime notifications.
	 *
	 * @param event Backend resource event emitted through `RendererBackendBridge`.
	 * @returns Nothing.
	 * @sideEffects May invalidate or destroy renderer-owned resources.
	 * @internal Backend resource lifetime bridge. Applications should subscribe
	 * to `backendresourceevent` instead of calling this method directly.
	 */
	public onBackendResourceEvent(event: RendererBackendResourceEvent): void {
		if (
			event.resource === "webgl-program" &&
			event.reason === "shader-compile-pending"
		) {
			this._markFrameDirty("postfx");
		}
		if (
			event.resource === "webgl-texture" &&
			event.reason === "texture-upload-pending"
		) {
			this._markFrameDirty("texture");
		}
		this.emit("backendresourceevent", event);
	}

	/**
	 * Prepares backend pipelines and resources for the current scene.
	 *
	 * @param options Warmup coverage, diagnostics, and scheduling controls.
	 * @returns A report after all requested warmup work completes.
	 * @constraints Callers that do not need to block initialization should use
	 * `scheduling: "idle"` and must handle the returned promise asynchronously.
	 * @sideEffects Synchronizes scene state and creates backend-owned resources.
	 */
	public async warmup(options: WarmupOptions = {}): Promise<WarmupReport> {
		const warmupStartDelay = getWarmupStartDelay(options);
		if (warmupStartDelay) {
			await warmupStartDelay;
		}
		this.scene.syncNodeToECS();
		this.scene.updateWorldMatrices();
		this.refreshReflectionProbeCaches();
		this._assertCameraInScene(this.scene, this.camera, "renderScene");
		this.camera.updateMatrices();
		const transient = createTransientStore();
		transient.set(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 0);
		transient.set(ANIMATION_SIM_DELTA_TIME_MS_KEY, 0);

		const resolved = resolveFeatureState(
			this.features,
			this.backend.capabilities,
			this.backend.type
		);
		const resolvedPostProcess = this.postProcess.createSnapshot(
			this.backend.type
		);
		for (const warning of [
			...resolved.warnings,
			...resolvedPostProcess.getWarnings(),
		]) {
			this.logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "Renderer",
				onceKey: warning.key,
			});
		}
		if (this.features.enableSH) {
			this.updateSH();
		}
		const frame = PreparedSceneBuilder.build(this);
		const { fullFrameRect, fullFrameTiles } = this._createFullFrameCoverage();
		const incrementalFrameContext = this._createInitialIncrementalFrameContext(
			renderDirtyReasonToMask("unknown"),
			fullFrameRect,
			fullFrameTiles
		);
		const context = this._createFrameContext(
			frame,
			resolved,
			resolvedPostProcess,
			transient,
			incrementalFrameContext
		);
		if (!this.backend.warmup) {
			const startedAt = Date.now();
			return {
				backend: this.backend.type,
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
		return this.backend.warmup(context, options);
	}

	public resizeCanvas(): void {
		const rect = this.canvas.getBoundingClientRect();
		this._deviceScaleFactor = window.devicePixelRatio || 1;
		this.canvas.width = rect.width * this._deviceScaleFactor;
		this.canvas.height = rect.height * this._deviceScaleFactor;

		this.backend.resize(this.canvas.width, this.canvas.height);
		this._occlusionCullingController.reset();
		this._markFrameDirty("resize");

		this.camera.aspectRatio = this._getSafeAspectRatio(
			this.canvas.width,
			this.canvas.height
		);
		this.camera.updateMatrices();
	}

	public requestRender(reason: RenderDirtyReason = "unknown"): void {
		this._markFrameDirty(reason);
	}

	public setScene(scene: Scene): void {
		this._assertCameraInScene(scene, this.camera, "setScene");
		this._scene = scene;
		this._lastKnownSceneVersion = scene.version;
		this._preparedSceneCache.reset();
		this._occlusionCullingController.reset();
		if (this._physicsSystem) {
			this._physicsSystem.setEntityNodeResolver((entityId) => {
				return this.scene.ecs.getNodeByEntity(entityId);
			});
			this._physicsSystem.bindSceneSpatial(this.scene);
		}
		this._markFrameDirty("unknown");
	}

	public setCamera(camera: Camera): void {
		this._assertCameraInScene(this.scene, camera, "setCamera");
		this._camera = camera;
		this._occlusionCullingController.reset();
		this._markFrameDirty("camera");
	}

	public setPhysicsSystem(physicsSystem: PhysicsSystem | null): void {
		if (this._physicsSystem && this._physicsSystem !== physicsSystem) {
			this._physicsSystem.bindSceneSpatial(null);
		}
		this._physicsSystem = physicsSystem;
		if (physicsSystem) {
			physicsSystem.setEntityNodeResolver((entityId) => {
				return this.scene.ecs.getNodeByEntity(entityId);
			});
			physicsSystem.bindSceneSpatial(this.scene);
		}
	}

	public registerFrameTransientContributor(
		contributor: FrameTransientContributor
	): void {
		this._frameTransientContributors.add(contributor);
	}

	public unregisterFrameTransientContributor(
		contributor: FrameTransientContributor
	): void {
		this._frameTransientContributors.delete(contributor);
	}

	public setIncrementalRendering(
		options: Partial<IncrementalRenderingOptions>
	): void {
		const next = mergeIncrementalRenderingOptions(this._incrementalOptions, options);
		if (
			next.enabled === this._incrementalOptions.enabled &&
			next.maxDirtyRects === this._incrementalOptions.maxDirtyRects &&
			next.dirtyTileSize === this._incrementalOptions.dirtyTileSize &&
			next.fullFrameFallbackAreaRatio ===
				this._incrementalOptions.fullFrameFallbackAreaRatio &&
			next.temporalPolicy === this._incrementalOptions.temporalPolicy
		) {
			return;
		}
		this._incrementalOptions = next;
		this._preparedSceneCache.reset();
		this._markFrameDirty("unknown");
	}

	public getIncrementalRenderingOptions(): IncrementalRenderingOptions {
		return { ...this._incrementalOptions };
	}

	public getLastIncrementalFrameStats(): IncrementalFrameStats | null {
		if (!this._lastIncrementalFrameStats) {
			return null;
		}
		return {
			...this._lastIncrementalFrameStats,
			dirtyRects: this._lastIncrementalFrameStats.dirtyRects.map((rect) => ({
				x: rect.x,
				y: rect.y,
				width: rect.width,
				height: rect.height,
			})),
			dirtyTiles: this._lastIncrementalFrameStats.dirtyTiles.slice(),
		};
	}

	public get backendType(): IRenderBackend["type"] {
		return this.backend.type;
	}

	public get canvas(): HTMLCanvasElement {
		return this._canvas;
	}

	public get shadowMaps(): Map<ShadowCastingLight, ShadowRenderSet> {
		return this._shadowMaps;
	}

	public get shCoeffs(): SHCoefficients {
		return this._shCoeffs;
	}

	public get shAmbientCoeffs(): SHCoefficients {
		return this._shAmbientCoeffs;
	}

	public get scene(): Scene {
		return this._scene;
	}

	public get camera(): Camera {
		return this._camera;
	}

	public get lastTime(): number {
		return this._lastTime;
	}

	private _markFrameDirty(reason: RenderDirtyReason = "unknown"): void {
		this._frameDirty = true;
		const reasonMask = renderDirtyReasonToMask(reason);
		this._pendingDirtyReasonMask |= reasonMask;
		this.scene.invalidate(reason);
	}

	private _consumeDirtyReasonMask(): number {
		const sceneReasonMask = this.scene.consumeDirtyReasonMask();
		const combinedMask = this._pendingDirtyReasonMask | sceneReasonMask;
		this._pendingDirtyReasonMask = 0;
		return combinedMask >>> 0;
	}

	private _createFullFrameCoverage(): {
		fullFrameRect: ReturnType<typeof makeFullScreenRect>;
		fullFrameTiles: DirtyTileCoverage;
	} {
		const fullFrameRect = makeFullScreenRect(
			this.canvas.width,
			this.canvas.height
		);
		const fullFrameTiles = buildDirtyTileCoverage(
			[fullFrameRect],
			fullFrameRect.width,
			fullFrameRect.height,
			this._incrementalOptions.dirtyTileSize
		);
		return {
			fullFrameRect,
			fullFrameTiles,
		};
	}

	private _createInitialIncrementalFrameContext(
		reasonMask: number,
		fullFrameRect: ReturnType<typeof makeFullScreenRect>,
		fullFrameTiles: DirtyTileCoverage
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

	private _createFrameContext(
		frame: ReturnType<typeof PreparedSceneBuilder.build>,
		resolved: ReturnType<typeof resolveFeatureState>,
		postProcess: ResolvedPostProcessState,
		transient: TransientStore,
		incremental: IncrementalFrameContext
	): FrameContext {
		const cameraPosition = this.camera.getWorldPosition(
			_tmpRendererCameraWorldPosition
		);
		const shadowFrameState = this.scene.shadows.buildFrameState({
			lights: frame.lights,
			enableShadows: resolved.enableShadows,
			cameraPosition,
			backendCapabilities: this._resolveShadowBackendCapabilities(),
		});
		this._shadowMaps = shadowFrameState.shadowMaps;

		const attachments = this.backend.getAttachments(
			this.canvas.width,
			this.canvas.height
		);
		return {
			camera: this.camera,
			attachments,
			features: resolved,
			postProcess,
			shadowMaps: this.shadowMaps,
			scene: frame,
			shCoeffs: this.shCoeffs,
			shAmbientCoeffs: this.shAmbientCoeffs,
			worldMatrix: this.features.worldMatrix || Matrix4.identity(),
			incremental,
			transient,
		};
	}

	private _resolveShadowBackendCapabilities(): IShadowBackendCapabilities {
		const supportsDirectionalCSM = this.backend.type !== "unknown";
		const supportsPositionalCSM = this.backend.type === "software";
		return {
			backendKey: this.backend.type,
			supportsFilterModes: ["pcf", "vsm"],
			supportsDirectionalCSM,
			supportsSpotCSM: supportsPositionalCSM,
			supportsPointCSM: supportsPositionalCSM,
			maxDynamicShadowCost: this._resolveShadowBudgetFromBackend(),
		};
	}

	private _resolveShadowBudgetFromBackend(): number {
		switch (this.backend.type) {
			case "webgpu":
				return 48;
			case "webgl":
				return 24;
			case "software":
				return 20;
			default:
				return 16;
		}
	}

	private _buildIncrementalFrameContext(
		plan: ReturnType<typeof IncrementalFramePlanner.plan>,
		prepared: PreparedSceneCacheBuildResult,
		initialFullFrameRect: ReturnType<typeof makeFullScreenRect>,
		initialFullFrameTiles: DirtyTileCoverage
	): IncrementalFrameContext {
		const enabled = this._incrementalOptions.enabled;
		const forceFullFrame = plan.forceFullFrame || prepared.forceFullFrame;
		const fullFrameRect = initialFullFrameRect;
		const fullFrameTiles = initialFullFrameTiles;
		let dirtyRects =
			enabled && !forceFullFrame ? prepared.dirtyRects.slice() : [fullFrameRect];
		let dirtyTiles =
			enabled && !forceFullFrame ?
				prepared.dirtyTiles.slice()
			:	fullFrameTiles.dirtyTiles.slice();
		let dirtyTileSize =
			enabled && !forceFullFrame ?
				prepared.dirtyTileSize
			:	fullFrameTiles.tileSize;
		let dirtyTileColumns =
			enabled && !forceFullFrame ?
				prepared.dirtyTileColumns
			:	fullFrameTiles.tileColumns;
		let dirtyTileRows =
			enabled && !forceFullFrame ?
				prepared.dirtyTileRows
			:	fullFrameTiles.tileRows;
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

	private _createStageExecutors(): Map<string, RendererStageExecutor> {
		return new Map<string, RendererStageExecutor>([
			[
				"feature-resolution",
				(state) => this._executeFeatureResolutionStage(state),
			],
			["sync-in", (state) => this._executeSyncInStage(state)],
			["animation-sim", (state) => this._executeAnimationStage(state)],
			["physics-sim", (state) => this._executePhysicsStage(state)],
			["transform-update", () => this._executeTransformUpdateStage()],
			["lod-resolve", () => this._resolveLODMeshes()],
			["csg-resolve", () => this._resolveCSGMeshes()],
			[
				"prepared-scene-build",
				(state) => this._executePreparedSceneBuildStage(state),
			],
			[
				"probe-capture",
				(state) => this._executeProbeCaptureStage(state),
			],
			["sync-out", () => this.scene.syncECSToNode()],
		]);
	}

	private _createRenderSceneFrameState(options: {
		now: number;
		deltaTimeSeconds: number;
		frameDirtyReasonMask: number;
		transient: TransientStore;
		hasActiveAnimations: boolean;
		hasParticleSystems: boolean;
	}): RenderSceneFrameState {
		const resolved = resolveFeatureState(
			this.features,
			this.backend.capabilities,
			this.backend.type
		);
		const resolvedPostProcess = this.postProcess.createSnapshot(
			this.backend.type
		);
		const {
			fullFrameRect: initialFullFrameRect,
			fullFrameTiles: initialFullFrameTiles,
		} = this._createFullFrameCoverage();
		const incrementalFrameContext = this._createInitialIncrementalFrameContext(
			options.frameDirtyReasonMask,
			initialFullFrameRect,
			initialFullFrameTiles
		);
		const stageOrder = this.pipeline.getExecutionOrder(
			{
				hasActiveAnimations:
					options.hasActiveAnimations && this.animationAutoRender,
				hasParticleSystems: options.hasParticleSystems,
			},
			(key, message) =>
				this.logger.warn(`[${key}] ${message}`, {
					scope: "Renderer",
					onceKey: key,
				})
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
			hasAnimationStage: stageOrder.some(
				(stage) => stage.id === "animation-sim"
			),
			initialFullFrameRect,
			initialFullFrameTiles,
			incrementalFrameContext,
			incrementalStartStageIndex: -1,
		};
	}

	private async _executeRenderStage(
		stageId: string,
		state: RenderSceneFrameState
	): Promise<void> {
		const executor = this._stageExecutors.get(stageId);
		if (executor) {
			await executor(state);
			return;
		}
		const stageKind = this.pipeline.getStageKind(stageId as FramePassStage);
		if (stageKind === "backend-pass" || stageKind === "shared-pass") {
			await this._executeFramePassStage(stageId, state);
			return;
		}
		if (stageKind === "renderer") {
			const key = `renderer-stage-executor-missing-${stageId}`;
			this.logger.warn(
				`[${key}] Renderer pipeline stage "${stageId}" has no internal executor; skipping stage`,
				{ scope: "Renderer", onceKey: key }
			);
			return;
		}
		const key = `renderer-stage-kind-missing-${stageId}`;
		this.logger.warn(
			`[${key}] Renderer pipeline stage "${stageId}" is not registered; skipping stage`,
			{ scope: "Renderer", onceKey: key }
		);
	}

	private _executeFeatureResolutionStage(
		state: RenderSceneFrameState
	): void {
		state.resolved = resolveFeatureState(
			this.features,
			this.backend.capabilities,
			this.backend.type
		);
		state.resolvedPostProcess = this.postProcess.createSnapshot(
			this.backend.type
		);
		for (const warning of [
			...state.resolved.warnings,
			...state.resolvedPostProcess.getWarnings(),
		]) {
			this.logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "Renderer",
				onceKey: warning.key,
			});
		}
	}

	private _executeSyncInStage(state: RenderSceneFrameState): void {
		this.scene.syncNodeToECS();
		if (!state.hasAnimationStage && !state.emittedPostAnimation) {
			this._emitPostAnimation(state);
		}
	}

	private _executeAnimationStage(state: RenderSceneFrameState): void {
		this._animationStage.execute(
			{
				scene: this.scene,
				transient: state.transient,
			},
			this._deltaTime
		);
		if (!state.emittedPostAnimation) {
			this._emitPostAnimation(state);
		}
	}

	private async _executePhysicsStage(
		state: RenderSceneFrameState
	): Promise<void> {
		if (this._physicsSystem) {
			await this._physicsSystem.stepAsync(state.deltaTimeSeconds);
		}
	}

	private _executeTransformUpdateStage(): void {
		this.scene.updateWorldMatrices();
		this._assertCameraInScene(this.scene, this.camera, "renderScene");
		this.camera.updateMatrices();
		this.refreshReflectionProbeCaches();
	}

	private async _executePreparedSceneBuildStage(
		state: RenderSceneFrameState
	): Promise<void> {
		if (this.features.enableSH) {
			this.updateSH();
		}
		const preparedResult = this._preparedSceneCache.build({
			renderer: this,
			viewportWidth: this.canvas.width,
			viewportHeight: this.canvas.height,
			features: state.resolved,
			postProcess: state.resolvedPostProcess,
			incrementalOptions: this._incrementalOptions,
			occlusionVisibilityProvider:
				this._occlusionCullingController.getVisibilityProvider(
					state.resolved
				),
			occlusionCullingOptions: state.resolved.occlusionCullingOptions,
		});
		state.frame = preparedResult.frame;
		const incrementalPlan = IncrementalFramePlanner.plan({
			enabled: this._incrementalOptions.enabled,
			reasonMask: state.frameDirtyReasonMask,
			features: state.resolved,
			postProcess: state.resolvedPostProcess,
			registry: this.pipeline.incremental,
		});
		state.incrementalFrameContext = this._buildIncrementalFrameContext(
			incrementalPlan,
			preparedResult,
			state.initialFullFrameRect,
			state.initialFullFrameTiles
		);
		state.incrementalStartStageIndex =
			state.incrementalFrameContext.enabled &&
			!state.incrementalFrameContext.forceFullFrame &&
			state.incrementalFrameContext.firstPass ?
				(state.stageIndexById.get(state.incrementalFrameContext.firstPass) ??
					-1)
			:	-1;
		this._recordIncrementalFrameStats(state.incrementalFrameContext);

		const context = this._createFrameContext(
			state.frame,
			state.resolved,
			state.resolvedPostProcess,
			state.transient,
			state.incrementalFrameContext
		);
		state.context = {
			...context,
			framePlan: this.pipeline.createFramePlan({
				stageOrder: state.stageOrder,
				frame: state.frame,
				features: state.resolved,
				postProcess: state.resolvedPostProcess,
				transient: state.transient,
				backendType: this.backend.type,
				backendCapabilities: this.backend.capabilities,
				incremental: state.incrementalFrameContext,
				frameContext: context,
				incrementalStartStageIndex: state.incrementalStartStageIndex,
			}),
		};
		state.frameStarted = true;
		await this.backend.beginFrame(state.context);
	}

	private async _executeProbeCaptureStage(
		state: RenderSceneFrameState
	): Promise<void> {
		const cameraWorldPosition = this.camera.getWorldPosition(
			_tmpRendererCameraWorldPosition
		);
		await this._probeCaptureRuntime.execute({
			scene: this.scene,
			nowMs: state.now,
			frameDirtyReasonMask: state.frameDirtyReasonMask,
			frameContext: state.context,
			cameraWorldPosition,
			webgpuCaptureSource:
				this.backend.type === "webgpu" ?
					(this.backend as unknown as ProbeWebGPUCaptureSource)
				:	null,
		});
	}

	private async _executeFramePassStage(
		stageId: string,
		state: RenderSceneFrameState
	): Promise<void> {
		if (!state.context || !state.frame) return;
		if (!this._isFramePassStage(stageId)) return;

		const pass =
			state.context.framePlan?.backendPasses.find(
				(candidate) => candidate.stage === stageId
			) ??
			this.pipeline.createFramePass(stageId as FramePassStage);
		if (!pass.enabled) {
			this.backend.skipPass?.(pass);
			return;
		}
		if (pass.executor === "shared") {
			if (!this.backend.executeSharedPass) {
				const key = `${this.backend.type}-shared-pass-${pass.stage}`;
				Logger.warn(
					`[${key}] ${this.backend.type} backend declared shared pass "${pass.stage}" without executeSharedPass implementation`,
					{ scope: "Renderer" }
				);
				return;
			}
			await this.backend.executeSharedPass(pass, state.context);
			return;
		}
		await this.backend.executePass(pass, state.context);
	}

	private _emitPostAnimation(state: RenderSceneFrameState): void {
		this.emit("postanimation", {
			now: state.now,
			deltaTime: this._deltaTime,
			scene: this.scene,
			transient: state.transient,
		});
		state.emittedPostAnimation = true;
	}

	private _recordIncrementalFrameStats(
		incrementalFrameContext: IncrementalFrameContext
	): void {
		this._lastIncrementalFrameStats = {
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
	}

	public async renderScene(now: number): Promise<void> {
		this._deltaTime = now - (this._lastTime || now);
		this._lastTime = now;

		this.emit("tick", { now, deltaTime: this._deltaTime });
		this.emit("framestart", { now, deltaTime: this._deltaTime });

		const hasParticleSystems = this.scene.getParticleSystems().length > 0;
		const hasActiveAnimations = this.animationSystem.hasActiveActions();
		if (hasParticleSystems) {
			this._pendingDirtyReasonMask |= renderDirtyReasonToMask("particles");
		}
		if (this.animationAutoRender && hasActiveAnimations) {
			this._pendingDirtyReasonMask |= renderDirtyReasonToMask("transform");
		}
		const hasDynamicTextureUpdates = Texture.updateDynamicTextures(now);
		if (hasDynamicTextureUpdates) {
			this._pendingDirtyReasonMask |= renderDirtyReasonToMask("texture");
			this._frameDirty = true;
		}
		if (
			this.scene.version !== this._lastKnownSceneVersion ||
			this.scene.dirtyReasonMask !== 0
		) {
			this._frameDirty = true;
			this._lastKnownSceneVersion = this.scene.version;
		}

		if (
			!this._frameDirty &&
			this.backend.frameScheduling === "on-demand" &&
			!hasParticleSystems &&
			!(this.animationAutoRender && hasActiveAnimations)
		) {
			this.emit("frameend", { now, deltaTime: this._deltaTime });
			requestAnimationFrame((time) => this.renderScene(time));
			return;
		}

		this._frameDirty = false;
		const frameDirtyReasonMask = this._consumeDirtyReasonMask();
		const transient = createTransientStore();
		const deltaTimeSeconds = Math.max(0, this._deltaTime) / 1000;
		transient.set(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, deltaTimeSeconds);
		transient.set(ANIMATION_SIM_DELTA_TIME_MS_KEY, this._deltaTime);
		if (this._frameTransientContributors.size > 0) {
			for (const contributor of this._frameTransientContributors) {
				contributor({
					now,
					deltaTime: this._deltaTime,
					scene: this.scene,
					camera: this.camera,
					transient,
				});
			}
		}
		const state = this._createRenderSceneFrameState({
			now,
			deltaTimeSeconds,
			frameDirtyReasonMask,
			transient,
			hasActiveAnimations,
			hasParticleSystems,
		});

		try {
			for (const stage of state.stageOrder) {
				await this._executeRenderStage(stage.id, state);
			}

			if (state.frameStarted) {
				await this.backend.endFrame();
				state.frameStarted = false;
			}
		} catch (error) {
			await this._abortFailedFrame(error, state.frameStarted);
			throw error;
		}

		this.emit("frameend", { now, deltaTime: this._deltaTime });
		requestAnimationFrame((time) => this.renderScene(time));
	}

	private async _abortFailedFrame(
		error: unknown,
		frameStarted: boolean
	): Promise<void> {
		if (!frameStarted || !this.backend.abortFrame) {
			return;
		}
		try {
			await this.backend.abortFrame(error);
		} catch (abortError) {
			const key = "renderer-backend-abort-failed";
			this.logger.warn(
				`[${key}] Failed to abort backend frame state: ${String(abortError)}`,
				{ scope: "Renderer" }
			);
		}
	}

	public updateSH(): void {
		const backendType = this.backend?.type ?? "software";
		let ambientProbeSH: SHCoefficients = SH.empty();
		let ambientR = 0;
		let ambientG = 0;
		let ambientB = 0;
		let hasAmbient = false;
		for (const light of this.scene.getLights()) {
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
					backendType !== "software" &&
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

		this._shAmbientCoeffs = ambientProbeSH.map((coefficient) => ({
			r: coefficient.r,
			g: coefficient.g,
			b: coefficient.b,
		})) as SHCoefficients;

		let totalSH: SHCoefficients = this._shAmbientCoeffs.map((coefficient) => ({
			r: coefficient.r,
			g: coefficient.g,
			b: coefficient.b,
		})) as SHCoefficients;

		for (const light of this.scene.getLights()) {
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

		this._shCoeffs = totalSH;
	}

	private _isFramePassStage(stageId: string): boolean {
		return this.pipeline.isFramePassStage(stageId as FramePassStage);
	}

	private _getSafeAspectRatio(width: number, height: number): number {
		return Math.max(width, 1) / Math.max(height, 1);
	}

	private _assertCameraInScene(
		scene: Scene,
		camera: Camera,
		caller: "setScene" | "setCamera" | "renderScene"
	): void {
		if (scene.contains(camera)) return;
		throw new Error(
			`Renderer.${caller} requires the active camera to belong to the active scene graph`
		);
	}

	public refreshReflectionProbeCaches(): void {
		for (const light of this.scene.getLights()) {
			if (light.type !== LightType.ReflectionProbe) continue;
			(light as ReflectionProbe).refreshRuntimeCache();
		}
	}

	private _resolveLODMeshes(): void {
		const cameraWorldPosition = this.camera.getWorldPosition(
			_tmpRendererCameraWorldPosition
		);
		const meshInstances = this.scene.getMeshInstances();
		for (const meshInstance of meshInstances) {
			if (!(meshInstance instanceof LODMeshInstance)) continue;
			meshInstance.updateLODForCamera(cameraWorldPosition, {
				notifyScene: false,
			});
		}
	}

	private async _resolveCSGMeshes(): Promise<void> {
		const meshInstances = this.scene.getMeshInstances();
		for (const meshInstance of meshInstances) {
			if (!(meshInstance instanceof CSGMeshInstance)) continue;
			if (!meshInstance.isCSGDirty) continue;
			if (meshInstance.physicsSync === "auto" && !meshInstance.physicsSystem) {
				meshInstance.physicsSystem = this._physicsSystem;
			}

			const flushResult = meshInstance.flushCSG();
			const result =
				flushResult instanceof Promise ? await flushResult : flushResult;
			for (const diagnostic of result.diagnostics) {
				if (diagnostic.severity === "info") continue;
				const key =
					`csg-diagnostic-${meshInstance.id}-` +
					`${diagnostic.code}-${diagnostic.message}`;
				Logger.warn(
					`[${key}] [CSG:${diagnostic.code}] ${diagnostic.message}`,
					{ scope: "Renderer" }
				);
			}
		}
	}
}
