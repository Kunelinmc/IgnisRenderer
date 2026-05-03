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
import { AnimationSimulationStage } from "../pipeline/AnimationSimulationStage";
import { PreparedSceneBuilder } from "../pipeline/PreparedSceneBuilder";
import {
	PreparedSceneCache,
	type PreparedSceneCacheBuildResult,
} from "../pipeline/PreparedSceneCache";
import {
	ReflectionProbeCaptureRuntime,
	type ReflectionProbeWebGPUCaptureSource,
} from "../pipeline/ReflectionProbeCaptureRuntime";
import {
	RendererStageGraph,
	type RendererStageDefinition,
} from "../pipeline/RendererStageGraph";
import { bakeEnvironmentIBLFromEnvironmentMap } from "../pipeline/EnvironmentIBLBaker";
import {
	DEFAULT_ENVIRONMENT_IBL_UPDATE_OPTIONS,
	EnvironmentIBLUpdateRuntime,
	normalizeEnvironmentIBLUpdateOptions,
	type EnvironmentIBLUpdateOptions,
} from "../pipeline/EnvironmentIBLUpdateRuntime";
import {
	ensureEnvironmentTextureEquirect,
	isTextureReadyForEnvironment,
} from "../pipeline/environmentMapRuntime";
import { isLocalizedLightProbe } from "../pipeline/lightProbeRuntime";
import { hasParticleShadowCasters } from "../pipeline/ParticleShadowVolume";
import {
	ANIMATION_SIM_DELTA_TIME_MS_KEY,
	createTransientStore,
	INTERACTION_TRANSIENT_STATE_KEY,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
} from "../pipeline/types";
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
import type { WebGPUComputeFacadeSource } from "./webgpu/ComputeFacade";
import type {
	FramePass,
	FrameContext,
	RendererFeatureFlags,
	RendererFeatureResolvedOptions,
	TransientStore,
} from "../pipeline/types";
import type {
	IRenderBackend,
	WarmupOptions,
	WarmupProgress,
	WarmupReport,
} from "./IRenderBackend";

export type {
	IncrementalFrameStats,
	IncrementalRenderingOptions,
	RenderDirtyReason,
} from "../pipeline/incremental";
export type { EnvironmentIBLUpdateOptions } from "../pipeline/EnvironmentIBLUpdateRuntime";

export interface RendererEvents {
	tick: [{ now: number; deltaTime: number }];
	framestart: [{ now: number; deltaTime: number }];
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

export class Renderer extends EventEmitter<RendererEvents> {
	public readonly backend: IRenderBackend;
	public readonly animationSystem: AnimationSystem;
	public readonly features: RendererFeatures;
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
	private _stageGraph: RendererStageGraph;
	private _physicsSystem: PhysicsSystem | null;
	private _frameTransientContributors: Set<FrameTransientContributor>;
	private _incrementalOptions: IncrementalRenderingOptions;
	private _lastIncrementalFrameStats: IncrementalFrameStats | null;
	private _preparedSceneCache: PreparedSceneCache;
	private _reflectionProbeCaptureRuntime: ReflectionProbeCaptureRuntime;
	private _environmentIBLUpdateRuntime: EnvironmentIBLUpdateRuntime;
	private _environmentIBLUpdateOptions: EnvironmentIBLUpdateOptions;
	private _environmentIBLUpdateRequestToken: number;
	private _pendingDirtyReasonMask: number;
	private _lastKnownSceneVersion: number;
	private _allowSkyboxSpecularFallback: boolean;

	constructor(
		backend: IRenderBackend,
		canvas: HTMLCanvasElement,
		camera: Camera | null = null
	) {
		super();
		this.backend = backend;
		this.animationSystem = new AnimationSystem();
		this._canvas = canvas;
		this.logger = Logger;
		this._deviceScaleFactor = window.devicePixelRatio || 1;
		this._deltaTime = 0;
		this._frameDirty = true;
		this.animationAutoRender = true;
		this._animationStage = new AnimationSimulationStage(this.animationSystem);
		this._stageGraph = new RendererStageGraph(createDefaultRendererStages());
		this._physicsSystem = null;
		this._frameTransientContributors = new Set();
		this._incrementalOptions = { ...DEFAULT_INCREMENTAL_RENDERING_OPTIONS };
		this._lastIncrementalFrameStats = null;
		this._preparedSceneCache = new PreparedSceneCache();
		this._reflectionProbeCaptureRuntime =
			new ReflectionProbeCaptureRuntime();
		this._environmentIBLUpdateRuntime = new EnvironmentIBLUpdateRuntime();
		this._environmentIBLUpdateOptions = {
			...DEFAULT_ENVIRONMENT_IBL_UPDATE_OPTIONS,
		};
		this._environmentIBLUpdateRequestToken = 0;
		this._pendingDirtyReasonMask = renderDirtyReasonToMask("unknown");
		this._lastKnownSceneVersion = 0;
		this._allowSkyboxSpecularFallback = true;

		this.features = {
			enableLighting: true,
			enableGamma: true,
			enableToneMapping: true,
			enableSH: false,
			enableShadows: true,
			enableReflection: true,
			enableSkybox: true,
			enableOIT: false,
			enableSSAO: false,
			enableSSGI: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableFog: false,
			enableMotionBlur: false,
			enableDOF: false,
			enableBloom: false,
			enableColorFilter: false,
			enableFXAA: false,
			enableClusteredLighting: false,
			ssrOptions: {},
			volumetricOptions: {},
			fogOptions: {},
			ssaoOptions: {},
			ssgiOptions: {},
			taaOptions: {},
			bloomOptions: {},
			motionBlurOptions: {},
			dofOptions: {},
			colorFilterOptions: {},
			clusteredLightingOptions: {},
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

	public async warmup(options: WarmupOptions = {}): Promise<WarmupReport> {
		this.scene.syncNodeToECS();
		this.scene.updateWorldMatrices();
		this.refreshReflectionProbeCaches();
		this._assertCameraInScene(this.scene, this.camera, "renderScene");
		this.camera.updateMatrices();
		const environmentIBLUpdated = await this._warmupBakeEnvironmentIBL(options);
		if (environmentIBLUpdated) {
			this.scene.syncNodeToECS();
			this.scene.updateWorldMatrices();
			this.camera.updateMatrices();
		}

		const transient = createTransientStore();
		transient.set(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 0);
		transient.set(ANIMATION_SIM_DELTA_TIME_MS_KEY, 0);

		const resolved = resolveFeatureState(
			this.features,
			this.backend.capabilities,
			this.backend.type
		);
		for (const warning of resolved.warnings) {
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

	private async _warmupBakeEnvironmentIBL(
		options: WarmupOptions
	): Promise<boolean> {
		if (typeof options.allowSkyboxSpecularFallback === "boolean") {
			this._setAllowSkyboxSpecularFallback(
				options.allowSkyboxSpecularFallback
			);
		}
		const includeEnvironmentIBLBake =
			options.includeEnvironmentIBLBake ?? !!options.environmentIBLBake;
		if (includeEnvironmentIBLBake === false) {
			return false;
		}

		const skybox = ensureEnvironmentTextureEquirect(this.scene.skybox);
		if (!skybox || !isTextureReadyForEnvironment(skybox)) {
			return false;
		}

		const bakeOptions = {
			...(options.environmentIBLBake ?? {}),
		};
		if (!bakeOptions.webgpuSource && this.backend.type === "webgpu") {
			bakeOptions.webgpuSource =
				this.backend as unknown as WebGPUComputeFacadeSource;
		}

		const bakedEnvironment = await bakeEnvironmentIBLFromEnvironmentMap(skybox, {
			...bakeOptions,
			onProgress: options.onProgress ?
				(progress) => {
					const event: WarmupProgress = {
						phase: `environment-ibl-bake:${progress.phase}`,
						completed: progress.completed,
						total: progress.total,
						detail: progress.detail,
					};
					options.onProgress?.(event);
				}
			:	undefined,
		});

		const lights = this.scene.getLights();
		const probes = lights.filter(
			(light): light is LightProbe => light.type === LightType.LightProbe
		);

		if (probes.length === 0) {
			probes.push(this.scene.add(new LightProbe()));
		}

		for (const probe of probes) {
			probe.sh = this._cloneSHCoefficients(bakedEnvironment.sh);
		}

		const reflectionProbes = lights.filter(
			(light): light is ReflectionProbe =>
				light.type === LightType.ReflectionProbe
		);
		for (const reflectionProbe of reflectionProbes) {
			if (reflectionProbe.source !== "skybox") continue;
			reflectionProbe.prefilteredMap = bakedEnvironment.prefilteredMap;
			reflectionProbe.markRuntimeDirty();
		}

		this._markFrameDirty("lighting");
		return true;
	}

	private _cloneSHCoefficients(source: SHCoefficients): SHCoefficients {
		return source.map((coefficient) => ({
			r: coefficient.r,
			g: coefficient.g,
			b: coefficient.b,
		})) as SHCoefficients;
	}

	public resizeCanvas(): void {
		const rect = this.canvas.getBoundingClientRect();
		this._deviceScaleFactor = window.devicePixelRatio || 1;
		this.canvas.width = rect.width * this._deviceScaleFactor;
		this.canvas.height = rect.height * this._deviceScaleFactor;

		this.backend.resize(this.canvas.width, this.canvas.height);
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
		this._environmentIBLUpdateRuntime.reset();
		this._environmentIBLUpdateRequestToken = 0;
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

	public setEnvironmentIBLUpdateOptions(
		options: Partial<EnvironmentIBLUpdateOptions>
	): void {
		const next = normalizeEnvironmentIBLUpdateOptions({
			...this._environmentIBLUpdateOptions,
			...(options ?? {}),
		});
		if (
			next.enabled === this._environmentIBLUpdateOptions.enabled &&
			next.autoUpdate === this._environmentIBLUpdateOptions.autoUpdate &&
			next.mipsPerFrame === this._environmentIBLUpdateOptions.mipsPerFrame &&
			next.temporalBlendFactor ===
				this._environmentIBLUpdateOptions.temporalBlendFactor &&
			next.temporalBlendEpsilon ===
				this._environmentIBLUpdateOptions.temporalBlendEpsilon &&
			next.acceleration === this._environmentIBLUpdateOptions.acceleration &&
			next.prefilterMaxSampleWidth ===
				this._environmentIBLUpdateOptions.prefilterMaxSampleWidth &&
			next.prefilterMaxSampleHeight ===
				this._environmentIBLUpdateOptions.prefilterMaxSampleHeight &&
			next.prefilterMaxMipLevels ===
				this._environmentIBLUpdateOptions.prefilterMaxMipLevels &&
			next.resetTemporalHistoryOnComplete ===
				this._environmentIBLUpdateOptions.resetTemporalHistoryOnComplete
		) {
			return;
		}
		this._environmentIBLUpdateOptions = next;
		this._markFrameDirty("environment-ibl");
	}

	public getEnvironmentIBLUpdateOptions(): EnvironmentIBLUpdateOptions {
		return { ...this._environmentIBLUpdateOptions };
	}

	public requestEnvironmentIBLUpdate(): void {
		this._environmentIBLUpdateRequestToken++;
		this._markFrameDirty("environment-ibl");
	}

	public setStageGraph(stages: RendererStageDefinition[]): void {
		this._stageGraph.setStages(stages);
	}

	public registerStage(stage: RendererStageDefinition): void {
		this._stageGraph.registerStage(stage);
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

	public get allowSkyboxSpecularFallback(): boolean {
		return this._allowSkyboxSpecularFallback;
	}

	public get lastTime(): number {
		return this._lastTime;
	}

	private _setAllowSkyboxSpecularFallback(value: boolean): void {
		if (this._allowSkyboxSpecularFallback === value) {
			return;
		}
		this._allowSkyboxSpecularFallback = value;
		this._preparedSceneCache.reset();
		this._markFrameDirty("lighting");
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
			reasonMask,
			temporalHistoryReset: true,
		};
	}

	private _createFrameContext(
		frame: ReturnType<typeof PreparedSceneBuilder.build>,
		resolved: ReturnType<typeof resolveFeatureState>,
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
			reasonMask: plan.reasonMask,
			temporalHistoryReset: plan.temporalHistoryReset || forceFullFrame,
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
		let resolved = resolveFeatureState(
			this.features,
			this.backend.capabilities,
			this.backend.type
		);
		let frame: ReturnType<typeof PreparedSceneBuilder.build> | null = null;
		let preparedResult: PreparedSceneCacheBuildResult | null = null;
		let context: FrameContext | null = null;
		let frameStarted = false;
		let emittedPostAnimation = false;
		const {
			fullFrameRect: initialFullFrameRect,
			fullFrameTiles: initialFullFrameTiles,
		} = this._createFullFrameCoverage();
		let incrementalFrameContext = this._createInitialIncrementalFrameContext(
			frameDirtyReasonMask,
			initialFullFrameRect,
			initialFullFrameTiles
		);

		const stageOrder = this._stageGraph.getExecutionOrder(
			{
				hasActiveAnimations: hasActiveAnimations && this.animationAutoRender,
				hasParticleSystems,
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
		let incrementalStartStageIndex = -1;
		const hasAnimationStage = stageOrder.some(
			(stage) => stage.id === "animation-sim"
		);

		for (const stage of stageOrder) {
			switch (stage.id) {
				case "feature-resolution": {
					resolved = resolveFeatureState(
						this.features,
						this.backend.capabilities,
						this.backend.type
					);
					for (const warning of resolved.warnings) {
						this.logger.warn(`[${warning.key}] ${warning.message}`, {
							scope: "Renderer",
							onceKey: warning.key,
						});
					}
					break;
				}
				case "environment-ibl-update": {
					const updateResult = this._environmentIBLUpdateRuntime.execute({
						scene: this.scene,
						requestToken: this._environmentIBLUpdateRequestToken,
						options: this._environmentIBLUpdateOptions,
						webgpuSource:
							this.backend.type === "webgpu" ?
								(this.backend as unknown as WebGPUComputeFacadeSource)
							:	null,
					});
					if (updateResult.dirtyReason) {
						this._markFrameDirty(updateResult.dirtyReason);
					}
					break;
				}
				case "sync-in": {
					this.scene.syncNodeToECS();
					if (!hasAnimationStage && !emittedPostAnimation) {
						this.emit("postanimation", {
							now,
							deltaTime: this._deltaTime,
							scene: this.scene,
							transient,
						});
						emittedPostAnimation = true;
					}
					break;
				}
				case "animation-sim": {
					this._animationStage.execute(
						{
							scene: this.scene,
							transient,
						},
						this._deltaTime
					);
					if (!emittedPostAnimation) {
						this.emit("postanimation", {
							now,
							deltaTime: this._deltaTime,
							scene: this.scene,
							transient,
						});
						emittedPostAnimation = true;
					}
					break;
				}
				case "physics-sim": {
					if (this._physicsSystem) {
						await this._physicsSystem.stepAsync(deltaTimeSeconds);
					}
					break;
				}
				case "transform-update": {
					this.scene.updateWorldMatrices();
					this._assertCameraInScene(this.scene, this.camera, "renderScene");
					this.camera.updateMatrices();
					this.refreshReflectionProbeCaches();
					break;
				}
				case "lod-resolve": {
					this._resolveLODMeshes();
					break;
				}
				case "csg-resolve": {
					await this._resolveCSGMeshes();
					break;
				}
				case "prepared-scene-build": {
					if (this.features.enableSH) {
						this.updateSH();
					}
					preparedResult = this._preparedSceneCache.build({
						renderer: this,
						viewportWidth: this.canvas.width,
						viewportHeight: this.canvas.height,
						features: resolved,
						incrementalOptions: this._incrementalOptions,
					});
					frame = preparedResult.frame;
					const incrementalPlan = IncrementalFramePlanner.plan({
						enabled: this._incrementalOptions.enabled,
						reasonMask: frameDirtyReasonMask,
						features: resolved,
					});
					incrementalFrameContext = this._buildIncrementalFrameContext(
						incrementalPlan,
						preparedResult,
						initialFullFrameRect,
						initialFullFrameTiles
					);
					incrementalStartStageIndex =
						incrementalFrameContext.enabled &&
						!incrementalFrameContext.forceFullFrame &&
						incrementalFrameContext.firstPass ?
							(stageIndexById.get(incrementalFrameContext.firstPass) ?? -1)
						:	-1;
					this._lastIncrementalFrameStats = {
						enabled: incrementalFrameContext.enabled,
						reasonMask: incrementalFrameContext.reasonMask,
						forceFullFrame: incrementalFrameContext.forceFullFrame,
						temporalHistoryReset:
							incrementalFrameContext.temporalHistoryReset,
						firstPass: incrementalFrameContext.firstPass,
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
					context = this._createFrameContext(
						frame,
						resolved,
						transient,
						incrementalFrameContext
					);
					await this.backend.beginFrame(context);
					frameStarted = true;
					break;
				}
				case "reflection-probe-capture": {
					const cameraWorldPosition = this.camera.getWorldPosition(
						_tmpRendererCameraWorldPosition
					);
						await this._reflectionProbeCaptureRuntime.execute({
							scene: this.scene,
							nowMs: now,
							frameDirtyReasonMask,
							frameContext: context,
							cameraWorldPosition,
							webgpuSource:
							this.backend.type === "webgpu" ?
								(this.backend as unknown as WebGPUComputeFacadeSource)
							:	null,
						webgpuCaptureSource:
							this.backend.type === "webgpu" ?
								(this.backend as unknown as ReflectionProbeWebGPUCaptureSource)
							:	null,
					});
					break;
				}
				case "sync-out": {
					this.scene.syncECSToNode();
					break;
				}
				default: {
					if (!context || !frame) break;
					if (!this._isBackendPassStage(stage.id)) break;
					if (
						incrementalStartStageIndex >= 0 &&
						(stageIndexById.get(stage.id) ?? Number.MAX_SAFE_INTEGER) <
							incrementalStartStageIndex
					) {
						const skippedPass = this._createBackendPass(stage.id);
						this.backend.skipPass?.(skippedPass);
						break;
					}
					if (
						!this._shouldRunBackendPass(
							stage.id,
							frame,
							resolved,
							transient,
							incrementalFrameContext
						)
					) {
						const skippedPass = this._createBackendPass(stage.id);
						this.backend.skipPass?.(skippedPass);
						break;
					}

						const pass = this._createBackendPass(stage.id);
						if (pass.executor === "shared") {
							if (!this.backend.executeSharedPass) {
								const key = `${this.backend.type}-shared-pass-${pass.stage}`;
								Logger.warn(
									`[${key}] ${this.backend.type} backend declared shared pass "${pass.stage}" without executeSharedPass implementation`,
									{ scope: "Renderer" }
								);
							break;
						}
						await this.backend.executeSharedPass(pass, context);
					} else {
						await this.backend.executePass(pass, context);
					}
					break;
				}
			}
		}

		if (frameStarted) {
			await this.backend.endFrame();
		}

		this.emit("frameend", { now, deltaTime: this._deltaTime });
		requestAnimationFrame((time) => this.renderScene(time));
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
				const intensity = light.intensity ?? 1;
				const coeffCount = Math.min(ambientProbeSH.length, probeSH.length);
				for (let i = 0; i < coeffCount; i++) {
					ambientProbeSH[i].r += probeSH[i].r * intensity;
					ambientProbeSH[i].g += probeSH[i].g * intensity;
					ambientProbeSH[i].b += probeSH[i].b * intensity;
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

	private _isBackendPassStage(stageId: string): boolean {
		return BACKEND_PASS_STAGES.has(stageId);
	}

	private _createBackendPass(stageId: string): FramePass {
		return {
			stage: stageId,
			executor: this.backend.passExecutors?.[stageId] ?? "backend",
			enabled: true,
		};
	}

	private _shouldRunBackendPass(
		stage: string,
		frame: ReturnType<typeof PreparedSceneBuilder.build>,
		features: ReturnType<typeof resolveFeatureState>,
		transient: TransientStore,
		incremental?: IncrementalFrameContext
	): boolean {
		if (
			incremental?.enabled &&
			!incremental.forceFullFrame &&
			incremental.dirtyRects.length === 0
		) {
			return false;
		}
		switch (stage) {
			case "particle-sim":
				return (frame.particleSystems?.length ?? 0) > 0;
			case "shadow":
				return (
					features.enableShadows &&
					(frame.shadowCasterPackets.length > 0 ||
						hasParticleShadowCasters(frame.particleSystems))
				);
			case "reflection":
				return features.enableReflection && frame.reflectivePackets.length > 0;
			case "main-opaque":
				return true;
			case "main-transparent":
				return frame.transparentPackets.length > 0;
			case "particles":
				return (frame.particleSystems?.length ?? 0) > 0;
			case "ssao":
				return features.enableSSAO;
			case "ssgi":
				return features.enableSSGI;
			case "taa":
				return features.enableTAA;
			case "ssr":
				return features.enableSSR;
			case "volumetric":
				return features.enableVolumetric;
			case "fog":
				return (
					features.enableFog &&
					(features.fogOptions?.application ?? "postprocess") !== "scene"
				);
			case "motion-blur":
				return features.enableMotionBlur;
			case "dof":
				return features.enableDOF;
			case "bloom":
				return features.enableBloom;
			case "tonemap":
				return features.enableToneMapping !== false;
			case "color-filter":
				return features.enableColorFilter;
			case "fxaa":
				return features.enableFXAA;
			case "interaction-outline": {
				const interaction = transient.get(INTERACTION_TRANSIENT_STATE_KEY);
				return (interaction?.selectedEntityIds?.length ?? 0) > 0;
			}
			case "gamma":
				return features.enableGamma;
			default:
				return false;
		}
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

const BACKEND_PASS_STAGES = new Set<string>([
	"particle-sim",
	"shadow",
	"reflection",
	"main-opaque",
	"main-transparent",
	"particles",
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"tonemap",
	"color-filter",
	"fxaa",
	"interaction-outline",
	"gamma",
]);

function createDefaultRendererStages(): RendererStageDefinition[] {
	return [
		{ id: "feature-resolution", dependsOn: [] },
		{ id: "environment-ibl-update", dependsOn: ["feature-resolution"] },
		{ id: "sync-in", dependsOn: ["environment-ibl-update"] },
		{
			id: "animation-sim",
			dependsOn: ["sync-in"],
			enabled: (context) => context.hasActiveAnimations,
		},
		{ id: "physics-sim", dependsOn: ["animation-sim", "sync-in"] },
		{
			id: "transform-update",
			dependsOn: ["physics-sim", "animation-sim", "sync-in"],
		},
		{ id: "lod-resolve", dependsOn: ["transform-update"] },
		{ id: "csg-resolve", dependsOn: ["lod-resolve"] },
		{ id: "prepared-scene-build", dependsOn: ["csg-resolve"] },
		{
			id: "particle-sim",
			dependsOn: ["prepared-scene-build"],
			enabled: (context) => context.hasParticleSystems,
		},
		{ id: "shadow", dependsOn: ["prepared-scene-build", "particle-sim"] },
		{
			id: "reflection-probe-capture",
			dependsOn: ["prepared-scene-build"],
		},
		{
			id: "reflection",
			dependsOn: ["prepared-scene-build", "reflection-probe-capture"],
		},
		{ id: "main-opaque", dependsOn: ["reflection", "shadow"] },
		{ id: "main-transparent", dependsOn: ["main-opaque"] },
		{ id: "particles", dependsOn: ["main-transparent"] },
		{ id: "ssao", dependsOn: ["particles"] },
		{ id: "ssgi", dependsOn: ["ssao"] },
		{ id: "taa", dependsOn: ["ssgi", "ssao"] },
		{ id: "ssr", dependsOn: ["taa"] },
		{ id: "volumetric", dependsOn: ["ssr"] },
		{ id: "fog", dependsOn: ["volumetric"] },
		{ id: "motion-blur", dependsOn: ["fog"] },
		{ id: "dof", dependsOn: ["motion-blur"] },
		{ id: "bloom", dependsOn: ["dof"] },
		{ id: "tonemap", dependsOn: ["bloom"] },
		{ id: "color-filter", dependsOn: ["tonemap"] },
		{ id: "fxaa", dependsOn: ["color-filter"] },
		{ id: "interaction-outline", dependsOn: ["fxaa"] },
		{ id: "gamma", dependsOn: ["interaction-outline", "tonemap"] },
		{ id: "sync-out", dependsOn: ["gamma"] },
	];
}
