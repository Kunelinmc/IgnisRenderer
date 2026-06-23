import { Camera } from "../cameras/Camera";
import { Scene } from "../core/Scene";
import { EventEmitter } from "../core/EventEmitter";
import { Logger, type LoggerStatic } from "../foundation/Logger";
import { AnimationSystem } from "../animation/AnimationSystem";
import type { PhysicsSystem } from "../physics";
import { RenderPipelineRegistry } from "../pipeline/RenderPipelineRegistry";
import { createDefaultPipelineStages } from "../pipeline/defaultPipeline";
import {
	GammaPass,
	PostProcessPassRegistry,
	ToneMappingPass,
} from "../postprocess";
import { resolveFeatureState } from "../pipeline/FeatureResolver";
import { PreparedSceneBuilder } from "../pipeline/PreparedSceneBuilder";
import {
	createTransientStore,
	ANIMATION_SIM_DELTA_TIME_MS_KEY,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	type TransientStore,
	type RendererFeatureFlags,
	type RendererFeatureResolvedOptions,
} from "../pipeline/types";

export type RendererFeatures = RendererFeatureFlags &
	RendererFeatureResolvedOptions & {
		worldMatrix: Matrix4;
	};

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

export type RenderFrameResult =
	| { rendered: true }
	| { rendered: false; reason: "clean" };

import { getWarmupStartDelay } from "../pipeline/WarmupScheduler";
import {
	buildDirtyTileCoverage,
	DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
	IncrementalFramePlanner,
	makeFullScreenRect,
	mergeIncrementalRenderingOptions,
	renderDirtyReasonToMask,
	type IncrementalFrameStats,
	type IncrementalRenderingOptions,
	type RenderDirtyReason,
	type DirtyTileCoverage,
} from "../pipeline/incremental";
import type {
	IRenderBackend,
	RenderBackendProfile,
	RenderBackendEvent,
	RenderBackendDeviceLostInfo,
	RenderBackendType,
	RendererBackendResourceEvent,
	WarmupOptions,
	WarmupReport,
} from "./IRenderBackend";
import type { RenderBackendExtensionKey } from "./BackendExtensions";
import {
	CustomRenderPassRegistry,
	RenderTargetRegistry,
	type RenderTargetReadbackOptions,
} from "./CustomRenderTargets";
import type { TextureReadbackResult } from "./IComputeRuntime";
import { RendererRuntime } from "./RendererRuntime";
import {
	FrameCoordinator,
	type FrameCoordinatorDelegate,
} from "./FrameCoordinator";
import type { SHCoefficients } from "../maths/types";
import {
	LightType,
	ReflectionProbe,
	type ShadowCastingLight,
	LightProbe,
} from "../lights";
import { isLocalizedLightProbe } from "../lights/runtime/lightProbeRuntime";
import { PBR_AMBIENT_FALLBACK_LINEAR } from "../lights/constants";
import { Vector3 } from "../maths/Vector3";
import type { ShadowRenderSet } from "../lights/shadows/ShadowMapping";
import { Matrix4 } from "../maths/Matrix4";
import { Texture } from "../core/Texture";
import { sRGBToLinear } from "../maths/Common";
import { SH } from "../maths/SH";

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
			backend: RenderBackendType;
			info?: RenderBackendDeviceLostInfo;
		},
	];
	devicerestored: [{ backend: RenderBackendType }];
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

export interface RendererOptions {
	readonly canvas: HTMLCanvasElement;
	readonly backend: IRenderBackend;
	readonly camera?: Camera | null;
}

export class Renderer extends EventEmitter<RendererEvents> implements FrameCoordinatorDelegate {
	public readonly backendProfile: RenderBackendProfile;
	public readonly animationSystem: AnimationSystem;
	public readonly features: RendererFeatures;
	public readonly pipeline: RenderPipelineRegistry;
	public readonly postProcess: PostProcessPassRegistry;
	public readonly renderTargets: RenderTargetRegistry;
	public readonly renderPasses: CustomRenderPassRegistry;
	public animationAutoRender: boolean;
	public readonly logger: Pick<LoggerStatic, "warn" | "error">;

	private readonly _canvas: HTMLCanvasElement;
	private _scene: Scene;
	private _camera: Camera;
	private _physicsSystem: PhysicsSystem | null = null;
	private readonly _frameTransientContributors = new Set<FrameTransientContributor>();
	private _incrementalOptions: IncrementalRenderingOptions;
	private _lastIncrementalFrameStats: IncrementalFrameStats | null = null;
	private readonly _runtime: RendererRuntime;
	private readonly _coordinator: FrameCoordinator;
	private _renderLoopStop: (() => void) | null = null;

	private _shadowMaps = new Map<ShadowCastingLight, ShadowRenderSet>();
	private _shCoeffs: SHCoefficients = [] as any;
	private _shAmbientCoeffs: SHCoefficients = [] as any;

	private _lastTime = 0;
	private _deltaTime = 0;
	private _frameDirty = true;
	private _pendingDirtyReasonMask = renderDirtyReasonToMask("unknown");
	private _lastKnownSceneVersion = 0;
	private _deviceScaleFactor = 1;

	constructor(options: RendererOptions | IRenderBackend, canvasParam?: HTMLCanvasElement, cameraParam?: Camera | null) {
		super();
		let backend: IRenderBackend;
		let canvas: HTMLCanvasElement;
		let camera: Camera | null = null;
		if (options && typeof options === "object" && "canvas" in options) {
			const opts = options as RendererOptions;
			backend = opts.backend;
			canvas = opts.canvas;
			camera = opts.camera ?? null;
		} else {
			backend = options as IRenderBackend;
			canvas = canvasParam!;
			camera = cameraParam ?? null;
		}

		this._canvas = canvas;
		this.logger = Logger;
		this.animationSystem = new AnimationSystem();
		this.pipeline = new RenderPipelineRegistry({
			stages: createDefaultPipelineStages(),
		});
		this.renderTargets = new RenderTargetRegistry({
			readColor: (
				id: string,
				attachmentIndex: number,
				options?: RenderTargetReadbackOptions
			): Promise<TextureReadbackResult> =>
				this._readRenderTargetColor(id, attachmentIndex, options),
		});
		this.renderPasses = new CustomRenderPassRegistry({
			registerPipelineStage: (pass) => {
				this.pipeline.registerPipelineStage({
					id: pass.id,
					kind: "backend-pass",
					dependsOn: pass.dependsOn ?? ["prepared-scene-build"],
					shouldRun: (context) => {
						if (!context.frameContext?.renderTargets.has(pass.target)) {
							return false;
						}
						return pass.shouldRun ? pass.shouldRun(context) : true;
					},
					incremental: pass.incremental,
				});
			},
			unregisterPipelineStage: (id) => this.pipeline.unregisterPipelineStage(id),
		});

		this.postProcess = new PostProcessPassRegistry();
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
		this.renderTargets.on("change", () => {
			if (this._scene) {
				this._markFrameDirty("unknown");
			} else {
				this._frameDirty = true;
				this._pendingDirtyReasonMask |= renderDirtyReasonToMask("unknown");
			}
		});
		this.renderPasses.on("change", () => {
			if (this._scene) {
				this._markFrameDirty("unknown");
			} else {
				this._frameDirty = true;
				this._pendingDirtyReasonMask |= renderDirtyReasonToMask("unknown");
			}
		});

		backend.attach({
				surface: { canvas },
				events: { emit: (event) => this._handleBackendEvent(event) },
		});

		this.backendProfile = backend.profile;

		this._runtime = new RendererRuntime(backend, (event) => this._handleBackendEvent(event));
		this._coordinator = new FrameCoordinator(backend, this.animationSystem);

		this._deviceScaleFactor = window.devicePixelRatio || 1;
		this.animationAutoRender = true;
		this._incrementalOptions = { ...DEFAULT_INCREMENTAL_RENDERING_OPTIONS };
		// Initialize default baseline features requested by the renderer.
		// While the `features` property is readonly to prevent reference reassignment,
		// its fields are mutable and can be configured by external consumers post-construction.
		// At runtime, these requested features are resolved and validated against the actual
		// backend capabilities (see `resolveFeatureState`).
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

		this._scene = new Scene();
		this._lastKnownSceneVersion = this._scene.version;
		this._camera = camera || new Camera();

		if (!this._camera.parent) {
			this._scene.add(this._camera);
		}

		if (!camera) {
			this._camera.position.set(0, 200, 200);
			this._camera.fov = 60;
		}

		this._camera.aspectRatio = this._getSafeAspectRatio(
			this.canvas.width,
			this.canvas.height
		);
		this._camera.updateMatrices();
	}

	/**
	 * Initializes the attached rendering backend and sets up default resource registries.
	 * Must be called before performing any rendering actions.
	 * 
	 * @throws {Error} If the renderer is already initialized or has been destroyed.
	 * @returns A promise that resolves when initialization is complete.
	 */
	public async initialize(): Promise<void> {
		this._runtime.assertNotDestroyed("initialize");
		if (this._runtime.isInitialized) {
			throw new Error("Renderer.initialize(): already initialized.");
		}
		await this._runtime.initialize();
		this.resizeCanvas();
	}

	/**
	 * Restores the renderer after context loss or disposal. Re-initializes runtime and resets coordinators.
	 * 
	 * @returns A promise that resolves when the restore process completes.
	 */
	public async restore(): Promise<void> {
		if (!this._runtime.isInitialized) {
			await this.initialize();
		}
		this._runtime.assertReady("restore");
		await this._runtime.restore();
		this._coordinator.reset();
		this.resizeCanvas();
	}

	/**
	 * Disposes all allocated GPU resources, stops the render loop, and cleans up event listeners.
	 * Once destroyed, the renderer instance can no longer be used.
	 * 
	 * @returns A promise that resolves when disposal is complete.
	 */
	public async destroy(): Promise<void> {
		this._renderLoopStop?.();
		await this._runtime.destroy();
		if (this._physicsSystem) {
			this._physicsSystem.bindSceneSpatial(null);
		}
		this._frameTransientContributors.clear();
	}

	/**
	 * Gets the canvas element associated with this renderer.
	 */
	public get canvas(): HTMLCanvasElement {
		return this._canvas;
	}

	/**
	 * Gets the currently active scene graph.
	 */
	public get scene(): Scene {
		return this._scene;
	}

	/**
	 * Gets the active rendering camera.
	 */
	public get camera(): Camera {
		return this._camera;
	}

	/**
	 * Gets the registered physics system instance, if any.
	 */
	public get physicsSystem(): PhysicsSystem | null {
		return this._physicsSystem;
	}

	/**
	 * Gets the current configuration options for incremental rendering.
	 */
	public get incrementalOptions(): IncrementalRenderingOptions {
		return this._incrementalOptions;
	}

	/**
	 * Gets the set of active frame transient contributors.
	 */
	public get frameTransientContributors(): Set<FrameTransientContributor> {
		return this._frameTransientContributors;
	}

	/**
	 * Gets the active shadow maps mapping.
	 * 
	 * @internal Owned by the pipeline stage system.
	 */
	public get shadowMaps(): Map<ShadowCastingLight, ShadowRenderSet> {
		return this._shadowMaps;
	}

	/**
	 * Gets the current spherical harmonics coefficients.
	 * 
	 * @internal Owned by the spherical harmonics lighting subsystem.
	 */
	public get shCoeffs(): SHCoefficients {
		return this._shCoeffs;
	}

	/**
	 * Gets the current ambient spherical harmonics coefficients.
	 * 
	 * @internal Owned by the spherical harmonics lighting subsystem.
	 */
	public get shAmbientCoeffs(): SHCoefficients {
		return this._shAmbientCoeffs;
	}

	/**
	 * Gets the timestamp of the last executed frame.
	 * 
	 * @internal Owned by the frame execution scheduling subsystem.
	 */
	public get lastTime(): number {
		return this._lastTime;
	}

	/**
	 * Sets the global spherical harmonics coefficients.
	 * 
	 * @internal Owned by the spherical harmonics lighting subsystem. Preferred alternative: update lighting sources in the scene.
	 * @param coeffs The new spherical harmonics coefficients.
	 */
	public setSHCoefficients(coeffs: SHCoefficients): void {
		this._shCoeffs = coeffs;
	}

	/**
	 * Sets the ambient spherical harmonics coefficients.
	 * 
	 * @internal Owned by the spherical harmonics lighting subsystem. Preferred alternative: update ambient lights in the scene.
	 * @param coeffs The new ambient spherical harmonics coefficients.
	 */
	public setSHAmbientCoefficients(coeffs: SHCoefficients): void {
		this._shAmbientCoeffs = coeffs;
	}

	/**
	 * Updates the active shadow map registry.
	 * 
	 * @internal Owned by the shadow mapping subsystem.
	 * @param shadowMaps The new shadow map collection mapping.
	 */
	public setShadowMaps(shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>): void {
		this._shadowMaps = shadowMaps;
	}

	/**
	 * Sets stats for the last incremental frame.
	 * 
	 * @internal Owned by the incremental rendering subsystem.
	 * @param stats The stats to store.
	 */
	public setLastIncrementalFrameStats(stats: IncrementalFrameStats | null): void {
		this._lastIncrementalFrameStats = stats;
	}

	/**
	 * Retrieves an extension interface supported by the active rendering backend.
	 * 
	 * @template TApi The extension api type.
	 * @param key The key identifying the requested extension.
	 * @returns The extension instance if supported, otherwise null.
	 */
	public getBackendExtension<TApi>(key: RenderBackendExtensionKey<TApi>): TApi | null {
		return this._runtime.backend.extensions.getBackendExtension(key);
	}

	/**
	 * Retrieves an extension interface supported by the active rendering backend, throwing if not available.
	 * 
	 * @template TApi The extension api type.
	 * @param key The key identifying the requested extension.
	 * @throws {Error} If the extension is not supported by the backend.
	 * @returns The extension instance.
	 */
	public requireBackendExtension<TApi>(key: RenderBackendExtensionKey<TApi>): TApi {
		return this._runtime.backend.extensions.requireBackendExtension(key);
	}

	/**
	 * Emits a post-animation event.
	 * 
	 * @internal Owned by the FrameCoordinator subsystem. Preferred alternative: subscribe to `RendererEvents.postanimation` instead of calling `emitPostAnimation` directly.
	 * @param now The current timestamp.
	 * @param deltaTime The elapsed time since the last frame.
	 * @param transient The current transient store.
	 */
	public emitPostAnimation(now: number, deltaTime: number, transient: TransientStore): void {
		this.emit("postanimation", {
			now,
			deltaTime,
			scene: this._scene,
			transient,
		});
	}

	/**
	 * Refreshes all runtime caches for reflection probes in the scene.
	 */
	public refreshReflectionProbeCaches(): void {
		for (const light of this._scene.getLights()) {
			if (light.type !== LightType.ReflectionProbe) continue;
			(light as ReflectionProbe).refreshRuntimeCache();
		}
	}

	/**
	 * Logs a warning event with the renderer logger prefix.
	 * 
	 * @internal Owned by the logging/diagnostics subsystem.
	 * @param key Unique key for deduping warnings.
	 * @param message Warning message.
	 */
	public warn(key: string, message: string): void {
		this.logger.warn(`[${key}] ${message}`, {
			scope: "Renderer",
			onceKey: key,
		});
	}

	/**
	 * Triggers the rendering of a single frame. This checks initialization and runs the execution coordinator.
	 * 
	 * @param nowMs The current animation frame timestamp in milliseconds.
	 * @throws {Error} If another frame is already rendering concurrently.
	 * @returns A promise resolving to the frame render result (rendered or skipped).
	 */
	public async renderFrame(nowMs: number): Promise<RenderFrameResult> {
		if (!this._runtime.isInitialized) {
			await this.initialize();
		}
		this._runtime.assertReady("renderFrame");
		if (this._runtime.activeFramePromise) {
			return Promise.reject(
				new Error("Renderer.renderFrame() cannot run concurrently.")
			);
		}
		const operation = this._renderFrame(nowMs);
		this._runtime.activeFramePromise = operation;
		operation.catch(() => {}).finally(() => {
			if (this._runtime.activeFramePromise === operation) {
				this._runtime.activeFramePromise = null;
			}
		});
		return operation;
	}

	/**
	 * Starts a serialized animation-frame loop for this renderer.
	 *
	 * Frame failures are logged and do not stop later frames. Repeated calls
	 * while the loop is active return the same stop function.
	 *
	 * @returns An idempotent function that stops the loop.
	 * @constraints The renderer must not have been destroyed.
	 * @sideEffects Schedules animation frames and logs rejected frame errors.
	 */
	public renderLoop(): () => void {
		this._runtime.assertNotDestroyed("renderLoop");
		if (this._renderLoopStop) {
			return this._renderLoopStop;
		}

		let active = true;
		let requestId: number | null = null;
		const scheduleNextFrame = (): void => {
			if (!active) return;
			requestId = requestAnimationFrame((nowMs) => {
				requestId = null;
				void this.renderFrame(nowMs)
					.catch((error) => {
						this.logger.error(
							["Renderer render loop frame failed.", error],
							{ scope: "Renderer" }
						);
					})
					.finally(scheduleNextFrame);
			});
		};
		const stop = (): void => {
			if (!active) return;
			active = false;
			if (requestId !== null && typeof cancelAnimationFrame === "function") {
				cancelAnimationFrame(requestId);
			}
			requestId = null;
			if (this._renderLoopStop === stop) {
				this._renderLoopStop = null;
			}
		};

		this._renderLoopStop = stop;
		scheduleNextFrame();
		return stop;
	}

	/**
	 * Renders one frame through the legacy scene-rendering entry point.
	 *
	 * @deprecated Use `renderFrame(nowMs)` for manual frame control or
	 * `renderLoop()` for automatic animation-frame scheduling.
	 * @param nowMs Animation-frame timestamp in milliseconds.
	 * @returns The result of the delegated `renderFrame(nowMs)` call.
	 * @sideEffects Performs the same rendering work as `renderFrame(nowMs)`.
	 */
	public async renderScene(nowMs: number): Promise<RenderFrameResult> {
		return this.renderFrame(nowMs);
	}

	private async _renderFrame(now: number): Promise<RenderFrameResult> {
		this._deltaTime = now - (this._lastTime || now);
		this._lastTime = now;

		this.emit("tick", { now, deltaTime: this._deltaTime });

		const hasParticleSystems = this._scene.getParticleSystems().length > 0;
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
			this._scene.version !== this._lastKnownSceneVersion ||
			this._scene.dirtyReasonMask !== 0
		) {
			this._frameDirty = true;
			this._lastKnownSceneVersion = this._scene.version;
		}

		if (
			!this._frameDirty &&
			this.backendProfile.frameScheduling === "on-demand" &&
			!hasParticleSystems &&
			!(this.animationAutoRender && hasActiveAnimations)
		) {
			return { rendered: false, reason: "clean" };
		}

		this.emit("framestart", { now, deltaTime: this._deltaTime });
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
					scene: this._scene,
					camera: this._camera,
					transient,
				});
			}
		}

		await this._coordinator.executeFrame(
			this,
			now,
			this._deltaTime,
			frameDirtyReasonMask,
			hasParticleSystems,
			hasActiveAnimations,
			deltaTimeSeconds,
			transient
		);

		this.emit("frameend", { now, deltaTime: this._deltaTime });
		return { rendered: true };
	}

	/**
	 * Warmup system execution to precompile shaders/pipelines and initialize cache buffers.
	 * 
	 * @param options Warmup planning options.
	 * @returns A promise resolving to the warmup report detailing compiled pipeline count.
	 */
	public async warmup(options: WarmupOptions = {}): Promise<WarmupReport> {
		const warmupStartDelay = getWarmupStartDelay(options);
		if (warmupStartDelay) {
			await warmupStartDelay;
		}
		this._scene.syncNodeToECS();
		this._scene.updateWorldMatrices();
		this.refreshReflectionProbeCaches();
		this._assertCameraInScene(this._scene, this._camera, "renderScene");
		this._camera.updateMatrices();
		const transient = createTransientStore();
		transient.set(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 0);
		transient.set(ANIMATION_SIM_DELTA_TIME_MS_KEY, 0);

		const resolved = resolveFeatureState(
			this.features,
			this.backendProfile.capabilities,
			this.backendProfile.id
		);
		const resolvedPostProcess = this.postProcess.createSnapshot(
			this.backendProfile.id
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
		const frame = PreparedSceneBuilder.build({
			scene: this._scene,
			camera: this._camera,
			shadowMaps: this._shadowMaps,
			hasActiveAnimations: this.animationSystem.hasActiveActions(),
		});
		const { fullFrameRect, fullFrameTiles } = this._createFullFrameCoverage();
		const incrementalFrameContext = this._createInitialIncrementalFrameContext(
			renderDirtyReasonToMask("unknown"),
			fullFrameRect,
			fullFrameTiles
		);
		const context = this._coordinator.createWarmupContext(
			this,
			frame,
			resolved,
			resolvedPostProcess,
			transient,
			incrementalFrameContext
		);
		return this._coordinator.warmup(this, context, options);
	}

	/**
	 * Resizes the canvas to match its bounding client rect dimensions scaled by device scale factor.
	 * Invalidates current backend viewport bounds and marks the next frame dirty.
	 */
	public resizeCanvas(): void {
		const rect = this._canvas.getBoundingClientRect();
		this._deviceScaleFactor = window.devicePixelRatio || 1;
		this._canvas.width = rect.width * this._deviceScaleFactor;
		this._canvas.height = rect.height * this._deviceScaleFactor;

		this._runtime.backend.resize({
			width: this._canvas.width,
			height: this._canvas.height,
		});
		this._coordinator.reset();
		this._markFrameDirty("resize");

		this._camera.aspectRatio = this._getSafeAspectRatio(
			this._canvas.width,
			this._canvas.height
		);
		this._camera.updateMatrices();
	}

	/**
	 * Requests a frame redraw by marking the frame as dirty with a specific reason.
	 * 
	 * @param reason The reason triggering the redraw request.
	 */
	public requestRender(reason: RenderDirtyReason = "unknown"): void {
		this._markFrameDirty(reason);
	}

	/**
	 * Binds a new active scene to the renderer and resets the execution coordinator.
	 * 
	 * @param scene The new Scene graph instance.
	 */
	public setScene(scene: Scene): void {
		this._assertCameraInScene(scene, this._camera, "setScene");
		this._scene = scene;
		this._lastKnownSceneVersion = scene.version;
		this._coordinator.reset();
		if (this._physicsSystem) {
			this._physicsSystem.setEntityNodeResolver((entityId) => {
				return this._scene.ecs.getNodeByEntity(entityId);
			});
			this._physicsSystem.bindSceneSpatial(this._scene);
		}
		this._markFrameDirty("unknown");
	}

	/**
	 * Binds a new active camera to the renderer and resets the execution coordinator.
	 * 
	 * @param camera The new Camera instance.
	 */
	public setCamera(camera: Camera): void {
		this._assertCameraInScene(this._scene, camera, "setCamera");
		this._camera = camera;
		this._coordinator.reset();
		this._markFrameDirty("camera");
	}

	/**
	 * Binds a physics simulation system instance to update and query colliders/rigid bodies.
	 * 
	 * @param physicsSystem The physics system instance or null to unbind.
	 */
	public setPhysicsSystem(physicsSystem: PhysicsSystem | null): void {
		if (this._physicsSystem && this._physicsSystem !== physicsSystem) {
			this._physicsSystem.bindSceneSpatial(null);
		}
		this._physicsSystem = physicsSystem;
		if (physicsSystem) {
			physicsSystem.setEntityNodeResolver((entityId) => {
				return this._scene.ecs.getNodeByEntity(entityId);
			});
			physicsSystem.bindSceneSpatial(this._scene);
		}
	}

	/**
	 * Registers a callback hook that populates custom transient store data before every frame.
	 * 
	 * @param contributor The contributor function.
	 */
	public registerFrameTransientContributor(contributor: FrameTransientContributor): void {
		this._frameTransientContributors.add(contributor);
	}

	/**
	 * Unregisters a previously registered frame transient contributor callback.
	 * 
	 * @param contributor The contributor function.
	 */
	public unregisterFrameTransientContributor(contributor: FrameTransientContributor): void {
		this._frameTransientContributors.delete(contributor);
	}

	private _readRenderTargetColor(
		id: string,
		attachmentIndex: number,
		options?: RenderTargetReadbackOptions
	): Promise<TextureReadbackResult> {
		const read = this._runtime.backend.readRenderTargetColor;
		if (typeof read !== "function") {
			return Promise.reject(
				new Error(
					`${this.backendProfile.id} backend does not support render target readback.`
				)
			);
		}
		return read.call(this._runtime.backend, id, attachmentIndex, options);
	}

	/**
	 * Configures incremental rendering options such as tile sizing and dirty rect maximums.
	 * 
	 * @param options Partial incremental rendering configurations.
	 */
	public setIncrementalRendering(options: Partial<IncrementalRenderingOptions>): void {
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
		this._coordinator.reset();
		this._markFrameDirty("unknown");
	}

	/**
	 * Retrieves the current active configuration options for incremental rendering.
	 * 
	 * @returns A copy of the incremental rendering options.
	 */
	public getIncrementalRenderingOptions(): IncrementalRenderingOptions {
		return { ...this._incrementalOptions };
	}

	/**
	 * Retrieves statistics for the last executed incremental frame.
	 * 
	 * @returns The last frame stats or null if none exist or incremental rendering is disabled.
	 */
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

	/**
	 * Updates spherical harmonics coefficients by projecting lighting data from the active scene.
	 */
	public updateSH(): void {
		if (this._coordinator) {
			this._coordinator.updateSH(this);
		} else {
			const scene = this._scene || (this as any).scene;
			if (!scene) return;

			let ambientProbeSH: SHCoefficients = SH.empty();
			let ambientR = 0;
			let ambientG = 0;
			let ambientB = 0;
			let hasAmbient = false;

			const bType = (this as any).backend?.type || (this as any)._backendType;
			const isGpu = bType === "webgl" || bType === "webgpu";

			for (const light of scene.getLights()) {
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
					if (isGpu && isLocalizedLightProbe(probe)) {
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

			for (const light of scene.getLights()) {
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

			if (typeof this.setSHCoefficients === "function") {
				this.setSHCoefficients(totalSH);
			} else {
				(this as any)._shCoeffs = totalSH;
			}
			if (typeof this.setSHAmbientCoefficients === "function") {
				this.setSHAmbientCoefficients(shAmbientCoeffs);
			} else {
				(this as any)._shAmbientCoeffs = shAmbientCoeffs;
			}
		}
	}

	private _handleBackendEvent(event: RenderBackendEvent): void {
		switch (event.type) {
			case "device-lost":
				this._coordinator.reset();
				this._markFrameDirty("unknown");
				this.emit("devicelost", {
					backend: this.backendProfile.id,
					info: event.info,
				});
				return;
			case "device-restored":
				this._coordinator.reset();
				this.resizeCanvas();
				this.emit("devicerestored", { backend: this.backendProfile.id });
				return;
			case "render-invalidated":
				this._markFrameDirty(event.reason);
				return;
			case "resource-lifecycle":
				this.emit("backendresourceevent", event.event);
		}
	}

	private _markFrameDirty(reason: RenderDirtyReason = "unknown"): void {
		this._frameDirty = true;
		const reasonMask = this.pipeline.incremental.getDirtyReasonMask(reason);
		this._pendingDirtyReasonMask |= reasonMask;
		this._scene.invalidate(reason);
	}

	private _consumeDirtyReasonMask(): number {
		const sceneReasonMask = this._scene.consumeDirtyReasonMask();
		const combinedMask = this._pendingDirtyReasonMask | sceneReasonMask;
		this._pendingDirtyReasonMask = 0;
		return combinedMask >>> 0;
	}

	private _createFullFrameCoverage(): {
		fullFrameRect: ReturnType<typeof makeFullScreenRect>;
		fullFrameTiles: DirtyTileCoverage;
	} {
		const fullFrameRect = makeFullScreenRect(
			this._canvas.width,
			this._canvas.height
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
	): any {
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
}
