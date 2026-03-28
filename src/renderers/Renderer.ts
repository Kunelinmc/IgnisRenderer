import { Camera } from "../cameras/Camera";
import { LightType, type LightProbe, type ShadowCastingLight } from "../lights";
import { Matrix4 } from "../maths/Matrix4";
import { SH } from "../maths/SH";
import { Vector3 } from "../maths/Vector3";
import { sRGBToLinear } from "../maths/Common";
import { ShadowMap } from "../lights/ShadowMapping";
import {
	BAKED_LIGHT_PROBE_SH_SCALE,
	PBR_AMBIENT_FALLBACK_LINEAR,
} from "../lights/constants";
import { EventEmitter } from "../core/EventEmitter";
import { Scene } from "../core/Scene";
import { Texture } from "../core/Texture";
import { resolveFeatureState } from "../pipeline/FeatureResolver";
import { AnimationSimulationStage } from "../pipeline/AnimationSimulationStage";
import { PreparedSceneBuilder } from "../pipeline/PreparedSceneBuilder";
import { getDirectionalLightWorldDirection } from "../pipeline/LightTransforms";
import {
	RendererStageGraph,
	type RendererStageDefinition,
} from "../pipeline/RendererStageGraph";
import {
	ANIMATION_SIM_DELTA_TIME_MS_KEY,
	INTERACTION_TRANSIENT_STATE_KEY,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	type InteractionTransientState,
} from "../pipeline/types";
import { AnimationSystem } from "../animation/AnimationSystem";
import type { PhysicsSystem } from "../physics";
import type { SHCoefficients } from "../maths/types";
import type {
	BloomOptions,
	ClusteredLightingOptions,
	DOFOptions,
	MotionBlurOptions,
	SSAOOptions,
	SSGIOptions,
	SSROptions,
	TAAOptions,
	VolumetricOptions,
	FramePass,
	FrameContext,
} from "../pipeline/types";
import type {
	IRenderBackend,
	WarmupOptions,
	WarmupReport,
} from "./IRenderBackend";

export interface RendererEvents {
	tick: [{ now: number; deltaTime: number }];
	framestart: [{ now: number; deltaTime: number }];
	postanimation: [
		{
			now: number;
			deltaTime: number;
			scene: Scene;
			transient: Map<string, any>;
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
	transient: Map<string, any>;
}

export type FrameTransientContributor = (
	context: FrameTransientContributorContext
) => void;

export interface RendererFeatures {
	enableLighting: boolean;
	enableGamma: boolean;
	enableSH: boolean;
	enableShadows: boolean;
	enableReflection: boolean;
	enableSkybox: boolean;
	enableSSAO: boolean;
	enableSSGI: boolean;
	enableTAA: boolean;
	enableSSR: boolean;
	enableVolumetric: boolean;
	enableMotionBlur: boolean;
	enableDOF: boolean;
	enableBloom: boolean;
	enableFXAA: boolean;
	enableClusteredLighting: boolean;
	ssrOptions: SSROptions;
	volumetricOptions: VolumetricOptions;
	ssaoOptions: SSAOOptions;
	ssgiOptions: SSGIOptions;
	taaOptions: TAAOptions;
	bloomOptions: BloomOptions;
	motionBlurOptions: MotionBlurOptions;
	dofOptions: DOFOptions;
	clusteredLightingOptions: ClusteredLightingOptions;
	worldMatrix: Matrix4;
}

const MAX_WARNING_KEYS = 1024;

export class Renderer extends EventEmitter<RendererEvents> {
	public canvas: HTMLCanvasElement;
	public readonly backend: IRenderBackend;
	public readonly animationSystem: AnimationSystem;
	public readonly features: RendererFeatures;
	public shadowMaps: Map<ShadowCastingLight, ShadowMap>;
	public shCoeffs: SHCoefficients;
	public shAmbientCoeffs: SHCoefficients;
	public scene: Scene;
	public camera: Camera;
	public lastTime: number;
	public animationAutoRender: boolean;

	private _warnings: Set<string>;
	private _deviceScaleFactor: number;
	private _deltaTime: number;
	private _frameDirty: boolean;
	private _animationStage: AnimationSimulationStage;
	private _stageGraph: RendererStageGraph;
	private _physicsSystem: PhysicsSystem | null;
	private _frameTransientContributors: Set<FrameTransientContributor>;

	constructor(
		backend: IRenderBackend,
		canvas: HTMLCanvasElement,
		camera: Camera | null = null
	) {
		super();
		this.backend = backend;
		this.animationSystem = new AnimationSystem();
		this.canvas = canvas;
		this._warnings = new Set();
		this._deviceScaleFactor = window.devicePixelRatio || 1;
		this._deltaTime = 0;
		this._frameDirty = true;
		this.animationAutoRender = true;
		this._animationStage = new AnimationSimulationStage(this.animationSystem);
		this._stageGraph = new RendererStageGraph(createDefaultRendererStages());
		this._physicsSystem = null;
		this._frameTransientContributors = new Set();

		this.features = {
			enableLighting: true,
			enableGamma: true,
			enableSH: false,
			enableShadows: true,
			enableReflection: true,
			enableSkybox: true,
			enableSSAO: false,
			enableSSGI: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableMotionBlur: false,
			enableDOF: false,
			enableBloom: false,
			enableFXAA: false,
			enableClusteredLighting: false,
			ssrOptions: {},
			volumetricOptions: {},
			ssaoOptions: {},
			ssgiOptions: {},
			taaOptions: {},
			bloomOptions: {},
			motionBlurOptions: {},
			dofOptions: {},
			clusteredLightingOptions: {},
			worldMatrix: Matrix4.identity(),
		};

		this.shadowMaps = new Map();
		this.shCoeffs = SH.empty();
		this.shAmbientCoeffs = SH.empty();
		this.scene = new Scene();
		this.camera = camera || new Camera();

		// Only add to the default internal scene if the camera doesn't already have a parent.
		// This prevents the constructor from "stealing" a camera that the user has already
		// placed in their own scene graph.
		if (!this.camera.parent) {
			this.scene.add(this.camera);
		}

		this.lastTime = 0;

		if (!camera) {
			this.camera.position.set(0, 200, 200);
			this.camera.fov = 60;
		}

		this.camera.aspectRatio = this._getSafeAspectRatio(
			this.canvas.width,
			this.canvas.height
		);
		this.camera.updateMatrices();
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
		this._assertCameraInScene(this.scene, this.camera, "renderScene");
		this.camera.updateMatrices();

		const transient = new Map<string, any>();
		transient.set(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 0);
		transient.set(ANIMATION_SIM_DELTA_TIME_MS_KEY, 0);

		const resolved = resolveFeatureState(
			this.features,
			this.backend.capabilities,
			this.backend.type
		);
		for (const warning of resolved.warnings) {
			this.warnOnce(warning.key, warning.message);
		}
		if (this.features.enableSH) {
			this.updateSH();
		}
		const frame = PreparedSceneBuilder.build(this);
		const attachments = this.backend.getAttachments(
			this.canvas.width,
			this.canvas.height
		);
		const context: FrameContext = {
			camera: this.camera,
			attachments,
			features: resolved,
			shadowMaps: this.shadowMaps,
			scene: frame,
			shCoeffs: this.shCoeffs,
			shAmbientCoeffs: this.shAmbientCoeffs,
			worldMatrix: this.features.worldMatrix || Matrix4.identity(),
			transient,
		};
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
		this._frameDirty = true;
		this.scene.invalidate();

		this.camera.aspectRatio = this._getSafeAspectRatio(
			this.canvas.width,
			this.canvas.height
		);
		this.camera.updateMatrices();
	}

	public requestRender(): void {
		this._frameDirty = true;
		this.scene.invalidate();
	}

	public setScene(scene: Scene): void {
		this._assertCameraInScene(scene, this.camera, "setScene");
		this.scene = scene;
		if (this._physicsSystem) {
			this._physicsSystem.setEntityNodeResolver((entityId) => {
				return this.scene.ecs.getNodeByEntity(entityId);
			});
		}
		this.scene.invalidate();
	}

	public setCamera(camera: Camera): void {
		this._assertCameraInScene(this.scene, camera, "setCamera");
		this.camera = camera;
		this.scene.invalidate();
	}

	public setPhysicsSystem(physicsSystem: PhysicsSystem | null): void {
		this._physicsSystem = physicsSystem;
		if (physicsSystem) {
			physicsSystem.setEntityNodeResolver((entityId) => {
				return this.scene.ecs.getNodeByEntity(entityId);
			});
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

	public setStageGraph(stages: RendererStageDefinition[]): void {
		this._stageGraph.setStages(stages);
	}

	public registerStage(stage: RendererStageDefinition): void {
		this._stageGraph.registerStage(stage);
	}

	public get backendType(): IRenderBackend["type"] {
		return this.backend.type;
	}

	public warnOnce(key: string, message: string): void {
		if (this._warnings.has(key)) return;
		if (this._warnings.size >= MAX_WARNING_KEYS) {
			const oldestKey = this._warnings.values().next().value as
				| string
				| undefined;
			if (oldestKey !== undefined) {
				this._warnings.delete(oldestKey);
			}
		}
		this._warnings.add(key);
		console.warn(message);
	}

	public async renderScene(now: number): Promise<void> {
		this._deltaTime = now - (this.lastTime || now);
		this.lastTime = now;

		this.emit("tick", { now, deltaTime: this._deltaTime });
		this.emit("framestart", { now, deltaTime: this._deltaTime });

		const hasParticleSystems = this.scene.getParticleSystems().length > 0;
		const hasActiveAnimations = this.animationSystem.hasActiveActions();
		const hasDynamicTextureUpdates = Texture.updateDynamicTextures(now);
		if (hasDynamicTextureUpdates) {
			this._frameDirty = true;
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
		const transient = new Map<string, any>();
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
		let context: FrameContext | null = null;
		let frameStarted = false;
		let emittedPostAnimation = false;

		const stageOrder = this._stageGraph.getExecutionOrder(
			{
				hasActiveAnimations: hasActiveAnimations && this.animationAutoRender,
				hasParticleSystems,
			},
			(key, message) => this.warnOnce(key, message)
		);
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
						this.warnOnce(warning.key, warning.message);
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
					break;
				}
				case "prepared-scene-build": {
					if (this.features.enableSH) {
						this.updateSH();
					}
					frame = PreparedSceneBuilder.build(this);
					const attachments = this.backend.getAttachments(
						this.canvas.width,
						this.canvas.height
					);
					context = {
						camera: this.camera,
						attachments: attachments,
						features: resolved,
						shadowMaps: this.shadowMaps,
						scene: frame,
						shCoeffs: this.shCoeffs,
						shAmbientCoeffs: this.shAmbientCoeffs,
						worldMatrix: this.features.worldMatrix || Matrix4.identity(),
						transient,
					};
					await this.backend.beginFrame(context);
					frameStarted = true;
					break;
				}
				case "sync-out": {
					this.scene.syncECSToNode();
					break;
				}
				default: {
					if (!context || !frame) break;
					if (!this._isBackendPassStage(stage.id)) break;
					if (!this._shouldRunBackendPass(stage.id, frame, resolved, transient)) {
						const skippedPass = this._createBackendPass(stage.id);
						this.backend.skipPass?.(skippedPass);
						break;
					}

					const pass = this._createBackendPass(stage.id);
					if (pass.executor === "shared") {
						if (!this.backend.executeSharedPass) {
							this.warnOnce(
								`${this.backend.type}-shared-pass-${pass.stage}`,
								`${this.backend.type} backend declared shared pass "${pass.stage}" without executeSharedPass implementation`
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
				const probeSH = probe.sh;
				const intensity = light.intensity ?? 1;
				const probeScale =
					probe.prefilteredMap ? BAKED_LIGHT_PROBE_SH_SCALE : 1;
				const coeffCount = Math.min(ambientProbeSH.length, probeSH.length);
				for (let i = 0; i < coeffCount; i++) {
					ambientProbeSH[i].r += probeSH[i].r * intensity * probeScale;
					ambientProbeSH[i].g += probeSH[i].g * intensity * probeScale;
					ambientProbeSH[i].b += probeSH[i].b * intensity * probeScale;
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

		this.shAmbientCoeffs = ambientProbeSH.map((coefficient) => ({
			r: coefficient.r,
			g: coefficient.g,
			b: coefficient.b,
		})) as SHCoefficients;

		let totalSH: SHCoefficients = this.shAmbientCoeffs.map((coefficient) => ({
			r: coefficient.r,
			g: coefficient.g,
			b: coefficient.b,
		})) as SHCoefficients;

		for (const light of this.scene.getLights()) {
			if (light.type !== LightType.Directional) continue;

			const worldDirection = getDirectionalLightWorldDirection(light);
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

		this.shCoeffs = totalSH;
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
		transient: Map<string, any>
	): boolean {
		switch (stage) {
			case "particle-sim":
				return (frame.particleSystems?.length ?? 0) > 0;
			case "shadow":
				return features.enableShadows && frame.shadowCasterPackets.length > 0;
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
			case "motion-blur":
				return features.enableMotionBlur;
			case "dof":
				return features.enableDOF;
			case "bloom":
				return features.enableBloom;
			case "fxaa":
				return features.enableFXAA;
			case "interaction-outline": {
				const interaction = transient.get(
					INTERACTION_TRANSIENT_STATE_KEY
				) as InteractionTransientState | null | undefined;
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
	"motion-blur",
	"dof",
	"bloom",
	"fxaa",
	"interaction-outline",
	"gamma",
]);

function createDefaultRendererStages(): RendererStageDefinition[] {
	return [
		{ id: "feature-resolution", dependsOn: [] },
		{ id: "sync-in", dependsOn: ["feature-resolution"] },
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
		{ id: "prepared-scene-build", dependsOn: ["transform-update"] },
		{
			id: "particle-sim",
			dependsOn: ["prepared-scene-build"],
			enabled: (context) => context.hasParticleSystems,
		},
		{ id: "shadow", dependsOn: ["prepared-scene-build", "particle-sim"] },
		{ id: "reflection", dependsOn: ["prepared-scene-build"] },
		{ id: "main-opaque", dependsOn: ["reflection", "shadow"] },
		{ id: "main-transparent", dependsOn: ["main-opaque"] },
		{ id: "particles", dependsOn: ["main-transparent"] },
		{ id: "ssao", dependsOn: ["particles"] },
		{ id: "ssgi", dependsOn: ["ssao"] },
		{ id: "taa", dependsOn: ["ssgi", "ssao"] },
		{ id: "ssr", dependsOn: ["taa"] },
		{ id: "volumetric", dependsOn: ["ssr"] },
		{ id: "motion-blur", dependsOn: ["volumetric"] },
		{ id: "dof", dependsOn: ["motion-blur"] },
		{ id: "bloom", dependsOn: ["dof"] },
		{ id: "fxaa", dependsOn: ["bloom"] },
		{ id: "interaction-outline", dependsOn: ["fxaa"] },
		{ id: "gamma", dependsOn: ["interaction-outline"] },
		{ id: "sync-out", dependsOn: ["gamma"] },
	];
}
