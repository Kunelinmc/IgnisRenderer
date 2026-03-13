import { Camera } from "../cameras/Camera";
import { LightType, type LightProbe, type ShadowCastingLight } from "../lights";
import { Matrix4 } from "../maths/Matrix4";
import { SH } from "../maths/SH";
import { Vector3 } from "../maths/Vector3";
import { sRGBToLinear } from "../maths/Common";
import { ShadowMap } from "../lights/ShadowMapping";
import { LightingConstants } from "../core/constants";
import { EventEmitter } from "../core/EventEmitter";
import { Scene } from "../core/Scene";
import { Texture } from "../core/Texture";
import { resolveFeatureState } from "../pipeline/FeatureResolver";
import { FramePlanner } from "../pipeline/FramePlanner";
import { AnimationSimulationStage } from "../pipeline/AnimationSimulationStage";
import { PreparedSceneBuilder } from "../pipeline/PreparedSceneBuilder";
import { getDirectionalLightWorldDirection } from "../pipeline/LightTransforms";
import {
	ANIMATION_SIM_DELTA_TIME_MS_KEY,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
} from "../pipeline/types";
import { AnimationSystem } from "../animation/AnimationSystem";
import type { SHCoefficients } from "../maths/types";
import type {
	SSAOOptions,
	SSROptions,
	TAAOptions,
	VolumetricOptions,
	FrameContext,
} from "../pipeline/types";
import type { IRenderBackend } from "./IRenderBackend";

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

export interface RendererFeatures {
	enableLighting: boolean;
	enableGamma: boolean;
	enableSH: boolean;
	enableShadows: boolean;
	enableReflection: boolean;
	enableSkybox: boolean;
	enableSSAO: boolean;
	enableTAA: boolean;
	enableSSR: boolean;
	enableVolumetric: boolean;
	enableFXAA: boolean;
	ssrOptions: SSROptions;
	volumetricOptions: VolumetricOptions;
	ssaoOptions: SSAOOptions;
	taaOptions: TAAOptions;
	worldMatrix: Matrix4;
}

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

		this.features = {
			enableLighting: true,
			enableGamma: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: true,
			enableSkybox: true,
			enableSSAO: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableFXAA: false,
			ssrOptions: {},
			volumetricOptions: {},
			ssaoOptions: {},
			taaOptions: {},
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
		this.scene.invalidate();
	}

	public setCamera(camera: Camera): void {
		this._assertCameraInScene(this.scene, camera, "setCamera");
		this.camera = camera;
		this.scene.invalidate();
	}

	public get backendType(): IRenderBackend["type"] {
		return this.backend.type;
	}

	public warnOnce(key: string, message: string): void {
		if (this._warnings.has(key)) return;
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
		this._animationStage.execute(
			{
				scene: this.scene,
				transient,
			},
			this._deltaTime
		);
		this.emit("postanimation", {
			now,
			deltaTime: this._deltaTime,
			scene: this.scene,
			transient,
		});
		this.scene.updateWorldMatrices();
		this._assertCameraInScene(this.scene, this.camera, "renderScene");
		this.camera.updateMatrices();

		if (this.features.enableSH) {
			this.updateSH();
		}

		const resolved = resolveFeatureState(
			this.features,
			this.backend.capabilities,
			this.backend.type
		);
		for (const warning of resolved.warnings) {
			this.warnOnce(warning.key, warning.message);
		}

		const frame = PreparedSceneBuilder.build(this);
		const attachments = this.backend.getAttachments(
			this.canvas.width,
			this.canvas.height
		);
		const context: FrameContext = {
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

		const framePlan = FramePlanner.build(
			frame,
			resolved,
			this.backend.passExecutors
		);

		// Execute shared passes before beginFrame so their results are available
		// for backend resource preparation.
		for (const pass of framePlan) {
			if (pass.enabled && pass.executor === "shared") {
				if (pass.stage === "animation-sim") {
					continue;
				}
				if (!this.backend.executeSharedPass) {
					this.warnOnce(
						`${this.backend.type}-shared-pass-${pass.stage}`,
						`${this.backend.type} backend declared shared pass "${pass.stage}" without executeSharedPass implementation`
					);
					continue;
				}
				await this.backend.executeSharedPass(pass, context);
			}
		}

		await this.backend.beginFrame(context);
		for (const pass of framePlan) {
			if (!pass.enabled || pass.executor === "shared") continue;

			await this.backend.executePass(pass, context);
		}
		await this.backend.endFrame();

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
					probe.prefilteredMap ?
						LightingConstants.BAKED_LIGHT_PROBE_SH_SCALE
					:	1;
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
			const fallbackLinear =
				LightingConstants.PBR_AMBIENT_FALLBACK_LINEAR * 255;
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
