import { Camera, CameraType } from "../cameras/Camera";
import {
	createLightContribution,
	evaluateLightContribution,
	LightType,
	type ShadowCastingLight,
} from "../lights";
import { Matrix4 } from "../maths/Matrix4";
import { SH } from "../maths/SH";
import { Vector3 } from "../maths/Vector3";
import { sRGBToLinear } from "../maths/Common";
import { ShadowMap } from "../utils/ShadowMapping";
import { LightingConstants } from "./constants";
import { EventEmitter } from "./EventEmitter";
import { Scene } from "./Scene";
import { ShadowRenderer } from "./ShadowRenderer";
import { Rasterizer, type RasterizerLike } from "./software/Rasterizer";
import { ReflectionRenderer } from "./software/ReflectionRenderer";
import { resolveFeatureState } from "./pipeline/FeatureResolver";
import { FramePlanner } from "./pipeline/FramePlanner";
import { PreparedSceneBuilder } from "./pipeline/PreparedSceneBuilder";
import type { SHCoefficients } from "../maths/types";
import type {
	SSAOOptions,
	VolumetricOptions,
	FrameContext,
	FramePassStage,
	PreparedScene,
	ResolvedFeatureState,
} from "./pipeline/types";
import {
	PostProcessor,
	type PostProcessorLike,
} from "./software/PostProcessor";
import type { IRenderBackend } from "./backend/IRenderBackend";

export interface RendererEvents {
	tick: [{ now: number; deltaTime: number }];
	framestart: [{ now: number; deltaTime: number }];
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
	enableVolumetric: boolean;
	enableFXAA: boolean;
	volumetricOptions: VolumetricOptions;
	ssaoOptions: SSAOOptions;
	worldMatrix: Matrix4;
}

export class Renderer extends EventEmitter<RendererEvents> {
	public canvas: HTMLCanvasElement;
	public readonly backend: IRenderBackend;
	public readonly features: RendererFeatures;
	public shadowMaps: Map<ShadowCastingLight, ShadowMap>;
	public shCoeffs: SHCoefficients;
	public shAmbientCoeffs: SHCoefficients;
	public scene: Scene;
	public camera: Camera;
	public lastTime: number;

	private _warnings: Set<string>;
	private _deviceScaleFactor: number;
	private _deltaTime: number;
	private _frameDirty: boolean;
	private _shadowRenderer: ShadowRenderer;
	private _postProcessor: PostProcessor;

	constructor(
		backend: IRenderBackend,
		canvas: HTMLCanvasElement,
		camera: Camera | null = null
	) {
		super();
		this.backend = backend;
		this.canvas = canvas;
		this._warnings = new Set();
		this._deviceScaleFactor = window.devicePixelRatio || 1;
		this._deltaTime = 0;
		this._frameDirty = true;

		this.features = {
			enableLighting: true,
			enableGamma: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: true,
			enableSkybox: true,
			enableSSAO: false,
			enableVolumetric: false,
			enableFXAA: false,
			volumetricOptions: {},
			ssaoOptions: {},
			worldMatrix: Matrix4.identity(),
		};

		this.shadowMaps = new Map();
		this.shCoeffs = SH.empty();
		this.shAmbientCoeffs = SH.empty();
		this.scene = new Scene();
		this.camera = camera || new Camera();
		this._shadowRenderer = new ShadowRenderer(this, new Rasterizer());
		this._postProcessor = new PostProcessor(this);
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

		const backendWithRenderer = this.backend as IRenderBackend & {
			setRenderer?: (renderer: Renderer) => void;
		};
		backendWithRenderer.setRenderer?.(this);
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

		if (!this._frameDirty && this.backend.type !== "software") {
			this.emit("frameend", { now, deltaTime: this._deltaTime });
			requestAnimationFrame((time) => this.renderScene(time));
			return;
		}

		this._frameDirty = false;
		this.camera.updateMatrices();
		this._updateLightWorldMatrices();

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
			transient: new Map(),
		};

		const framePlan = FramePlanner.build(frame, resolved);

		// Execute shared passes (CPU-based) before beginFrame so their results (like shadow maps)
		// are available for backend resource preparation.
		for (const pass of framePlan) {
			if (pass.enabled && pass.executor === "shared") {
				await this._executeSharedPass(pass.stage, context);
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

	public get postProcessor(): PostProcessorLike {
		return this._postProcessor;
	}

	public updateSH(): void {
		let ambientProbeSH: SHCoefficients = SH.empty();
		let ambientR = 0;
		let ambientG = 0;
		let ambientB = 0;
		let hasAmbient = false;

		const featureWorldMatrix = (
			this as Renderer & {
				params?: { worldMatrix?: Matrix4 };
			}
		).features?.worldMatrix;
		const legacyWorldMatrix = (
			this as Renderer & {
				params?: { worldMatrix?: Matrix4 };
			}
		).params?.worldMatrix;
		const worldMatrix =
			featureWorldMatrix || legacyWorldMatrix || Matrix4.identity();

		for (const light of this.scene.lights) {
			light.updateWorldMatrix(worldMatrix);

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
				const probeSH = (light as any).sh;
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
		const lightContribution = createLightContribution();

		for (const light of this.scene.lights) {
			if (light.type !== LightType.Directional) continue;

			const contribution = evaluateLightContribution(
				light,
				{ position: { x: 0, y: 0, z: 0 } },
				lightContribution
			);
			if (!contribution?.direction) continue;

			const direction = Vector3.normalize(contribution.direction);
			const intensity = contribution.intensity ?? 1;
			const lightSH = SH.projectDirectionalLight(direction, {
				r: contribution.color.r * intensity,
				g: contribution.color.g * intensity,
				b: contribution.color.b * intensity,
			});
			totalSH = SH.addCoeffs(totalSH, lightSH);
		}

		this.shCoeffs = totalSH;
	}

	private async _executeSharedPass(
		stage: FramePassStage,
		context: FrameContext
	): Promise<void> {
		if (stage !== "shadow") return;
		this._shadowRenderer.render(context);
	}

	private _getSafeAspectRatio(width: number, height: number): number {
		return Math.max(width, 1) / Math.max(height, 1);
	}

	private _updateLightWorldMatrices(): void {
		const worldMatrix = this.features.worldMatrix || Matrix4.identity();
		for (const light of this.scene.lights) {
			light.updateWorldMatrix(worldMatrix);
		}
	}
}
