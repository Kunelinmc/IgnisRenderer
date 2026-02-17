import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import { SH } from "../maths/SH";
import { Camera } from "../cameras/Camera";
import { Scene } from "./Scene";
import { EventEmitter } from "./EventEmitter";
import { ShadowMap } from "../utils/ShadowMapping";
import { Projector } from "./Projector";
import { ShadowRenderer } from "./ShadowRenderer";
import { ReflectionRenderer } from "./ReflectionRenderer";
import { Rasterizer } from "./Rasterizer";
import { PostProcessor } from "./PostProcessor";
import { PostProcessConstants } from "./Constants";
import { LightType, type ShadowCastingLight } from "../lights";
import type { SHCoefficients } from "../maths/types";
import type { PostProcessorLike, VolumetricOptions } from "./PostProcessor";
import type { RasterizerLike } from "./Rasterizer";
import type { IModel, ProjectedFace } from "./types";

/**
 * CORE RENDERING CONVENTIONS:
 * - Coordinate System: Right-Handed (X: Right, Y: Up, Z: Towards Viewer)
 * - World Space: Standard Cartesian units
 * - View Space: Eye at origin, -Z is forward
 * - Depth Buffer: Stores linear camera-space depth (positive distance from camera plane)
 * - Screen Space: (0,0) at top-left, (W,H) at bottom-right, pixel centers at +0.5
 */

export class Renderer extends EventEmitter {
	public canvas: HTMLCanvasElement;
	private _ctx: CanvasRenderingContext2D;

	private _sf: number;
	private _deltaTime: number;

	public lastTime: number;

	/** Depth buffer storing linear camera-space distance (near to far) */
	public depthBuffer: Float32Array | null;

	private _offscreenCanvas: HTMLCanvasElement;
	private _offscreenCtx: CanvasRenderingContext2D;

	public params: {
		offset: { x: number; y: number };
		cacheInvalid: boolean;
		enableFXAA: boolean;
		enableLighting: boolean;
		enableSH: boolean;
		enableShadows: boolean;
		enableVolumetric: boolean;
		volumetricOptions: VolumetricOptions;
		enableGamma: boolean;
		enableReflection: boolean;
		worldMatrix?: Matrix4;
	};

	public shadowMaps: Map<ShadowCastingLight, ShadowMap>;
	public shCoeffs: SHCoefficients;
	public shAmbientCoeffs: SHCoefficients;

	public scene: Scene;
	public camera: Camera;

	public rasterizer: RasterizerLike;
	public reflectionRenderer: ReflectionRenderer;

	private _shadowRenderer: ShadowRenderer;
	private _postProcessor: PostProcessorLike;

	private _projectedModels: Map<IModel, ProjectedFace[]> = new Map();

	constructor(canvas: HTMLCanvasElement, camera: Camera | null = null) {
		super();
		this.canvas = canvas;
		this._ctx = canvas.getContext("2d")!;
		this._sf = window.devicePixelRatio || 1;

		this.lastTime = 0;
		this._deltaTime = 0;

		this.depthBuffer = null;
		this._offscreenCanvas = document.createElement("canvas");
		this._offscreenCtx = this._offscreenCanvas.getContext("2d", {
			willReadFrequently: true,
		})!;

		this.params = {
			offset: { x: 0, y: 0 },
			cacheInvalid: true,
			enableFXAA: false,
			enableLighting: true,
			enableSH: false,
			enableShadows: false,
			enableVolumetric: false,
			volumetricOptions: {},
			enableGamma: true,
			enableReflection: true,
			worldMatrix: Matrix4.identity(),
		};

		this.shadowMaps = new Map();

		this.shCoeffs = SH.empty();
		this.shAmbientCoeffs = SH.empty();

		this.scene = new Scene();
		this.camera = camera || new Camera();

		// Initial camera setup if not provided
		if (!camera) {
			this.camera.position.set(0, 200, 200);
			this.camera.fov = 60;
		}

		this.camera.aspectRatio = this.canvas.width / this.canvas.height;
		this.camera.updateMatrices();

		this.rasterizer = new Rasterizer(this);
		this._shadowRenderer = new ShadowRenderer(this);
		this.reflectionRenderer = new ReflectionRenderer(this);
		this._postProcessor = new PostProcessor(this);
	}

	public init(): void {
		this.resizeCanvas();
		requestAnimationFrame((time) => this.renderScene(time));
	}

	/**
	 * Picks a model at the given screen coordinates.
	 * @param {number} x - The x-coordinate of the screen.
	 * @param {number} y - The y-coordinate of the screen.
	 * @returns {IModel | null} The model at the given screen coordinates, or null if no model is found.
	 */
	public pick(x: number, y: number): IModel | null {
		const rect = this.canvas.getBoundingClientRect();
		const canvasX = (x - rect.left) * this._sf;
		const canvasY = (y - rect.top) * this._sf;

		let nearestModel: IModel | null = null;
		let minDepth = Infinity;

		for (const model of this.scene.models) {
			const faces = this._projectedModels.get(model);
			if (!faces) continue;

			const face = Projector.getFaceAtPoint(faces, canvasX, canvasY);
			if (face) {
				const depth = face.depthInfo.avg;
				if (depth < minDepth) {
					minDepth = depth;
					nearestModel = model;
				}
			}
		}

		return nearestModel;
	}

	/**
	 * Resizes the canvas and updates the renderer's parameters.
	 */
	public resizeCanvas(): void {
		const rect = this.canvas.getBoundingClientRect();
		this._sf = window.devicePixelRatio || 1;
		this.canvas.width = rect.width * this._sf;
		this.canvas.height = rect.height * this._sf;
		this._offscreenCanvas.width = this.canvas.width;
		this._offscreenCanvas.height = this.canvas.height;
		this._clearDepthBuffer();
		this.params.cacheInvalid = true;

		if (this.camera) {
			this.camera.aspectRatio = this.canvas.width / this.canvas.height;
			this.camera.updateMatrices();
		}
	}

	private _clearDepthBuffer(): void {
		const size = this.canvas.width * this.canvas.height;
		if (!this.depthBuffer || this.depthBuffer.length !== size) {
			this.depthBuffer = new Float32Array(size);
		}
		this.depthBuffer.fill(Infinity);
	}

	/**
	 * Requests a render of the scene.
	 */
	public requestRender(): void {
		this.params.cacheInvalid = true;
	}

	public renderScene(now: number): void {
		this._deltaTime = now - (this.lastTime || now);
		this.lastTime = now;

		this.emit("tick", { now, deltaTime: this._deltaTime });
		this.emit("framestart", { now, deltaTime: this._deltaTime });

		if (!this.params.cacheInvalid) {
			this.emit("frameend", {
				now,
				deltaTime: this._deltaTime,
			});
			requestAnimationFrame((time) => this.renderScene(time));
			return;
		}

		this.params.cacheInvalid = false;

		// Keep camera matrices current for every rendering pass (shadow/reflection/main).
		this.camera.updateMatrices();

		// Update light matrices once per frame
		if (this.scene.lights) {
			const worldMat = this.params.worldMatrix || Matrix4.identity();
			for (const light of this.scene.lights) {
				light.updateWorldMatrix(worldMat);
			}
		}

		this._shadowRenderer.render();
		if (this.params.enableReflection) {
			this.reflectionRenderer.render();
		}

		this._offscreenCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
		this._clearDepthBuffer();

		const imageData = this._offscreenCtx.getImageData(
			0,
			0,
			this.canvas.width,
			this.canvas.height
		);
		const pixels = imageData.data;

		this._projectedModels.clear();
		for (const model of this.scene.models) {
			const faces = Projector.projectModel(model, this);
			this._projectedModels.set(model, faces);
		}

		const opaqueFaces: ProjectedFace[] = [];
		const transparentFaces: ProjectedFace[] = [];
		for (const model of this.scene.models) {
			const faces = this._projectedModels.get(model) || [];
			for (let i = 0, len = faces.length; i < len; i++) {
				const face = faces[i];
				const alpha = face.color?.a ?? face.material?.opacity ?? 1;
				const explicitAlphaMode = face.material?.alphaMode;
				const alphaMode = explicitAlphaMode || "OPAQUE";
				if (
					alphaMode === "BLEND" ||
					(explicitAlphaMode === undefined && alpha < 0.99)
				) {
					transparentFaces.push(face);
				} else {
					opaqueFaces.push(face);
				}
			}
		}

		for (const face of opaqueFaces) {
			const projected = face.projected;
			for (let i = 1; i < projected.length - 1; i++) {
				this.rasterizer.drawTriangle(
					[projected[0], projected[i], projected[i + 1]],
					face,
					pixels,
					false
				);
			}
		}

		transparentFaces.sort((a, b) => b.depthInfo.avg - a.depthInfo.avg);

		for (const face of transparentFaces) {
			const projected = face.projected;
			for (let i = 1; i < projected.length - 1; i++) {
				this.rasterizer.drawTriangle(
					[projected[0], projected[i], projected[i + 1]],
					face,
					pixels,
					true
				);
			}
		}

		if (this.params.enableFXAA) {
			this._postProcessor.applyFXAA(
				this._offscreenCtx,
				this._offscreenCanvas,
				pixels
			);
		}

		if (this.params.enableVolumetric) {
			this._postProcessor.applyVolumetricLight(
				this._offscreenCtx,
				this._offscreenCanvas,
				pixels,
				this.depthBuffer,
				this.params.volumetricOptions
			);
		}

		if (this.params.enableGamma) {
			this._postProcessor.applyGamma(
				this._offscreenCtx,
				this._offscreenCanvas,
				PostProcessConstants.DEFAULT_GAMMA,
				pixels
			);
		}

		this._offscreenCtx.putImageData(imageData, 0, 0);
		this._ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
		this._ctx.drawImage(this._offscreenCanvas, 0, 0);

		this.emit("frameend", {
			now,
			deltaTime: this._deltaTime,
		});

		requestAnimationFrame((time) => this.renderScene(time));
	}

	/**
	 * Updates the Spherical Harmonics (SH) coefficients based on the current lights in the scene.
	 * IMPORTANT: This must be called AFTER all lights are added to the scene (scene.addLight)
	 * and whenever dynamic lights change their intensity or direction.
	 */
	public updateSH(): void {
		let ambientProbeSH: SHCoefficients = SH.empty();
		let ambientR = 0,
			ambientG = 0,
			ambientB = 0;
		let hasAmbient = false;
		const gamma = PostProcessConstants.DEFAULT_GAMMA;

		const worldMatrix = this.params.worldMatrix || Matrix4.identity();

		if (this.scene.lights) {
			for (const light of this.scene.lights) {
				// Update light's world matrix once per frame/update
				light.updateWorldMatrix(worldMatrix);

				if (light.type === LightType.Ambient) {
					const color = light.color || { r: 255, g: 255, b: 255 };
					const intensity = light.intensity ?? 1;
					ambientR += Math.pow(color.r / 255, gamma) * 255 * intensity;
					ambientG += Math.pow(color.g / 255, gamma) * 255 * intensity;
					ambientB += Math.pow(color.b / 255, gamma) * 255 * intensity;
					hasAmbient = true;
				} else if (light.type === LightType.LightProbe) {
					const probeSH = light.sh;
					const intensity = light.intensity ?? 1;
					for (let i = 0; i < 9; i++) {
						ambientProbeSH[i].r += probeSH[i].r * intensity;
						ambientProbeSH[i].g += probeSH[i].g * intensity;
						ambientProbeSH[i].b += probeSH[i].b * intensity;
					}
				}
			}
		}

		if (
			!hasAmbient &&
			ambientProbeSH[0].r === 0 &&
			ambientProbeSH[0].g === 0 &&
			ambientProbeSH[0].b === 0
		) {
			const fallbackSrgb = 51 / 255;
			const fallbackLinear = Math.pow(fallbackSrgb, gamma) * 255;
			ambientR = fallbackLinear;
			ambientG = fallbackLinear;
			ambientB = fallbackLinear;
		}

		ambientProbeSH[0].r += ambientR / Math.PI / 0.282095;
		ambientProbeSH[0].g += ambientG / Math.PI / 0.282095;
		ambientProbeSH[0].b += ambientB / Math.PI / 0.282095;

		// Keep an ambient/probe-only SH set for PBR ambient IBL
		this.shAmbientCoeffs = ambientProbeSH.map((c) => ({
			r: c.r,
			g: c.g,
			b: c.b,
		})) as SHCoefficients;

		// Full SH (includes directional), kept for compatibility/debug tooling.
		let totalSH: SHCoefficients = this.shAmbientCoeffs.map((c) => ({
			r: c.r,
			g: c.g,
			b: c.b,
		})) as SHCoefficients;

		if (this.scene.lights) {
			for (const light of this.scene.lights) {
				if (light.type === LightType.Directional) {
					const contrib = light.computeContribution({ x: 0, y: 0, z: 0 });
					if (contrib) {
						const dir = Vector3.normalize(contrib.direction!);
						const lightSH = SH.projectDirectionalLight(dir, contrib.color);
						totalSH = SH.addCoeffs(totalSH, lightSH);
					}
				}
			}
		}
		this.shCoeffs = totalSH;
	}
}
