import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import { SH } from "../maths/SH";
import { Camera, CameraType } from "../cameras/Camera";
import { Scene } from "./Scene";
import { EventEmitter } from "./EventEmitter";
import { ShadowMap } from "../utils/ShadowMapping";
import { Projector } from "./software/Projector";
import { ShadowRenderer } from "./ShadowRenderer";
import { ReflectionRenderer } from "./software/ReflectionRenderer";
import { PostProcessor } from "./software/PostProcessor";
import { Rasterizer, type RasterizerLike } from "./software/Rasterizer";
import { LightingConstants, PostProcessConstants } from "./Constants";
import { sRGBToLinear } from "../maths/Common";
import { LightType, type ShadowCastingLight } from "../lights";
import type { SHCoefficients } from "../maths/types";
import type {
	PostProcessorLike,
	VolumetricOptions,
	SSAOOptions,
} from "./software/PostProcessor";
import type { IModel, ProjectedFace } from "./types";

// RAL Imports
import { IDevice } from "./ral/IDevice";
import { ResourceManager } from "./bridge/ResourceManager";
import { resolveWebGPUFeatureState } from "./bridge/webgpuUtils";

/**
 * CORE RENDERING CONVENTIONS:
 * - Backend-Agnostic Orchestrator
 */

export interface RendererEvents {
	tick: [{ now: number; deltaTime: number }];
	framestart: [{ now: number; deltaTime: number }];
	frameend: [{ now: number; deltaTime: number }];
	[key: string]: any[];
}

export class Renderer extends EventEmitter<RendererEvents> {
	public canvas: HTMLCanvasElement;
	public pixels: Uint8ClampedArray;
	public depthBuffer: Float32Array;
	public normalBuffer: Float32Array | null;
	private _ctx: CanvasRenderingContext2D | null;
	private _device: IDevice;
	private _resourceManager: ResourceManager;
	private _warnings: Set<string>;

	private _sf: number;
	private _deltaTime: number;
	public lastTime: number;

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
		enableSkybox: boolean;
		enableSSAO: boolean;
		ssaoOptions: SSAOOptions;
		worldMatrix?: Matrix4;
	};

	public shadowMaps: Map<ShadowCastingLight, ShadowMap>;
	public shCoeffs: SHCoefficients;
	public shAmbientCoeffs: SHCoefficients;

	public scene: Scene = new Scene();
	public camera: Camera;
	public rasterizer: RasterizerLike;

	private _shadowRenderer: ShadowRenderer;
	private _postProcessor: PostProcessorLike;
	public reflectionRenderer: ReflectionRenderer;

	constructor(
		device: IDevice,
		canvas: HTMLCanvasElement,
		camera: Camera | null = null
	) {
		super();
		this._device = device;
		this.canvas = canvas;
		this._ctx =
			this._device.type === "software" ? canvas.getContext("2d") : null;
		this._warnings = new Set();
		this.pixels = new Uint8ClampedArray(0);
		this.depthBuffer = new Float32Array(0);

		if (this._device.type === "software" && (this._device as any).setRenderer) {
			(this._device as any).setRenderer(this);
		}

		this._resourceManager = new ResourceManager(device);
		this._sf = window.devicePixelRatio || 1;

		this.lastTime = 0;
		this._deltaTime = 0;
		this.normalBuffer = null;

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
			enableSkybox: true,
			enableSSAO: false,
			ssaoOptions: {},
			worldMatrix: Matrix4.identity(),
		};

		this.shadowMaps = new Map();
		this.shCoeffs = SH.empty();
		this.shAmbientCoeffs = SH.empty();

		this.camera = camera || new Camera();
		if (!camera) {
			this.camera.position.set(0, 200, 200);
			this.camera.fov = 60;
		}

		this.camera.aspectRatio = this._getSafeAspectRatio(
			this.canvas.width,
			this.canvas.height
		);
		this.camera.updateMatrices();

		this.rasterizer =
			((this._device as any)._rasterizer as RasterizerLike) ??
			new Rasterizer(this as any);
		this._shadowRenderer = new ShadowRenderer(this as any);
		this.reflectionRenderer = new ReflectionRenderer(this as any);
		this._postProcessor = new PostProcessor(this as any);
	}

	public async init(): Promise<void> {
		await this._device.init();
		await this._resourceManager.init();
		this.resizeCanvas();
		requestAnimationFrame((time) => this.renderScene(time));
	}

	public resizeCanvas(): void {
		const rect = this.canvas.getBoundingClientRect();
		this._sf = window.devicePixelRatio || 1;
		this.canvas.width = rect.width * this._sf;
		this.canvas.height = rect.height * this._sf;

		if (this._device.type === "software") {
			this._ctx = this._ctx ?? this.canvas.getContext("2d");
			this.pixels = new Uint8ClampedArray(
				this.canvas.width * this.canvas.height * 4
			);
			this.depthBuffer = new Float32Array(
				this.canvas.width * this.canvas.height
			);
			this.depthBuffer.fill(Infinity);
			this.normalBuffer = new Float32Array(
				this.canvas.width * this.canvas.height * 3
			);
		}

		this._device.resize(this.canvas.width, this.canvas.height);
		this.params.cacheInvalid = true;

		if (this.camera) {
			this.camera.aspectRatio = this._getSafeAspectRatio(
				this.canvas.width,
				this.canvas.height
			);
			this.camera.updateMatrices();
		}
	}

	private _getSafeAspectRatio(width: number, height: number): number {
		return Math.max(width, 1) / Math.max(height, 1);
	}

	public requestRender(): void {
		this.params.cacheInvalid = true;
	}

	public get backendType(): IDevice["type"] {
		return this._device.type;
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

		if (!this.params.cacheInvalid && this._device.type !== "software") {
			this.emit("frameend", { now, deltaTime: this._deltaTime });
			requestAnimationFrame((time) => this.renderScene(time));
			return;
		}

		this.params.cacheInvalid = false;
		this.camera.updateMatrices();

		const worldMatrix = this.params.worldMatrix || Matrix4.identity();
		for (const light of this.scene.lights) {
			light.updateWorldMatrix(worldMatrix);
		}

		if (this._device.type === "software") {
			await this._renderSoftwareScene();
			this.emit("frameend", { now, deltaTime: this._deltaTime });
			requestAnimationFrame((time) => this.renderScene(time));
			return;
		}

		await this._renderWebGPUScene();

		this.emit("frameend", { now, deltaTime: this._deltaTime });
		requestAnimationFrame((time) => this.renderScene(time));
	}

	private async _renderWebGPUScene(): Promise<void> {
		const featureState = resolveWebGPUFeatureState(this.params);

		if (featureState.enableShadows) {
			this._shadowRenderer.render();
		}

		this._resourceManager.prepareWebGPUFrame(this, featureState);

		const encoder = this._device.createCommandEncoder();
		encoder.beginRenderPass({
			colorAttachments: [
				{
					view: null,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: null,
				depthClearValue: 1,
				depthLoadOp: "clear",
				depthStoreOp: "store",
			},
		});

		for (const model of this.scene.models) {
			const resources = await this._resourceManager.getWebGPUDrawResources(
				model,
				this
			);
			if (!resources) continue;

			encoder.setPipeline(resources.pipeline);
			encoder.setBindingGroup(0, resources.frameBinding);
			encoder.setBindingGroup(1, resources.modelBinding);
			encoder.setVertexBuffer(0, resources.vertexBuffer);
			encoder.setIndexBuffer(resources.indexBuffer, "uint32");
			encoder.drawIndexed(resources.indexCount);
		}

		encoder.endRenderPass();
		this._device.submit([encoder.finish()]);
	}

	private async _renderSoftwareScene(): Promise<void> {
		if (!this._ctx) {
			throw new Error("Software renderer requires a 2D canvas context.");
		}

		if (!this.pixels || !this.depthBuffer) return;

		if (this.params.enableShadows) {
			this._shadowRenderer.render();
		}
		if (this.params.enableReflection) {
			this.reflectionRenderer.render();
		}

		this._clearSoftwareBuffers();

		if (this.params.enableSkybox && this.scene.skybox) {
			this.renderSkybox(this.pixels);
		}

		const opaqueFaces: ProjectedFace[] = [];
		const transparentFaces: ProjectedFace[] = [];

		for (const model of this.scene.models) {
			const faces = Projector.projectModel(model, this);
			for (const face of faces) {
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
					this.pixels,
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
					this.pixels,
					true
				);
			}
		}

		if (this.params.enableSSAO) {
			this._postProcessor.applySSAO(
				this.pixels,
				this.depthBuffer,
				this.normalBuffer,
				this.params.ssaoOptions
			);
		}

		if (this.params.enableGamma) {
			this._postProcessor.applyGamma(
				this._ctx,
				this.canvas,
				PostProcessConstants.DEFAULT_GAMMA,
				this.pixels
			);
		}

		const imageData = this._ctx.createImageData(
			this.canvas.width,
			this.canvas.height
		);
		imageData.data.set(this.pixels);
		this._ctx.putImageData(imageData, 0, 0);
	}

	private _clearSoftwareBuffers(): void {
		const size = this.canvas.width * this.canvas.height;
		for (let i = 0; i < size; i++) {
			const idx = i << 2;
			this.pixels[idx] = 0;
			this.pixels[idx + 1] = 0;
			this.pixels[idx + 2] = 0;
			this.pixels[idx + 3] = 255;
		}

		this.depthBuffer.fill(Infinity);
		this.normalBuffer?.fill(0);
	}

	public renderSkybox(
		pixels: Uint8ClampedArray,
		width?: number,
		height?: number
	): void {
		const skybox = this.scene.skybox;
		if (!skybox) return;

		const w = width ?? this.canvas.width;
		const h = height ?? this.canvas.height;
		const camera = this.camera;
		const view = camera.viewMatrix.elements;
		const right = { x: view[0][0], y: view[0][1], z: view[0][2] };
		const up = { x: view[1][0], y: view[1][1], z: view[1][2] };
		const backward = { x: view[2][0], y: view[2][1], z: view[2][2] };
		const isOrthographic = camera.type === CameraType.Orthographic;
		const fovRad = (camera.fov * Math.PI) / 180;
		const tanHalfFov = isOrthographic ? 0 : Math.tan(fovRad * 0.5);
		const aspect =
			width && height ? width / height : camera.aspectRatio || w / h;

		for (let y = 0; y < h; y++) {
			const ndcY = 1 - ((y + 0.5) / h) * 2;
			const cy = ndcY * tanHalfFov;
			const rowBase = y * w * 4;

			for (let x = 0; x < w; x++) {
				const ndcX = ((x + 0.5) / w) * 2 - 1;
				const cx = ndcX * aspect * tanHalfFov;
				const dirX = right.x * cx + up.x * cy - backward.x;
				const dirY = right.y * cx + up.y * cy - backward.y;
				const dirZ = right.z * cx + up.z * cy - backward.z;
				const invLen = 1 / Math.sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
				const dx = dirX * invLen;
				const dy = dirY * invLen;
				const dz = dirZ * invLen;
				const phi = Math.atan2(dx, dz);
				const theta = Math.acos(Math.max(-1, Math.min(1, dy)));
				const u = (phi + Math.PI) / (2 * Math.PI);
				const v = theta / Math.PI;
				const color = skybox.sample(u, v);
				const idx = rowBase + x * 4;
				pixels[idx] = color.r;
				pixels[idx + 1] = color.g;
				pixels[idx + 2] = color.b;
				pixels[idx + 3] = 255;
			}
		}
	}

	// ... Other methods (updateSH, etc.) kept for metadata logic
	public updateSH(): void {
		let ambientProbeSH: SHCoefficients = SH.empty();
		let ambientR = 0,
			ambientG = 0,
			ambientB = 0;
		let hasAmbient = false;

		const worldMatrix = this.params.worldMatrix || Matrix4.identity();

		if (this.scene.lights) {
			for (const light of this.scene.lights) {
				light.updateWorldMatrix(worldMatrix);

				if (light.type === LightType.Ambient) {
					const color = light.color || { r: 255, g: 255, b: 255 };
					const intensity = light.intensity ?? 1;
					ambientR += sRGBToLinear(color.r / 255) * 255 * intensity;
					ambientG += sRGBToLinear(color.g / 255) * 255 * intensity;
					ambientB += sRGBToLinear(color.b / 255) * 255 * intensity;
					hasAmbient = true;
				} else if (light.type === LightType.LightProbe) {
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

		this.shAmbientCoeffs = ambientProbeSH.map((c) => ({
			r: c.r,
			g: c.g,
			b: c.b,
		})) as SHCoefficients;

		let totalSH: SHCoefficients = this.shAmbientCoeffs.map((c) => ({
			r: c.r,
			g: c.g,
			b: c.b,
		})) as SHCoefficients;

		if (this.scene.lights) {
			for (const light of this.scene.lights) {
				if (light.type === LightType.Directional) {
					const contrib = light.computeContribution({
						position: { x: 0, y: 0, z: 0 },
					});
					if (contrib?.direction) {
						const dir = Vector3.normalize(contrib.direction);
						const intensity = contrib.intensity ?? 1.0;
						const lightSH = SH.projectDirectionalLight(dir, {
							r: contrib.color.r * intensity,
							g: contrib.color.g * intensity,
							b: contrib.color.b * intensity,
						});
						totalSH = SH.addCoeffs(totalSH, lightSH);
					}
				}
			}
		}

		this.shCoeffs = totalSH;
	}
}
