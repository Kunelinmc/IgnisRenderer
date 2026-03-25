import type { IRenderBackend, RendererBackendBridge } from "./IRenderBackend";
import {
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	type FrameContext,
	type FramePass,
} from "../pipeline/types";
import { Rasterizer } from "./software/Rasterizer";
import { PostProcessor } from "./software/PostProcessor";
import { SoftwareMainPass } from "./software/passes/SoftwareMainPass";
import { SoftwareParticlePass } from "./software/passes/SoftwareParticlePass";
import { SoftwareReflectionPass } from "./software/passes/SoftwareReflectionPass";
import { SoftwareShadowPass } from "./software/passes/SoftwareShadowPass";
import { SkyboxRenderer } from "./software/SkyboxRenderer";
import { isShadowCastingLight } from "../lights";
import {
	resolveShadowCasterBounds,
	syncShadowMapRegistry,
	updateShadowMapMetadata,
} from "../pipeline/ShadowMetadata";
import { FrameAttachments } from "../pipeline/types";
import { DefaultParticleSimulator } from "../simulation/particles/DefaultParticleSimulator";
import {
	DEFAULT_SOFTWARE_RASTER_MODE,
	type SoftwareBackendOptions,
	type SoftwareRasterMode,
} from "./software/SoftwareRasterConfig";

export type {
	SoftwareBackendOptions,
	SoftwareRasterMode,
	SoftwareTileOptions,
} from "./software/SoftwareRasterConfig";

export class SoftwareBackend implements IRenderBackend {
	public readonly type = "software";
	public readonly frameScheduling = "on-demand";
	public readonly passExecutors = {
		"animation-sim": "shared",
		"particle-sim": "backend",
	} as const;
	public readonly capabilities = {
		sh: true,
		shadows: true,
		reflection: true,
		skybox: true,
		ssao: true,
		taa: false,
		ssr: false,
		volumetric: true,
		bloom: false,
	};
	public readonly requestedRasterMode: SoftwareRasterMode;

	private _renderer: RendererBackendBridge | null = null;
	private _ctx: CanvasRenderingContext2D | null = null;
	private _rasterizer: Rasterizer | null = null;
	private _mainPass: SoftwareMainPass | null = null;
	private _particlePass: SoftwareParticlePass | null = null;
	private _shadowPass: SoftwareShadowPass | null = null;
	private _reflectionPass: SoftwareReflectionPass | null = null;
	private _postProcessor: PostProcessor | null = null;
	private _framePixelsShared = false;
	private _pixels: Uint8ClampedArray | null = null;
	private _depthBuffer: Float32Array | null = null;
	private _normalBuffer: Float32Array | null = null;
	private _frameImageData: ImageData | null = null;
	private _framePixels: Uint8ClampedArray | null = null;
	private _frameWidth = 0;
	private _frameHeight = 0;
	private _particleSimulator: DefaultParticleSimulator | null = null;
	private _offscreenCanvas: OffscreenCanvas | null = null;
	private _offscreenCtx: OffscreenCanvasRenderingContext2D | null = null;
	private _options: SoftwareBackendOptions;
	private _activeRasterMode: SoftwareRasterMode;

	public constructor(options: SoftwareBackendOptions = {}) {
		this._options = options;
		this.requestedRasterMode =
			options.rasterMode ?? DEFAULT_SOFTWARE_RASTER_MODE;
		this._activeRasterMode = this.requestedRasterMode;
	}

	public get activeRasterMode(): SoftwareRasterMode {
		return this._activeRasterMode;
	}

	public async init(canvas: HTMLCanvasElement): Promise<void> {
		this._ctx = canvas.getContext("2d");
	}

	public setRenderer(renderer: RendererBackendBridge): void {
		this._renderer = renderer;
		this._rasterizer = new Rasterizer();
		this._shadowPass = new SoftwareShadowPass(this._rasterizer);
		this._mainPass = new SoftwareMainPass(this._rasterizer, {
			mode: this.requestedRasterMode,
			tile: this._options.tile,
			warnOnce: (key, message) => renderer.warnOnce(key, message),
		});
		this._particlePass = new SoftwareParticlePass();
		this._reflectionPass = new SoftwareReflectionPass(this._rasterizer);
		this._postProcessor = new PostProcessor(renderer);
		this._particleSimulator = new DefaultParticleSimulator({
			backendTag: this.type,
		});
		this._syncActiveRasterMode();
	}

	public getAttachments(width: number, height: number): FrameAttachments {
		if (
			!this._pixels ||
			this._pixels.length !== width * height * 4 ||
			!this._depthBuffer ||
			this._depthBuffer.length !== width * height
		) {
			this._pixels = new Uint8ClampedArray(width * height * 4);
			this._depthBuffer = new Float32Array(width * height);
			this._normalBuffer = new Float32Array(width * height * 3);
		}
		this._frameWidth = width;
		this._frameHeight = height;
		return {
			pixels: this._pixels,
			depthBuffer: this._depthBuffer,
			normalBuffer: this._normalBuffer,
			width,
			height,
		};
	}

	public resize(width: number, height: number): void {
		this._frameImageData = null;
		this._framePixels = null;
		this._framePixelsShared = false;

		if (!this._offscreenCanvas) {
			this._offscreenCanvas = new OffscreenCanvas(width, height);
			this._offscreenCtx = this._offscreenCanvas.getContext(
				"2d"
			) as OffscreenCanvasRenderingContext2D | null;
		} else {
			this._offscreenCanvas.width = width;
			this._offscreenCanvas.height = height;
		}
	}

	public beginFrame(context: FrameContext): void {
		this._particleSimulator?.beginFrame(context);

		const pixels = context.attachments.pixels!;
		const size = pixels.length >> 2;
		for (let i = 0; i < size; i++) {
			const index = i << 2;
			pixels[index] = 0;
			pixels[index + 1] = 0;
			pixels[index + 2] = 0;
			pixels[index + 3] = 255;
		}
		context.attachments.depthBuffer.fill(Infinity);
		context.attachments.normalBuffer?.fill(0);

		const shadowLights = context.scene.lights.filter(isShadowCastingLight);
		syncShadowMapRegistry(context.shadowMaps, shadowLights);
		const shadowCasterBounds = resolveShadowCasterBounds(
			context.scene.shadowCasterPackets,
			context.scene.sceneBounds,
			context.scene.camera
		);
		for (const shadowLight of shadowLights) {
			const shadowMap = context.shadowMaps.get(shadowLight);
			if (shadowMap) {
				updateShadowMapMetadata(shadowMap, shadowLight, shadowCasterBounds);
			}
		}

		if (context.features.enableSkybox && context.scene.skybox) {
			SkyboxRenderer.render(
				context.scene.skybox,
				pixels,
				context.camera,
				context.attachments.width,
				context.attachments.height
			);
		}
	}

	public async executePass(
		pass: FramePass,
		context: FrameContext
	): Promise<void> {
		if (!this._renderer || !this._mainPass || !this._reflectionPass) return;

		switch (pass.stage) {
			case "animation-sim":
				break;
			case "particle-sim":
				this._particleSimulator?.simulate(
					context,
					this._resolveParticleDeltaTime(context)
				);
				this._particleSimulator?.emitRenderBatches(context);
				break;
			case "shadow":
				this._shadowPass?.render(context);
				break;
			case "reflection":
				this._reflectionPass.render(context);
				break;
			case "main-opaque":
				await this._mainPass.render(
					context,
					context.scene.opaquePackets,
					false
				);
				this._syncActiveRasterMode();
				break;
			case "main-transparent":
				await this._mainPass.render(
					context,
					context.scene.transparentPackets,
					true
				);
				this._syncActiveRasterMode();
				break;
			case "particles":
				this._particlePass?.render(context);
				break;
			case "ssao":
				this._postProcessor?.applySSAO(context);
				break;
			case "taa":
				break;
			case "ssr":
				break;
			case "volumetric":
				if (this._ctx) {
					this._postProcessor?.applyVolumetricLight(context, this._ctx);
				}
				break;
			case "fxaa":
				if (this._ctx) {
					this._postProcessor?.applyFXAA(context, this._ctx);
				}
				break;
			case "gamma":
				if (this._ctx) {
					this._postProcessor?.applyGamma(context, this._ctx);
				}
				break;
		}
	}

	public endFrame(): void {
		if (!this._renderer || !this._ctx) return;

		this._particleSimulator?.endFrame();

		const imageData = this._getFrameImageData(this._renderer);
		if (this._offscreenCtx && this._offscreenCanvas) {
			this._offscreenCtx.putImageData(imageData, 0, 0);
			const bitmap = this._offscreenCanvas.transferToImageBitmap();
			this._ctx.drawImage(bitmap, 0, 0);
			bitmap.close();
		} else {
			this._ctx.putImageData(imageData, 0, 0);
		}
	}

	private _resolveParticleDeltaTime(context: FrameContext): number {
		const value = context.transient.get(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY);
		if (typeof value !== "number" || !Number.isFinite(value)) {
			return 0;
		}
		return Math.max(0, value);
	}

	private _getFrameImageData(renderer: RendererBackendBridge): ImageData {
		const pixels = this._resolveFramePixels(renderer);
		const { width, height } = this._resolveFrameDimensions(renderer, pixels);

		if (
			!this._frameImageData ||
			this._framePixels !== pixels ||
			this._frameImageData.width !== width ||
			this._frameImageData.height !== height
		) {
			this._frameImageData = this._createFrameImageData(pixels, width, height);
			this._framePixels = pixels;
		}

		if (!this._framePixelsShared) {
			this._frameImageData.data.set(pixels);
		}

		return this._frameImageData;
	}

	private _resolveFrameDimensions(
		renderer: RendererBackendBridge,
		pixels: Uint8ClampedArray
	): { width: number; height: number } {
		const canvasWidth = renderer.canvas.width;
		const canvasHeight = renderer.canvas.height;
		if (pixels.length === canvasWidth * canvasHeight * 4) {
			return {
				width: canvasWidth,
				height: canvasHeight,
			};
		}

		if (
			this._frameWidth > 0 &&
			this._frameHeight > 0 &&
			pixels.length === this._frameWidth * this._frameHeight * 4
		) {
			return {
				width: this._frameWidth,
				height: this._frameHeight,
			};
		}

		if (
			this._frameImageData &&
			pixels.length ===
				this._frameImageData.width * this._frameImageData.height * 4
		) {
			return {
				width: this._frameImageData.width,
				height: this._frameImageData.height,
			};
		}

		const pixelCount = Math.floor(pixels.length / 4);
		return {
			width: Math.max(1, pixelCount),
			height: 1,
		};
	}

	private _resolveFramePixels(
		renderer: RendererBackendBridge
	): Uint8ClampedArray {
		const legacyPixels = (
			renderer as RendererBackendBridge & {
				pixels?: Uint8ClampedArray | null;
			}
		).pixels;
		const pixels = this._pixels || legacyPixels;

		if (!pixels) {
			throw new Error("Software backend frame buffer is not initialized.");
		}

		return pixels;
	}

	private _createFrameImageData(
		pixels: Uint8ClampedArray,
		width: number,
		height: number
	): ImageData {
		try {
			const imageData = new ImageData(pixels as ImageDataArray, width, height);
			this._framePixelsShared =
				imageData.data === pixels || imageData.data.buffer === pixels.buffer;
			return imageData;
		} catch {
			const imageData = new ImageData(width, height);
			const copyLength = Math.min(imageData.data.length, pixels.length);
			imageData.data.set(pixels.subarray(0, copyLength));
			this._framePixelsShared = false;
			return imageData;
		}
	}

	public destroy(): void {
		this._mainPass?.destroy();
		this._mainPass = null;
		this._particlePass = null;
		this._shadowPass = null;
		this._reflectionPass = null;
		this._postProcessor = null;
		this._particleSimulator = null;
		this._rasterizer = null;
	}

	private _syncActiveRasterMode(): void {
		const mode = this._mainPass?.getActiveMode();
		this._activeRasterMode = mode ?? this.requestedRasterMode;
	}
}
