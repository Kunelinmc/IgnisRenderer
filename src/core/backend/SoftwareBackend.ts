import type {
	IRenderBackend,
	RendererBackendBridge,
} from "./IRenderBackend";
import type { FrameContext, FramePass } from "../pipeline/types";
import { Rasterizer } from "./software/Rasterizer";
import { PostProcessor } from "./software/PostProcessor";
import { SoftwareMainPass } from "./software/passes/SoftwareMainPass";
import { SoftwareParticlePass } from "./software/passes/SoftwareParticlePass";
import { SoftwareReflectionPass } from "./software/passes/SoftwareReflectionPass";
import { SoftwareShadowPass } from "./software/passes/SoftwareShadowPass";
import { SkyboxRenderer } from "./software/SkyboxRenderer";

export class SoftwareBackend implements IRenderBackend {
	public readonly type = "software";
	public readonly frameScheduling = "always";
	public readonly passExecutors = {
		"particle-sim": "shared",
		shadow: "shared",
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
	};

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

	public async init(canvas: HTMLCanvasElement): Promise<void> {
		this._ctx = canvas.getContext("2d");
	}

	public setRenderer(renderer: RendererBackendBridge): void {
		this._renderer = renderer;
		this._rasterizer = new Rasterizer();
		this._shadowPass = new SoftwareShadowPass(this._rasterizer);
		this._mainPass = new SoftwareMainPass(this._rasterizer);
		this._particlePass = new SoftwareParticlePass();
		this._reflectionPass = new SoftwareReflectionPass(this._rasterizer);
		this._postProcessor = new PostProcessor(renderer);
	}

	public getAttachments(width: number, height: number): any {
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
		return {
			pixels: this._pixels,
			depthBuffer: this._depthBuffer,
			normalBuffer: this._normalBuffer,
			width,
			height,
		};
	}

	public resize(_width: number, _height: number): void {
		this._frameImageData = null;
		this._framePixels = null;
		this._framePixelsShared = false;
	}

	public beginFrame(context: FrameContext): void {
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

	public executeSharedPass(pass: FramePass, context: FrameContext): void {
		if (pass.stage !== "shadow") return;
		this._shadowPass?.render(context);
	}

	public executePass(pass: FramePass, context: FrameContext): void {
		if (!this._renderer || !this._mainPass || !this._reflectionPass) return;

		switch (pass.stage) {
			case "reflection":
				this._reflectionPass.render(context);
				break;
			case "main-opaque":
				this._mainPass.render(context, context.scene.opaquePackets, false);
				break;
			case "main-transparent":
				this._mainPass.render(context, context.scene.transparentPackets, true);
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

		const imageData = this._getFrameImageData(this._renderer);
		this._ctx.putImageData(imageData, 0, 0);
	}

	private _getFrameImageData(renderer: RendererBackendBridge): ImageData {
		const width = renderer.canvas.width;
		const height = renderer.canvas.height;
		const pixels = this._resolveFramePixels(renderer);

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
			imageData.data.set(pixels);
			this._framePixelsShared = false;
			return imageData;
		}
	}
}
