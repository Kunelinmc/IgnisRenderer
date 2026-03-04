import type { Renderer } from "../Renderer";
import type { IRenderBackend } from "./IRenderBackend";
import type {
	FramePass,
	PreparedScene,
	ResolvedFeatureState,
} from "../pipeline/types";
import { Rasterizer } from "../software/Rasterizer";
import { SoftwareMainPass } from "../software/passes/SoftwareMainPass";
import { SoftwareReflectionPass } from "../software/passes/SoftwareReflectionPass";

export class SoftwareBackend implements IRenderBackend {
	public readonly type = "software";
	public readonly capabilities = {
		sh: true,
		shadows: true,
		reflection: true,
		skybox: true,
		ssao: true,
		volumetric: true,
	};

	private _renderer: Renderer | null = null;
	private _ctx: CanvasRenderingContext2D | null = null;
	private _rasterizer: Rasterizer | null = null;
	private _mainPass: SoftwareMainPass | null = null;
	private _reflectionPass: SoftwareReflectionPass | null = null;
	private _resolvedFeatures: ResolvedFeatureState | null = null;
	private _frameImageData: ImageData | null = null;
	private _framePixels: Uint8ClampedArray | null = null;
	private _framePixelsShared = false;

	public async init(canvas: HTMLCanvasElement): Promise<void> {
		this._ctx = canvas.getContext("2d");
	}

	public setRenderer(renderer: Renderer): void {
		this._renderer = renderer;
		this._rasterizer = new Rasterizer(renderer);
		this._mainPass = new SoftwareMainPass(renderer);
		this._reflectionPass = new SoftwareReflectionPass(renderer);
		renderer.rasterizer = this._rasterizer;
	}

	public resize(_width: number, _height: number): void {
		this._frameImageData = null;
		this._framePixels = null;
		this._framePixelsShared = false;
	}

	public beginFrame(
		_frame: PreparedScene,
		features: ResolvedFeatureState
	): void {
		this._resolvedFeatures = features;
		if (!this._renderer) return;

		const pixels = this._renderer.pixels;
		const size = this._renderer.canvas.width * this._renderer.canvas.height;
		for (let i = 0; i < size; i++) {
			const index = i << 2;
			pixels[index] = 0;
			pixels[index + 1] = 0;
			pixels[index + 2] = 0;
			pixels[index + 3] = 255;
		}
		this._renderer.depthBuffer.fill(Infinity);
		this._renderer.normalBuffer?.fill(0);

		if (features.enableSkybox && this._renderer.scene.skybox) {
			this._renderer.renderSkybox(this._renderer.pixels);
		}
	}

	public executePass(pass: FramePass, frame: PreparedScene): void {
		if (!this._renderer || !this._mainPass || !this._reflectionPass) return;

		switch (pass.stage) {
			case "reflection":
				this._reflectionPass.render(frame);
				break;
			case "main-opaque":
				this._mainPass.render(frame.opaquePackets, false);
				break;
			case "main-transparent":
				this._mainPass.render(frame.transparentPackets, true);
				break;
			case "ssao":
				this._renderer.postProcessor.applySSAO(
					this._renderer.pixels,
					this._renderer.depthBuffer,
					this._renderer.normalBuffer,
					this._renderer.features.ssaoOptions
				);
				break;
			case "volumetric":
				if (this._ctx) {
					this._renderer.postProcessor.applyVolumetricLight(
						this._ctx,
						this._renderer.canvas,
						this._renderer.pixels,
						this._renderer.depthBuffer,
						this._renderer.features.volumetricOptions
					);
				}
				break;
			case "fxaa":
				if (this._ctx) {
					this._renderer.postProcessor.applyFXAA(
						this._ctx,
						this._renderer.canvas,
						this._renderer.pixels
					);
				}
				break;
			case "gamma":
				if (this._ctx) {
					this._renderer.postProcessor.applyGamma(
						this._ctx,
						this._renderer.canvas,
						undefined,
						this._renderer.pixels
					);
				}
				break;
		}
	}

	public endFrame(): void {
		if (!this._renderer || !this._ctx) return;

		const imageData = this._getFrameImageData(this._renderer);
		this._ctx.putImageData(imageData, 0, 0);
	}

	private _getFrameImageData(renderer: Renderer): ImageData {
		const width = renderer.canvas.width;
		const height = renderer.canvas.height;
		const pixels = renderer.pixels;

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
