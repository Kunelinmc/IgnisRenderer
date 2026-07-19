import type { FrameAttachments } from "../../pipeline/types";
import type { RenderSurfaceSize } from "../IRenderBackend";

/** @internal Owns Software backend canvas presentation and CPU frame attachments. */
export class SoftwareSurfaceRuntime {
	private _canvas: HTMLCanvasElement | null = null;
	private _ctx: CanvasRenderingContext2D | null = null;
	private _pixels: Uint8ClampedArray | null = null;
	private _depthBuffer: Float32Array | null = null;
	private _normalBuffer: Float32Array | null = null;
	private _motionBuffer: Float32Array | null = null;
	private _frameImageData: ImageData | null = null;
	private _framePixels: Uint8ClampedArray | null = null;
	private _framePixelsShared = false;
	private _frameWidth = 0;
	private _frameHeight = 0;
	private _offscreenCanvas: OffscreenCanvas | null = null;
	private _offscreenCtx: OffscreenCanvasRenderingContext2D | null = null;

	public attach(canvas: HTMLCanvasElement): void {
		this._canvas = canvas;
	}

	public initialize(): void {
		const canvas = this._canvas as
			| (HTMLCanvasElement & { getContext?: HTMLCanvasElement["getContext"] })
			| null;
		this._ctx = typeof canvas?.getContext === "function" ? canvas.getContext("2d") : null;
	}

	public getCanvasContext(): CanvasRenderingContext2D | null {
		return this._ctx;
	}

	public getAttachments(size: RenderSurfaceSize): FrameAttachments {
		const { width, height } = size;
		if (
			!this._pixels ||
			this._pixels.length !== width * height * 4 ||
			!this._depthBuffer ||
			this._depthBuffer.length !== width * height
		) {
			this._pixels = new Uint8ClampedArray(width * height * 4);
			this._depthBuffer = new Float32Array(width * height);
			this._normalBuffer = new Float32Array(width * height * 3);
			this._motionBuffer = new Float32Array(width * height * 4);
		}
		this._frameWidth = width;
		this._frameHeight = height;
		return {
			pixels: this._pixels,
			depthBuffer: this._depthBuffer,
			normalBuffer: this._normalBuffer,
			motionBuffer: this._motionBuffer,
			width,
			height,
		};
	}

	public resize(size: RenderSurfaceSize): void {
		const { width, height } = size;
		this._frameImageData = null;
		this._framePixels = null;
		this._framePixelsShared = false;
		if (!this._offscreenCanvas && typeof OffscreenCanvas !== "undefined") {
			this._offscreenCanvas = new OffscreenCanvas(width, height);
			this._offscreenCtx = this._offscreenCanvas.getContext(
				"2d",
			) as OffscreenCanvasRenderingContext2D | null;
		} else if (this._offscreenCanvas) {
			this._offscreenCanvas.width = width;
			this._offscreenCanvas.height = height;
		}
	}

	public present(): void {
		if (!this._ctx) return;
		const pixels = this._requirePixels();
		const { width, height } = this._resolveFrameDimensions(pixels);
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
		if (this._offscreenCtx && this._offscreenCanvas) {
			this._offscreenCtx.putImageData(this._frameImageData, 0, 0);
			const bitmap = this._offscreenCanvas.transferToImageBitmap();
			this._ctx.drawImage(bitmap, 0, 0);
			bitmap.close();
			return;
		}
		this._ctx.putImageData(this._frameImageData, 0, 0);
	}

	public destroy(): void {
		this._ctx = null;
		this._canvas = null;
		this._pixels = null;
		this._depthBuffer = null;
		this._normalBuffer = null;
		this._motionBuffer = null;
		this._frameImageData = null;
		this._framePixels = null;
		this._offscreenCanvas = null;
		this._offscreenCtx = null;
		this._framePixelsShared = false;
		this._frameWidth = 0;
		this._frameHeight = 0;
	}

	private _requirePixels(): Uint8ClampedArray {
		if (!this._pixels) {
			throw new Error("Software backend frame buffer is not initialized.");
		}
		return this._pixels;
	}

	private _resolveFrameDimensions(pixels: Uint8ClampedArray): { width: number; height: number } {
		const canvas = this._canvas;
		if (canvas && pixels.length === canvas.width * canvas.height * 4) {
			return { width: canvas.width, height: canvas.height };
		}
		if (
			this._frameWidth > 0 &&
			this._frameHeight > 0 &&
			pixels.length === this._frameWidth * this._frameHeight * 4
		) {
			return { width: this._frameWidth, height: this._frameHeight };
		}
		const pixelCount = Math.floor(pixels.length / 4);
		return { width: Math.max(1, pixelCount), height: 1 };
	}

	private _createFrameImageData(
		pixels: Uint8ClampedArray,
		width: number,
		height: number,
	): ImageData {
		try {
			const imageData = new ImageData(pixels as ImageDataArray, width, height);
			this._framePixelsShared =
				imageData.data === pixels || imageData.data.buffer === pixels.buffer;
			return imageData;
		} catch {
			const imageData = new ImageData(width, height);
			imageData.data.set(pixels.subarray(0, Math.min(imageData.data.length, pixels.length)));
			this._framePixelsShared = false;
			return imageData;
		}
	}
}
