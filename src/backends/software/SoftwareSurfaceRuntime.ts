import type { FrameAttachments } from "../../pipeline/types";
import type { PostProcessColorDomain } from "../../postprocess/PostProcessPass";
import { warnIncompletePresentation } from "../../rendering/DisplayOutput";
import type { RenderSurfaceSize } from "../IRenderBackend";
import {
	SOFTWARE_HDR_IMAGE_DATA_SETTINGS,
	SoftwareDisplayOutputManager,
} from "./SoftwareDisplayOutputManager";
import type { SoftwareFrameView } from "./SoftwareFrameView";

const ALPHA_EPSILON = 1e-6;

/** @internal Presents through a backend-owned context and owns CPU frame targets. */
export class SoftwareSurfaceRuntime {
	private _context: CanvasRenderingContext2D | null = null;
	private _pixels: Uint8ClampedArray | null = null;
	private _sceneColor: Float32Array | null = null;
	private _depthBuffer: Float32Array | null = null;
	private _normalBuffer: Float32Array | null = null;
	private _motionBuffer: Float32Array | null = null;
	private _sdrImageData: ImageData | null = null;
	private _hdrImageData: ImageData | null = null;
	private _hdrPixels: Float16Array | null = null;
	private _frameWidth = 0;
	private _frameHeight = 0;

	constructor(private readonly _displayOutput: SoftwareDisplayOutputManager) {}

	public initialize(context: CanvasRenderingContext2D | null): void {
		this._releaseFrameTargets();
		this._context = context;
	}

	public getCanvasContext(): CanvasRenderingContext2D | null {
		return this._context;
	}

	public getSceneColorTarget(): Float32Array {
		if (!this._sceneColor) {
			throw new Error("Software backend scene color target is not initialized.");
		}
		return this._sceneColor;
	}

	public getAttachments(size: RenderSurfaceSize): FrameAttachments {
		const { width, height } = size;
		const pixelCount = width * height;
		if (
			!this._pixels ||
			this._pixels.length !== pixelCount * 4 ||
			!this._sceneColor ||
			this._sceneColor.length !== pixelCount * 4 ||
			!this._depthBuffer ||
			this._depthBuffer.length !== pixelCount
		) {
			this._pixels = new Uint8ClampedArray(pixelCount * 4);
			this._sceneColor = new Float32Array(pixelCount * 4);
			this._depthBuffer = new Float32Array(pixelCount);
			this._normalBuffer = new Float32Array(pixelCount * 3);
			this._motionBuffer = new Float32Array(pixelCount * 4);
			this._sdrImageData = null;
			this._hdrImageData = null;
			this._hdrPixels = null;
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
		this._frameWidth = size.width;
		this._frameHeight = size.height;
		this._releaseFrameTargets();
	}

	public present(frame: SoftwareFrameView, colorDomain: PostProcessColorDomain): void {
		warnIncompletePresentation(colorDomain, "software");
		const context = this._context;
		const state = this._displayOutput.state;
		if (!context) {
			this._copyRegions(frame, null, this._requirePixels());
			return;
		}
		if (state.activeDynamicRange === "hdr") {
			const imageData = this._prepareHDRImageData(frame);
			context.putImageData(imageData, 0, 0);
			return;
		}
		const imageData = this._prepareSDRImageData(frame);
		context.putImageData(imageData, 0, 0);
	}

	public destroy(): void {
		this._context = null;
		this._releaseFrameTargets();
		this._frameWidth = 0;
		this._frameHeight = 0;
	}

	private _prepareSDRImageData(frame: SoftwareFrameView): ImageData {
		const pixels = this._requirePixels();
		this._copyRegions(frame, null, pixels);
		if (
			!this._sdrImageData ||
			this._sdrImageData.width !== frame.attachments.width ||
			this._sdrImageData.height !== frame.attachments.height
		) {
			try {
				this._sdrImageData = new ImageData(
					pixels as ImageDataArray,
					frame.attachments.width,
					frame.attachments.height,
					{ colorSpace: "srgb" },
				);
			} catch {
				this._sdrImageData = new ImageData(
					frame.attachments.width,
					frame.attachments.height,
				);
			}
		}
		if (this._sdrImageData.data !== pixels) {
			this._sdrImageData.data.set(pixels);
		}
		return this._sdrImageData;
	}

	private _prepareHDRImageData(frame: SoftwareFrameView): ImageData {
		const elementCount = frame.attachments.width * frame.attachments.height * 4;
		if (!this._hdrPixels || this._hdrPixels.length !== elementCount) {
			this._hdrPixels = new Float16Array(elementCount);
			this._hdrImageData = null;
		}
		this._copyRegions(frame, this._hdrPixels, this._requirePixels());
		if (
			!this._hdrImageData ||
			this._hdrImageData.width !== frame.attachments.width ||
			this._hdrImageData.height !== frame.attachments.height
		) {
			this._hdrImageData = new ImageData(
				this._hdrPixels as unknown as ImageDataArray,
				frame.attachments.width,
				frame.attachments.height,
				SOFTWARE_HDR_IMAGE_DATA_SETTINGS as ImageDataSettings,
			);
		}
		return this._hdrImageData;
	}

	private _copyRegions(
		frame: SoftwareFrameView,
		hdrPixels: Float16Array | null,
		preview: Uint8ClampedArray,
	): void {
		const color = frame.attachments.color;
		const width = frame.attachments.width;
		for (const region of frame.clipRegions) {
			for (let y = region.minY; y < region.maxYExclusive; y++) {
				for (let x = region.minX; x < region.maxXExclusive; x++) {
					const index = (y * width + x) << 2;
					const transparent = frame.presentationAlphaMode === "premultiplied";
					const alpha = transparent
						? Math.min(1, Math.max(0, finiteOrZero(color[index + 3])))
						: 1;
					const inverseAlpha = alpha > ALPHA_EPSILON ? 1 / alpha : 0;
					// Canvas ImageData stores straight RGB, even for transparent output.
					const scale = transparent ? inverseAlpha : 1;
					const red = Math.max(0, finiteOrZero(color[index] * scale));
					const green = Math.max(0, finiteOrZero(color[index + 1] * scale));
					const blue = Math.max(0, finiteOrZero(color[index + 2] * scale));
					if (hdrPixels) {
						hdrPixels[index] = red;
						hdrPixels[index + 1] = green;
						hdrPixels[index + 2] = blue;
						hdrPixels[index + 3] = alpha;
					}
					preview[index] = Math.round(Math.min(1, Math.max(0, red)) * 255);
					preview[index + 1] = Math.round(Math.min(1, Math.max(0, green)) * 255);
					preview[index + 2] = Math.round(Math.min(1, Math.max(0, blue)) * 255);
					preview[index + 3] = Math.round(alpha * 255);
				}
			}
		}
	}

	private _requirePixels(): Uint8ClampedArray {
		if (!this._pixels) {
			throw new Error("Software backend frame buffer is not initialized.");
		}
		return this._pixels;
	}

	private _releaseFrameTargets(): void {
		this._pixels = null;
		this._sceneColor = null;
		this._depthBuffer = null;
		this._normalBuffer = null;
		this._motionBuffer = null;
		this._sdrImageData = null;
		this._hdrImageData = null;
		this._hdrPixels = null;
	}
}

function finiteOrZero(value: number): number {
	return Number.isFinite(value) ? value : 0;
}
