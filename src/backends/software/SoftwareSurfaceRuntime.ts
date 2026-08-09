import type { FrameAttachments } from "../../pipeline/types";
import type { PostProcessColorDomain } from "../../postprocess/PostProcessPass";
import {
	encodeLinearSRGB,
	linearSrgbToDisplayP3,
} from "../../postprocess/passes/GammaPass";
import {
	applyACESToneMapping,
	applyHDRSoftShoulder,
} from "../../postprocess/passes/ToneMappingPass";
import type { DisplayOutputState } from "../../rendering/DisplayOutput";
import type { RenderSurfaceSize } from "../IRenderBackend";
import {
	SOFTWARE_HDR_IMAGE_DATA_SETTINGS,
	SoftwareDisplayOutputManager,
} from "./SoftwareDisplayOutputManager";
import type { SoftwareFrameView } from "./SoftwareFrameView";

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
	private readonly _inputColor: [number, number, number] = [0, 0, 0];
	private readonly _mappedColor: [number, number, number] = [0, 0, 0];
	private readonly _p3Color: [number, number, number] = [0, 0, 0];
	private readonly _encodedColor: [number, number, number] = [0, 0, 0];

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
		const context = this._context;
		const state = this._displayOutput.state;
		if (!context) {
			this._convertRegions(frame, colorDomain, state, null, this._requirePixels());
			return;
		}
		if (state.activeDynamicRange === "hdr") {
			const imageData = this._prepareHDRImageData(frame, colorDomain, state);
			context.putImageData(imageData, 0, 0);
			return;
		}
		const imageData = this._prepareSDRImageData(frame, colorDomain, state);
		context.putImageData(imageData, 0, 0);
	}

	public destroy(): void {
		this._context = null;
		this._releaseFrameTargets();
		this._frameWidth = 0;
		this._frameHeight = 0;
	}

	private _prepareSDRImageData(
		frame: SoftwareFrameView,
		colorDomain: PostProcessColorDomain,
		state: DisplayOutputState,
	): ImageData {
		const pixels = this._requirePixels();
		this._convertRegions(frame, colorDomain, state, null, pixels);
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

	private _prepareHDRImageData(
		frame: SoftwareFrameView,
		colorDomain: PostProcessColorDomain,
		state: DisplayOutputState,
	): ImageData {
		const elementCount = frame.attachments.width * frame.attachments.height * 4;
		if (!this._hdrPixels || this._hdrPixels.length !== elementCount) {
			this._hdrPixels = new Float16Array(elementCount);
			this._hdrImageData = null;
		}
		this._convertRegions(frame, colorDomain, state, this._hdrPixels, this._requirePixels());
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

	private _convertRegions(
		frame: SoftwareFrameView,
		colorDomain: PostProcessColorDomain,
		state: DisplayOutputState,
		hdrPixels: Float16Array | null,
		preview: Uint8ClampedArray,
	): void {
		const color = frame.attachments.color;
		const width = frame.attachments.width;
		for (const region of frame.clipRegions) {
			for (let y = region.minY; y < region.maxYExclusive; y++) {
				for (let x = region.minX; x < region.maxXExclusive; x++) {
					const index = (y * width + x) << 2;
					const encoded = this._completeColor(
						color[index],
						color[index + 1],
						color[index + 2],
						colorDomain,
						state,
					);
					const alpha = Math.min(1, Math.max(0, finiteOrZero(color[index + 3])));
					if (hdrPixels) {
						hdrPixels[index] = finiteOrZero(encoded[0]);
						hdrPixels[index + 1] = finiteOrZero(encoded[1]);
						hdrPixels[index + 2] = finiteOrZero(encoded[2]);
						hdrPixels[index + 3] = alpha;
					}
					preview[index] = Math.round(Math.min(1, Math.max(0, encoded[0])) * 255);
					preview[index + 1] = Math.round(Math.min(1, Math.max(0, encoded[1])) * 255);
					preview[index + 2] = Math.round(Math.min(1, Math.max(0, encoded[2])) * 255);
					preview[index + 3] = Math.round(alpha * 255);
				}
			}
		}
	}

	private _completeColor(
		red: number,
		green: number,
		blue: number,
		domain: PostProcessColorDomain,
		state: DisplayOutputState,
	): [number, number, number] {
		const encoded = this._encodedColor;
		if (domain === "display-encoded") {
			encoded[0] = finiteOrZero(red);
			encoded[1] = finiteOrZero(green);
			encoded[2] = finiteOrZero(blue);
			return encoded;
		}
		const hdr = state.activeDynamicRange === "hdr";
		const input = this._inputColor;
		input[0] = finiteOrZero(red);
		input[1] = finiteOrZero(green);
		input[2] = finiteOrZero(blue);
		let linear = input;
		if (domain === "scene-linear-hdr") {
			linear = hdr
				? applyHDRSoftShoulder(
						input,
						state.requested.exposure,
						state.requested.hdrHeadroom,
						this._mappedColor,
					)
				: applyACESToneMapping(linear, state.requested.exposure, this._mappedColor);
		}
		if (hdr) linear = linearSrgbToDisplayP3(linear, this._p3Color);
		const upperBound = hdr ? state.requested.hdrHeadroom : 1;
		encoded[0] = encodeLinearSRGB(Math.min(upperBound, Math.max(0, linear[0])), hdr);
		encoded[1] = encodeLinearSRGB(Math.min(upperBound, Math.max(0, linear[1])), hdr);
		encoded[2] = encodeLinearSRGB(Math.min(upperBound, Math.max(0, linear[2])), hdr);
		return encoded;
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
