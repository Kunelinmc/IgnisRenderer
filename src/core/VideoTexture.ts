import { Texture, type TextureColorSpace } from "./Texture";

type VideoTextureContext2D =
	| CanvasRenderingContext2D
	| OffscreenCanvasRenderingContext2D;

const VIDEO_INVALIDATION_EVENTS = [
	"loadeddata",
	"loadedmetadata",
	"seeking",
	"seeked",
	"ratechange",
] as const;

const VIDEO_FRAME_CALLBACK_EVENTS = [
	"play",
	"pause",
	"ended",
	"emptied",
] as const;

export interface VideoTextureParams {
	colorSpace?: TextureColorSpace;
}

/**
 * Dynamic texture that pulls frames from an HTMLVideoElement.
 */
export class VideoTexture extends Texture {
	public readonly video: HTMLVideoElement;

	private _canvas: HTMLCanvasElement | OffscreenCanvas;
	private _context: VideoTextureContext2D;
	private _lastVideoTime: number;
	private _forceRefresh: boolean;
	private _pendingVideoFrame: boolean;
	private _videoFrameCallbackHandle: number | null;
	private _supportsVideoFrameCallback: boolean;
	private _isDisposed: boolean;
	private _onVideoInvalidated: () => void;
	private _onPlaybackStateChanged: () => void;
	private _onVideoFramePresented: (now: number, metadata: unknown) => void;

	constructor(video: HTMLVideoElement, params: VideoTextureParams = {}) {
		super(null, 0, 0, params.colorSpace ?? "sRGB");
		if (!video) {
			throw new Error("VideoTexture requires a valid HTMLVideoElement");
		}

		this.video = video;
		this._canvas = this._createCanvas();
		this._context = this._createContext(this._canvas);
		this._lastVideoTime = -1;
		this._forceRefresh = true;
		this._pendingVideoFrame = true;
		this._videoFrameCallbackHandle = null;
		this._supportsVideoFrameCallback = this._hasVideoFrameCallbackSupport();
		this._isDisposed = false;
		this._onVideoInvalidated = () => {
			this._forceRefresh = true;
			this._pendingVideoFrame = true;
			this._scheduleVideoFrameCallback();
		};
		this._onPlaybackStateChanged = () => {
			this._scheduleVideoFrameCallback();
		};
		this._onVideoFramePresented = () => {
			this._videoFrameCallbackHandle = null;
			this._pendingVideoFrame = true;
			this._forceRefresh = true;
			this._scheduleVideoFrameCallback();
		};

		this._registerAsDynamicTexture();
		this._bindVideoEvents();
		this._scheduleVideoFrameCallback();
		this.update();
	}

	public invalidate(): void {
		this._forceRefresh = true;
		this._pendingVideoFrame = true;
	}

	public override update(_timeMs: number = 0): boolean {
		const video = this.video;
		if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
			return false;
		}

		const width = video.videoWidth | 0;
		const height = video.videoHeight | 0;
		if (width <= 0 || height <= 0) {
			return false;
		}

		const currentTime = video.currentTime;
		const hasNewFrame =
			this._supportsVideoFrameCallback ?
				this._pendingVideoFrame
			:	currentTime !== this._lastVideoTime;
		if (!this._forceRefresh && !hasNewFrame) {
			return false;
		}

		if (this._canvas.width !== width || this._canvas.height !== height) {
			this._canvas.width = width;
			this._canvas.height = height;
		}

		try {
			this._context.drawImage(video, 0, 0, width, height);
			const frameData = this._context.getImageData(0, 0, width, height).data;

			if (
				!(this.data instanceof Uint8ClampedArray) ||
				this.data.length !== frameData.length
			) {
				this.data = new Uint8ClampedArray(frameData);
			} else {
				this.data.set(frameData);
			}
		} catch (error) {
			const reason =
				error instanceof Error ? error.message : "Unknown canvas error";
			throw new Error(
				`VideoTexture failed to read frame data. Ensure video CORS/canvas access is allowed: ${reason}`
			);
		}

		this.width = width;
		this.height = height;
		this.mipmaps = this.data ? [this.data] : [];
		this._lastVideoTime = currentTime;
		this._forceRefresh = false;
		this._pendingVideoFrame = false;
		this.markNeedsUpdate();
		return true;
	}

	public override dispose(): void {
		this._isDisposed = true;
		this._unbindVideoEvents();
		this._cancelVideoFrameCallback();
		super.dispose();
	}

	private _bindVideoEvents(): void {
		for (const eventName of VIDEO_INVALIDATION_EVENTS) {
			this.video.addEventListener(eventName, this._onVideoInvalidated);
		}
		for (const eventName of VIDEO_FRAME_CALLBACK_EVENTS) {
			this.video.addEventListener(eventName, this._onPlaybackStateChanged);
		}
	}

	private _unbindVideoEvents(): void {
		for (const eventName of VIDEO_INVALIDATION_EVENTS) {
			this.video.removeEventListener(eventName, this._onVideoInvalidated);
		}
		for (const eventName of VIDEO_FRAME_CALLBACK_EVENTS) {
			this.video.removeEventListener(eventName, this._onPlaybackStateChanged);
		}
	}

	private _hasVideoFrameCallbackSupport(): boolean {
		return (
			typeof (this.video as any).requestVideoFrameCallback === "function" &&
			typeof (this.video as any).cancelVideoFrameCallback === "function"
		);
	}

	private _scheduleVideoFrameCallback(): void {
		if (!this._supportsVideoFrameCallback || this._isDisposed) {
			return;
		}
		if (this._videoFrameCallbackHandle !== null) {
			return;
		}
		const requestCallback = (this.video as any).requestVideoFrameCallback as
			| ((cb: (now: number, metadata: unknown) => void) => number)
			| undefined;
		if (typeof requestCallback !== "function") {
			return;
		}
		this._videoFrameCallbackHandle = requestCallback.call(
			this.video,
			this._onVideoFramePresented
		);
	}

	private _cancelVideoFrameCallback(): void {
		if (!this._supportsVideoFrameCallback) {
			return;
		}
		if (this._videoFrameCallbackHandle === null) {
			return;
		}
		const cancelCallback = (this.video as any).cancelVideoFrameCallback as
			| ((handle: number) => void)
			| undefined;
		if (typeof cancelCallback === "function") {
			cancelCallback.call(this.video, this._videoFrameCallbackHandle);
		}
		this._videoFrameCallbackHandle = null;
	}

	private _createCanvas(): HTMLCanvasElement | OffscreenCanvas {
		if (typeof OffscreenCanvas !== "undefined") {
			return new OffscreenCanvas(1, 1);
		}
		if (
			typeof document !== "undefined" &&
			typeof document.createElement === "function"
		) {
			const canvas = document.createElement("canvas");
			canvas.width = 1;
			canvas.height = 1;
			return canvas;
		}
		throw new Error(
			"VideoTexture requires OffscreenCanvas or document.createElement('canvas') support"
		);
	}

	private _createContext(
		canvas: HTMLCanvasElement | OffscreenCanvas
	): VideoTextureContext2D {
		const context = canvas.getContext("2d", {
			willReadFrequently: true,
		} as CanvasRenderingContext2DSettings);
		if (!this._isVideoTextureContext2D(context)) {
			throw new Error("VideoTexture failed to acquire a 2D canvas context");
		}
		return context;
	}

	private _isVideoTextureContext2D(
		context: unknown
	): context is VideoTextureContext2D {
		if (!context || typeof context !== "object") {
			return false;
		}
		const maybe2D = context as {
			drawImage?: unknown;
			getImageData?: unknown;
		};
		return (
			typeof maybe2D.drawImage === "function" &&
			typeof maybe2D.getImageData === "function"
		);
	}
}
