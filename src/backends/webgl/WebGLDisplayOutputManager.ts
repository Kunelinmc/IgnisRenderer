import { Platform, type PlatformMediaQueryList } from "../../foundation/Platform";
import {
	createSDRDisplayOutputState,
	resolveDisplayOutputOptions,
	type DisplayOutputFallbackReason,
	type DisplayOutputOptions,
	type DisplayOutputState,
	type ResolvedDisplayOutputOptions,
} from "../../rendering/DisplayOutput";

interface ExtendedWebGL2RenderingContext extends WebGL2RenderingContext {
	readonly drawingBufferFormat?: number;
	drawingBufferColorSpace: PredefinedColorSpace;
	drawingBufferStorage?: (sizedFormat: number, width: number, height: number) => void;
}

/** @internal Resolves verified WebGL Display HDR output and SDR fallback. */
export class WebGLDisplayOutputManager {
	private _requested: ResolvedDisplayOutputOptions;
	private _state: DisplayOutputState;
	private _mediaQuery: PlatformMediaQueryList | null = null;
	private _mediaQueryListener: (() => void) | null = null;

	constructor(requested: ResolvedDisplayOutputOptions) {
		this._requested = requested;
		this._state = createSDRDisplayOutputState(requested);
	}

	public get requested(): ResolvedDisplayOutputOptions {
		return this._requested;
	}

	public get state(): DisplayOutputState {
		return this._state;
	}

	public setRequested(options: DisplayOutputOptions): ResolvedDisplayOutputOptions {
		this._requested = resolveDisplayOutputOptions(options, this._requested);
		return this._requested;
	}

	public observeDynamicRange(onChange: () => void): void {
		this.stopObservingDynamicRange();
		const mediaQuery = Platform.getHighDynamicRangeMediaQuery();
		if (!mediaQuery) return;
		const listener = (): void => onChange();
		mediaQuery.addEventListener?.("change", listener);
		this._mediaQuery = mediaQuery;
		this._mediaQueryListener = listener;
	}

	public stopObservingDynamicRange(): void {
		if (this._mediaQuery && this._mediaQueryListener) {
			this._mediaQuery.removeEventListener?.("change", this._mediaQueryListener);
		}
		this._mediaQuery = null;
		this._mediaQueryListener = null;
	}

	public configure(
		context: WebGL2RenderingContext,
		width: number,
		height: number,
	): DisplayOutputState {
		const gl = context as ExtendedWebGL2RenderingContext;
		const resolvedWidth = Math.max(1, Math.floor(width));
		const resolvedHeight = Math.max(1, Math.floor(height));
		if (this._requested.mode === "sdr") {
			return this._configureSDR(gl, resolvedWidth, resolvedHeight);
		}

		const displayIsHDR =
			this._mediaQuery?.matches ?? Platform.getHighDynamicRangeMediaQuery()?.matches ?? false;
		if (!displayIsHDR) {
			return this._configureSDR(
				gl,
				resolvedWidth,
				resolvedHeight,
				this._requested.mode === "hdr" ? "display-not-hdr-capable" : undefined,
			);
		}
		if (!this._supportsHDRDrawingBuffer(gl)) {
			return this._configureSDR(
				gl,
				resolvedWidth,
				resolvedHeight,
				"canvas-hdr-output-unsupported",
			);
		}

		try {
			this._clearErrors(gl);
			gl.drawingBufferColorSpace = "display-p3";
			gl.drawingBufferStorage!(gl.RGBA16F, resolvedWidth, resolvedHeight);
			if (
				this._readError(gl) !== gl.NO_ERROR ||
				gl.drawingBufferFormat !== gl.RGBA16F ||
				gl.drawingBufferColorSpace !== "display-p3"
			) {
				return this._configureSDR(
					gl,
					resolvedWidth,
					resolvedHeight,
					"hdr-context-configuration-failed",
				);
			}
			this._state = Object.freeze({
				requested: this._requested,
				activeDynamicRange: "hdr",
				colorSpace: "display-p3",
			});
			return this._state;
		} catch {
			return this._configureSDR(
				gl,
				resolvedWidth,
				resolvedHeight,
				"hdr-context-configuration-failed",
			);
		}
	}

	public destroy(): void {
		this.stopObservingDynamicRange();
	}

	private _supportsHDRDrawingBuffer(gl: ExtendedWebGL2RenderingContext): boolean {
		return (
			typeof gl.drawingBufferStorage === "function" &&
			typeof gl.drawingBufferFormat === "number" &&
			"drawingBufferColorSpace" in gl &&
			Boolean(gl.getExtension("EXT_color_buffer_float"))
		);
	}

	private _configureSDR(
		gl: ExtendedWebGL2RenderingContext,
		width: number,
		height: number,
		fallbackReason?: DisplayOutputFallbackReason,
	): DisplayOutputState {
		this._clearErrors(gl);
		try {
			if ("drawingBufferColorSpace" in gl) {
				gl.drawingBufferColorSpace = "srgb";
			}
			if (typeof gl.drawingBufferStorage === "function") {
				gl.drawingBufferStorage(gl.RGBA8, width, height);
				if (
					this._readError(gl) !== gl.NO_ERROR ||
					(typeof gl.drawingBufferFormat === "number" &&
						gl.drawingBufferFormat !== gl.RGBA8)
				) {
					throw new Error("WebGL failed to restore its RGBA8 drawing buffer.");
				}
			}
		} catch (error) {
			throw new Error("WebGL failed to restore SDR presentation.", { cause: error });
		}
		this._state = createSDRDisplayOutputState(this._requested, fallbackReason);
		return this._state;
	}

	private _clearErrors(gl: WebGL2RenderingContext): void {
		for (let attempt = 0; attempt < 8; attempt++) {
			if (this._readError(gl) === gl.NO_ERROR) return;
		}
	}

	private _readError(gl: WebGL2RenderingContext): number {
		const getError = (gl as Partial<WebGL2RenderingContext>).getError;
		return typeof getError === "function" ? getError.call(gl) : gl.NO_ERROR;
	}
}
