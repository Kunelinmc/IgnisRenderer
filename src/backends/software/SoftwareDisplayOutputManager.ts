import { Platform, type PlatformMediaQueryList } from "../../foundation/Platform";
import {
	createSDRDisplayOutputState,
	resolveDisplayOutputOptions,
	type DisplayOutputOptions,
	type DisplayOutputState,
	type ResolvedDisplayOutputOptions,
} from "../../rendering/DisplayOutput";

interface ExtendedCanvasRenderingContext2DSettings {
	readonly colorSpace: "display-p3";
	readonly colorType: "float16";
}

interface ExtendedImageDataSettings {
	readonly colorSpace: "display-p3";
	readonly pixelFormat: "rgba-float16";
}

interface ExtendedCanvasContextAttributes {
	readonly colorSpace?: string;
	readonly colorType?: string;
}

type SoftwareCanvasContext = CanvasRenderingContext2D & {
	getContextAttributes?(): ExtendedCanvasContextAttributes;
	getImageData(
		sx: number,
		sy: number,
		sw: number,
		sh: number,
		settings?: ExtendedImageDataSettings,
	): ImageData;
};

/**
 * @internal Owned by `SoftwareBackend`; applications use `Renderer`.
 */
const HDR_PROBE_CONTEXT_SETTINGS: CanvasRenderingContext2DSettings &
	ExtendedCanvasRenderingContext2DSettings = Object.freeze({
	willReadFrequently: true,
	colorSpace: "display-p3",
	colorType: "float16",
});

const HDR_IMAGE_DATA_SETTINGS: ExtendedImageDataSettings = {
	colorSpace: "display-p3",
	pixelFormat: "rgba-float16",
};

/** @internal Resolves Canvas 2D Display HDR capability for SoftwareBackend. */
export class SoftwareDisplayOutputManager {
	private _requested: ResolvedDisplayOutputOptions;
	private _state: DisplayOutputState;
	private _hdrCanvasSupported = false;
	private _hdrConfigurationFailed = false;
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

	/** Probes HDR with a detached canvas without requesting the visible context. */
	public detect(canvas: HTMLCanvasElement): {
		readonly hdrCanvasSupported: boolean;
		readonly contextSettings: CanvasRenderingContext2DSettings;
	} {
		this._hdrConfigurationFailed = false;
		this._hdrCanvasSupported = this._probeHDRCanvas(canvas);
		return {
			hdrCanvasSupported: this._hdrCanvasSupported,
			contextSettings: HDR_PROBE_CONTEXT_SETTINGS,
		};
	}

	/** Verifies and resolves output for the backend-owned visible context. */
	public configure(context: CanvasRenderingContext2D | null): DisplayOutputState {
		this._hdrConfigurationFailed =
			this._hdrCanvasSupported &&
			(!context || !this._hasHDRAttributes(context as SoftwareCanvasContext));
		this._state = this._resolveState();
		return this._state;
	}

	public setRequested(options: DisplayOutputOptions): DisplayOutputState {
		this._requested = resolveDisplayOutputOptions(options, this._requested);
		this._state = this._resolveState();
		return this._state;
	}

	public refreshDynamicRange(): DisplayOutputState {
		this._state = this._resolveState();
		return this._state;
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

	public destroy(): void {
		this.stopObservingDynamicRange();
		this._hdrCanvasSupported = false;
		this._hdrConfigurationFailed = false;
	}

	private _resolveState(): DisplayOutputState {
		if (this._requested.mode === "sdr") {
			return createSDRDisplayOutputState(this._requested);
		}
		if (!this._hdrCanvasSupported || this._hdrConfigurationFailed) {
			return createSDRDisplayOutputState(
				this._requested,
				this._hdrConfigurationFailed
					? "hdr-context-configuration-failed"
					: "canvas-hdr-output-unsupported",
			);
		}
		const displayIsHDR =
			this._mediaQuery?.matches ?? Platform.getHighDynamicRangeMediaQuery()?.matches ?? false;
		if (!displayIsHDR) {
			return createSDRDisplayOutputState(
				this._requested,
				this._requested.mode === "hdr" ? "display-not-hdr-capable" : undefined,
			);
		}
		return Object.freeze({
			requested: this._requested,
			activeDynamicRange: "hdr",
			colorSpace: "display-p3",
		});
	}

	private _probeHDRCanvas(canvas: HTMLCanvasElement): boolean {
		if (typeof Float16Array === "undefined" || typeof ImageData === "undefined") {
			return false;
		}
		const probe = canvas.ownerDocument?.createElement?.("canvas");
		if (!probe || typeof probe.getContext !== "function") return false;
		probe.width = 1;
		probe.height = 1;
		try {
			const context = probe.getContext(
				"2d",
				HDR_PROBE_CONTEXT_SETTINGS as CanvasRenderingContext2DSettings,
			) as SoftwareCanvasContext | null;
			if (!context || !this._hasHDRAttributes(context)) return false;
			const values = new Float16Array([2, 0.25, 0.5, 1]);
			const image = new ImageData(
				values as unknown as ImageDataArray,
				1,
				1,
				HDR_IMAGE_DATA_SETTINGS as ImageDataSettings,
			);
			context.putImageData(image, 0, 0);
			const readback = context.getImageData(0, 0, 1, 1, HDR_IMAGE_DATA_SETTINGS);
			return readback.data instanceof Float16Array && readback.data[0] > 1;
		} catch {
			return false;
		}
	}

	private _hasHDRAttributes(context: SoftwareCanvasContext): boolean {
		if (typeof context.getContextAttributes !== "function") return false;
		const attributes = context.getContextAttributes() as ExtendedCanvasContextAttributes;
		return attributes.colorSpace === "display-p3" && attributes.colorType === "float16";
	}
}

export const SOFTWARE_HDR_IMAGE_DATA_SETTINGS = HDR_IMAGE_DATA_SETTINGS;
