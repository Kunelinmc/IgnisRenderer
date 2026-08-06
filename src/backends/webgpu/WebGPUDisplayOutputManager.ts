import {
	createSDRDisplayOutputState,
	resolveDisplayOutputOptions,
	type DisplayOutputFallbackReason,
	type DisplayOutputOptions,
	type DisplayOutputState,
	type ResolvedDisplayOutputOptions,
} from "../../rendering/DisplayOutput";
import {
	Platform,
	type PlatformMediaQueryList,
} from "../../foundation/Platform";
import type { TextureFormat } from "../types";

export interface WebGPUDisplayOutputConfiguration {
	readonly state: DisplayOutputState;
	readonly canvas: GPUCanvasConfiguration;
	readonly format: TextureFormat;
}

interface ExtendedGPUCanvasConfiguration extends GPUCanvasConfiguration {
	colorSpace?: PredefinedColorSpace;
	toneMapping?: { mode: "standard" | "extended" };
}

/**
 * Resolves WebGPU Display HDR capability and produces verified canvas configs.
 *
 * @internal Owned by `WebGPUBackend`.
 */
export class WebGPUDisplayOutputManager {
	private _requested: ResolvedDisplayOutputOptions;
	private _mediaQuery: PlatformMediaQueryList | null = null;
	private _mediaQueryListener: (() => void) | null = null;

	public constructor(requested: ResolvedDisplayOutputOptions) {
		this._requested = requested;
	}

	public get requested(): ResolvedDisplayOutputOptions {
		return this._requested;
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
		context: GPUCanvasContext,
		device: GPUDevice,
		preferredFormat: TextureFormat,
	): WebGPUDisplayOutputConfiguration {
		if (this._requested.mode === "sdr") {
			return this._configureSDR(context, device, preferredFormat);
		}
		const displayIsHDR = this._mediaQuery?.matches ??
			Platform.getHighDynamicRangeMediaQuery()?.matches ?? false;
		if (!displayIsHDR) {
			return this._configureSDRFallback(
				context,
				device,
				preferredFormat,
				this._requested.mode === "hdr" ?
					"display-not-hdr-capable" : undefined,
			);
		}
		if (typeof context.getConfiguration !== "function") {
			return this._configureSDRFallback(
				context,
				device,
				preferredFormat,
				"canvas-tone-mapping-unsupported",
			);
		}

		const configuration: ExtendedGPUCanvasConfiguration = {
			device,
			format: "rgba16float",
			colorSpace: "display-p3",
			toneMapping: { mode: "extended" },
			alphaMode: "premultiplied",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
		};
		try {
			context.configure(configuration);
			const active = context.getConfiguration() as
				| ExtendedGPUCanvasConfiguration
				| null;
			if (
				active?.format !== "rgba16float" ||
				active.colorSpace !== "display-p3" ||
				active.toneMapping?.mode !== "extended"
			) {
				return this._configureSDRFallback(
					context,
					device,
					preferredFormat,
					"hdr-context-configuration-failed",
				);
			}
			const state: DisplayOutputState = Object.freeze({
				requested: this._requested,
				activeDynamicRange: "hdr",
				colorSpace: "display-p3",
			});
			return Object.freeze({
				state,
				canvas: configuration,
				format: "rgba16float" as TextureFormat,
			});
		} catch {
			return this._configureSDRFallback(
				context,
				device,
				preferredFormat,
				"hdr-context-configuration-failed",
			);
		}
	}

	private _configureSDR(
		context: GPUCanvasContext,
		device: GPUDevice,
		preferredFormat: TextureFormat,
		fallbackReason?: DisplayOutputFallbackReason,
	): WebGPUDisplayOutputConfiguration {
		const configuration: ExtendedGPUCanvasConfiguration = {
			device,
			format: preferredFormat as GPUTextureFormat,
			colorSpace: "srgb",
			alphaMode: "premultiplied",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
		};
		if (typeof context.getConfiguration === "function") {
			configuration.toneMapping = { mode: "standard" };
		}
		try {
			context.configure(configuration);
		} catch (error) {
			if (!configuration.toneMapping) throw error;
			delete configuration.toneMapping;
			context.configure(configuration);
		}
		return Object.freeze({
			state: createSDRDisplayOutputState(this._requested, fallbackReason),
			canvas: configuration,
			format: preferredFormat,
		});
	}

	private _configureSDRFallback(
		context: GPUCanvasContext,
		device: GPUDevice,
		preferredFormat: TextureFormat,
		fallbackReason?: DisplayOutputFallbackReason,
	): WebGPUDisplayOutputConfiguration {
		try {
			context.unconfigure();
		} catch {
			// A failed HDR configure may leave no active configuration.
		}
		return this._configureSDR(
			context,
			device,
			preferredFormat,
			fallbackReason,
		);
	}
}
