import { WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE, WEBGPU_MRT_COLOR_TARGET_COUNT } from "./constants";
import { TextureFormat } from "../types";
import type { WebGPUObjectIdentity } from "./WebGPUObjectIdentity";

const WEBGPU_MSAA_SAMPLE_CANDIDATES = [16, 8, 4, 2, 1];
const WEBGPU_DEFAULT_MSAA_SAMPLE_COUNT = 4;

/** @internal WebGPU rendering dependency contract. */
export interface WebGPUMSAAContext {
	readonly sampleCount: number;
	resolveSupportedSampleCount(
		requested: number,
		probeFormats?: readonly GPUTextureFormat[],
	): number;
	fallbackToSingleSample(): boolean;
}

/** @internal Default MSAA context for isolated WebGPU runtime tests. */
export const SINGLE_SAMPLE_WEBGPU_MSAA_CONTEXT: WebGPUMSAAContext = {
	sampleCount: 1,
	resolveSupportedSampleCount: () => 1,
	fallbackToSingleSample: () => false,
};

/** @internal WebGPU MSAA controller host contract. */
export interface WebGPUMSAAControllerHost {
	readonly device: GPUDevice | null;
	readonly canvasFormat: GPUTextureFormat;
	readonly canvasDepthFormat: TextureFormat;
	readonly objectIdentity: WebGPUObjectIdentity;
	onRuntimeFallback(): void;
}

/**
 * Owns WebGPU MSAA selection and device-scoped fallback state.
 *
 * @internal WebGPU backend implementation detail; configure MSAA through
 * `WebGPUBackendOptions.msaaSampleCount` instead.
 */
export class WebGPUMSAAController implements WebGPUMSAAContext {
	private readonly _configuredSampleCount: number;
	private _sampleCount = 1;
	private _runtimeFallbackActive = false;
	private _selectionCache = new Map<string, number>();

	constructor(host: WebGPUMSAAControllerHost, configuredSampleCount?: number) {
		this._host = host;
		this._configuredSampleCount = this._normalizeConfiguredSampleCount(configuredSampleCount);
	}

	private readonly _host: WebGPUMSAAControllerHost;

	public get sampleCount(): number {
		return this._sampleCount;
	}

	public get runtimeFallbackActive(): boolean {
		return this._runtimeFallbackActive;
	}

	/** Activates the configured sample count for a newly initialized device. */
	public activateDevice(): void {
		this._runtimeFallbackActive = false;
		this._sampleCount = this.resolveSupportedSampleCount(this._configuredSampleCount);
	}

	/** Resets device-scoped state while preserving constructor configuration. */
	public resetDevice(): void {
		this._sampleCount = 1;
		this._runtimeFallbackActive = false;
		this._selectionCache.clear();
	}

	public clearCapabilityCache(): void {
		this._selectionCache.clear();
	}

	public fallbackToSingleSample(): boolean {
		if (this._sampleCount === 1) {
			return false;
		}
		this._sampleCount = 1;
		this._runtimeFallbackActive = true;
		this._host.onRuntimeFallback();
		return true;
	}

	public resolveSupportedSampleCount(
		requested: number,
		probeFormats?: readonly GPUTextureFormat[],
	): number {
		const normalized = Math.max(1, Math.floor(requested));
		const device = this._host.device;
		const maxColorAttachments = device?.limits?.maxColorAttachments ?? 0;
		const maxColorAttachmentBytesPerSample =
			device?.limits?.maxColorAttachmentBytesPerSample ?? 0;
		const formats = this._getProbeFormats(probeFormats);
		const cacheKey = [
			`device:${this._host.objectIdentity.getCacheToken(device)}`,
			`requested:${normalized}`,
			`maxAttachments:${maxColorAttachments}`,
			`maxBytes:${maxColorAttachmentBytesPerSample}`,
			`formats:${formats.join(",")}`,
		].join("|");
		const cached = this._selectionCache.get(cacheKey);
		if (cached !== undefined) {
			return cached;
		}

		const candidates = Array.from(
			new Set([
				normalized,
				...WEBGPU_MSAA_SAMPLE_CANDIDATES.filter((sampleCount) => sampleCount <= normalized),
				1,
			]),
		).sort((left, right) => right - left);
		let selected = 1;
		for (const candidate of candidates) {
			if (this._isSampleCountSupported(candidate, formats)) {
				selected = candidate;
				break;
			}
		}
		this._selectionCache.set(cacheKey, selected);
		return selected;
	}

	private _normalizeConfiguredSampleCount(sampleCount: number | undefined): number {
		if (sampleCount === undefined) {
			return WEBGPU_DEFAULT_MSAA_SAMPLE_COUNT;
		}
		if (!Number.isFinite(sampleCount)) {
			throw new Error("WebGPU msaaSampleCount must be a finite number.");
		}
		return Math.max(1, Math.floor(sampleCount));
	}

	private _isSampleCountSupported(
		sampleCount: number,
		probeFormats: readonly GPUTextureFormat[],
	): boolean {
		if (!Number.isInteger(sampleCount) || sampleCount < 1) {
			return false;
		}
		if (sampleCount === 1) {
			return true;
		}
		const device = this._host.device;
		if (!device || typeof device.createTexture !== "function") {
			return false;
		}
		if (
			(device.limits?.maxColorAttachments ?? 0) < WEBGPU_MRT_COLOR_TARGET_COUNT ||
			(device.limits?.maxColorAttachmentBytesPerSample ?? 0) <
				WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE
		) {
			return false;
		}
		for (const format of probeFormats) {
			if (!this._probeSampleCountForFormat(sampleCount, format)) {
				return false;
			}
		}
		return true;
	}

	private _getProbeFormats(probeFormats?: readonly GPUTextureFormat[]): GPUTextureFormat[] {
		const formats = new Set<GPUTextureFormat>([
			this._host.canvasFormat,
			this._host.canvasDepthFormat as GPUTextureFormat,
			TextureFormat.RGBA16Float as GPUTextureFormat,
			TextureFormat.RGBA8Unorm as GPUTextureFormat,
			TextureFormat.Depth32Float as GPUTextureFormat,
		]);
		for (const format of probeFormats ?? []) {
			formats.add(format);
		}
		return Array.from(formats);
	}

	private _probeSampleCountForFormat(sampleCount: number, format: GPUTextureFormat): boolean {
		const device = this._host.device;
		if (!device) {
			return false;
		}
		try {
			const probeTexture = device.createTexture({
				size: [1, 1, 1],
				sampleCount,
				format,
				usage: GPUTextureUsage.RENDER_ATTACHMENT,
				label: `WebGPUMSAAProbe_${format}_${sampleCount}`,
			});
			probeTexture.destroy();
			return true;
		} catch {
			return false;
		}
	}
}
