import { WEBGPU_DEFAULT_SAMPLE_COUNT } from "./constants";
import type { WebGPUObjectIdentity } from "./WebGPUObjectIdentity";

const WEBGPU_SAMPLE_COUNT_CANDIDATES = [16, 8, 4, 2, 1];

/** @internal Device surface required by WebGPU sample-count resolution. */
export interface WebGPUSampleCountResolverHost {
	readonly device: GPUDevice | null;
	readonly objectIdentity: WebGPUObjectIdentity;
}

/** @internal Domain-scoped resolved WebGPU sample-count state. */
export interface WebGPUSampleCountSelection {
	readonly domain: string;
	readonly requestedSampleCount: number;
	readonly sampleCount: number;
	readonly signature: string;
	readonly runtimeFallbackActive: boolean;
}

/** @internal Attachment limits participating in a sample-count selection. */
export interface WebGPUSampleCountProbeConstraints {
	readonly colorAttachmentCount?: number;
	readonly colorAttachmentBytesPerSample?: number;
}

/**
 * Resolves device-supported sample counts without owning one backend-wide count.
 *
 * @internal WebGPU backend implementation detail.
 */
export class WebGPUSampleCountResolver {
	private _capabilityCache = new Map<string, number>();
	private _runtimeFallbacks = new Set<string>();

	public constructor(private readonly _host: WebGPUSampleCountResolverHost) {}

	public normalizeRequestedSampleCount(
		sampleCount: number | undefined,
		label = "WebGPU sampleCount",
	): number {
		if (sampleCount === undefined) {
			return WEBGPU_DEFAULT_SAMPLE_COUNT;
		}
		if (!Number.isFinite(sampleCount)) {
			throw new Error(`${label} must be a finite number.`);
		}
		return Math.max(1, Math.floor(sampleCount));
	}

	public resolveSupportedSampleCount(
		requested: number,
		probeFormats: readonly GPUTextureFormat[],
		constraints: WebGPUSampleCountProbeConstraints = {},
	): number {
		const normalized = this.normalizeRequestedSampleCount(requested, "WebGPU sampleCount");
		const device = this._host.device;
		const formats = Array.from(new Set(probeFormats)).sort();
		const maxColorAttachments = device?.limits?.maxColorAttachments ?? 0;
		const maxColorAttachmentBytesPerSample =
			device?.limits?.maxColorAttachmentBytesPerSample ?? 0;
		const requiredColorAttachmentCount = Math.max(
			0,
			Math.floor(constraints.colorAttachmentCount ?? 0),
		);
		const requiredColorAttachmentBytesPerSample = Math.max(
			0,
			Math.floor(constraints.colorAttachmentBytesPerSample ?? 0),
		);
		const cacheKey = [
			`device:${this._host.objectIdentity.getCacheToken(device)}`,
			`requested:${normalized}`,
			`maxAttachments:${maxColorAttachments}`,
			`maxBytes:${maxColorAttachmentBytesPerSample}`,
			`requiredAttachments:${requiredColorAttachmentCount}`,
			`requiredBytes:${requiredColorAttachmentBytesPerSample}`,
			`formats:${formats.join(",")}`,
		].join("|");
		const cached = this._capabilityCache.get(cacheKey);
		if (cached !== undefined) {
			return cached;
		}

		const candidates = Array.from(
			new Set([
				normalized,
				...WEBGPU_SAMPLE_COUNT_CANDIDATES.filter((count) => count <= normalized),
				1,
			]),
		).sort((left, right) => right - left);
		let selected = 1;
		for (const candidate of candidates) {
			if (this._isSampleCountSupported(candidate, formats, {
				colorAttachmentCount: requiredColorAttachmentCount,
				colorAttachmentBytesPerSample: requiredColorAttachmentBytesPerSample,
			})) {
				selected = candidate;
				break;
			}
		}
		this._capabilityCache.set(cacheKey, selected);
		return selected;
	}

	public resolveDomainSampleCount(
		domain: string,
		requested: number,
		probeFormats: readonly GPUTextureFormat[],
		constraints: WebGPUSampleCountProbeConstraints = {},
	): WebGPUSampleCountSelection {
		const normalized = this.normalizeRequestedSampleCount(requested, "WebGPU sampleCount");
		const formats = Array.from(new Set(probeFormats)).sort();
		const colorAttachmentCount = Math.max(
			0,
			Math.floor(constraints.colorAttachmentCount ?? 0),
		);
		const colorAttachmentBytesPerSample = Math.max(
			0,
			Math.floor(constraints.colorAttachmentBytesPerSample ?? 0),
		);
		const signature = [
			`device:${this._host.objectIdentity.getCacheToken(this._host.device)}`,
			`domain:${domain}`,
			`requested:${normalized}`,
			`formats:${formats.join(",")}`,
			`colorAttachments:${colorAttachmentCount}`,
			`colorBytes:${colorAttachmentBytesPerSample}`,
		].join("|");
		const runtimeFallbackActive = this._runtimeFallbacks.has(signature);
		return {
			domain,
			requestedSampleCount: normalized,
			sampleCount: runtimeFallbackActive
				? 1
				: this.resolveSupportedSampleCount(normalized, formats, {
						colorAttachmentCount,
						colorAttachmentBytesPerSample,
					}),
			signature,
			runtimeFallbackActive,
		};
	}

	public fallbackToSingleSample(signature: string): boolean {
		if (this._runtimeFallbacks.has(signature)) {
			return false;
		}
		this._runtimeFallbacks.add(signature);
		return true;
	}

	public clearCapabilityCache(): void {
		this._capabilityCache.clear();
	}

	public resetDevice(): void {
		this._capabilityCache.clear();
		this._runtimeFallbacks.clear();
	}

	private _isSampleCountSupported(
		sampleCount: number,
		probeFormats: readonly GPUTextureFormat[],
		constraints: Required<WebGPUSampleCountProbeConstraints>,
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
			(device.limits?.maxColorAttachments ?? 0) < constraints.colorAttachmentCount ||
			(device.limits?.maxColorAttachmentBytesPerSample ?? 0) <
				constraints.colorAttachmentBytesPerSample
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

	private _probeSampleCountForFormat(
		sampleCount: number,
		format: GPUTextureFormat,
	): boolean {
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
				label: `WebGPUSampleCountProbe_${format}_${sampleCount}`,
			});
			probeTexture.destroy();
			return true;
		} catch {
			return false;
		}
	}
}
