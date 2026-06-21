import { LightType } from "..";
import type { ShadowConfig } from "./ShadowMapping";
import { ShadowMapBase } from "./ShadowMapBase";
import type {
	CascadedShadowMapDefaults,
	PagedShadowFeedbackMode,
	PagedShadowMapOptions,
	ShadowBoundLightType,
} from "./types";
import type { PagedShadowLayoutMetadata } from "./ShadowMapping";

const DEFAULT_CASCADE_COUNTS: CascadedShadowMapDefaults = {
	directional: 4,
	spot: 1,
	point: 1,
};
const DEFAULT_VIRTUAL_RESOLUTION = 16384;
const DEFAULT_PAGE_SIZE = 128;
const DEFAULT_PHYSICAL_PAGE_COUNT = 2048;
const DEFAULT_CLIPMAP_LEVELS = 6;
const DEFAULT_MAX_PAGES_PER_FRAME = 256;
const DEFAULT_CACHE_FRAMES = 120;

/**
 * Describes a first-version paged shadow map request.
 *
 * Paged shadows use CSM-compatible legacy metadata in v1 so unsupported
 * backends and the WebGPU stub path can keep rendering atlas shadows.
 */
export class PagedShadowMap extends ShadowMapBase {
	public override readonly kind = "paged-shadow" as const;
	public virtualResolution: number;
	public pageSize: number;
	public physicalPageCount: number;
	public clipmapLevels: number;
	public maxPagesPerFrame: number;
	public cacheFrames: number;
	public feedbackMode: PagedShadowFeedbackMode;
	public cascadeCounts: CascadedShadowMapDefaults;
	public lambda: number;
	public maxDistance?: number;
	public blendRatio: number;
	public stabilize: boolean;

	/**
	 * Creates a paged shadow map request.
	 *
	 * @param options Paged shadow budget, fallback CSM, sampling, and bias options.
	 * @returns A shadow map object that can be bound through `Scene.shadows`.
	 * @remarks The v1 implementation does not allocate virtual shadow pages; it
	 * stores intent and emits CSM-compatible fallback metadata.
	 */
	public constructor(options: PagedShadowMapOptions = {}) {
		super(options);
		this.virtualResolution = normalizePowerOfTwo(
			options.virtualResolution,
			DEFAULT_VIRTUAL_RESOLUTION
		);
		this.pageSize = normalizePowerOfTwo(options.pageSize, DEFAULT_PAGE_SIZE);
		this.physicalPageCount = Math.max(
			1,
			Math.floor(resolveFinite(options.physicalPageCount, DEFAULT_PHYSICAL_PAGE_COUNT))
		);
		this.clipmapLevels = Math.max(
			1,
			Math.floor(resolveFinite(options.clipmapLevels, DEFAULT_CLIPMAP_LEVELS))
		);
		this.maxPagesPerFrame = Math.max(
			1,
			Math.floor(resolveFinite(options.maxPagesPerFrame, DEFAULT_MAX_PAGES_PER_FRAME))
		);
		this.cacheFrames = Math.max(
			0,
			Math.floor(resolveFinite(options.cacheFrames, DEFAULT_CACHE_FRAMES))
		);
		this.feedbackMode = options.feedbackMode ?? "conservative";
		this.cascadeCounts = {
			directional: clampCascadeCount(
				resolveFinite(options.cascadeCounts?.directional, DEFAULT_CASCADE_COUNTS.directional)
			),
			spot: clampCascadeCount(
				resolveFinite(options.cascadeCounts?.spot, DEFAULT_CASCADE_COUNTS.spot)
			),
			point: clampCascadeCount(
				resolveFinite(options.cascadeCounts?.point, DEFAULT_CASCADE_COUNTS.point)
			),
		};
		this.lambda = clamp01(resolveFinite(options.lambda, 0.65));
		this.maxDistance =
			typeof options.maxDistance === "number" &&
			Number.isFinite(options.maxDistance) ?
				Math.max(0.01, options.maxDistance)
			:	undefined;
		this.blendRatio = clamp01(resolveFinite(options.blendRatio, 0.1));
		this.stabilize = options.stabilize !== false;
	}

	/**
	 * @internal Shadow scheduling hook used by `ShadowManager`.
	 *
	 * @param lightType Light type for the current binding.
	 * @returns The fallback cascade count requested for that light type.
	 */
	public override resolveCascadeCount(lightType: LightType): number {
		return this.getCascadeCountForBoundType(this.resolveBoundLightType(lightType));
	}

	/**
	 * Resolves the fallback cascade count for a normalized shadow light type.
	 *
	 * @param boundType Normalized light type used by shadow scheduling.
	 * @returns The configured cascade count, clamped to the supported CSM range.
	 */
	public getCascadeCountForBoundType(boundType: ShadowBoundLightType): number {
		switch (boundType) {
			case "directional":
				return this.cascadeCounts.directional;
			case "spot":
				return this.cascadeCounts.spot;
			case "point":
				return this.cascadeCounts.point;
			default:
				return 1;
		}
	}

	/**
	 * Estimates scheduling cost from the physical page budget.
	 *
	 * @returns A normalized cost based on physical pages and page size.
	 * @remarks Virtual resolution is intentionally ignored in v1 so a large
	 * virtual address space does not make the fallback path appear unaffordable.
	 */
	public override estimateCost(): number {
		const normalizedPageSize = this.pageSize / DEFAULT_PAGE_SIZE;
		const normalizedPageCount =
			this.physicalPageCount / DEFAULT_PHYSICAL_PAGE_COUNT;
		return Math.max(0.01, normalizedPageCount * normalizedPageSize * normalizedPageSize);
	}

	/**
	 * Builds the CSM-compatible fallback configuration used by legacy render paths.
	 *
	 * @param lightType Light type for the current binding.
	 * @param overrides Optional budget-selected size or cascade count.
	 * @returns A CSM shadow config that existing atlas renderers can consume.
	 */
	public override toLegacyShadowConfig(
		lightType: LightType,
		overrides?: {
			size?: number;
			cascadeCount?: number;
		}
	): ShadowConfig {
		const cascadeCount =
			overrides?.cascadeCount ?? this.resolveCascadeCount(lightType);
		return this.createCSMLegacyConfig(cascadeCount, {
			size: overrides?.size,
			lambda: this.lambda,
			maxDistance: this.maxDistance,
			blendRatio: this.blendRatio,
			stabilize: this.stabilize,
		});
	}

	/**
	 * @internal Shadow scheduling metadata consumed by `ShadowManager`.
	 */
	public toLayoutMetadata(): PagedShadowLayoutMetadata {
		return {
			virtualResolution: this.virtualResolution,
			pageSize: this.pageSize,
			pageGridSize: Math.max(1, Math.floor(this.virtualResolution / this.pageSize)),
			physicalPageCount: this.physicalPageCount,
			maxPagesPerFrame: this.maxPagesPerFrame,
			clipmapLevels: this.clipmapLevels,
			cacheFrames: this.cacheFrames,
			feedbackMode: this.feedbackMode,
		};
	}
}

function resolveFinite(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return value;
}

function clamp01(value: number): number {
	return Math.max(0, Math.min(1, value));
}

function clampCascadeCount(value: number): number {
	return Math.max(1, Math.min(4, Math.floor(value)));
}

function normalizePowerOfTwo(value: unknown, fallback: number): number {
	const resolved = Math.max(1, Math.floor(resolveFinite(value, fallback)));
	let power = 1;
	while (power < resolved) {
		power <<= 1;
	}
	const lower = power >> 1;
	if (lower >= 1 && resolved - lower < power - resolved) {
		return lower;
	}
	return power;
}
