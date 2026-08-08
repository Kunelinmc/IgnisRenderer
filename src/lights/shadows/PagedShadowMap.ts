import { LightType } from "..";
import { ShadowMapBase } from "./ShadowMapBase";
import type {
	CascadedShadowMapDefaults,
	PagedShadowFeedbackMode,
	PreparedPagedShadowSettings,
	PagedShadowMapOptions,
	ShadowBoundLightType,
	ShadowProjectionSnapshot,
	ShadowStoragePreference,
} from "./types";

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
	private _virtualResolution: number;
	private _pageSize: number;
	private _physicalPageCount: number;
	private _clipmapLevels: number;
	private _maxPagesPerFrame: number;
	private _cacheFrames: number;
	private _feedbackMode: PagedShadowFeedbackMode;
	private readonly _cascadeCounts: CascadedShadowMapDefaults;
	private _lambda: number;
	private _maxDistance?: number;
	private _blendRatio: number;
	private _stabilize: boolean;

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
		this._virtualResolution = normalizePowerOfTwo(
			options.virtualResolution,
			DEFAULT_VIRTUAL_RESOLUTION
		);
		this._pageSize = normalizePowerOfTwo(options.pageSize, DEFAULT_PAGE_SIZE);
		this._physicalPageCount = Math.max(
			1,
			Math.floor(resolveFinite(options.physicalPageCount, DEFAULT_PHYSICAL_PAGE_COUNT))
		);
		this._clipmapLevels = Math.max(
			1,
			Math.floor(resolveFinite(options.clipmapLevels, DEFAULT_CLIPMAP_LEVELS))
		);
		this._maxPagesPerFrame = Math.max(
			1,
			Math.floor(resolveFinite(options.maxPagesPerFrame, DEFAULT_MAX_PAGES_PER_FRAME))
		);
		this._cacheFrames = Math.max(
			0,
			Math.floor(resolveFinite(options.cacheFrames, DEFAULT_CACHE_FRAMES))
		);
		this._feedbackMode = options.feedbackMode ?? "conservative";
		this._cascadeCounts = this._createObservableSettings({
			directional: clampCascadeCount(
				resolveFinite(options.cascadeCounts?.directional, DEFAULT_CASCADE_COUNTS.directional)
			),
			spot: clampCascadeCount(
				resolveFinite(options.cascadeCounts?.spot, DEFAULT_CASCADE_COUNTS.spot)
			),
			point: clampCascadeCount(
				resolveFinite(options.cascadeCounts?.point, DEFAULT_CASCADE_COUNTS.point)
			),
		}, {
			directional: (value) => clampCascadeCount(
				resolveFinite(value, DEFAULT_CASCADE_COUNTS.directional)
			),
			spot: (value) => clampCascadeCount(
				resolveFinite(value, DEFAULT_CASCADE_COUNTS.spot)
			),
			point: (value) => clampCascadeCount(
				resolveFinite(value, DEFAULT_CASCADE_COUNTS.point)
			),
		});
		this._lambda = clamp01(resolveFinite(options.lambda, 0.65));
		this._maxDistance =
			typeof options.maxDistance === "number" &&
			Number.isFinite(options.maxDistance) ?
				Math.max(0.01, options.maxDistance)
			:	undefined;
		this._blendRatio = clamp01(resolveFinite(options.blendRatio, 0.1));
		this._stabilize = options.stabilize !== false;
	}

	public get virtualResolution(): number {
		return this._virtualResolution;
	}

	public set virtualResolution(value: number) {
		this._setNumber("virtualResolution", normalizePowerOfTwo(value, DEFAULT_VIRTUAL_RESOLUTION));
	}

	public get pageSize(): number {
		return this._pageSize;
	}

	public set pageSize(value: number) {
		this._setNumber("pageSize", normalizePowerOfTwo(value, DEFAULT_PAGE_SIZE));
	}

	public get physicalPageCount(): number {
		return this._physicalPageCount;
	}

	public set physicalPageCount(value: number) {
		this._setNumber(
			"physicalPageCount",
			Math.max(1, Math.floor(resolveFinite(value, DEFAULT_PHYSICAL_PAGE_COUNT)))
		);
	}

	public get clipmapLevels(): number {
		return this._clipmapLevels;
	}

	public set clipmapLevels(value: number) {
		this._setNumber(
			"clipmapLevels",
			Math.max(1, Math.floor(resolveFinite(value, DEFAULT_CLIPMAP_LEVELS)))
		);
	}

	public get maxPagesPerFrame(): number {
		return this._maxPagesPerFrame;
	}

	public set maxPagesPerFrame(value: number) {
		this._setNumber(
			"maxPagesPerFrame",
			Math.max(1, Math.floor(resolveFinite(value, DEFAULT_MAX_PAGES_PER_FRAME)))
		);
	}

	public get cacheFrames(): number {
		return this._cacheFrames;
	}

	public set cacheFrames(value: number) {
		this._setNumber(
			"cacheFrames",
			Math.max(0, Math.floor(resolveFinite(value, DEFAULT_CACHE_FRAMES)))
		);
	}

	public get feedbackMode(): PagedShadowFeedbackMode {
		return this._feedbackMode;
	}

	public set feedbackMode(value: PagedShadowFeedbackMode) {
		const normalized = value === "screen-feedback" ? value : "conservative";
		if (this._feedbackMode === normalized) return;
		this._feedbackMode = normalized;
		this._markDefinitionChanged();
	}

	public get cascadeCounts(): CascadedShadowMapDefaults {
		return this._cascadeCounts;
	}

	public set cascadeCounts(value: Partial<CascadedShadowMapDefaults>) {
		this._assignObservableSettings(this._cascadeCounts, value);
	}

	public get lambda(): number {
		return this._lambda;
	}

	public set lambda(value: number) {
		this._setNumber("lambda", clamp01(resolveFinite(value, 0.65)));
	}

	public get maxDistance(): number | undefined {
		return this._maxDistance;
	}

	public set maxDistance(value: number | undefined) {
		const normalized = typeof value === "number" && Number.isFinite(value) ?
			Math.max(0.01, value)
		: undefined;
		if (Object.is(this._maxDistance, normalized)) return;
		this._maxDistance = normalized;
		this._markDefinitionChanged();
	}

	public get blendRatio(): number {
		return this._blendRatio;
	}

	public set blendRatio(value: number) {
		this._setNumber("blendRatio", clamp01(resolveFinite(value, 0.1)));
	}

	public get stabilize(): boolean {
		return this._stabilize;
	}

	public set stabilize(value: boolean) {
		const normalized = value !== false;
		if (this._stabilize === normalized) return;
		this._stabilize = normalized;
		this._markDefinitionChanged();
	}

	public override update(options: Partial<PagedShadowMapOptions>): this {
		return this._runDefinitionUpdate(() => {
			super.update(options);
			if (options.virtualResolution !== undefined) {
				this.virtualResolution = options.virtualResolution;
			}
			if (options.pageSize !== undefined) this.pageSize = options.pageSize;
			if (options.physicalPageCount !== undefined) {
				this.physicalPageCount = options.physicalPageCount;
			}
			if (options.clipmapLevels !== undefined) {
				this.clipmapLevels = options.clipmapLevels;
			}
			if (options.maxPagesPerFrame !== undefined) {
				this.maxPagesPerFrame = options.maxPagesPerFrame;
			}
			if (options.cacheFrames !== undefined) this.cacheFrames = options.cacheFrames;
			if (options.feedbackMode !== undefined) this.feedbackMode = options.feedbackMode;
			if (options.cascadeCounts !== undefined) {
				this.cascadeCounts = options.cascadeCounts;
			}
			if (options.lambda !== undefined) this.lambda = options.lambda;
			if ("maxDistance" in options) this.maxDistance = options.maxDistance;
			if (options.blendRatio !== undefined) this.blendRatio = options.blendRatio;
			if (options.stabilize !== undefined) this.stabilize = options.stabilize;
		});
	}

	protected override get storagePreference(): ShadowStoragePreference {
		return "paged";
	}

	protected override createProjectionSnapshot(): ShadowProjectionSnapshot {
		return {
			technique: "cascaded",
			cascadeCounts: Object.freeze({ ...this.cascadeCounts }),
			lambda: this.lambda,
			maxDistance: this.maxDistance,
			blendRatio: this.blendRatio,
			stabilize: this.stabilize,
		};
	}

	protected override createPagedSettingsSnapshot(): Readonly<PreparedPagedShadowSettings> {
		return Object.freeze({
			virtualResolution: this.virtualResolution,
			pageSize: this.pageSize,
			pageGridSize: Math.max(1, Math.floor(this.virtualResolution / this.pageSize)),
			physicalPageCount: this.physicalPageCount,
			clipmapLevels: this.clipmapLevels,
			maxPagesPerFrame: this.maxPagesPerFrame,
			cacheFrames: this.cacheFrames,
			feedbackMode: this.feedbackMode,
		});
	}

	private _setNumber(
		key:
			| "virtualResolution"
			| "pageSize"
			| "physicalPageCount"
			| "clipmapLevels"
			| "maxPagesPerFrame"
			| "cacheFrames"
			| "lambda"
			| "blendRatio",
		value: number
	): void {
		const privateKey = `_${key}` as
			| "_virtualResolution"
			| "_pageSize"
			| "_physicalPageCount"
			| "_clipmapLevels"
			| "_maxPagesPerFrame"
			| "_cacheFrames"
			| "_lambda"
			| "_blendRatio";
		if (Object.is(this[privateKey], value)) return;
		this[privateKey] = value;
		this._markDefinitionChanged();
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
