/**
 * Shared shadow metadata definitions.
 *
 * Runtime shadow buffers and sampling logic are implemented inside each backend.
 */

import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import type { PagedShadowFeedbackMode } from "./types";

export interface ShadowParams {
	shadowBias?: number;
	shadowSlopeBias?: number;
	shadowNormalBias?: number;
	shadowNormalBiasMin?: number;
	shadowTexelBias?: number;
	shadowMaxBias?: number;
	shadowPCF?: number;
	shadowStrength?: number;
	shadowRadius?: number;
	shadowSamples?: number;
	shadowSearchSamples?: number;
	[key: string]: unknown;
}

const DEFAULT_SHADOW_PARAMS: ShadowParams = {
	shadowBias: 0,
	shadowSlopeBias: 0.01,
	shadowNormalBias: 0.01,
	shadowNormalBiasMin: 0.01,
	shadowTexelBias: 1.0,
	shadowMaxBias: 0.1,
	shadowPCF: 1,
	shadowStrength: 1,
};

export type ShadowStrategyType = "single-map" | "csm";
export type CascadedCascadeCount = 1 | 2 | 3 | 4;
export type ShadowStorageMode = "atlas" | "paged";
export type ShadowRegionKind = "single" | "cascade" | "paged-page";

interface BaseShadowConfig {
	strategy: ShadowStrategyType;
	size?: number;
	params?: ShadowParams;
	priority?: number;
}

export interface SingleMapShadowConfig extends BaseShadowConfig {
	strategy: "single-map";
}

export interface CascadedShadowConfig extends BaseShadowConfig {
	strategy: "csm";
	cascadeCount?: CascadedCascadeCount;
	splitMode?: "practical";
	lambda?: number;
	maxDistance?: number;
	blendRatio?: number;
	stabilize?: boolean;
}

export type ShadowConfig = SingleMapShadowConfig | CascadedShadowConfig;

export const DEFAULT_SINGLE_MAP_SHADOW_CONFIG: SingleMapShadowConfig = {
	strategy: "single-map",
	size: 1024,
	priority: 0,
};

export const DEFAULT_CASCADED_SHADOW_CONFIG: CascadedShadowConfig = {
	strategy: "csm",
	size: 1024,
	priority: 0,
	cascadeCount: 4,
	splitMode: "practical",
	lambda: 0.65,
	maxDistance: undefined,
	blendRatio: 0.1,
	stabilize: true,
};

function clampFinite(value: unknown, fallback: number, min = -Infinity, max = Infinity): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(max, Math.max(min, value));
}

function resolveCascadeCount(value: unknown, fallback: CascadedCascadeCount): CascadedCascadeCount {
	if (value === 1 || value === 2 || value === 3 || value === 4) {
		return value;
	}
	return fallback;
}

export function normalizeShadowConfig(config?: ShadowConfig): ShadowConfig {
	if (!config || config.strategy === "single-map") {
		const source = config ?? DEFAULT_SINGLE_MAP_SHADOW_CONFIG;
		return {
			strategy: "single-map",
			size: Math.max(1, Math.floor(clampFinite(source.size, 1024, 1))),
			params: {
				...DEFAULT_SHADOW_PARAMS,
				...(source.params ?? {}),
			},
			priority: clampFinite(source.priority, 0),
		};
	}

	const source = config;
	const maxDistance =
		typeof source.maxDistance === "number" && Number.isFinite(source.maxDistance) ?
			Math.max(0.01, source.maxDistance)
		: 	undefined;

	return {
		strategy: "csm",
		size: Math.max(1, Math.floor(clampFinite(source.size, 1024, 1))),
		params: {
			...DEFAULT_SHADOW_PARAMS,
			...(source.params ?? {}),
		},
		priority: clampFinite(source.priority, 0),
		cascadeCount: resolveCascadeCount(source.cascadeCount, 4),
		splitMode: "practical",
		lambda: clampFinite(source.lambda, 0.65, 0, 1),
		maxDistance,
		blendRatio: clampFinite(source.blendRatio, 0.1, 0, 1),
		stabilize: source.stabilize !== false,
	};
}

export function shadowConfigSignature(config: ShadowConfig): string {
	const normalized = normalizeShadowConfig(config);
	if (normalized.strategy === "single-map") {
		return JSON.stringify({
			strategy: normalized.strategy,
			size: normalized.size,
			priority: normalized.priority,
			params: normalized.params,
		});
	}

	return JSON.stringify({
		strategy: normalized.strategy,
		size: normalized.size,
		priority: normalized.priority,
		params: normalized.params,
		cascadeCount: normalized.cascadeCount,
		splitMode: normalized.splitMode,
		lambda: normalized.lambda,
		maxDistance: normalized.maxDistance,
		blendRatio: normalized.blendRatio,
		stabilize: normalized.stabilize,
	});
}

export class ShadowMap {
	public size: number;
	public params: ShadowParams;
	public viewMatrix: Matrix4 | null = null;
	public projectionMatrix: Matrix4 | null = null;
	public viewProjectionMatrix: Matrix4 | null = null;
	public latestLightDir: IVector3 = { x: 0, y: -1, z: 0 };
	public stabilizedBoundsRadius: number | null = null;
	public csmStableCenterLightX: number | null = null;
	public csmStableCenterLightY: number | null = null;
	public csmStableLightDir: IVector3 | null = null;

	constructor(size = 1024, params: ShadowParams = {}) {
		this.size = size;
		this.params = {
			...DEFAULT_SHADOW_PARAMS,
			...params,
		};
	}
}

export interface ShadowAtlasRect {
	offsetX: number;
	offsetY: number;
	size: number;
	tileSize: number;
	localTileX: number;
	localTileY: number;
	localTileSpan: number;
}

export interface PagedShadowRegionMetadata {
	level: number;
	pageX: number;
	pageY: number;
	pageSize: number;
	virtualResolution: number;
	physicalPageIndex: number;
	dirty: boolean;
	resident: boolean;
}

export interface PagedShadowLayoutMetadata {
	virtualResolution: number;
	pageSize: number;
	pageGridSize: number;
	physicalPageCount: number;
	maxPagesPerFrame: number;
	clipmapLevels: number;
	cacheFrames: number;
	feedbackMode: PagedShadowFeedbackMode;
	pageTableBase?: number;
	pageTableCascadeStride?: number;
	physicalAtlasSize?: number;
	physicalGridSize?: number;
	physicalPageSize?: number;
}

export interface ShadowRegionDescriptor {
	id: number;
	kind: ShadowRegionKind;
	view: Matrix4 | null;
	projection: Matrix4 | null;
	viewProjection: Matrix4 | null;
	lightDir: IVector3;
	splitNear: number;
	splitFar: number;
	atlasRect?: ShadowAtlasRect;
	sourceSliceIndex: number;
	paged?: PagedShadowRegionMetadata;
}

export interface ShadowLayout {
	storageMode: ShadowStorageMode;
	regions: ShadowRegionDescriptor[];
	paged?: PagedShadowLayoutMetadata;
}

export interface ShadowSlice {
	index: number;
	shadowMap: ShadowMap;
	splitNear: number;
	splitFar: number;
	atlasRect: ShadowAtlasRect | null;
}

export interface ShadowRenderSet {
	requestedStrategyType: ShadowStrategyType;
	effectiveStrategyType: ShadowStrategyType;
	resolvedConfig: ShadowConfig;
	configSignature: string;
	size: number;
	slices: ShadowSlice[];
	storageMode: ShadowStorageMode;
	layout: ShadowLayout;
	metadataVersion: number;
}

export interface ShadowRenderSetOptions {
	storageMode?: ShadowStorageMode;
	paged?: PagedShadowLayoutMetadata;
}

function isShadowMapLike(value: unknown): value is ShadowMap {
	if (!value || typeof value !== "object") {
		return false;
	}
	return (
		"size" in value &&
		"params" in value &&
		"viewProjectionMatrix" in value
	);
}

function resolveSliceCount(config: ShadowConfig): number {
	if (config.strategy !== "csm") {
		return 1;
	}
	return config.cascadeCount ?? DEFAULT_CASCADED_SHADOW_CONFIG.cascadeCount ?? 4;
}

function resolveSliceSize(config: ShadowConfig, baseSize: number): number {
	if (config.strategy !== "csm") {
		return baseSize;
	}
	return Math.max(1, Math.floor(baseSize / 2));
}

function createShadowSlice(index: number, sliceSize: number, params: ShadowParams): ShadowSlice {
	return {
		index,
		shadowMap: new ShadowMap(sliceSize, params),
		splitNear: 0,
		splitFar: 0,
		atlasRect: null,
	};
}

function resolveRegionKind(
	config: ShadowConfig,
	storageMode: ShadowStorageMode
): ShadowRegionKind {
	if (storageMode === "paged") {
		return "paged-page";
	}
	return config.strategy === "csm" ? "cascade" : "single";
}

export function syncShadowLayout(renderSet: ShadowRenderSet): ShadowLayout {
	const regionKind = resolveRegionKind(
		renderSet.resolvedConfig,
		renderSet.storageMode
	);
	const regions = renderSet.slices.map((slice) => ({
		id: slice.index,
		kind: regionKind,
		view: slice.shadowMap.viewMatrix,
		projection: slice.shadowMap.projectionMatrix,
		viewProjection: slice.shadowMap.viewProjectionMatrix,
		lightDir: slice.shadowMap.latestLightDir,
		splitNear: slice.splitNear,
		splitFar: slice.splitFar,
		atlasRect: slice.atlasRect ?? undefined,
		sourceSliceIndex: slice.index,
	}));
	renderSet.layout = {
		storageMode: renderSet.storageMode,
		regions,
		paged: renderSet.storageMode === "paged" ? renderSet.layout.paged : undefined,
	};
	return renderSet.layout;
}

export function createShadowRenderSet(
	config?: ShadowConfig,
	options: ShadowRenderSetOptions = {}
): ShadowRenderSet {
	const resolvedConfig = normalizeShadowConfig(config);
	const sliceCount = resolveSliceCount(resolvedConfig);
	const baseSize = resolvedConfig.size ?? 1024;
	const sliceSize = resolveSliceSize(resolvedConfig, baseSize);
	const slices: ShadowSlice[] = [];

	for (let index = 0; index < sliceCount; index++) {
		slices.push(createShadowSlice(index, sliceSize, resolvedConfig.params ?? {}));
	}

	const storageMode = options.storageMode ?? "atlas";
	const renderSet: ShadowRenderSet = {
		requestedStrategyType: resolvedConfig.strategy,
		effectiveStrategyType: resolvedConfig.strategy,
		resolvedConfig,
		configSignature: shadowConfigSignature(resolvedConfig),
		size: baseSize,
		slices,
		storageMode,
		layout: {
			storageMode,
			regions: [],
			paged: storageMode === "paged" ? options.paged : undefined,
		},
		metadataVersion: 0,
	};
	syncShadowLayout(renderSet);
	return renderSet;
}

export function ensureShadowRenderSetMatchesConfig(
	renderSet: ShadowRenderSet,
	config?: ShadowConfig,
	options: ShadowRenderSetOptions = {}
): ShadowRenderSet {
	const resolvedConfig = normalizeShadowConfig(config);
	const signature = shadowConfigSignature(resolvedConfig);
	const storageMode = options.storageMode ?? renderSet.storageMode ?? "atlas";
	const paged = storageMode === "paged" ? options.paged ?? renderSet.layout.paged : undefined;
	if (
		renderSet.configSignature === signature &&
		renderSet.storageMode === storageMode
	) {
		renderSet.requestedStrategyType = resolvedConfig.strategy;
		renderSet.resolvedConfig = resolvedConfig;
		renderSet.size = resolvedConfig.size ?? renderSet.size;
		renderSet.storageMode = storageMode;
		renderSet.layout.storageMode = storageMode;
		renderSet.layout.paged = paged;
		syncShadowLayout(renderSet);
		return renderSet;
	}

	const next = createShadowRenderSet(resolvedConfig, { storageMode, paged });
	next.metadataVersion = renderSet.metadataVersion + 1;
	return next;
}

export function getPrimaryShadowSlice(
	renderSet: ShadowRenderSet | ShadowMap | null | undefined
): ShadowSlice | null {
	if (!renderSet) {
		return null;
	}
	if (isShadowMapLike(renderSet)) {
		return {
			index: 0,
			shadowMap: renderSet,
			splitNear: 0,
			splitFar: 1,
			atlasRect: null,
		};
	}
	if (renderSet.slices.length <= 0) {
		return null;
	}
	return renderSet.slices[0];
}

export function getPrimaryShadowMap(
	renderSet: ShadowRenderSet | ShadowMap | null | undefined
): ShadowMap | null {
	return getPrimaryShadowSlice(renderSet)?.shadowMap ?? null;
}
