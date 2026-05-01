import {
	isFogPostProcessEnabled,
	type BuiltinFramePassStage,
	type FramePassStage,
	type ResolvedFeatureState,
} from "./types";

export const RENDER_DIRTY_REASON_MASK = {
	unknown: 1 << 0,
	resize: 1 << 1,
	camera: 1 << 2,
	transform: 1 << 3,
	material: 1 << 4,
	texture: 1 << 5,
	lighting: 1 << 6,
	shadow: 1 << 7,
	postfx: 1 << 8,
	interaction: 1 << 9,
	physics: 1 << 10,
	particles: 1 << 11,
	"postfx-light": 1 << 12,
	"postfx-standard": 1 << 13,
	"postfx-cinematic": 1 << 14,
	"reflection-probe": 1 << 15,
} as const;

export type RenderDirtyReason = keyof typeof RENDER_DIRTY_REASON_MASK;

export const RENDER_DIRTY_GROUP = {
	postfx:
		RENDER_DIRTY_REASON_MASK.postfx |
		RENDER_DIRTY_REASON_MASK["postfx-light"] |
		RENDER_DIRTY_REASON_MASK["postfx-standard"] |
		RENDER_DIRTY_REASON_MASK["postfx-cinematic"],
	shading:
		RENDER_DIRTY_REASON_MASK.material |
		RENDER_DIRTY_REASON_MASK.texture |
		RENDER_DIRTY_REASON_MASK.lighting |
		RENDER_DIRTY_REASON_MASK.shadow |
		RENDER_DIRTY_REASON_MASK["reflection-probe"],
} as const;

export interface DirtyRect {
	x: number;
	y: number;
	width: number;
	height: number;
}

export interface DirtyTileCoverage {
	tileSize: number;
	tileColumns: number;
	tileRows: number;
	dirtyTiles: number[];
}

export interface IncrementalRenderingOptions {
	enabled: boolean;
	maxDirtyRects: number;
	dirtyTileSize: number;
	fullFrameFallbackAreaRatio: number;
	temporalPolicy: "conservative-reset";
}

export interface IncrementalFrameStats {
	enabled: boolean;
	reasonMask: number;
	forceFullFrame: boolean;
	temporalHistoryReset: boolean;
	firstPass: FramePassStage | null;
	dirtyRectCount: number;
	dirtyTileCount: number;
	dirtyTileSize: number;
	dirtyTileColumns: number;
	dirtyTileRows: number;
	dirtyAreaRatio: number;
	dirtyRects: DirtyRect[];
	dirtyTiles: number[];
}

export interface IncrementalFrameContext {
	enabled: boolean;
	forceFullFrame: boolean;
	dirtyRects: DirtyRect[];
	dirtyTileSize: number;
	dirtyTileColumns: number;
	dirtyTileRows: number;
	dirtyTiles: number[];
	dirtyAreaRatio: number;
	firstPass: FramePassStage | null;
	reasonMask: number;
	temporalHistoryReset: boolean;
}

export interface IncrementalPlanInput {
	enabled: boolean;
	reasonMask: number;
	features: ResolvedFeatureState;
}

export interface IncrementalPlan {
	firstPass: FramePassStage | null;
	forceFullFrame: boolean;
	temporalHistoryReset: boolean;
	reasonMask: number;
}

export const POST_PROCESS_GRADES = [
	"none",
	"light",
	"standard",
	"cinematic",
] as const;
export type PostProcessGrade = (typeof POST_PROCESS_GRADES)[number];

const FRAME_PASS_STAGE_ORDER: FramePassStage[] = [
	"particle-sim",
	"shadow",
	"reflection",
	"main-opaque",
	"main-transparent",
	"particles",
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"tonemap",
	"color-filter",
	"fxaa",
	"interaction-outline",
	"gamma",
];

const FRAME_PASS_STAGE_INDEX = new Map<FramePassStage, number>();
for (let index = 0; index < FRAME_PASS_STAGE_ORDER.length; index++) {
	FRAME_PASS_STAGE_INDEX.set(FRAME_PASS_STAGE_ORDER[index], index);
}

type PostProcessStage = Extract<
	BuiltinFramePassStage,
	| "ssao"
	| "ssgi"
	| "taa"
	| "ssr"
	| "volumetric"
	| "fog"
	| "motion-blur"
	| "dof"
	| "bloom"
	| "tonemap"
	| "color-filter"
	| "fxaa"
	| "gamma"
>;

type PostProcessFeatureFlag = keyof Pick<
	ResolvedFeatureState,
	| "enableSSAO"
	| "enableSSGI"
	| "enableTAA"
	| "enableSSR"
	| "enableVolumetric"
	| "enableFog"
	| "enableMotionBlur"
	| "enableDOF"
	| "enableBloom"
	| "enableToneMapping"
	| "enableColorFilter"
	| "enableFXAA"
	| "enableGamma"
>;

const POST_PROCESS_STAGE_FEATURE_ORDER: ReadonlyArray<
	readonly [PostProcessStage, PostProcessFeatureFlag]
> = [
	["ssao", "enableSSAO"],
	["ssgi", "enableSSGI"],
	["taa", "enableTAA"],
	["ssr", "enableSSR"],
	["volumetric", "enableVolumetric"],
	["fog", "enableFog"],
	["motion-blur", "enableMotionBlur"],
	["dof", "enableDOF"],
	["bloom", "enableBloom"],
	["tonemap", "enableToneMapping"],
	["color-filter", "enableColorFilter"],
	["fxaa", "enableFXAA"],
	["gamma", "enableGamma"],
];

const POSTFX_REASON_MASK = RENDER_DIRTY_GROUP.postfx;
const SHADING_REASON_MASK = RENDER_DIRTY_GROUP.shading;

const POST_PROCESS_GRADE_INFLATION_RADIUS: Record<PostProcessGrade, number> = {
	none: 0,
	light: 2,
	standard: 12,
	cinematic: 24,
};

const POST_PROCESS_GRADE_FALLBACK_SCALE: Record<PostProcessGrade, number> = {
	none: 1,
	light: 1,
	standard: 0.9,
	cinematic: 0.8,
};

const TEMPORAL_RESET_MASK =
	RENDER_DIRTY_REASON_MASK.resize |
	RENDER_DIRTY_REASON_MASK.camera |
	RENDER_DIRTY_REASON_MASK.transform |
	RENDER_DIRTY_REASON_MASK.material |
	RENDER_DIRTY_REASON_MASK.lighting |
	RENDER_DIRTY_REASON_MASK.shadow |
	RENDER_DIRTY_REASON_MASK["reflection-probe"] |
	RENDER_DIRTY_REASON_MASK.physics |
	RENDER_DIRTY_REASON_MASK["postfx-cinematic"] |
	RENDER_DIRTY_REASON_MASK.unknown;

const FORCE_FULL_FRAME_MASK =
	RENDER_DIRTY_REASON_MASK.resize |
	RENDER_DIRTY_REASON_MASK.camera |
	RENDER_DIRTY_REASON_MASK.lighting |
	RENDER_DIRTY_REASON_MASK.shadow |
	RENDER_DIRTY_REASON_MASK["reflection-probe"] |
	RENDER_DIRTY_REASON_MASK.unknown;

export const DEFAULT_INCREMENTAL_DIRTY_TILE_SIZE = 32;

export const DEFAULT_INCREMENTAL_RENDERING_OPTIONS: IncrementalRenderingOptions =
	{
		enabled: false,
		maxDirtyRects: 16,
		dirtyTileSize: DEFAULT_INCREMENTAL_DIRTY_TILE_SIZE,
		fullFrameFallbackAreaRatio: 0.3,
		temporalPolicy: "conservative-reset",
	};

export function renderDirtyReasonToMask(
	reason: RenderDirtyReason | undefined
): number {
	if (!reason) {
		return RENDER_DIRTY_REASON_MASK.unknown;
	}
	return RENDER_DIRTY_REASON_MASK[reason] ?? RENDER_DIRTY_REASON_MASK.unknown;
}

export function hasAnyDirtyReason(
	mask: number,
	...reasons: RenderDirtyReason[]
): boolean {
	for (const reason of reasons) {
		if ((mask & renderDirtyReasonToMask(reason)) !== 0) {
			return true;
		}
	}
	return false;
}

export function normalizeIncrementalRenderingOptions(
	options?: Partial<IncrementalRenderingOptions> | null
): IncrementalRenderingOptions {
	const source = options ?? {};
	return {
		enabled: source.enabled ?? DEFAULT_INCREMENTAL_RENDERING_OPTIONS.enabled,
		maxDirtyRects: clampInteger(
			source.maxDirtyRects,
			1,
			256,
			DEFAULT_INCREMENTAL_RENDERING_OPTIONS.maxDirtyRects
		),
		dirtyTileSize: clampInteger(
			source.dirtyTileSize,
			4,
			512,
			DEFAULT_INCREMENTAL_RENDERING_OPTIONS.dirtyTileSize
		),
		fullFrameFallbackAreaRatio: clampNumber(
			source.fullFrameFallbackAreaRatio,
			0.01,
			1,
			DEFAULT_INCREMENTAL_RENDERING_OPTIONS.fullFrameFallbackAreaRatio
		),
		temporalPolicy: "conservative-reset",
	};
}

export function mergeIncrementalRenderingOptions(
	current: IncrementalRenderingOptions,
	next?: Partial<IncrementalRenderingOptions> | null
): IncrementalRenderingOptions {
	if (!next) {
		return current;
	}
	return normalizeIncrementalRenderingOptions({
		...current,
		...next,
	});
}

export class IncrementalFramePlanner {
	public static plan(input: IncrementalPlanInput): IncrementalPlan {
		const reasonMask = input.reasonMask >>> 0;
		if (!input.enabled) {
			return {
				firstPass: null,
				forceFullFrame: true,
				temporalHistoryReset: true,
				reasonMask,
			};
		}

		if (reasonMask === 0) {
			return {
				firstPass: null,
				forceFullFrame: false,
				temporalHistoryReset: false,
				reasonMask,
			};
		}

		const temporalHistoryReset = (reasonMask & TEMPORAL_RESET_MASK) !== 0;
		const forceFullFrame = (reasonMask & FORCE_FULL_FRAME_MASK) !== 0;

		const candidates: FramePassStage[] = [];

		if (
			(reasonMask &
				(RENDER_DIRTY_REASON_MASK.particles |
					RENDER_DIRTY_REASON_MASK.physics)) !==
			0
		) {
			candidates.push("particle-sim");
		}

		if (
			(reasonMask &
				(RENDER_DIRTY_REASON_MASK.resize |
					RENDER_DIRTY_REASON_MASK.camera |
					RENDER_DIRTY_REASON_MASK.transform |
					SHADING_REASON_MASK)) !==
			0
		) {
			candidates.push(
				input.features.enableShadows ? "shadow" : "main-opaque"
			);
		}

		if ((reasonMask & POSTFX_REASON_MASK) !== 0) {
			candidates.push(
				resolveFirstEnabledPostProcessStage(input.features) ?? "gamma"
			);
		}

		if ((reasonMask & RENDER_DIRTY_REASON_MASK.interaction) !== 0) {
			candidates.push("interaction-outline");
		}

		if (candidates.length === 0) {
			candidates.push("main-opaque");
		}

		return {
			firstPass: pickEarliestPass(candidates),
			forceFullFrame,
			temporalHistoryReset,
			reasonMask,
		};
	}
}

export function makeFullScreenRect(width: number, height: number): DirtyRect {
	return {
		x: 0,
		y: 0,
		width: Math.max(1, Math.floor(width)),
		height: Math.max(1, Math.floor(height)),
	};
}

export function clampDirtyRect(
	rect: DirtyRect,
	width: number,
	height: number
): DirtyRect | null {
	const maxWidth = Math.max(1, Math.floor(width));
	const maxHeight = Math.max(1, Math.floor(height));
	const minX = Math.max(0, Math.floor(rect.x));
	const minY = Math.max(0, Math.floor(rect.y));
	const maxX = Math.min(maxWidth, Math.ceil(rect.x + rect.width));
	const maxY = Math.min(maxHeight, Math.ceil(rect.y + rect.height));
	const clampedWidth = maxX - minX;
	const clampedHeight = maxY - minY;
	if (clampedWidth <= 0 || clampedHeight <= 0) {
		return null;
	}
	return {
		x: minX,
		y: minY,
		width: clampedWidth,
		height: clampedHeight,
	};
}

export function buildDirtyTileCoverage(
	rects: DirtyRect[],
	width: number,
	height: number,
	tileSize: number
): DirtyTileCoverage {
	const resolvedWidth = Math.max(1, Math.floor(width));
	const resolvedHeight = Math.max(1, Math.floor(height));
	const resolvedTileSize = clampInteger(
		tileSize,
		4,
		512,
		DEFAULT_INCREMENTAL_DIRTY_TILE_SIZE
	);
	const tileColumns = Math.max(1, Math.ceil(resolvedWidth / resolvedTileSize));
	const tileRows = Math.max(1, Math.ceil(resolvedHeight / resolvedTileSize));
	const tileCount = tileColumns * tileRows;
	const visited = new Uint8Array(tileCount);
	const dirtyTiles: number[] = [];

	for (const rect of rects) {
		const clamped = clampDirtyRect(rect, resolvedWidth, resolvedHeight);
		if (!clamped) {
			continue;
		}
		const minTileX = Math.floor(clamped.x / resolvedTileSize);
		const minTileY = Math.floor(clamped.y / resolvedTileSize);
		const maxTileX = Math.floor(
			(clamped.x + clamped.width - 1) / resolvedTileSize
		);
		const maxTileY = Math.floor(
			(clamped.y + clamped.height - 1) / resolvedTileSize
		);
		for (let tileY = minTileY; tileY <= maxTileY; tileY++) {
			for (let tileX = minTileX; tileX <= maxTileX; tileX++) {
				const tileIndex = tileY * tileColumns + tileX;
				if (tileIndex < 0 || tileIndex >= tileCount || visited[tileIndex] !== 0) {
					continue;
				}
				visited[tileIndex] = 1;
				dirtyTiles.push(tileIndex);
			}
		}
	}

	dirtyTiles.sort((left, right) => left - right);
	return {
		tileSize: resolvedTileSize,
		tileColumns,
		tileRows,
		dirtyTiles,
	};
}

export function tileCoverageToDirtyRects(
	coverage: DirtyTileCoverage,
	maxRects: number,
	width: number,
	height: number
): DirtyRect[] {
	if (coverage.dirtyTiles.length === 0) {
		return [];
	}
	const resolvedWidth = Math.max(1, Math.floor(width));
	const resolvedHeight = Math.max(1, Math.floor(height));
	const tileRects: DirtyRect[] = [];
	for (const tileIndex of coverage.dirtyTiles) {
		const tileX = tileIndex % coverage.tileColumns;
		const tileY = Math.floor(tileIndex / coverage.tileColumns);
		if (
			tileX < 0 ||
			tileY < 0 ||
			tileX >= coverage.tileColumns ||
			tileY >= coverage.tileRows
		) {
			continue;
		}
		const x = tileX * coverage.tileSize;
		const y = tileY * coverage.tileSize;
		tileRects.push({
			x,
			y,
			width: Math.min(coverage.tileSize, resolvedWidth - x),
			height: Math.min(coverage.tileSize, resolvedHeight - y),
		});
	}
	return mergeDirtyRects(tileRects, maxRects, resolvedWidth, resolvedHeight);
}

export function getDirtyTileCoverageAreaRatio(
	coverage: DirtyTileCoverage,
	width: number,
	height: number
): number {
	const resolvedWidth = Math.max(1, Math.floor(width));
	const resolvedHeight = Math.max(1, Math.floor(height));
	const area = resolvedWidth * resolvedHeight;
	let dirtyArea = 0;
	for (const tileIndex of coverage.dirtyTiles) {
		const tileX = tileIndex % coverage.tileColumns;
		const tileY = Math.floor(tileIndex / coverage.tileColumns);
		if (
			tileX < 0 ||
			tileY < 0 ||
			tileX >= coverage.tileColumns ||
			tileY >= coverage.tileRows
		) {
			continue;
		}
		const x = tileX * coverage.tileSize;
		const y = tileY * coverage.tileSize;
		const tileWidth = Math.min(coverage.tileSize, resolvedWidth - x);
		const tileHeight = Math.min(coverage.tileSize, resolvedHeight - y);
		if (tileWidth > 0 && tileHeight > 0) {
			dirtyArea += tileWidth * tileHeight;
		}
	}
	return Math.max(0, Math.min(1, dirtyArea / area));
}

export function inflateDirtyRects(
	rects: DirtyRect[],
	amount: number,
	width: number,
	height: number
): DirtyRect[] {
	const inflateAmount = Math.max(0, Math.floor(amount));
	if (inflateAmount <= 0 || rects.length === 0) {
		return rects.slice();
	}
	const result: DirtyRect[] = [];
	for (const rect of rects) {
		const inflated = clampDirtyRect(
			{
				x: rect.x - inflateAmount,
				y: rect.y - inflateAmount,
				width: rect.width + inflateAmount * 2,
				height: rect.height + inflateAmount * 2,
			},
			width,
			height
		);
		if (inflated) {
			result.push(inflated);
		}
	}
	return result;
}

export function mergeDirtyRects(
	rects: DirtyRect[],
	maxRects: number,
	width: number,
	height: number
): DirtyRect[] {
	const normalized: DirtyRect[] = [];
	for (const rect of rects) {
		const clamped = clampDirtyRect(rect, width, height);
		if (!clamped) continue;
		normalized.push(clamped);
	}
	if (normalized.length <= 1) {
		return normalized;
	}

	normalized.sort((left, right) => {
		if (left.x !== right.x) return left.x - right.x;
		return left.y - right.y;
	});

	const merged: DirtyRect[] = [];
	for (const rect of normalized) {
		let current = rect;
		for (let i = merged.length - 1; i >= 0; i--) {
			const previous = merged[i];
			if (!dirtyRectsIntersectOrTouch(previous, current)) {
				continue;
			}
			current = unionDirtyRect(previous, current);
			merged.splice(i, 1);
		}
		merged.push(current);
	}

	const cappedMaxRects = clampInteger(maxRects, 1, 256, 16);
	while (merged.length > cappedMaxRects) {
		let bestLeft = 0;
		let bestRight = 1;
		let bestGrowth = Number.POSITIVE_INFINITY;
		for (let leftIndex = 0; leftIndex < merged.length; leftIndex++) {
			for (
				let rightIndex = leftIndex + 1;
				rightIndex < merged.length;
				rightIndex++
			) {
				const left = merged[leftIndex];
				const right = merged[rightIndex];
				const union = unionDirtyRect(left, right);
				const growth = getDirtyRectArea(union) -
					(getDirtyRectArea(left) + getDirtyRectArea(right));
				if (growth < bestGrowth) {
					bestGrowth = growth;
					bestLeft = leftIndex;
					bestRight = rightIndex;
				}
			}
		}
		const union = unionDirtyRect(merged[bestLeft], merged[bestRight]);
		merged.splice(bestRight, 1);
		merged.splice(bestLeft, 1, union);
	}

	return merged;
}

export function getDirtyRectsAreaRatio(
	rects: DirtyRect[],
	width: number,
	height: number
): number {
	const area = Math.max(1, Math.floor(width) * Math.floor(height));
	let dirtyArea = 0;
	for (const rect of rects) {
		dirtyArea += getDirtyRectArea(rect);
	}
	return Math.max(0, Math.min(1, dirtyArea / area));
}

export function computePostProcessInflationRadius(
	features: ResolvedFeatureState
): number {
	let radius = getPostProcessGradeInflationRadius(resolvePostProcessGrade(features));
	if (features.enableSSAO) radius = Math.max(radius, 8);
	if (features.enableSSGI) radius = Math.max(radius, 12);
	if (features.enableTAA) radius = Math.max(radius, 8);
	if (features.enableSSR) radius = Math.max(radius, 16);
	if (features.enableVolumetric) radius = Math.max(radius, 16);
	if (isFogPostProcessEnabled(features)) radius = Math.max(radius, 20);
	if (features.enableMotionBlur) radius = Math.max(radius, 24);
	if (features.enableDOF) radius = Math.max(radius, 32);
	if (features.enableBloom) radius = Math.max(radius, 48);
	if (features.enableColorFilter) radius = Math.max(radius, 2);
	if (features.enableFXAA) radius = Math.max(radius, 2);
	return radius;
}

export function resolvePostProcessGrade(
	features: ResolvedFeatureState
): PostProcessGrade {
	if (
		features.enableTAA ||
		features.enableSSR ||
		features.enableVolumetric ||
		isFogPostProcessEnabled(features) ||
		features.enableMotionBlur ||
		features.enableDOF
	) {
		return "cinematic";
	}
	if (features.enableSSAO || features.enableSSGI || features.enableBloom) {
		return "standard";
	}
	if (
		features.enableToneMapping ||
		features.enableColorFilter ||
		features.enableFXAA ||
		features.enableGamma
	) {
		return "light";
	}
	return "none";
}

export function getPostProcessGradeInflationRadius(
	grade: PostProcessGrade
): number {
	return POST_PROCESS_GRADE_INFLATION_RADIUS[grade] ?? 0;
}

export function scaleFullFrameFallbackAreaRatioForPostProcess(
	baseRatio: number,
	features: ResolvedFeatureState
): number {
	const normalizedBaseRatio = clampNumber(
		baseRatio,
		0.01,
		1,
		DEFAULT_INCREMENTAL_RENDERING_OPTIONS.fullFrameFallbackAreaRatio
	);
	const grade = resolvePostProcessGrade(features);
	const scale = POST_PROCESS_GRADE_FALLBACK_SCALE[grade] ?? 1;
	return clampNumber(normalizedBaseRatio * scale, 0.01, 1, normalizedBaseRatio);
}

export function unionDirtyRect(left: DirtyRect, right: DirtyRect): DirtyRect {
	const minX = Math.min(left.x, right.x);
	const minY = Math.min(left.y, right.y);
	const maxX = Math.max(left.x + left.width, right.x + right.width);
	const maxY = Math.max(left.y + left.height, right.y + right.height);
	return {
		x: minX,
		y: minY,
		width: maxX - minX,
		height: maxY - minY,
	};
}

function resolveFirstEnabledPostProcessStage(
	features: ResolvedFeatureState
): PostProcessStage | null {
	for (const [stage, featureFlag] of POST_PROCESS_STAGE_FEATURE_ORDER) {
		if (stage === "fog" && !isFogPostProcessEnabled(features)) {
			continue;
		}
		if (features[featureFlag]) {
			return stage;
		}
	}
	return null;
}

function pickEarliestPass(candidates: FramePassStage[]): FramePassStage {
	let earliest = candidates[0];
	let earliestIndex = FRAME_PASS_STAGE_INDEX.get(earliest) ?? Number.MAX_SAFE_INTEGER;
	for (let index = 1; index < candidates.length; index++) {
		const candidate = candidates[index];
		const candidateIndex = FRAME_PASS_STAGE_INDEX.get(candidate) ?? Number.MAX_SAFE_INTEGER;
		if (candidateIndex < earliestIndex) {
			earliest = candidate;
			earliestIndex = candidateIndex;
		}
	}
	return earliest;
}

function getDirtyRectArea(rect: DirtyRect): number {
	return Math.max(0, rect.width) * Math.max(0, rect.height);
}

function dirtyRectsIntersectOrTouch(left: DirtyRect, right: DirtyRect): boolean {
	return (
		left.x <= right.x + right.width &&
		left.x + left.width >= right.x &&
		left.y <= right.y + right.height &&
		left.y + left.height >= right.y
	);
}

function clampNumber(
	value: number | undefined,
	min: number,
	max: number,
	fallback: number
): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(max, Math.max(min, value));
}

function clampInteger(
	value: number | undefined,
	min: number,
	max: number,
	fallback: number
): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(max, Math.max(min, Math.floor(value)));
}
