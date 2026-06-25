import type { RGB } from "../../foundation/Color";
import type {
	ShadowMap,
	ShadowRenderSet,
	ShadowStrategyType,
} from "../../lights/shadows/ShadowMapping";
import { clamp, sRGBToLinear } from "../../maths/Common";
import type { Matrix4 } from "../../maths/Matrix4";

type RGBLike = Partial<RGB>;

export interface ResolvedShadowData {
	enabled: boolean;
	strategyType: ShadowStrategyType;
	cascadeCount: number;
	cascadeBlendRatio: number;
	cascadeViewProjectionMatrices: Array<Matrix4 | null>;
	cascadeSplits: Array<[number, number, number, number]>;
	viewProjectionMatrix: Matrix4 | null;
	depthBias: number;
	slopeBias: number;
	normalBias: number;
	normalBiasMin: number;
	pcfRadius: number;
	pcssEnabled: boolean;
	pcssRadius: number;
	shadowSamples: number;
	shadowSearchSamples: number;
	shadowStrength: number;
	shadowMapBaseSize: number;
	shadowMapSize: number;
	atlasTileSize: number;
	storageMode: "atlas" | "paged";
	pagedPageTableBase: number;
	pagedPageTableCascadeStride: number;
	pagedPageGridSize: number;
	pagedPageSize: number;
	pagedPhysicalAtlasSize: number;
	pagedPhysicalGridSize: number;
	pagedPhysicalPageSize: number;
	shadowMap: ShadowMap | null;
}

export interface ResolveShadowDataOptions {
	keepShadowMapWhenDisabled?: boolean;
}

const LIGHT_PROBE_DC_IRRADIANCE_SCALE = Math.PI * 0.282095;

export function toLinearLightColor(
	color: RGBLike | null | undefined,
	intensity: number
): [number, number, number] {
	const resolvedIntensity = resolveFiniteNumber(intensity, 1);
	return [
		sRGBToLinear(resolveColorChannel(color?.r) / 255) * resolvedIntensity,
		sRGBToLinear(resolveColorChannel(color?.g) / 255) * resolvedIntensity,
		sRGBToLinear(resolveColorChannel(color?.b) / 255) * resolvedIntensity,
	];
}

export function accumulateAmbientLightColor(
	ambientColor: [number, number, number],
	color: RGBLike | null | undefined,
	intensity: number
): void {
	const linear = toLinearLightColor(color, intensity);
	ambientColor[0] += linear[0];
	ambientColor[1] += linear[1];
	ambientColor[2] += linear[2];
}

export function accumulateLightProbeFallbackAmbientColor(
	ambientColor: [number, number, number],
	dc: RGBLike | null | undefined,
	intensity: number
): void {
	if (!dc) return;
	const resolvedIntensity = resolveFiniteNumber(intensity, 1);
	ambientColor[0] +=
		(Math.max(0, resolveFiniteNumber(dc.r, 0) * LIGHT_PROBE_DC_IRRADIANCE_SCALE) /
			255) *
		resolvedIntensity;
	ambientColor[1] +=
		(Math.max(0, resolveFiniteNumber(dc.g, 0) * LIGHT_PROBE_DC_IRRADIANCE_SCALE) /
			255) *
		resolvedIntensity;
	ambientColor[2] +=
		(Math.max(0, resolveFiniteNumber(dc.b, 0) * LIGHT_PROBE_DC_IRRADIANCE_SCALE) /
			255) *
		resolvedIntensity;
}

export function resolveShadowData(
	enableShadows: boolean,
	renderSetInput?: ShadowRenderSet | ShadowMap,
	options: ResolveShadowDataOptions = {}
): ResolvedShadowData {
	const renderSet =
		renderSetInput &&
		typeof renderSetInput === "object" &&
		Array.isArray((renderSetInput as { slices?: unknown }).slices) ?
			(renderSetInput as ShadowRenderSet)
		:	null;
	const legacyShadowMap =
		!renderSet &&
		renderSetInput &&
		typeof renderSetInput === "object" &&
		"viewProjectionMatrix" in renderSetInput ?
			(renderSetInput as ShadowMap)
		:	null;
	const primarySlice = renderSet?.slices[0] ?? null;
	const shadowMap = primarySlice?.shadowMap ?? null;
	const resolvedShadowMap = shadowMap ?? legacyShadowMap;

	if (!enableShadows || !resolvedShadowMap?.viewProjectionMatrix) {
		return {
			enabled: false,
			strategyType: "single-map",
			cascadeCount: 1,
			cascadeBlendRatio: 0,
			cascadeViewProjectionMatrices: [null, null, null, null],
			cascadeSplits: [
				[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0],
			],
			viewProjectionMatrix: null,
			depthBias: 0,
			slopeBias: 0,
			normalBias: 0,
			normalBiasMin: 0,
			pcfRadius: 0,
			pcssEnabled: false,
			pcssRadius: 0,
			shadowSamples: 0,
			shadowSearchSamples: 0,
			shadowStrength: 0,
			shadowMapBaseSize: 0,
			shadowMapSize: 0,
			atlasTileSize: 0,
			storageMode: "atlas",
			pagedPageTableBase: 0,
			pagedPageTableCascadeStride: 0,
			pagedPageGridSize: 0,
			pagedPageSize: 0,
			pagedPhysicalAtlasSize: 0,
			pagedPhysicalGridSize: 0,
			pagedPhysicalPageSize: 0,
			shadowMap: options.keepShadowMapWhenDisabled ? resolvedShadowMap : null,
		};
	}

	const size = Math.max(1, resolvedShadowMap.size | 0);
	const texelBias =
		(resolvedShadowMap.params.shadowTexelBias ?? 1.0) * (1.0 / size);
	const maxBias = resolvedShadowMap.params.shadowMaxBias ?? 0.05;
	const depthBias = Math.min(
		maxBias,
		(resolvedShadowMap.params.shadowBias ?? 0.008) + texelBias
	);
	const pcfRadius = Math.max(1, resolvedShadowMap.params.shadowPCF ?? 1);
	const pcssRadius = Math.max(0, resolvedShadowMap.params.shadowRadius ?? 0);
	const pcssEnabled = pcssRadius > 0;
	const shadowSamples = Math.max(
		1,
		Math.min(64, Math.floor(resolvedShadowMap.params.shadowSamples ?? 16))
	);
	const shadowSearchSamples = Math.max(
		1,
		Math.min(64, Math.floor(resolvedShadowMap.params.shadowSearchSamples ?? 16))
	);

	const cascadeViewProjectionMatrices: Array<Matrix4 | null> = [
		null,
		null,
		null,
		null,
	];
	const cascadeSplits: Array<[number, number, number, number]> = [
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],
	];
	if (renderSet) {
		for (let index = 0; index < Math.min(renderSet.slices.length, 4); index++) {
			const slice = renderSet.slices[index];
			cascadeViewProjectionMatrices[index] =
				slice.shadowMap.viewProjectionMatrix ?? null;
			const localTileX = index % 2;
			const localTileY = Math.floor(index / 2);
			cascadeSplits[index] = [
				Math.max(0, slice.splitNear),
				Math.max(0, slice.splitFar),
				localTileX,
				localTileY,
			];
		}
	} else {
		cascadeViewProjectionMatrices[0] = resolvedShadowMap.viewProjectionMatrix;
		cascadeSplits[0] = [0, 1, 0, 0];
	}

	const strategyType = renderSet?.effectiveStrategyType ?? "single-map";
	const availableCascadeCount = cascadeViewProjectionMatrices.reduce(
		(count, matrix) => (matrix ? count + 1 : count),
		0
	);
	const cascadeCount =
		strategyType === "csm" ?
			Math.max(1, Math.min(4, availableCascadeCount || 1))
		:	1;
	const cascadeBlendRatio =
		strategyType === "csm" &&
		cascadeCount > 1 &&
		renderSet &&
		renderSet.resolvedConfig.strategy === "csm" ?
			Math.max(0, Math.min(1, renderSet.resolvedConfig.blendRatio ?? 0.1))
		:	0;
	const paged = renderSet?.storageMode === "paged" ? renderSet.layout.paged : null;

	return {
		enabled: true,
		strategyType,
		cascadeCount,
		cascadeBlendRatio,
		cascadeViewProjectionMatrices,
		cascadeSplits,
		viewProjectionMatrix: resolvedShadowMap.viewProjectionMatrix,
		depthBias,
		slopeBias: Math.max(0, resolvedShadowMap.params.shadowSlopeBias ?? 0.03),
		normalBias: Math.max(0, resolvedShadowMap.params.shadowNormalBias ?? 1.0),
		normalBiasMin: Math.max(
			0,
			resolvedShadowMap.params.shadowNormalBiasMin ?? 0.05
		),
		pcfRadius: Math.max(1, pcfRadius),
		pcssEnabled,
		pcssRadius,
		shadowSamples,
		shadowSearchSamples,
		shadowStrength: clamp(resolvedShadowMap.params.shadowStrength ?? 1.0, 0, 1),
		shadowMapBaseSize: Math.max(1, renderSet?.size ?? resolvedShadowMap.size),
		shadowMapSize: size,
		atlasTileSize: 0,
		storageMode: paged ? "paged" : "atlas",
		pagedPageTableBase: paged?.pageTableBase ?? 0,
		pagedPageTableCascadeStride: paged?.pageTableCascadeStride ?? 0,
		pagedPageGridSize: paged?.pageGridSize ?? 0,
		pagedPageSize: paged?.pageSize ?? 0,
		pagedPhysicalAtlasSize: paged?.physicalAtlasSize ?? 0,
		pagedPhysicalGridSize: paged?.physicalGridSize ?? 0,
		pagedPhysicalPageSize: paged?.physicalPageSize ?? 0,
		shadowMap: resolvedShadowMap,
	};
}

function resolveColorChannel(value: unknown): number {
	return clamp(resolveFiniteNumber(value, 255), 0, 255);
}

function resolveFiniteNumber(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return value;
}
