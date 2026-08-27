import type { RGB } from "../../foundation/Color";
import type { PreparedShadowLight } from "../shadows/ShadowFramePlan";
import { clamp, sRGBToLinear } from "../../maths/Common";
import type { Matrix4 } from "../../maths/Matrix4";
import type {
	ShadowFilterMode,
	ShadowSamplingQuality,
} from "../shadows/types";
import {
	resolveShadowDepthProjectionParams,
	type ShadowDepthProjectionParams,
} from "../shadows/shadowSampling";

type RGBLike = Partial<RGB>;

export type ResolvedShadowStrategy = "single-map" | "csm";

/**
 * Backend-neutral shadow sampling information derived solely from a prepared
 * frame plan. Backend allocations are deliberately kept out of this record.
 */
export interface ResolvedShadowData {
	enabled: boolean;
	strategyType: ResolvedShadowStrategy;
	cascadeCount: number;
	cascadeBlendRatio: number;
	cascadeViewProjectionMatrices: Array<Matrix4 | null>;
	cascadeSplits: Array<[number, number, number, number]>;
	depthProjectionParams: Array<ShadowDepthProjectionParams>;
	viewProjectionMatrix: Matrix4 | null;
	depthBias: number;
	slopeBias: number;
	normalBias: number;
	normalBiasMin: number;
	filterMode: ShadowFilterMode;
	samplingQuality: ShadowSamplingQuality;
	shadowStrength: number;
	shadowMapBaseSize: number;
	shadowMapSize: number;
	storageMode: "atlas" | "paged";
	pagedPageGridSize: number;
	pagedPageSize: number;
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
	prepared?: PreparedShadowLight | null
): ResolvedShadowData {
	const primarySlice = prepared?.slices[0] ?? null;
	if (!enableShadows || !prepared || !primarySlice) {
		return createDisabledShadowData();
	}

	const baseSize = Math.max(1, prepared.effectiveResolution | 0);
	const sliceSize = Math.max(1, primarySlice.resolution | 0);
	const bias = prepared.definition.bias;
	const texelBias = (bias.texel ?? 1) * (1 / sliceSize);
	const maxBias = bias.max ?? 0.05;
	const depthBias = Math.min(maxBias, (bias.constant ?? 0.008) + texelBias);
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
	const depthProjectionParams: Array<ShadowDepthProjectionParams> = [
		[0, 0, 0, 1],
		[0, 0, 0, 1],
		[0, 0, 0, 1],
		[0, 0, 0, 1],
	];
	for (let index = 0; index < Math.min(prepared.slices.length, 4); index++) {
		const slice = prepared.slices[index];
		cascadeViewProjectionMatrices[index] = slice.viewProjection;
		depthProjectionParams[index] = resolveShadowDepthProjectionParams(
			slice.projection,
		);
		cascadeSplits[index] = [
			Math.max(0, slice.splitNear),
			Math.max(0, slice.splitFar),
			index % 2,
			Math.floor(index / 2),
		];
	}

	const strategyType: ResolvedShadowStrategy =
		prepared.effectiveTechnique === "cascaded" && prepared.slices.length > 1 ?
			"csm"
		: 	"single-map";
	const cascadeCount = strategyType === "csm" ?
		Math.max(1, Math.min(4, prepared.slices.length))
	: 1;
	return {
		enabled: true,
		strategyType,
		cascadeCount,
		cascadeBlendRatio:
			strategyType === "csm" ? clamp(prepared.definition.projection.blendRatio ?? 0.1, 0, 1) : 0,
		cascadeViewProjectionMatrices,
		cascadeSplits,
		depthProjectionParams,
		viewProjectionMatrix: primarySlice.viewProjection,
		depthBias,
		slopeBias: Math.max(0, bias.slope ?? 0.03),
		normalBias: Math.max(0, bias.normal ?? 1),
		normalBiasMin: Math.max(0, bias.normalMin ?? 0.05),
		filterMode: prepared.effectiveFilterMode,
		samplingQuality: prepared.sampling.quality,
		shadowStrength: clamp(prepared.definition.strength, 0, 1),
		// A CSM atlas tile is the logical base resolution, while each
		// rasterized slice occupies one quadrant of it. WGSL uses the latter
		// for texel addressing and filter/bias calculations.
		shadowMapBaseSize: baseSize,
		shadowMapSize: sliceSize,
		storageMode: prepared.storage,
		pagedPageGridSize: prepared.pagedSettings?.pageGridSize ?? 0,
		pagedPageSize: prepared.pagedSettings?.pageSize ?? 0,
	};
}

function createDisabledShadowData(): ResolvedShadowData {
	return {
		enabled: false,
		strategyType: "single-map",
		cascadeCount: 1,
		cascadeBlendRatio: 0,
		cascadeViewProjectionMatrices: [null, null, null, null],
		cascadeSplits: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
		depthProjectionParams: [
			[0, 0, 0, 1],
			[0, 0, 0, 1],
			[0, 0, 0, 1],
			[0, 0, 0, 1],
		],
		viewProjectionMatrix: null,
		depthBias: 0,
		slopeBias: 0,
		normalBias: 0,
		normalBiasMin: 0,
		filterMode: "pcf",
		samplingQuality: "medium",
		shadowStrength: 0,
		shadowMapBaseSize: 0,
		shadowMapSize: 0,
		storageMode: "atlas",
		pagedPageGridSize: 0,
		pagedPageSize: 0,
	};
}

function resolveColorChannel(value: unknown): number {
	return clamp(resolveFiniteNumber(value, 255), 0, 255);
}

function resolveFiniteNumber(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
