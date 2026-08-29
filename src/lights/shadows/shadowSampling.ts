import type { Matrix4 } from "../../maths/Matrix4";
import type { ShadowFilterMode, ShadowSamplingQuality } from "./types";

export const SHADOW_PCF_RADIUS_TEXELS = 1.5;
export const SHADOW_PCSS_SEARCH_RADIUS_TEXELS = 5;
export const SHADOW_PCSS_MAX_PENUMBRA_TEXELS = 5;
export const SHADOW_PCSS_CONTACT_THRESHOLD_TEXELS = 0.75;

export const SHADOW_FILTER_MODE_CODE: Readonly<Record<ShadowFilterMode, number>> = {
	pcf: 0,
	pcss: 1,
};

export const SHADOW_QUALITY_CODE: Readonly<Record<ShadowSamplingQuality, number>> = {
	low: 0,
	medium: 1,
	high: 2,
};

export interface ShadowSamplingPreset {
	readonly pcfSamples: number;
	readonly pcssSearchSamples: number;
	readonly pcssFilterSamples: number;
}

export const SHADOW_SAMPLING_PRESETS: Readonly<
	Record<ShadowSamplingQuality, ShadowSamplingPreset>
> = Object.freeze({
	low: Object.freeze({
		pcfSamples: 1,
		pcssSearchSamples: 4,
		pcssFilterSamples: 3,
	}),
	medium: Object.freeze({
		pcfSamples: 3,
		pcssSearchSamples: 8,
		pcssFilterSamples: 5,
	}),
	high: Object.freeze({
		pcfSamples: 5,
		pcssSearchSamples: 12,
		pcssFilterSamples: 7,
	}),
});

export const SHADOW_FILTER_DISK_SAMPLES: readonly (readonly [number, number])[] =
	Object.freeze([
		// Low: one centered tap.
		Object.freeze([0, 0] as const),
		// Medium: centered equilateral triangle.
		Object.freeze([0, 0.53] as const),
		Object.freeze([-0.458993464, -0.265] as const),
		Object.freeze([0.458993464, -0.265] as const),
		// High PCF / medium PCSS: center plus a symmetric cross.
		Object.freeze([0, 0] as const),
		Object.freeze([0.66, 0] as const),
		Object.freeze([-0.66, 0] as const),
		Object.freeze([0, 0.66] as const),
		Object.freeze([0, -0.66] as const),
		// High PCSS: center plus a symmetric hexagonal ring.
		Object.freeze([0, 0] as const),
		Object.freeze([0.68, 0] as const),
		Object.freeze([0.34, 0.588897275] as const),
		Object.freeze([-0.34, 0.588897275] as const),
		Object.freeze([-0.68, 0] as const),
		Object.freeze([-0.34, -0.588897275] as const),
		Object.freeze([0.34, -0.588897275] as const),
	]);

export const SHADOW_SEARCH_DISK_SAMPLES: readonly (readonly [number, number])[] =
	Object.freeze([
		// Opposite pairs keep the low, medium, and high prefixes centered.
		Object.freeze([-0.191063595, 0.710747050] as const),
		Object.freeze([0.191063595, -0.710747050] as const),
		Object.freeze([0.790556673, 0.288710029] as const),
		Object.freeze([-0.790556673, -0.288710029] as const),
		Object.freeze([-0.822442486, 0.339492303] as const),
		Object.freeze([0.822442486, -0.339492303] as const),
		Object.freeze([-0.364378997, -0.701589586] as const),
		Object.freeze([0.364378997, 0.701589586] as const),
		Object.freeze([0.396471625, -0.847236833] as const),
		Object.freeze([-0.396471625, 0.847236833] as const),
		Object.freeze([0.571225035, -0.363366609] as const),
		Object.freeze([-0.571225035, 0.363366609] as const),
	]);

export type ShadowDepthProjectionParams = readonly [number, number, number, number];

/** @internal Extracts coefficients used to reconstruct projection-space depth. */
export function resolveShadowDepthProjectionParams(
	projection: Matrix4,
): ShadowDepthProjectionParams {
	const values = projection.elements;
	return Object.freeze([
		values[2][2],
		values[2][3],
		values[3][2],
		values[3][3],
	]);
}

/** @internal Reconstructs positive light-view distance from OpenGL NDC depth. */
export function linearizeShadowNdcDepth(
	ndcDepth: number,
	params: ShadowDepthProjectionParams,
): number {
	const [m22, m23, m32, m33] = params;
	const denominator = ndcDepth * m32 - m22;
	if (!Number.isFinite(denominator) || Math.abs(denominator) <= 1e-8) {
		return Number.POSITIVE_INFINITY;
	}
	return Math.abs((m23 - ndcDepth * m33) / denominator);
}

/** @internal Resolves one centered filter sample for the active tap count. */
export function resolveShadowFilterDiskSample(
	index: number,
	sampleCount: number,
	rotationCos: number,
	rotationSin: number,
): readonly [number, number] {
	const normalizedCount = sampleCount <= 1 ? 1 : sampleCount <= 3 ? 3 :
		sampleCount <= 5 ? 5 : 7;
	const start = normalizedCount === 1 ? 0 : normalizedCount === 3 ? 1 :
		normalizedCount === 5 ? 4 : 9;
	const sample = SHADOW_FILTER_DISK_SAMPLES[
		start + Math.max(0, Math.min(normalizedCount - 1, index | 0))
	];
	return rotateShadowDiskSample(sample, rotationCos, rotationSin);
}

/** @internal Resolves one centered blocker-search disk sample. */
export function resolveShadowSearchDiskSample(
	index: number,
	rotationCos: number,
	rotationSin: number,
): readonly [number, number] {
	const sample = SHADOW_SEARCH_DISK_SAMPLES[
		Math.max(0, Math.min(SHADOW_SEARCH_DISK_SAMPLES.length - 1, index | 0))
	];
	return rotateShadowDiskSample(sample, rotationCos, rotationSin);
}

function rotateShadowDiskSample(
	sample: readonly [number, number],
	rotationCos: number,
	rotationSin: number,
): readonly [number, number] {
	return [
		sample[0] * rotationCos - sample[1] * rotationSin,
		sample[0] * rotationSin + sample[1] * rotationCos,
	];
}

/** @internal Spatially stable rotation shared by CPU and GPU implementations. */
export function resolveShadowSampleRotation(
	lightIndex: number,
	cascadeIndex: number,
): number {
	let hash = 0x9e3779b9;
	hash ^= Math.imul(lightIndex | 0, 0xcb1ab31f);
	hash ^= Math.imul(cascadeIndex | 0, 0x165667b1);
	hash ^= hash >>> 16;
	hash = Math.imul(hash, 0x7feb352d);
	hash ^= hash >>> 15;
	return ((hash >>> 0) / 0x100000000) * Math.PI * 2;
}
