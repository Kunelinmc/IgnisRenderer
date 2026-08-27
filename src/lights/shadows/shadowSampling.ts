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

export const SHADOW_DISK_SAMPLES: readonly (readonly [number, number])[] =
	Object.freeze([
		Object.freeze([0, 0] as const),
		Object.freeze([-0.191063595, 0.710747050] as const),
		Object.freeze([0.328594541, 0.428593391] as const),
		Object.freeze([-0.822442486, 0.339492303] as const),
		Object.freeze([-0.260699267, 0.238821884] as const),
		Object.freeze([-0.364378997, -0.701589586] as const),
		Object.freeze([-0.603011395, -0.106664225] as const),
		Object.freeze([0.396471625, -0.847236833] as const),
		Object.freeze([0.039904201, -0.454687792] as const),
		Object.freeze([0.790556673, 0.288710029] as const),
		Object.freeze([0.571225035, -0.363366609] as const),
		Object.freeze([0.292982446, 0.934074205] as const),
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

/** @internal Rotates one precomputed low-discrepancy disk sample. */
export function resolveShadowDiskSample(
	index: number,
	rotationCos: number,
	rotationSin: number,
): readonly [number, number] {
	const sample = SHADOW_DISK_SAMPLES[
		Math.max(0, Math.min(SHADOW_DISK_SAMPLES.length - 1, index | 0))
	];
	return [
		sample[0] * rotationCos - sample[1] * rotationSin,
		sample[0] * rotationSin + sample[1] * rotationCos,
	];
}

/** @internal Stable receiver rotation shared by CPU and GPU implementations. */
export function resolveShadowSampleRotation(
	texelX: number,
	texelY: number,
	lightIndex: number,
	cascadeIndex: number,
): number {
	let hash = Math.imul(texelX | 0, 0x8da6b343);
	hash ^= Math.imul(texelY | 0, 0xd8163841);
	hash ^= Math.imul(lightIndex | 0, 0xcb1ab31f);
	hash ^= Math.imul(cascadeIndex | 0, 0x165667b1);
	hash ^= hash >>> 16;
	hash = Math.imul(hash, 0x7feb352d);
	hash ^= hash >>> 15;
	return ((hash >>> 0) / 0x100000000) * Math.PI * 2;
}
