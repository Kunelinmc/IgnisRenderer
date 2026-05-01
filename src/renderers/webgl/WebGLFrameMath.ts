import { clamp } from "../../maths/Common";
import { Matrix4 } from "../../maths/Matrix4";
import type { Matrix3Arr } from "../../maths/types";
import { TAA_JITTER_SEQUENCE_LENGTH } from "../constants";
import {
	WEBGL_MAX_LOCAL_LIGHT_PROBES,
	WEBGL_MAX_REFLECTION_PROBES,
} from "./constants";
import type {
	WebGLLocalLightProbeUniform,
	WebGLReflectionProbeUniform,
	WebGLShadowData,
} from "./WebGLLightCollector";

const TAA_HALTON_X = [
	0.5,
	0.25,
	0.75,
	0.125,
	0.625,
	0.375,
	0.875,
	0.0625,
	0.5625,
	0.3125,
	0.8125,
	0.1875,
	0.6875,
	0.4375,
	0.9375,
	0.03125,
];
const TAA_HALTON_Y = [
	0.333333,
	0.666667,
	0.111111,
	0.444444,
	0.777778,
	0.222222,
	0.555556,
	0.888889,
	0.037037,
	0.37037,
	0.703704,
	0.148148,
	0.481481,
	0.814815,
	0.259259,
	0.592593,
];

export function computeHaltonJitterNDC(
	index: number,
	width: number,
	height: number
): [number, number] {
	const idx = index % TAA_JITTER_SEQUENCE_LENGTH;
	return [
		((TAA_HALTON_X[idx] - 0.5) / width) * 2,
		((TAA_HALTON_Y[idx] - 0.5) / height) * 2,
	];
}

export function flattenVec4<T>(
	values: T[],
	mapper: (value: T) => [number, number, number, number],
	maxCount: number
): Float32Array {
	const resolvedMaxCount =
		typeof maxCount === "number" && Number.isFinite(maxCount) ?
			Math.max(0, Math.floor(maxCount))
		:	0;
	const packed = new Float32Array(resolvedMaxCount * 4);
	const count = Math.min(resolvedMaxCount, values.length);
	for (let i = 0; i < count; i++) {
		const value = mapper(values[i]);
		const offset = i * 4;
		packed[offset] = value[0];
		packed[offset + 1] = value[1];
		packed[offset + 2] = value[2];
		packed[offset + 3] = value[3];
	}
	return packed;
}

export function flattenShadowViewProjection(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 16);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const matrix = values[i]?.viewProjectionMatrix;
		if (!matrix) {
			continue;
		}
		packed.set(toColumnMajorMat4(matrix), i * 16);
	}
	return packed;
}

export function flattenShadowCascadeViewProjection(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const cascadesPerLight = 4;
	const packed = new Float32Array(maxCount * cascadesPerLight * 16);
	const count = Math.min(maxCount, values.length);
	for (let lightIndex = 0; lightIndex < count; lightIndex++) {
		const shadow = values[lightIndex];
		if (!shadow?.enabled) {
			continue;
		}
		const cascades = shadow.cascadeViewProjectionMatrices ?? [];
		for (
			let cascadeIndex = 0;
			cascadeIndex < Math.min(cascadesPerLight, cascades.length);
			cascadeIndex++
		) {
			const matrix = cascades[cascadeIndex];
			if (!matrix) {
				continue;
			}
			const offset = (lightIndex * cascadesPerLight + cascadeIndex) * 16;
			packed.set(toColumnMajorMat4(matrix), offset);
		}
	}
	return packed;
}

export function flattenShadowCascadeSplits(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const cascadesPerLight = 4;
	const packed = new Float32Array(maxCount * cascadesPerLight * 4);
	const count = Math.min(maxCount, values.length);
	for (let lightIndex = 0; lightIndex < count; lightIndex++) {
		const shadow = values[lightIndex];
		const splits = shadow.cascadeSplits ?? [];
		for (
			let cascadeIndex = 0;
			cascadeIndex < Math.min(cascadesPerLight, splits.length);
			cascadeIndex++
		) {
			const split = splits[cascadeIndex];
			const offset = (lightIndex * cascadesPerLight + cascadeIndex) * 4;
			packed[offset] = finiteOr(split[0], 0);
			packed[offset + 1] = finiteOr(split[1], 0);
			packed[offset + 2] = finiteOr(split[2], 0);
			packed[offset + 3] = finiteOr(split[3], 0);
		}
	}
	return packed;
}

export function flattenShadowParamsA(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 4);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const shadow = values[i];
		const offset = i * 4;
		packed[offset] = shadow.enabled ? 1 : 0;
		packed[offset + 1] = finiteOr(shadow.depthBias, 0);
		packed[offset + 2] = finiteOr(shadow.normalBias, 0);
		packed[offset + 3] = finiteOr(shadow.normalBiasMin, 0);
	}
	return packed;
}

export function flattenShadowParamsB(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 4);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const shadow = values[i];
		const offset = i * 4;
		packed[offset] = finiteOr(shadow.pcfRadius, 0);
		packed[offset + 1] = finiteOr(shadow.shadowStrength, 0);
		packed[offset + 2] = finiteOr(shadow.shadowMapSize, 0);
		packed[offset + 3] = finiteOr(shadow.atlasTileSize, 0);
	}
	return packed;
}

export function flattenShadowParamsC(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 4);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const shadow = values[i];
		const offset = i * 4;
		const isCSM =
			shadow.enabled &&
			shadow.strategyType === "csm" &&
			shadow.cascadeCount > 1;
		const cascadeCount =
			isCSM ? Math.max(1, Math.min(4, shadow.cascadeCount | 0)) : 1;
		packed[offset] = finiteOr(shadow.slopeBias, 0);
		packed[offset + 1] = isCSM ? 1 : 0;
		packed[offset + 2] = cascadeCount;
		packed[offset + 3] =
			isCSM ? clamp(finiteOr(shadow.cascadeBlendRatio, 0), 0, 1) : 0;
	}
	return packed;
}

export function flattenShadowParamsD(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 4);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const shadow = values[i];
		const offset = i * 4;
		packed[offset] = shadow.pcssEnabled ? 1 : 0;
		packed[offset + 1] = finiteOr(shadow.pcssRadius, 0);
		packed[offset + 2] = finiteOr(shadow.shadowSamples, 0);
		packed[offset + 3] = finiteOr(shadow.shadowSearchSamples, 0);
	}
	return packed;
}

export function flattenReflectionProbeRows(
	values: WebGLReflectionProbeUniform[],
	matrixKey: "worldToProbeMatrix" | "probeToWorldMatrix",
	row: 0 | 1 | 2
): Float32Array {
	const packed = new Float32Array(WEBGL_MAX_REFLECTION_PROBES * 4);
	const count = Math.min(WEBGL_MAX_REFLECTION_PROBES, values.length);
	for (let i = 0; i < count; i++) {
		const matrix = values[i][matrixKey].elements;
		const offset = i * 4;
		packed[offset] = finiteOr(matrix[row][0], 0);
		packed[offset + 1] = finiteOr(matrix[row][1], 0);
		packed[offset + 2] = finiteOr(matrix[row][2], 0);
		packed[offset + 3] = finiteOr(matrix[row][3], 0);
	}
	return packed;
}

export function flattenReflectionProbeVec4(
	values: WebGLReflectionProbeUniform[],
	mapper: (probe: WebGLReflectionProbeUniform) => [number, number, number, number]
): Float32Array {
	const packed = new Float32Array(WEBGL_MAX_REFLECTION_PROBES * 4);
	const count = Math.min(WEBGL_MAX_REFLECTION_PROBES, values.length);
	for (let i = 0; i < count; i++) {
		const mapped = mapper(values[i]);
		const offset = i * 4;
		packed[offset] = finiteOr(mapped[0], 0);
		packed[offset + 1] = finiteOr(mapped[1], 0);
		packed[offset + 2] = finiteOr(mapped[2], 0);
		packed[offset + 3] = finiteOr(mapped[3], 0);
	}
	return packed;
}

export function flattenLocalLightProbeRows(
	values: WebGLLocalLightProbeUniform[],
	row: 0 | 1 | 2
): Float32Array {
	const packed = new Float32Array(WEBGL_MAX_LOCAL_LIGHT_PROBES * 4);
	const count = Math.min(WEBGL_MAX_LOCAL_LIGHT_PROBES, values.length);
	for (let i = 0; i < count; i++) {
		const matrix = values[i].worldToProbeMatrix.elements;
		const offset = i * 4;
		packed[offset] = finiteOr(matrix[row][0], 0);
		packed[offset + 1] = finiteOr(matrix[row][1], 0);
		packed[offset + 2] = finiteOr(matrix[row][2], 0);
		packed[offset + 3] = finiteOr(matrix[row][3], 0);
	}
	return packed;
}

export function flattenLocalLightProbeVec4(
	values: WebGLLocalLightProbeUniform[],
	mapper: (probe: WebGLLocalLightProbeUniform) => [number, number, number, number]
): Float32Array {
	const packed = new Float32Array(WEBGL_MAX_LOCAL_LIGHT_PROBES * 4);
	const count = Math.min(WEBGL_MAX_LOCAL_LIGHT_PROBES, values.length);
	for (let i = 0; i < count; i++) {
		const mapped = mapper(values[i]);
		const offset = i * 4;
		packed[offset] = finiteOr(mapped[0], 0);
		packed[offset + 1] = finiteOr(mapped[1], 0);
		packed[offset + 2] = finiteOr(mapped[2], 0);
		packed[offset + 3] = finiteOr(mapped[3], 0);
	}
	return packed;
}

export function getMaxShadowSize(values: WebGLShadowData[]): number {
	let maxSize = 0;
	for (const shadow of values) {
		if (!shadow.enabled || !shadow.shadowMap) {
			continue;
		}
		maxSize = Math.max(maxSize, shadow.shadowMapBaseSize | 0);
	}
	return maxSize;
}

export function toColumnMajorMat4(matrix: Matrix4 | number[][]): Float32Array {
	const elements = matrix instanceof Array ? matrix : matrix.elements;
	return new Float32Array([
		elements[0][0],
		elements[1][0],
		elements[2][0],
		elements[3][0],
		elements[0][1],
		elements[1][1],
		elements[2][1],
		elements[3][1],
		elements[0][2],
		elements[1][2],
		elements[2][2],
		elements[3][2],
		elements[0][3],
		elements[1][3],
		elements[2][3],
		elements[3][3],
	]);
}

export function toFiniteColumnMajorMat4(
	matrix: Matrix4 | number[][]
): Float32Array | null {
	const values = toColumnMajorMat4(matrix);
	for (let i = 0; i < values.length; i++) {
		if (!Number.isFinite(values[i])) {
			return null;
		}
	}
	return values;
}

export function toColumnMajorMat3(matrix: Matrix4 | Matrix3Arr): Float32Array | null {
	const rows: number[][] =
		matrix instanceof Array ? matrix : (matrix as Matrix4).elements;
	if (!rows || rows.length < 3) {
		return null;
	}
	const values = [
		rows[0][0],
		rows[1][0],
		rows[2][0],
		rows[0][1],
		rows[1][1],
		rows[2][1],
		rows[0][2],
		rows[1][2],
		rows[2][2],
	];
	for (let i = 0; i < values.length; i++) {
		if (!Number.isFinite(values[i])) {
			return null;
		}
	}
	return new Float32Array(values);
}

export function isFiniteMatrix(matrix: Matrix4): boolean {
	const elements = matrix.elements;
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			if (!Number.isFinite(elements[row][col])) {
				return false;
			}
		}
	}
	return true;
}

export function finiteOr(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function sanitizeFiniteClamped(
	value: unknown,
	fallback: number,
	minValue: number,
	maxValue: number
): number {
	return clamp(finiteOr(value, fallback), minValue, maxValue);
}

export function sanitizeFloat32Array(
	values: Float32Array,
	fallback: number
): {
	values: Float32Array;
	hadInvalid: boolean;
} {
	let hadInvalid = false;
	for (let i = 0; i < values.length; i++) {
		if (!Number.isFinite(values[i])) {
			values[i] = fallback;
			hadInvalid = true;
		}
	}
	return { values, hadInvalid };
}

export function clampDownsample(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(8, Math.max(1, Math.floor(value)));
}

export function toSafeDimension(value: unknown): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return 1;
	}
	return Math.max(1, Math.floor(value));
}
