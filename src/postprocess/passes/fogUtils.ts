import { clamp } from "../../maths/Common";
import { finiteOr } from "../../maths/Misc";
import {
	DEFAULT_FOG_OPTIONS,
	type FogOptions,
} from "../../pipeline/types";

/**
 * Resolves the numeric fog mode code shared by WebGPU and WebGL shaders.
 *
 * @param mode User-provided fog mode.
 * @returns `0` for linear, `1` for exponential, and `2` for exponential-squared.
 * @sideEffects None.
 */
export function resolveFogModeCode(mode: FogOptions["mode"] | undefined): number {
	switch (mode) {
		case "exp":
			return 1;
		case "exp2":
			return 2;
		default:
			return 0;
	}
}

/**
 * Resolves fog uniforms shared by scene and post-process fog paths.
 *
 * @param options User-provided fog options.
 * @param enabled Whether fog strength should be applied.
 * @param params0 Destination for mode/start/end/density.
 * @param params1 Destination for color/strength.
 * @returns The destination tuple passed in by the caller.
 * @sideEffects Mutates `params0` and `params1`.
 */
export function resolveFogUniformParams(
	options: FogOptions | undefined,
	enabled: boolean,
	params0: Float32Array,
	params1: Float32Array
): readonly [Float32Array, Float32Array] {
	const source = options ?? DEFAULT_FOG_OPTIONS;
	const color = source.color ?? DEFAULT_FOG_OPTIONS.color;
	const start = Math.max(
		0,
		finiteOr(source.start, DEFAULT_FOG_OPTIONS.start)
	);
	const end = Math.max(
		start + 1e-4,
		finiteOr(source.end, DEFAULT_FOG_OPTIONS.end)
	);
	const density = Math.max(
		0,
		finiteOr(source.density, DEFAULT_FOG_OPTIONS.density)
	);
	const strength = enabled ?
		Math.max(0, finiteOr(source.strength, DEFAULT_FOG_OPTIONS.strength))
	:	0;

	params0[0] = resolveFogModeCode(source.mode);
	params0[1] = start;
	params0[2] = end;
	params0[3] = density;

	params1[0] = clamp(finiteOr(color[0], DEFAULT_FOG_OPTIONS.color[0]), 0, 1);
	params1[1] = clamp(finiteOr(color[1], DEFAULT_FOG_OPTIONS.color[1]), 0, 1);
	params1[2] = clamp(finiteOr(color[2], DEFAULT_FOG_OPTIONS.color[2]), 0, 1);
	params1[3] = strength;
	return [params0, params1];
}

/**
 * Resolves fog uniforms into one contiguous WebGPU parameter buffer.
 *
 * @param options User-provided fog options.
 * @param enabled Whether fog strength should be applied.
 * @param data Destination with at least eight floats.
 * @returns The populated destination.
 * @sideEffects Mutates `data`.
 */
export function resolveFogParamData(
	options: FogOptions | undefined,
	enabled: boolean,
	data: Float32Array
): Float32Array {
	const params0 = data.subarray(0, 4);
	const params1 = data.subarray(4, 8);
	resolveFogUniformParams(options, enabled, params0, params1);
	return data;
}
