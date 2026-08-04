import { Texture } from "../../core/Texture";
import { sRGBToLinear } from "../../maths/Common";
import { SH } from "../../maths/SH";
import type { SHCoefficients } from "../../maths/types";
import {
	IBL_PREFILTER_MAX_SAMPLE_HEIGHT,
	IBL_PREFILTER_MAX_SAMPLE_WIDTH,
} from "./IBLPrefilter";
import {
	ensureEnvironmentTextureEquirect,
	isTextureReadyForEnvironment,
} from "../runtime/environmentMapRuntime";

const SRGB_TO_LINEAR_LUT = createSRGBToLinearLUT();

export interface EnvironmentSHProjectionOptions {
	signal?: AbortSignal | null;
	maxSampleWidth?: number;
	maxSampleHeight?: number;
}

function createEnvironmentSHProjectionAbortError(): Error {
	const error = new Error("Environment SH projection was aborted");
	error.name = "AbortError";
	return error;
}

function assertProjectionNotAborted(signal?: AbortSignal | null): void {
	if (!signal?.aborted) return;
	throw createEnvironmentSHProjectionAbortError();
}

function sanitizeProjectionDimension(
	value: number | undefined,
	fallback: number
): number {
	if (!Number.isFinite(value)) {
		return Math.max(1, Math.floor(fallback));
	}
	return Math.max(1, Math.floor(value as number));
}

function createSRGBToLinearLUT(): Float32Array {
	const lut = new Float32Array(256);
	for (let i = 0; i < lut.length; i++) {
		lut[i] = sRGBToLinear(i / 255);
	}
	return lut;
}

function decodeSRGBToLinear01(value255: number): number {
	if (value255 >= 0 && value255 <= 255 && Number.isInteger(value255)) {
		return SRGB_TO_LINEAR_LUT[value255];
	}
	return sRGBToLinear(value255 / 255);
}

function decodeEnvironmentChannelToLinear255(
	value: number,
	colorSpace: Texture["colorSpace"],
	sourceIsFloat: boolean
): number {
	if (sourceIsFloat) {
		const linearValue = colorSpace === "sRGB" ? sRGBToLinear(value) : value;
		return linearValue * 255;
	}
	if (colorSpace === "sRGB") {
		return decodeSRGBToLinear01(value) * 255;
	}
	return value;
}

/**
 * Projects an environment texture into radiance spherical harmonics.
 *
 * @param envMap Source 2D equirectangular or cubemap environment texture.
 * @param options Sampling limits and cancellation options.
 * @returns Radiance SH coefficients sampled from the environment.
 * @constraints Source texture must contain ready pixel data.
 * @sideEffects None.
 */
export function projectEnvironmentTextureToSH(
	envMap: Texture,
	options: EnvironmentSHProjectionOptions = {}
): SHCoefficients {
	assertProjectionNotAborted(options.signal);
	const sampledEnvironment = ensureEnvironmentTextureEquirect(envMap);
	if (
		!sampledEnvironment ||
		!isTextureReadyForEnvironment(sampledEnvironment)
	) {
		throw new Error(
			"Environment SH projection requires a valid environment texture (2D equirect or cubemap)."
		);
	}

	const { width, height, data } = sampledEnvironment;
	if (!data) {
		return SH.empty();
	}

	const sh = SH.empty();
	const sourceIsFloat = data instanceof Float32Array;
	const maxSampleWidth = sanitizeProjectionDimension(
		options.maxSampleWidth,
		IBL_PREFILTER_MAX_SAMPLE_WIDTH
	);
	const maxSampleHeight = sanitizeProjectionDimension(
		options.maxSampleHeight,
		IBL_PREFILTER_MAX_SAMPLE_HEIGHT
	);
	const sampleWidth = Math.min(width, maxSampleWidth);
	const sampleHeight = Math.min(height, maxSampleHeight);
	const stepX = width / sampleWidth;
	const stepY = height / sampleHeight;
	const dTheta = Math.PI / sampleHeight;
	const dPhi = (2 * Math.PI) / sampleWidth;

	let totalWeight = 0;
	for (let sj = 0; sj < sampleHeight; sj++) {
		assertProjectionNotAborted(options.signal);
		const theta = (sj + 0.5) * dTheta;
		const sinTheta = Math.sin(theta);
		const cosTheta = Math.cos(theta);
		const weight = sinTheta * dTheta * dPhi;
		const j = Math.floor((sj + 0.5) * stepY);

		for (let si = 0; si < sampleWidth; si++) {
			const phi = (si + 0.5) * dPhi - Math.PI;
			const x = sinTheta * Math.sin(phi);
			const y = cosTheta;
			const z = sinTheta * Math.cos(phi);
			const basis = SH.evalBasis({ x, y, z });
			const i = Math.floor((si + 0.5) * stepX);
			const idx = (j * width + i) * 4;
			const r = decodeEnvironmentChannelToLinear255(
				data[idx],
				sampledEnvironment.colorSpace,
				sourceIsFloat
			);
			const g = decodeEnvironmentChannelToLinear255(
				data[idx + 1],
				sampledEnvironment.colorSpace,
				sourceIsFloat
			);
			const b = decodeEnvironmentChannelToLinear255(
				data[idx + 2],
				sampledEnvironment.colorSpace,
				sourceIsFloat
			);

			for (let k = 0; k < sh.length; k++) {
				const basisWeight = basis[k] * weight;
				sh[k].r += r * basisWeight;
				sh[k].g += g * basisWeight;
				sh[k].b += b * basisWeight;
			}

			totalWeight += weight;
		}
	}

	const normFactor = (4 * Math.PI) / totalWeight;
	for (let k = 0; k < sh.length; k++) {
		sh[k].r *= normFactor;
		sh[k].g *= normFactor;
		sh[k].b *= normFactor;
	}

	return sh;
}
