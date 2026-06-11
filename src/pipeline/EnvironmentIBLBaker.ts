import { Texture } from "../core/Texture";
import { sRGBToLinear } from "../maths/Common";
import { SH } from "../maths/SH";
import type { SHCoefficients } from "../maths/types";
import {
	IBL_PREFILTER_MAX_MIP_LEVELS,
	IBL_PREFILTER_MAX_SAMPLE_HEIGHT,
	IBL_PREFILTER_MAX_SAMPLE_WIDTH,
	IBLPrefilter,
	resolveIBLPrefilterOptions,
	type IBLPrefilterAcceleration,
	type IBLPrefilterOptions,
} from "./IBLPrefilter";
import {
	ensureEnvironmentTextureEquirect,
	isTextureReadyForEnvironment,
} from "../lights/runtime/environmentMapRuntime";

export {
	IBL_PREFILTER_MAX_MIP_LEVELS,
	IBL_PREFILTER_MAX_SAMPLE_HEIGHT,
	IBL_PREFILTER_MAX_SAMPLE_WIDTH,
	IBLPrefilter,
	buildPrefilteredTexture,
	prefilterEnvironmentIBL,
	prefilterEnvMapCPU,
	prefilterEnvMapMipLevel,
	prefilterEnvMapWithWebGPU,
	resolveIBLPrefilterOptions,
	resolvePrefilterBaseDimensions,
	type IBLPrefilterAcceleration,
	type IBLPrefilterBackendSource,
	type IBLPrefilterConstructorOptions,
	type IBLPrefilterMipData,
	type IBLPrefilterOptions,
	type IBLPrefilterProgress,
	type ResolvedIBLPrefilterOptions,
} from "./IBLPrefilter";

export const ENVIRONMENT_IBL_MAX_SAMPLE_WIDTH = IBL_PREFILTER_MAX_SAMPLE_WIDTH;
export const ENVIRONMENT_IBL_MAX_SAMPLE_HEIGHT = IBL_PREFILTER_MAX_SAMPLE_HEIGHT;
export const ENVIRONMENT_IBL_MAX_MIP_LEVELS = IBL_PREFILTER_MAX_MIP_LEVELS;

const SRGB_TO_LINEAR_LUT = createSRGBToLinearLUT();

export type EnvironmentIBLBakeAcceleration = IBLPrefilterAcceleration;

export interface EnvironmentIBLBakeProgress {
	phase: "project-sh" | "prefilter" | "finalize";
	completed: number;
	total: number;
	detail?: string;
}

export interface EnvironmentIBLBakeOptions
	extends Omit<IBLPrefilterOptions, "onProgress"> {
	onProgress?: (progress: EnvironmentIBLBakeProgress) => void;
	prefilter?: IBLPrefilter | null;
}

export interface BakedEnvironmentIBL {
	sh: SHCoefficients;
	prefilteredMap: Texture;
}

function createEnvironmentIBLBakeAbortError(): Error {
	const error = new Error("Environment IBL bake was aborted");
	error.name = "AbortError";
	return error;
}

function assertBakeNotAborted(signal?: AbortSignal | null): void {
	if (!signal?.aborted) return;
	throw createEnvironmentIBLBakeAbortError();
}

function resolveTextureIsLinear(texture: Texture): boolean {
	return texture.colorSpace === "HDR" || texture.colorSpace === "Linear";
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

export function projectEquirectTextureToSH(
	envMap: Texture,
	signal?: AbortSignal | null
): SHCoefficients {
	if (!envMap || !envMap.data) {
		return SH.empty();
	}

	const { width, height, data } = envMap;
	const sh = SH.empty();
	const sourceIsLinear = resolveTextureIsLinear(envMap);

	const sampleWidth = Math.min(width, ENVIRONMENT_IBL_MAX_SAMPLE_WIDTH);
	const sampleHeight = Math.min(height, ENVIRONMENT_IBL_MAX_SAMPLE_HEIGHT);

	const stepX = width / sampleWidth;
	const stepY = height / sampleHeight;

	const dTheta = Math.PI / sampleHeight;
	const dPhi = (2 * Math.PI) / sampleWidth;

	let totalWeight = 0;
	for (let sj = 0; sj < sampleHeight; sj++) {
		assertBakeNotAborted(signal);
		const theta = (sj + 0.5) * dTheta;
		const sinTheta = Math.sin(theta);
		const cosTheta = Math.cos(theta);
		const weight = sinTheta * dTheta * dPhi;
		const j = Math.floor((sj + 0.5) * stepY);

		for (let si = 0; si < sampleWidth; si++) {
			const phi = (si + 0.5) * dPhi;
			const x = sinTheta * Math.sin(phi);
			const y = cosTheta;
			const z = sinTheta * Math.cos(phi);
			const basis = SH.evalBasis({ x, y, z });
			const i = Math.floor((si + 0.5) * stepX);
			const idx = (j * width + i) * 4;

			const r =
				sourceIsLinear ?
					data[idx] * 255
				: 	decodeSRGBToLinear01(data[idx]) * 255;
			const g =
				sourceIsLinear ?
					data[idx + 1] * 255
				: 	decodeSRGBToLinear01(data[idx + 1]) * 255;
			const b =
				sourceIsLinear ?
					data[idx + 2] * 255
				: 	decodeSRGBToLinear01(data[idx + 2]) * 255;

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

function emitProgress(
	options: EnvironmentIBLBakeOptions,
	progress: EnvironmentIBLBakeProgress
): void {
	options.onProgress?.(progress);
}

export async function bakeEnvironmentIBLFromEnvironmentMap(
	envMap: Texture,
	options: EnvironmentIBLBakeOptions = {}
): Promise<BakedEnvironmentIBL> {
	assertBakeNotAborted(options.signal);
	const sampledEnvironment = ensureEnvironmentTextureEquirect(envMap);
	if (
		!sampledEnvironment ||
		!isTextureReadyForEnvironment(sampledEnvironment)
	) {
		throw new Error(
			"Environment IBL bake requires a valid environment texture (2D equirect or cubemap)."
		);
	}
	const prefilterOptions = resolveIBLPrefilterOptions(options);
	const totalMipLevels = prefilterOptions.maxMipLevels;
	const totalProgress = totalMipLevels + 2;
	let completed = 0;

	const sh = projectEquirectTextureToSH(
		sampledEnvironment,
		options.signal ?? null
	);
	completed++;
	emitProgress(options, {
		phase: "project-sh",
		completed,
		total: totalProgress,
	});

	const prefilter =
		options.prefilter ??
		new IBLPrefilter({
			backend: options.backend ?? null,
			computeSource: options.computeSource ?? null,
		});
	const prefiltered = await prefilter.prefilter(sampledEnvironment, {
		signal: options.signal ?? null,
		acceleration: options.acceleration ?? "auto",
		workerCount: options.workerCount,
		backend: options.backend ?? null,
		computeSource: options.computeSource ?? null,
		maxSampleWidth: prefilterOptions.maxSampleWidth,
		maxSampleHeight: prefilterOptions.maxSampleHeight,
		maxMipLevels: totalMipLevels,
		onProgress: (progress) => {
			completed++;
			emitProgress(options, {
				phase: "prefilter",
				completed,
				total: totalProgress,
				detail: progress.detail,
			});
		},
	});

	assertBakeNotAborted(options.signal);
	completed++;
	emitProgress(options, {
		phase: "finalize",
		completed,
		total: totalProgress,
	});
	return {
		sh,
		prefilteredMap: prefiltered,
	};
}
