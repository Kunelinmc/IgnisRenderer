import { Texture } from "../core/Texture";
import { SH } from "../maths/SH";
import { sRGBToLinear, lerp } from "../maths/Common";
import { Vector3 } from "../maths/Vector3";
import { hammersley, importanceSampleGGX_VNDF } from "../maths/Sampling";
import type { SHCoefficients, IVector3 } from "../maths/types";

export const LIGHT_PROBE_MAX_SAMPLE_WIDTH = 128;
export const LIGHT_PROBE_MAX_SAMPLE_HEIGHT = 64;
export const LIGHT_PROBE_MAX_MIP_LEVELS = 5;

const SRGB_TO_LINEAR_LUT = createSRGBToLinearLUT();

interface MutableRGB {
	r: number;
	g: number;
	b: number;
}

export interface LightProbePrefilterMipData {
	level: number;
	width: number;
	height: number;
	data: Float32Array;
}

function createAbortError(): Error {
	const error = new Error("Operation was aborted");
	error.name = "AbortError";
	return error;
}

function assertNotAborted(signal?: AbortSignal | null): void {
	if (!signal?.aborted) return;
	throw createAbortError();
}

function createSRGBToLinearLUT(): Float32Array {
	const lut = new Float32Array(256);
	for (let i = 0; i < lut.length; i++) {
		lut[i] = sRGBToLinear(i / 255);
	}
	return lut;
}

function decodeSRGBToLinear01(value255: number): number {
	if (
		value255 >= 0 &&
		value255 <= 255 &&
		Number.isInteger(value255)
	) {
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

	const sampleWidth = Math.min(width, LIGHT_PROBE_MAX_SAMPLE_WIDTH);
	const sampleHeight = Math.min(height, LIGHT_PROBE_MAX_SAMPLE_HEIGHT);

	const stepX = width / sampleWidth;
	const stepY = height / sampleHeight;

	const dTheta = Math.PI / sampleHeight;
	const dPhi = (2 * Math.PI) / sampleWidth;
	const isLinear =
		envMap.colorSpace === "HDR" || envMap.colorSpace === "Linear";

	let totalWeight = 0;
	for (let sj = 0; sj < sampleHeight; sj++) {
		assertNotAborted(signal);
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
				isLinear ? data[idx] * 255 : decodeSRGBToLinear01(data[idx]) * 255;
			const g =
				isLinear ?
					data[idx + 1] * 255
				:	decodeSRGBToLinear01(data[idx + 1]) * 255;
			const b =
				isLinear ?
					data[idx + 2] * 255
				:	decodeSRGBToLinear01(data[idx + 2]) * 255;

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

export function resolvePrefilterBaseDimensions(envMap: Texture): {
	baseWidth: number;
	baseHeight: number;
} {
	return {
		baseWidth: Math.min(envMap.width, LIGHT_PROBE_MAX_SAMPLE_WIDTH),
		baseHeight: Math.min(envMap.height, LIGHT_PROBE_MAX_SAMPLE_HEIGHT),
	};
}

export function prefilterEnvMapMipLevel(
	envMap: Texture,
	level: number,
	baseWidth: number,
	baseHeight: number,
	maxMipLevels: number = LIGHT_PROBE_MAX_MIP_LEVELS,
	signal?: AbortSignal | null
): LightProbePrefilterMipData {
	assertNotAborted(signal);
	const roughness = level / (maxMipLevels - 1);
	const sampleCount = Math.floor(lerp(1024, 64, roughness));
	const width = Math.max(1, baseWidth >> level);
	const height = Math.max(1, baseHeight >> level);
	const data = new Float32Array(width * height * 4);

	const normal: IVector3 = { x: 0, y: 0, z: 0 };
	const radiance: MutableRGB = { r: 0, g: 0, b: 0 };

	for (let j = 0; j < height; j++) {
		assertNotAborted(signal);
		const theta = ((j + 0.5) / height) * Math.PI;
		for (let i = 0; i < width; i++) {
			const phi = ((i + 0.5) / width) * 2 * Math.PI;
			normal.x = Math.sin(theta) * Math.sin(phi);
			normal.y = Math.cos(theta);
			normal.z = Math.sin(theta) * Math.cos(phi);

			prefilterSpecular(envMap, normal, roughness, sampleCount, radiance);
			const idx = (j * width + i) * 4;
			data[idx] = radiance.r;
			data[idx + 1] = radiance.g;
			data[idx + 2] = radiance.b;
			data[idx + 3] = 1;
		}
	}

	return {
		level,
		width,
		height,
		data,
	};
}

export function buildPrefilteredTexture(
	baseWidth: number,
	baseHeight: number,
	mipData: LightProbePrefilterMipData[]
): Texture {
	const prefiltered = new Texture(null, baseWidth, baseHeight, "HDR");
	const sorted = [...mipData].sort((left, right) => left.level - right.level);
	prefiltered.mipmaps = sorted.map((mip) => mip.data);
	prefiltered.data = prefiltered.mipmaps[0] ?? null;
	return prefiltered;
}

export function prefilterEnvMapCPU(
	envMap: Texture,
	signal?: AbortSignal | null,
	onMipComplete?: (level: number, total: number) => void
): Texture {
	const { baseWidth, baseHeight } = resolvePrefilterBaseDimensions(envMap);
	const mipmaps: LightProbePrefilterMipData[] = [];
	for (let level = 0; level < LIGHT_PROBE_MAX_MIP_LEVELS; level++) {
		const mip = prefilterEnvMapMipLevel(
			envMap,
			level,
			baseWidth,
			baseHeight,
			LIGHT_PROBE_MAX_MIP_LEVELS,
			signal
		);
		mipmaps.push(mip);
		onMipComplete?.(level, LIGHT_PROBE_MAX_MIP_LEVELS);
	}
	return buildPrefilteredTexture(baseWidth, baseHeight, mipmaps);
}

function prefilterSpecular(
	envMap: Texture,
	normal: IVector3,
	roughness: number,
	sampleCount: number,
	outColor: MutableRGB
): void {
	let totalWeight = 0;
	outColor.r = 0;
	outColor.g = 0;
	outColor.b = 0;

	for (let i = 0; i < sampleCount; i++) {
		const xi = hammersley(i, sampleCount);
		const view = normal;
		const half = importanceSampleGGX_VNDF(xi, view, normal, roughness);
		const nDotH = Math.max(Vector3.dot(normal, half), 0);
		const lightDir = Vector3.normalize({
			x: 2.0 * nDotH * half.x - view.x,
			y: 2.0 * nDotH * half.y - view.y,
			z: 2.0 * nDotH * half.z - view.z,
		});

		const nDotL = Math.max(Vector3.dot(normal, lightDir), 0);
		if (nDotL <= 0) continue;

		const phi = Math.atan2(lightDir.x, lightDir.z);
		const theta = Math.acos(Math.max(-1, Math.min(1, lightDir.y)));
		const u = (phi + Math.PI) / (2 * Math.PI);
		const v = theta / Math.PI;
		const sample = envMap.sample(u, v);

		const isLinear =
			envMap.colorSpace === "HDR" || envMap.colorSpace === "Linear";
		const r = isLinear ? sample.r / 255 : decodeSRGBToLinear01(sample.r);
		const g = isLinear ? sample.g / 255 : decodeSRGBToLinear01(sample.g);
		const b = isLinear ? sample.b / 255 : decodeSRGBToLinear01(sample.b);

		outColor.r += r * nDotL;
		outColor.g += g * nDotL;
		outColor.b += b * nDotL;
		totalWeight += nDotL;
	}

	if (totalWeight > 0) {
		outColor.r /= totalWeight;
		outColor.g /= totalWeight;
		outColor.b /= totalWeight;
	}
}
