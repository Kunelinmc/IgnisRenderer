import type { Texture } from "../../core/Texture";
import { lerp } from "../../maths/Common";
import { hammersley, importanceSampleGGX_VNDF } from "../../maths/Sampling";
import type { IVector3 } from "../../maths/types";
import { Vector3 } from "../../maths/Vector3";

import { sampleEnvironmentTextureLevelLinear } from "../runtime/environmentMapRuntime";
import {
	assertIBLPrefilterNotAborted,
	assertIBLPrefilterSourceRevision,
	type IBLPrefilterExecutionRequest,
	type IBLPrefilterExecutorAvailability,
	type IBLPrefilterExecutorLike,
	type IBLPrefilterMipData,
	type IBLPrefilterMipPlan,
	type IBLPrefilterPlan,
} from "./IBLPrefilterExecutor";

const CPU_MAX_SAMPLE_COUNT = 1024;
const CPU_MIN_SAMPLE_COUNT = 64;
const PREFILTER_EPSILON = 1e-6;
const EQUIRECT_DISTORTION_EPSILON = 1e-4;

interface MutableRGB {
	r: number;
	g: number;
	b: number;
}

/** @internal Lighting-owned synchronous IBL executor. */
export class SingleThreadIBLPrefilterExecutor
	implements IBLPrefilterExecutorLike {
	public readonly id = "single-thread" as const;

	public getAvailability(): IBLPrefilterExecutorAvailability {
		return {
			state: "ready",
			acceptsRequests: true,
			reason: null,
		};
	}

	public execute(request: IBLPrefilterExecutionRequest): IBLPrefilterMipData[] {
		assertIBLPrefilterSourceRevision(
			request.envMap,
			request.sourceRevision,
		);
		return prefilterEnvMapCPUWithPlan(
			request.envMap,
			request.plan,
			request.signal,
			request.onMipComplete,
		);
	}
}

export function prefilterEnvMapMipLevel(
	envMap: Texture,
	mipPlan: IBLPrefilterMipPlan,
	signal?: AbortSignal | null,
): IBLPrefilterMipData {
	assertIBLPrefilterNotAborted(signal);
	const { level, width, height, roughness } = mipPlan;
	const sampleCount = resolveSampleCountByRoughness(
		roughness,
		CPU_MAX_SAMPLE_COUNT,
		CPU_MIN_SAMPLE_COUNT,
	);
	const data = new Float32Array(width * height * 4);
	const normal: IVector3 = { x: 0, y: 0, z: 0 };
	const radiance: MutableRGB = { r: 0, g: 0, b: 0 };

	for (let j = 0; j < height; j++) {
		assertIBLPrefilterNotAborted(signal);
		const theta = ((j + 0.5) / height) * Math.PI;
		for (let i = 0; i < width; i++) {
			const phi = ((i + 0.5) / width) * 2 * Math.PI - Math.PI;
			normal.x = Math.sin(theta) * Math.sin(phi);
			normal.y = Math.cos(theta);
			normal.z = Math.sin(theta) * Math.cos(phi);

			prefilterSpecular(
				envMap,
				normal,
				roughness,
				sampleCount,
				radiance,
			);
			const index = (j * width + i) * 4;
			data[index] = radiance.r;
			data[index + 1] = radiance.g;
			data[index + 2] = radiance.b;
			data[index + 3] = 1;
		}
	}

	return { level, width, height, data };
}

export function prefilterEnvMapCPUWithPlan(
	envMap: Texture,
	plan: IBLPrefilterPlan,
	signal?: AbortSignal | null,
	onMipComplete?: (level: number, total: number) => void,
): IBLPrefilterMipData[] {
	const totalMipLevels = plan.mipLevels.length;
	const mipmaps: IBLPrefilterMipData[] = [];
	for (const mipPlan of plan.mipLevels) {
		const mip = prefilterEnvMapMipLevel(
			envMap,
			mipPlan,
			signal,
		);
		mipmaps.push(mip);
		onMipComplete?.(mipPlan.level, totalMipLevels);
	}
	return mipmaps;
}

function resolveSampleCountByRoughness(
	roughness: number,
	maxSamples: number,
	minSamples: number,
): number {
	const sampleCount = Math.floor(lerp(maxSamples, minSamples, roughness));
	return Math.max(minSamples, Math.min(maxSamples, sampleCount));
}

function distributionGGX(nDotH: number, roughness: number): number {
	const alpha = Math.max(roughness * roughness, 1e-4);
	const alpha2 = alpha * alpha;
	const denominator = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
	return alpha2 /
		Math.max(Math.PI * denominator * denominator, PREFILTER_EPSILON);
}

function computeGGXSamplePDF(
	nDotH: number,
	vDotH: number,
	roughness: number,
): number {
	if (nDotH <= 0 || vDotH <= 0) return PREFILTER_EPSILON;
	const distribution = distributionGGX(nDotH, roughness);
	return Math.max(
		(distribution * nDotH) / Math.max(4 * vDotH, PREFILTER_EPSILON),
		PREFILTER_EPSILON,
	);
}

function computeEquirectTexelSolidAngle(
	sourceWidth: number,
	sourceHeight: number,
	directionY: number,
): number {
	const safeWidth = Math.max(1, sourceWidth);
	const safeHeight = Math.max(1, sourceHeight);
	const sinTheta = Math.sqrt(Math.max(0, 1 - directionY * directionY));
	return (
		2 *
		Math.PI *
		Math.PI *
		Math.max(sinTheta, EQUIRECT_DISTORTION_EPSILON)
	) / (safeWidth * safeHeight);
}

function resolvePrefilterSampleMipLevel(
	envMap: Texture,
	roughness: number,
	sampleCount: number,
	pdf: number,
	directionY: number,
): number {
	const mipCount = Math.max(1, envMap.mipmaps.length || 1);
	if (mipCount <= 1 || roughness <= PREFILTER_EPSILON) return 0;
	const texelSolidAngle = computeEquirectTexelSolidAngle(
		envMap.width,
		envMap.height,
		directionY,
	);
	const sampleSolidAngle = 1 / Math.max(sampleCount * pdf, PREFILTER_EPSILON);
	const lod = 0.5 * Math.log2(sampleSolidAngle / texelSolidAngle);
	return Math.max(0, Math.min(mipCount - 1, lod));
}

function prefilterSpecular(
	envMap: Texture,
	normal: IVector3,
	roughness: number,
	sampleCount: number,
	outColor: MutableRGB,
): void {
	let totalWeight = 0;
	outColor.r = 0;
	outColor.g = 0;
	outColor.b = 0;

	for (let index = 0; index < sampleCount; index++) {
		const xi = hammersley(index, sampleCount);
		const view = normal;
		const half = importanceSampleGGX_VNDF(xi, view, normal, roughness);
		const nDotH = Math.max(Vector3.dot(normal, half), 0);
		const vDotH = Math.max(Vector3.dot(view, half), 0);
		if (vDotH <= 0) continue;
		const lightDirection = Vector3.normalize({
			x: 2 * nDotH * half.x - view.x,
			y: 2 * nDotH * half.y - view.y,
			z: 2 * nDotH * half.z - view.z,
		});

		const nDotL = Math.max(Vector3.dot(normal, lightDirection), 0);
		if (nDotL <= 0) continue;
		const pdf = computeGGXSamplePDF(nDotH, vDotH, roughness);
		const sampleMipLevel = resolvePrefilterSampleMipLevel(
			envMap,
			roughness,
			sampleCount,
			pdf,
			lightDirection.y,
		);
		const sample = sampleEnvironmentTextureLevelLinear(
			envMap,
			lightDirection,
			sampleMipLevel,
		);
		outColor.r += sample.r * nDotL;
		outColor.g += sample.g * nDotL;
		outColor.b += sample.b * nDotL;
		totalWeight += nDotL;
	}

	if (totalWeight <= 0) return;
	outColor.r /= totalWeight;
	outColor.g /= totalWeight;
	outColor.b /= totalWeight;
}
