import { Matrix4 } from "../../../maths/Matrix4";
import { Vector3 } from "../../../maths/Vector3";
import type { IVector3 } from "../../../maths/types";
import type { RGB } from "../../../foundation/Color";
import type { ShadowMap, ShadowParams } from "../../../lights/ShadowMapping";
import { SoftwareShadowConstants } from "./constants";
import type { SoftwareShadowRenderTarget } from "./types";
import { clamp } from "../../../maths/Common";

interface SoftwareShadowSampleContext {
	worldPoint: IVector3;
	normal?: IVector3 | null;
	shadowMap: ShadowMap;
	runtimeTarget: SoftwareShadowRenderTarget;
	params: ShadowParams;
}

function getVogelSample(index: number, numSamples: number, theta: number) {
	const goldenAngle = 2.400049405230919;
	const r = Math.sqrt((index + 0.5) / numSamples);
	const angle = index * goldenAngle + theta;
	return { x: r * Math.cos(angle), y: r * Math.sin(angle) };
}

function calculateShadowFactor(ctx: SoftwareShadowSampleContext): RGB {
	const { worldPoint, normal, shadowMap, runtimeTarget, params } = ctx;
	const { viewProjectionMatrix, latestLightDir, projectionMatrix } = shadowMap;
	const buffer = runtimeTarget.depthBuffer;
	const transmissionBuffer = runtimeTarget.transmissionBuffer;
	const size = runtimeTarget.size;

	if (!viewProjectionMatrix) return { r: 1.0, g: 1.0, b: 1.0 };

	const L = Vector3.normalize({
		x: -latestLightDir.x,
		y: -latestLightDir.y,
		z: -latestLightDir.z,
	});

	const normalBias = params.shadowNormalBias ?? 1.0;
	const normalBiasMin = params.shadowNormalBiasMin ?? 0.05;

	let offsetPoint = worldPoint;
	if (normal) {
		const N = Vector3.normalize(normal);
		const cosTheta = Math.max(0, Vector3.dot(N, L));
		const normalOffset =
			normalBiasMin + (normalBias - normalBiasMin) * (1.0 - cosTheta);
		offsetPoint = {
			x: worldPoint.x + N.x * normalOffset,
			y: worldPoint.y + N.y * normalOffset,
			z: worldPoint.z + N.z * normalOffset,
		};
	} else {
		const volumeOffset = normalBiasMin;
		offsetPoint = {
			x: worldPoint.x + L.x * volumeOffset,
			y: worldPoint.y + L.y * volumeOffset,
			z: worldPoint.z + L.z * volumeOffset,
		};
	}

	const lightSpacePos = Matrix4.transformPoint(
		viewProjectionMatrix,
		offsetPoint
	);
	const w = lightSpacePos.w;
	if (w <= SoftwareShadowConstants.MIN_CLIP_W) {
		return { r: 1.0, g: 1.0, b: 1.0 };
	}
	const invW = 1 / w;
	const ndcX = lightSpacePos.x * invW;
	const ndcY = lightSpacePos.y * invW;
	const ndcZ = lightSpacePos.z * invW;

	const u = ndcX * 0.5 + 0.5;
	const v = 0.5 - ndcY * 0.5;
	const currentDepth = ndcZ;

	if (
		u < 0 ||
		u > 1 ||
		v < 0 ||
		v > 1 ||
		currentDepth < SoftwareShadowConstants.MIN_NDC_DEPTH ||
		currentDepth > SoftwareShadowConstants.MAX_NDC_DEPTH
	) {
		return { r: 1.0, g: 1.0, b: 1.0 };
	}

	const constantBias = params.shadowBias ?? 0.008;
	const slopeBias = params.shadowSlopeBias ?? 0.03;
	const texelBias = (params.shadowTexelBias ?? 1.0) * (2.0 / size);
	const maxBias = params.shadowMaxBias ?? 0.05;

	const m = projectionMatrix ? projectionMatrix.elements : null;
	const isPerspective = m ? Math.abs(m[3][2] + 1.0) < 1e-6 : false;
	const linearizeDepth = (zNdc: number): number => {
		if (!m) return zNdc;
		if (isPerspective) {
			return m[2][3] / (zNdc + m[2][2]);
		}
		return (m[2][3] - zNdc) / m[2][2];
	};

	const bias =
		normal ?
			Math.min(
				maxBias,
				constantBias +
					slopeBias * (1.0 - Vector3.dot(Vector3.normalize(normal), L)) +
					texelBias
			)
		:	Math.min(maxBias, constantBias + texelBias);

	const strength = clamp(params.shadowStrength ?? 1.0);
	const pcfRadiusParams = params.shadowRadius ?? 0;
	const texelSize = 1.0 / size;

	let visibilityR = 0;
	let visibilityG = 0;
	let visibilityB = 0;
	let validSampleCount = 0;

	if (pcfRadiusParams > 0) {
		const theta =
			(worldPoint.x * 12.9898 + worldPoint.y * 78.233 + worldPoint.z * 37.719) %
			(Math.PI * 2);
		const numSearchSamples = Math.floor(params.shadowSearchSamples ?? 16);
		const numSamples = Math.floor(params.shadowSamples ?? 16);
		const maxRadiusUV = pcfRadiusParams * texelSize;

		let numBlockers = 0;
		let avgBlockerDepth = 0;
		for (let i = 0; i < numSearchSamples; i++) {
			const offset = getVogelSample(i, numSearchSamples, theta);
			const su = u + offset.x * maxRadiusUV;
			const sv = v + offset.y * maxRadiusUV;
			if (su >= 0 && su <= 1 && sv >= 0 && sv <= 1) {
				const tx = Math.max(0, Math.min(size - 1, Math.floor(su * (size - 1))));
				const ty = Math.max(0, Math.min(size - 1, Math.floor(sv * (size - 1))));
				const shadowDepth = buffer[ty * size + tx];
				if (currentDepth - bias > shadowDepth) {
					numBlockers++;
					avgBlockerDepth += shadowDepth;
				}
			}
		}

		if (numBlockers === 0) {
			return { r: 1.0, g: 1.0, b: 1.0 };
		}

		avgBlockerDepth /= numBlockers;
		const linCurrent = linearizeDepth(currentDepth);
		const linBlocker = linearizeDepth(avgBlockerDepth);

		let penumbraRatio = 1.0;
		if (linCurrent > linBlocker) {
			const divergence = isPerspective ? linBlocker || 1e-6 : 100.0;
			penumbraRatio = (linCurrent - linBlocker) / divergence;
			penumbraRatio = Math.max(0.0, Math.min(1.0, penumbraRatio));
		} else {
			penumbraRatio = 0;
		}

		const filterRadiusUV = maxRadiusUV * penumbraRatio;
		if (filterRadiusUV < texelSize * 0.1) {
			return calculateShadowFactor({
				...ctx,
				params: {
					...params,
					shadowRadius: 0,
				},
			});
		}

		for (let i = 0; i < numSamples; i++) {
			const offset = getVogelSample(i, numSamples, theta);
			const su = u + offset.x * filterRadiusUV;
			const sv = v + offset.y * filterRadiusUV;
			if (su < 0 || su > 1 || sv < 0 || sv > 1) continue;

			const tx = Math.max(0, Math.min(size - 1, Math.floor(su * (size - 1))));
			const ty = Math.max(0, Math.min(size - 1, Math.floor(sv * (size - 1))));
			const idx = ty * size + tx;
			const shadowDepth = buffer[idx];

			validSampleCount++;
			const isOccluded = currentDepth - bias > shadowDepth;
			if (isOccluded) {
				visibilityR += 1.0 - strength;
				visibilityG += 1.0 - strength;
				visibilityB += 1.0 - strength;
				continue;
			}

			const cIdx = idx * 3;
			const transSampleR = transmissionBuffer[cIdx];
			const transSampleG = transmissionBuffer[cIdx + 1];
			const transSampleB = transmissionBuffer[cIdx + 2];
			visibilityR += 1.0 - strength + strength * transSampleR;
			visibilityG += 1.0 - strength + strength * transSampleG;
			visibilityB += 1.0 - strength + strength * transSampleB;
		}
	} else {
		const theta =
			(worldPoint.x * 12.9898 + worldPoint.y * 78.233 + worldPoint.z * 37.719) %
			(Math.PI * 2);
		const pcfRadius = params.shadowPCF ?? 1.5;
		const numSamples = Math.floor(params.shadowSamples ?? 16);
		const radiusUV = pcfRadius * texelSize;

		for (let i = 0; i < numSamples; i++) {
			const offset = getVogelSample(i, numSamples, theta);
			const su = u + offset.x * radiusUV;
			const sv = v + offset.y * radiusUV;
			if (su < 0 || su > 1 || sv < 0 || sv > 1) continue;

			const tx = Math.max(0, Math.min(size - 1, Math.floor(su * (size - 1))));
			const ty = Math.max(0, Math.min(size - 1, Math.floor(sv * (size - 1))));
			const idx = ty * size + tx;
			const shadowDepth = buffer[idx];

			validSampleCount++;
			const isOccluded = currentDepth - bias > shadowDepth;
			if (isOccluded) {
				visibilityR += 1.0 - strength;
				visibilityG += 1.0 - strength;
				visibilityB += 1.0 - strength;
				continue;
			}

			const cIdx = idx * 3;
			const transSampleR = transmissionBuffer[cIdx];
			const transSampleG = transmissionBuffer[cIdx + 1];
			const transSampleB = transmissionBuffer[cIdx + 2];
			visibilityR += 1.0 - strength + strength * transSampleR;
			visibilityG += 1.0 - strength + strength * transSampleG;
			visibilityB += 1.0 - strength + strength * transSampleB;
		}
	}

	if (validSampleCount === 0) return { r: 1.0, g: 1.0, b: 1.0 };

	const invCount = 1.0 / validSampleCount;
	return {
		r: clamp(visibilityR * invCount),
		g: clamp(visibilityG * invCount),
		b: clamp(visibilityB * invCount),
	};
}

export function sampleSoftwareShadow(
	shadowMap: ShadowMap,
	runtimeTarget: SoftwareShadowRenderTarget,
	worldPoint: IVector3,
	normal?: IVector3 | null
): RGB {
	return calculateShadowFactor({
		worldPoint,
		normal,
		shadowMap,
		runtimeTarget,
		params: shadowMap.params,
	});
}
