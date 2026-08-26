import { Matrix4 } from "../../maths/Matrix4";
import { clamp } from "../../maths/Common";
import type { IVector3 } from "../../maths/types";
import { Vector3 } from "../../maths/Vector3";
import type { RGB } from "../../foundation/Color";
import type { ShadowCastingLight } from "../../lights";
import type {
	PreparedShadowLight,
	PreparedShadowSlice,
	ShadowFramePlan,
} from "../../lights/shadows/ShadowFramePlan";
import type {
	SoftwareShadowRenderTarget,
	SoftwareShadowRuntimeMap,
	SoftwareShadowSampler,
	SoftwareShadowSamplerCamera,
} from "./SoftwareShadowContracts";
import { sampleParticleShadowVolumeTransmittance } from "../../pipeline/ParticleShadowVolume";
import { SoftwareShadowConstants } from "./SoftwareShadowConstants";

interface SoftwareShadowSampleContext {
	worldPoint: IVector3;
	normal?: IVector3 | null;
	shadow: PreparedShadowLight;
	slice: PreparedShadowSlice;
	runtimeTarget: SoftwareShadowRenderTarget;
}

export type SoftwareShadowSampleKernel = (
	shadow: PreparedShadowLight,
	slice: PreparedShadowSlice,
	runtimeTarget: SoftwareShadowRenderTarget,
	worldPoint: IVector3,
	normal?: IVector3 | null,
) => RGB;

interface SoftwareShadowSamplerOptions {
	readonly camera?: SoftwareShadowSamplerCamera | null;
}

/** @internal Selects Software shadow slices and delegates target sampling. */
export function createSoftwareShadowSampler(
	plan: ShadowFramePlan,
	runtimeMap: SoftwareShadowRuntimeMap,
	sampleKernel: SoftwareShadowSampleKernel,
	options: SoftwareShadowSamplerOptions = {},
): SoftwareShadowSampler {
	return (
		light: ShadowCastingLight,
		worldPoint: IVector3,
		normal?: IVector3 | null,
	): RGB => {
		const shadow = plan.lights.find((candidate) => candidate.light === light);
		if (!shadow) return { r: 1, g: 1, b: 1 };
		const runtimeTargets = runtimeMap.get(light);
		return samplePreparedShadow(
			light,
			shadow,
			runtimeTargets,
			worldPoint,
			normal,
			sampleKernel,
			options,
		);
	};
}

function samplePreparedShadow(
	light: ShadowCastingLight,
	shadow: PreparedShadowLight,
	runtimeTargets: SoftwareShadowRenderTarget[] | undefined,
	worldPoint: IVector3,
	normal: IVector3 | null | undefined,
	sampleKernel: SoftwareShadowSampleKernel,
	options: SoftwareShadowSamplerOptions,
): RGB {
	if (!runtimeTargets || runtimeTargets.length === 0) {
		return { r: 1, g: 1, b: 1 };
	}

	const isCSM = shadow.effectiveTechnique === "cascaded" && shadow.slices.length > 1;
	if (!isCSM) {
		const slice = shadow.slices[0];
		const target = runtimeTargets[0];
		return slice && target ?
			sampleKernel(shadow, slice, target, worldPoint, normal)
		: { r: 1, g: 1, b: 1 };
	}

	if (light.type === "directional") {
		const selectedIndex = resolveDirectionalCSMSliceIndex(
			shadow,
			worldPoint,
			options.camera,
		);
		const slice = shadow.slices[selectedIndex];
		const target = runtimeTargets[selectedIndex];
		if (selectedIndex >= 0 && slice && target) {
			return sampleKernel(shadow, slice, target, worldPoint, normal);
		}
	}

	let fallbackSlice: PreparedShadowSlice | null = null;
	let fallbackTarget: SoftwareShadowRenderTarget | null = null;
	for (let index = 0; index < shadow.slices.length; index++) {
		const slice = shadow.slices[index];
		const target = runtimeTargets[index];
		if (!target) continue;
		fallbackSlice ??= slice;
		fallbackTarget ??= target;
		if (isWorldPointInsideShadowMap(slice, worldPoint)) {
			return sampleKernel(shadow, slice, target, worldPoint, normal);
		}
	}

	return fallbackSlice && fallbackTarget ?
		sampleKernel(shadow, fallbackSlice, fallbackTarget, worldPoint, normal)
	: { r: 1, g: 1, b: 1 };
}

function isWorldPointInsideShadowMap(
	slice: PreparedShadowSlice,
	worldPoint: IVector3,
): boolean {
	const clip = Matrix4.transformPoint(slice.viewProjection, worldPoint);
	if (clip.w <= 1e-6) return false;
	const invW = 1 / clip.w;
	const uvX = clip.x * invW * 0.5 + 0.5;
	const uvY = 0.5 - clip.y * invW * 0.5;
	const ndcZ = clip.z * invW;
	return uvX >= 0 && uvX <= 1 && uvY >= 0 && uvY <= 1 && ndcZ >= -1 && ndcZ <= 1;
}

function resolveDirectionalCSMSliceIndex(
	shadow: PreparedShadowLight,
	worldPoint: IVector3,
	camera?: SoftwareShadowSamplerCamera | null,
): number {
	const viewDepth = resolveCameraViewDepth(worldPoint, camera);
	if (!Number.isFinite(viewDepth)) return -1;
	let fallbackIndex = -1;
	for (let index = 0; index < shadow.slices.length; index++) {
		fallbackIndex = fallbackIndex < 0 ? index : fallbackIndex;
		if (viewDepth <= shadow.slices[index].splitFar + 1e-6) return index;
	}
	return fallbackIndex;
}

function resolveCameraViewDepth(
	worldPoint: IVector3,
	camera?: SoftwareShadowSamplerCamera | null,
): number {
	if (!camera) return Number.NaN;
	if (camera.viewMatrix) {
		return Math.max(0, -Matrix4.transformPoint(camera.viewMatrix, worldPoint).z);
	}
	const position = camera.getWorldPosition?.() ?? camera.position;
	if (!position) return Number.NaN;
	if (camera.getWorldDirection) {
		const forward = camera.getWorldDirection({ x: 0, y: 0, z: -1 });
		return Math.max(
			0,
			(worldPoint.x - position.x) * forward.x +
				(worldPoint.y - position.y) * forward.y +
				(worldPoint.z - position.z) * forward.z,
		);
	}
	return Math.max(0, position.z - worldPoint.z);
}

function getVogelSample(index: number, numSamples: number, theta: number) {
	const goldenAngle = 2.400049405230919;
	const r = Math.sqrt((index + 0.5) / numSamples);
	const angle = index * goldenAngle + theta;
	return { x: r * Math.cos(angle), y: r * Math.sin(angle) };
}

function resolveDirectionalCascadeDepthBiasScale(viewProjection: Matrix4): number {
	const row = viewProjection.elements[2];
	const reciprocalDepthRange = Math.hypot(row[0], row[1], row[2]) * 0.5;
	if (!Number.isFinite(reciprocalDepthRange) || reciprocalDepthRange <= 0) {
		return 1;
	}
	return Math.min(1, reciprocalDepthRange);
}


function calculateShadowFactor(ctx: SoftwareShadowSampleContext): RGB {
	const { worldPoint, normal, shadow, slice, runtimeTarget } = ctx;
	const {
		viewProjection: viewProjectionMatrix,
		lightDirection: latestLightDir,
		projection: projectionMatrix,
	} = slice;
	const { bias: biasSettings, sampling: params } = shadow.definition;
	const buffer = runtimeTarget.depthBuffer;
	const transmissionBuffer = runtimeTarget.transmissionBuffer;
	const size = runtimeTarget.size;

	if (!viewProjectionMatrix) return { r: 1.0, g: 1.0, b: 1.0 };

	const L = Vector3.normalize({
		x: -latestLightDir.x,
		y: -latestLightDir.y,
		z: -latestLightDir.z,
	});

	const normalBias = biasSettings.normal ?? 1.0;
	const normalBiasMin = biasSettings.normalMin ?? 0.05;

	let offsetPoint = worldPoint;
	if (normal) {
		const N = Vector3.normalize(normal);
		const cosTheta = Math.max(0, Vector3.dot(N, L));
		const normalOffset = normalBiasMin + (normalBias - normalBiasMin) * (1.0 - cosTheta);
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

	const lightSpacePos = Matrix4.transformPoint(viewProjectionMatrix, offsetPoint);
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

	const constantBias = biasSettings.constant ?? 0.008;
	const slopeBias = biasSettings.slope ?? 0.03;
	const texelBias = (biasSettings.texel ?? 1.0) * (2.0 / size);
	const maxBias = biasSettings.max ?? 0.05;

	const m = projectionMatrix ? projectionMatrix.elements : null;
	const isPerspective = m ? Math.abs(m[3][2] + 1.0) < 1e-6 : false;
	const linearizeDepth = (zNdc: number): number => {
		if (!m) return zNdc;
		if (isPerspective) {
			return m[2][3] / (zNdc + m[2][2]);
		}
		return (m[2][3] - zNdc) / m[2][2];
	};

	const rawBias = normal
		? Math.min(
				maxBias,
				constantBias +
					slopeBias * (1.0 - Vector3.dot(Vector3.normalize(normal), L)) +
					texelBias,
			)
		: Math.min(maxBias, constantBias + texelBias);
	const depthBiasScale =
		shadow.effectiveTechnique === "cascaded" && shadow.slices.length > 1 && !isPerspective
			? Math.min(1, resolveDirectionalCascadeDepthBiasScale(viewProjectionMatrix) * 2)
			: 1;
	const bias = rawBias * depthBiasScale;

	const strength = clamp(params.strength ?? 1.0);
	const pcfRadiusParams = params.radius ?? 0;
	const texelSize = 1.0 / size;

	let visibilityR = 0;
	let visibilityG = 0;
	let visibilityB = 0;
	let validSampleCount = 0;

	if (pcfRadiusParams > 0) {
		const theta =
			(worldPoint.x * 12.9898 + worldPoint.y * 78.233 + worldPoint.z * 37.719) %
			(Math.PI * 2);
		const numSearchSamples = Math.floor(params.searchSamples ?? 16);
		const numSamples = Math.floor(params.samples ?? 16);
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
				shadow: {
					...shadow,
					definition: {
						...shadow.definition,
						sampling: { ...params, radius: 0 },
					},
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
		const pcfRadius = params.pcfRadius ?? 1.5;
		const numSamples = Math.floor(params.samples ?? 16);
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
	shadow: PreparedShadowLight,
	slice: PreparedShadowSlice,
	runtimeTarget: SoftwareShadowRenderTarget,
	worldPoint: IVector3,
	normal?: IVector3 | null,
): RGB {
	const base = calculateShadowFactor({
		worldPoint,
		normal,
		shadow,
		slice,
		runtimeTarget,
	});
	const volumeTransmittance = sampleParticleShadowVolumeTransmittance(
		runtimeTarget.particleVolume,
		slice.viewProjection,
		worldPoint
	);
	return {
		r: base.r * volumeTransmittance,
		g: base.g * volumeTransmittance,
		b: base.b * volumeTransmittance,
	};
}
