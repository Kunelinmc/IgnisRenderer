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
import {
	MAX_NDC_DEPTH,
	MIN_CLIP_W,
	MIN_NDC_DEPTH,
} from "./SoftwareShadowConstants";
import {
	linearizeShadowNdcDepth,
	resolveShadowDepthProjectionParams,
	resolveShadowFilterDiskSample,
	resolveShadowSampleRotation,
	resolveShadowSearchDiskSample,
	SHADOW_PCF_RADIUS_TEXELS,
	SHADOW_PCSS_CONTACT_THRESHOLD_TEXELS,
	SHADOW_PCSS_MAX_PENUMBRA_TEXELS,
	SHADOW_PCSS_SEARCH_RADIUS_TEXELS,
	SHADOW_SAMPLING_PRESETS,
} from "../../lights/shadows/shadowSampling";

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

	if (w <= MIN_CLIP_W) {
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
		currentDepth < MIN_NDC_DEPTH ||
		currentDepth > MAX_NDC_DEPTH
	) {
		return { r: 1.0, g: 1.0, b: 1.0 };
	}

	const constantBias = biasSettings.constant ?? 0.008;
	const slopeBias = biasSettings.slope ?? 0.03;
	const texelBias = (biasSettings.texel ?? 1.0) * (2.0 / size);
	const maxBias = biasSettings.max ?? 0.05;

	const rawBias = normal
		? Math.min(
				maxBias,
				constantBias +
					slopeBias * (1.0 - Vector3.dot(Vector3.normalize(normal), L)) +
					texelBias,
			)
		: Math.min(maxBias, constantBias + texelBias);
	const isPerspective = Math.abs(projectionMatrix.elements[3][2] + 1) < 1e-6;
	const depthBiasScale =
		shadow.effectiveTechnique === "cascaded" &&
		shadow.slices.length > 1 &&
		!isPerspective
			? Math.min(1, resolveDirectionalCascadeDepthBiasScale(viewProjectionMatrix) * 2)
			: 1;
	const bias = rawBias * depthBiasScale;
	const quality = params.quality ?? "medium";
	const preset = SHADOW_SAMPLING_PRESETS[quality];
	const texelPositionX = u * (size - 1);
	const texelPositionY = v * (size - 1);
	const rotation = resolveShadowSampleRotation(0, slice.index);
	const rotationCos = Math.cos(rotation);
	const rotationSin = Math.sin(rotation);
	const projectionParams = resolveShadowDepthProjectionParams(projectionMatrix);

	const sampleFiltered = (sampleCount: number, radius: number): RGB => {
		let visibilityR = 0;
		let visibilityG = 0;
		let visibilityB = 0;
		for (let index = 0; index < sampleCount; index++) {
			const offset = resolveShadowFilterDiskSample(
				index,
				sampleCount,
				rotationCos,
				rotationSin,
			);
			const sampleX = Math.max(
				0,
				Math.min(size - 1, texelPositionX + offset[0] * radius),
			);
			const sampleY = Math.max(
				0,
				Math.min(size - 1, texelPositionY + offset[1] * radius),
			);
			const x0 = Math.floor(sampleX);
			const y0 = Math.floor(sampleY);
			const x1 = Math.min(size - 1, x0 + 1);
			const y1 = Math.min(size - 1, y0 + 1);
			const fx = sampleX - x0;
			const fy = sampleY - y0;
			const compare = (x: number, y: number): number =>
				currentDepth - bias <= buffer[y * size + x] ? 1 : 0;
			const top = compare(x0, y0) * (1 - fx) + compare(x1, y0) * fx;
			const bottom = compare(x0, y1) * (1 - fx) + compare(x1, y1) * fx;
			const visibility = top * (1 - fy) + bottom * fy;
			const tx = Math.max(0, Math.min(size - 1, Math.round(sampleX)));
			const ty = Math.max(0, Math.min(size - 1, Math.round(sampleY)));
			const transmittanceOffset = (ty * size + tx) * 3;
			visibilityR += visibility * transmissionBuffer[transmittanceOffset];
			visibilityG += visibility * transmissionBuffer[transmittanceOffset + 1];
			visibilityB += visibility * transmissionBuffer[transmittanceOffset + 2];
		}
		const invCount = 1 / Math.max(sampleCount, 1);
		const strength = clamp(shadow.definition.strength);
		return {
			r: clamp(1 - strength + strength * visibilityR * invCount),
			g: clamp(1 - strength + strength * visibilityG * invCount),
			b: clamp(1 - strength + strength * visibilityB * invCount),
		};
	};

	if (shadow.effectiveFilterMode !== "pcss") {
		return sampleFiltered(preset.pcfSamples, SHADOW_PCF_RADIUS_TEXELS);
	}

	let blockerCount = 0;
	let blockerDistanceSum = 0;
	for (let index = 0; index < preset.pcssSearchSamples; index++) {
		const offset = resolveShadowSearchDiskSample(
			index,
			rotationCos,
			rotationSin,
		);
		const sampleX = Math.max(0, Math.min(
			size - 1,
			Math.round(texelPositionX + offset[0] * SHADOW_PCSS_SEARCH_RADIUS_TEXELS),
		));
		const sampleY = Math.max(0, Math.min(
			size - 1,
			Math.round(texelPositionY + offset[1] * SHADOW_PCSS_SEARCH_RADIUS_TEXELS),
		));
		const sampleDepth = buffer[sampleY * size + sampleX];
		if (currentDepth - bias > sampleDepth) {
			const blockerDistance = linearizeShadowNdcDepth(
				sampleDepth,
				projectionParams,
			);
			if (Number.isFinite(blockerDistance)) {
				blockerDistanceSum += blockerDistance;
				blockerCount++;
			}
		}
	}
	if (blockerCount === 0) return { r: 1, g: 1, b: 1 };

	const receiverDistance = linearizeShadowNdcDepth(
		currentDepth,
		projectionParams,
	);
	const blockerDistance = blockerDistanceSum / blockerCount;
	const penumbraRatio = Math.max(0, Math.min(
		1,
		(receiverDistance - blockerDistance) / Math.max(blockerDistance, 1e-6),
	));
	const filterRadius = penumbraRatio * SHADOW_PCSS_MAX_PENUMBRA_TEXELS;
	if (filterRadius < SHADOW_PCSS_CONTACT_THRESHOLD_TEXELS) {
		return sampleFiltered(preset.pcfSamples, SHADOW_PCF_RADIUS_TEXELS);
	}
	return sampleFiltered(preset.pcssFilterSamples, filterRadius);
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
