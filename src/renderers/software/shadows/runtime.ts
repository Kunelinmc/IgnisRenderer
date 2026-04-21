import type { IVector3 } from "../../../maths/types";
import type { RGB } from "../../../foundation/Color";
import type { ShadowCastingLight } from "../../../lights";
import {
	getPrimaryShadowMap,
	type ShadowMap,
	type ShadowRenderSet,
} from "../../../lights/ShadowMapping";
import { Matrix4 } from "../../../maths/Matrix4";
import { sampleSoftwareShadow } from "./sampling";
import {
	SOFTWARE_SHADOW_RUNTIME_KEY,
	type SoftwareShadowRenderTarget,
	type SoftwareShadowRuntimeMap,
} from "./types";

export function getSoftwareShadowRuntimeMap(
	transient: Map<string, any>
): SoftwareShadowRuntimeMap | null {
	return (
		(transient.get(SOFTWARE_SHADOW_RUNTIME_KEY) as SoftwareShadowRuntimeMap) ??
		null
	);
}

export function setSoftwareShadowRuntimeMap(
	transient: Map<string, any>,
	runtimeMap: SoftwareShadowRuntimeMap
): void {
	transient.set(SOFTWARE_SHADOW_RUNTIME_KEY, runtimeMap);
}

export function ensureSoftwareShadowRenderTarget(
	runtimeMap: SoftwareShadowRuntimeMap,
	light: ShadowCastingLight,
	sliceIndex: number,
	size: number
): SoftwareShadowRenderTarget {
	let targets = runtimeMap.get(light);
	if (!targets) {
		targets = [];
		runtimeMap.set(light, targets);
	}
	let target = targets[sliceIndex];
	if (!target || target.size !== size) {
		target = {
			size,
			depthBuffer: new Float32Array(size * size),
			transmissionBuffer: new Float32Array(size * size * 3),
		};
		targets[sliceIndex] = target;
	}

	return target;
}

export function clearSoftwareShadowRenderTarget(
	target: SoftwareShadowRenderTarget
): void {
	target.depthBuffer.fill(Infinity);
	target.transmissionBuffer.fill(1.0);
}

export function syncSoftwareShadowRuntimeMap(
	runtimeMap: SoftwareShadowRuntimeMap,
	activeLights: ShadowCastingLight[]
): void {
	for (const [light] of runtimeMap) {
		if (!activeLights.includes(light)) {
			runtimeMap.delete(light);
		}
	}
}

export function trimSoftwareShadowRuntimeTargets(
	runtimeMap: SoftwareShadowRuntimeMap,
	light: ShadowCastingLight,
	sliceCount: number
): void {
	const targets = runtimeMap.get(light);
	if (!targets) {
		return;
	}
	targets.length = Math.max(0, sliceCount | 0);
}

function isWorldPointInsideShadowMap(
	shadowMap: ShadowMap,
	worldPoint: IVector3
): boolean {
	if (!shadowMap.viewProjectionMatrix) {
		return false;
	}
	const clip = Matrix4.transformPoint(shadowMap.viewProjectionMatrix, worldPoint);
	if (clip.w <= 1e-6) {
		return false;
	}
	const invW = 1 / clip.w;
	const ndcX = clip.x * invW;
	const ndcY = clip.y * invW;
	const ndcZ = clip.z * invW;
	const uvX = ndcX * 0.5 + 0.5;
	const uvY = 0.5 - ndcY * 0.5;
	return (
		uvX >= 0 &&
		uvX <= 1 &&
		uvY >= 0 &&
		uvY <= 1 &&
		ndcZ >= -1 &&
		ndcZ <= 1
	);
}

function sampleFromRenderSet(
	renderSet: ShadowRenderSet,
	runtimeTargets: SoftwareShadowRenderTarget[] | undefined,
	worldPoint: IVector3,
	normal?: IVector3 | null
): RGB {
	if (!runtimeTargets || runtimeTargets.length <= 0) {
		return { r: 1, g: 1, b: 1 };
	}

	const isCSM =
		renderSet.effectiveStrategyType === "csm" &&
		renderSet.slices.length > 1;
	if (!isCSM) {
		const shadowMap = getPrimaryShadowMap(renderSet);
		const runtimeTarget = runtimeTargets[0];
		if (!shadowMap || !runtimeTarget) {
			return { r: 1, g: 1, b: 1 };
		}
		return sampleSoftwareShadow(shadowMap, runtimeTarget, worldPoint, normal);
	}

	let fallbackShadowMap: ShadowMap | null = null;
	let fallbackRuntimeTarget: SoftwareShadowRenderTarget | null = null;
	for (let index = 0; index < renderSet.slices.length; index++) {
		const slice = renderSet.slices[index];
		const runtimeTarget = runtimeTargets[index];
		if (!runtimeTarget || !slice.shadowMap.viewProjectionMatrix) {
			continue;
		}
		if (!fallbackShadowMap) {
			fallbackShadowMap = slice.shadowMap;
			fallbackRuntimeTarget = runtimeTarget;
		}
		if (isWorldPointInsideShadowMap(slice.shadowMap, worldPoint)) {
			return sampleSoftwareShadow(
				slice.shadowMap,
				runtimeTarget,
				worldPoint,
				normal
			);
		}
	}

	if (fallbackShadowMap && fallbackRuntimeTarget) {
		return sampleSoftwareShadow(
			fallbackShadowMap,
			fallbackRuntimeTarget,
			worldPoint,
			normal
		);
	}

	return { r: 1, g: 1, b: 1 };
}

export function createSoftwareShadowSampler(
	shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>,
	runtimeMap: SoftwareShadowRuntimeMap | null
): (
	light: ShadowCastingLight,
	worldPoint: IVector3,
	normal?: IVector3 | null
) => RGB {
	return (
		light: ShadowCastingLight,
		worldPoint: IVector3,
		normal?: IVector3 | null
	): RGB => {
		if (!runtimeMap) return { r: 1, g: 1, b: 1 };

		const shadowRenderSet = shadowMaps.get(light);
		if (!shadowRenderSet) return { r: 1, g: 1, b: 1 };
		const runtimeTargets = runtimeMap.get(light);
		return sampleFromRenderSet(
			shadowRenderSet,
			runtimeTargets,
			worldPoint,
			normal
		);
	};
}
