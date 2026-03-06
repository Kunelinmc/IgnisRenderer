import type { IVector3 } from "../../../../maths/types";
import type { RGB } from "../../../../utils/Color";
import type { ShadowCastingLight } from "../../../../lights";
import type { ShadowMap } from "../../../../utils/ShadowMapping";
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
	size: number
): SoftwareShadowRenderTarget {
	let target = runtimeMap.get(light);
	if (!target || target.size !== size) {
		target = {
			size,
			depthBuffer: new Float32Array(size * size),
			transmissionBuffer: new Float32Array(size * size * 3),
		};
		runtimeMap.set(light, target);
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

export function createSoftwareShadowSampler(
	shadowMaps: Map<ShadowCastingLight, ShadowMap>,
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

		const shadowMap = shadowMaps.get(light);
		const runtimeTarget = runtimeMap.get(light);
		if (!shadowMap || !runtimeTarget) return { r: 1, g: 1, b: 1 };

		return sampleSoftwareShadow(shadowMap, runtimeTarget, worldPoint, normal);
	};
}
