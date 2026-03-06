import { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import type { ShadowCastingLight } from "../../lights";
import { ShadowMap } from "../../utils/ShadowMapping";

interface SceneBounds {
	center: IVector3;
	radius: number;
}

function resetShadowMapMetadata(shadowMap: ShadowMap): void {
	shadowMap.viewMatrix = null;
	shadowMap.projectionMatrix = null;
	shadowMap.viewProjectionMatrix = null;
	shadowMap.latestLightDir = { x: 0, y: -1, z: 0 };
}

export function syncShadowMapRegistry(
	shadowMaps: Map<ShadowCastingLight, ShadowMap>,
	activeLights: ShadowCastingLight[]
): void {
	for (const [light] of shadowMaps) {
		if (!activeLights.includes(light)) {
			shadowMaps.delete(light);
		}
	}

	for (const light of activeLights) {
		if (!shadowMaps.has(light)) {
			shadowMaps.set(light, new ShadowMap());
		}
	}
}

export function updateShadowMapMetadata(
	shadowMap: ShadowMap,
	light: ShadowCastingLight,
	sceneBounds: SceneBounds,
	worldMatrix: Matrix4
): void {
	if (!light.shadow) {
		resetShadowMapMetadata(shadowMap);
		return;
	}

	const config = light.shadow.setupShadowCamera({
		sceneBounds,
		worldMatrix: worldMatrix ?? light.worldMatrix,
	});
	if (!config) {
		resetShadowMapMetadata(shadowMap);
		return;
	}

	shadowMap.viewMatrix = config.view;
	shadowMap.projectionMatrix = config.projection;
	shadowMap.latestLightDir = config.lightDir;
	shadowMap.viewProjectionMatrix = Matrix4.multiply(
		config.projection,
		config.view
	);
}
