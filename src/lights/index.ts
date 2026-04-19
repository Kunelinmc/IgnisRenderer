import { AmbientLight } from "./AmbientLight";
import { DirectionalLight } from "./DirectionalLight";
import { PointLight } from "./PointLight";
import { SpotLight } from "./SpotLight";
import { LightProbe } from "./LightProbe";
import { ReflectionProbe } from "./ReflectionProbe";
import { AreaLight } from "./AreaLight";
import type { ShadowConfig } from "./ShadowMapping";

export * from "./constants";
export * from "./Light";
export * from "./AmbientLight";
export * from "./DirectionalLight";
export * from "./PointLight";
export * from "./SpotLight";
export * from "./LightProbe";
export * from "./ReflectionProbe";
export * from "./AreaLight";

export type SceneLight =
	| AmbientLight
	| DirectionalLight
	| PointLight
	| SpotLight
	| LightProbe
	| ReflectionProbe
	| AreaLight;

export type ShadowCastingLight = SceneLight & { shadow: ShadowConfig };

export function isShadowCastingLight(
	light: SceneLight
): light is ShadowCastingLight {
	return light.castShadow && light.shadow !== undefined;
}
