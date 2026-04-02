import { ShadowCaster } from "./Light";
import { AmbientLight } from "./AmbientLight";
import { DirectionalLight } from "./DirectionalLight";
import { PointLight } from "./PointLight";
import { SpotLight } from "./SpotLight";
import { LightProbe } from "./LightProbe";
import { ReflectionProbe } from "./ReflectionProbe";
import { AreaLight } from "./AreaLight";

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

export type ShadowCastingLight = SceneLight & { shadow: ShadowCaster };

export function isShadowCastingLight(
	light: SceneLight
): light is ShadowCastingLight {
	return light.castShadow && light.shadow !== undefined;
}
