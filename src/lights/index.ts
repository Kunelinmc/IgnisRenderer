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

export type ShadowCastingLight =
	| DirectionalLight
	| PointLight
	| SpotLight
	| AreaLight;

export function isShadowCastingLight(
	light: SceneLight
): light is ShadowCastingLight {
	if (
		light.type !== "directional" &&
		light.type !== "point" &&
		light.type !== "spot" &&
		light.type !== "rectArea"
	) {
		return false;
	}
	return light.scene?.shadows.getBoundShadowMap(light) !== undefined;
}
