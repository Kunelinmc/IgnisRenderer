import { ShadowCaster } from "./Light";
import { AmbientLight } from "./AmbientLight";
import { DirectionalLight } from "./DirectionalLight";
import { PointLight } from "./PointLight";
import { SpotLight } from "./SpotLight";
import { LightProbe } from "./LightProbe";

export * from "./Light";
export * from "./AmbientLight";
export * from "./DirectionalLight";
export * from "./PointLight";
export * from "./SpotLight";
export * from "./LightProbe";

export type SceneLight =
	| AmbientLight
	| DirectionalLight
	| PointLight
	| SpotLight
	| LightProbe;

export type ShadowCastingLight = SceneLight & { shadow: ShadowCaster };

export function isShadowCastingLight(
	light: SceneLight
): light is ShadowCastingLight {
	return light.castShadow && light.shadow !== undefined;
}
