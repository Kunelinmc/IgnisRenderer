import type { SceneLight, ShadowCastingLight } from "../types";

/** Returns whether a scene light can be bound to a shadow definition. */
export function isShadowCastingLight(light: SceneLight): light is ShadowCastingLight {
	return light.type === "directional" || light.type === "spot" ||
		light.type === "point" || light.type === "rectArea";
}
