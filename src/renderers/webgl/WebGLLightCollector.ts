import { clamp, sRGBToLinear } from "../../maths/Common";
import type { Texture } from "../../core/Texture";
import {
	LightType,
	type LightProbe,
	type SceneLight,
	type ShadowCastingLight,
} from "../../lights";
import type { ShadowMap } from "../../lights/ShadowMapping";
import type { Matrix4 } from "../../maths/Matrix4";
import {
	getDirectionalLightWorldDirection,
	getPointLightWorldPosition,
	getSpotLightInnerAngle,
	getSpotLightWorldDirection,
	getSpotLightWorldPosition,
} from "../../pipeline/LightTransforms";
import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_POINT_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
} from "./constants";

export interface WebGLDirectionalLight {
	direction: [number, number, number];
	color: [number, number, number];
}

export interface WebGLPointLight {
	position: [number, number, number];
	range: number;
	color: [number, number, number];
}

export interface WebGLSpotLight {
	position: [number, number, number];
	range: number;
	direction: [number, number, number];
	outerCos: number;
	innerCos: number;
	color: [number, number, number];
}

export interface WebGLShadowData {
	enabled: boolean;
	viewProjectionMatrix: Matrix4 | null;
	depthBias: number;
	slopeBias: number;
	normalBias: number;
	normalBiasMin: number;
	pcfRadius: number;
	shadowStrength: number;
	shadowMapSize: number;
	atlasTileSize: number;
	shadowMap: ShadowMap | null;
}

export interface WebGLLightState {
	ambientColor: [number, number, number];
	directionalLights: WebGLDirectionalLight[];
	directionalShadows: WebGLShadowData[];
	pointLights: WebGLPointLight[];
	spotLights: WebGLSpotLight[];
	spotShadows: WebGLShadowData[];
	envSpecularMap: Texture | null;
}

type WarnFn = (key: string, message: string) => void;
const LIGHT_PROBE_DC_IRRADIANCE_SCALE = Math.PI * 0.282095;

export function collectWebGLLights(
	lights: SceneLight[],
	enableLighting: boolean,
	warn: WarnFn,
	enableShadows = false,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowMap>,
	enableSH = false
): WebGLLightState {
	const state: WebGLLightState = {
		ambientColor: [0, 0, 0],
		directionalLights: [],
		directionalShadows: [],
		pointLights: [],
		spotLights: [],
		spotShadows: [],
		envSpecularMap: null,
	};
	if (!enableLighting) {
		return state;
	}

	for (const light of lights) {
		switch (light.type) {
			case LightType.Ambient: {
				state.ambientColor[0] +=
					sRGBToLinear((light.color.r ?? 255) / 255) * (light.intensity ?? 1);
				state.ambientColor[1] +=
					sRGBToLinear((light.color.g ?? 255) / 255) * (light.intensity ?? 1);
				state.ambientColor[2] +=
					sRGBToLinear((light.color.b ?? 255) / 255) * (light.intensity ?? 1);
				break;
			}
			case LightType.Directional: {
				if (state.directionalLights.length >= WEBGL_MAX_DIRECTIONAL_LIGHTS) {
					warn(
						"webgl-directional-light-limit",
						`WebGL forward shading supports at most ${WEBGL_MAX_DIRECTIONAL_LIGHTS} directional lights; extra lights are ignored`
					);
					break;
				}
				const direction = getDirectionalLightWorldDirection(light);
				const intensity = light.intensity ?? 1;
				state.directionalLights.push({
					direction: [-direction.x, -direction.y, -direction.z],
					color: [
						sRGBToLinear((light.color.r ?? 255) / 255) * intensity,
						sRGBToLinear((light.color.g ?? 255) / 255) * intensity,
						sRGBToLinear((light.color.b ?? 255) / 255) * intensity,
					],
				});
				state.directionalShadows.push(
					resolveWebGLShadowData(
						enableShadows,
						shadowMaps?.get(light as ShadowCastingLight)
					)
				);
				break;
			}
			case LightType.Point: {
				if (state.pointLights.length >= WEBGL_MAX_POINT_LIGHTS) {
					warn(
						"webgl-point-light-limit",
						`WebGL forward shading supports at most ${WEBGL_MAX_POINT_LIGHTS} point lights; extra lights are ignored`
					);
					break;
				}
				const position = getPointLightWorldPosition(light);
				const intensity = light.intensity ?? 1;
				state.pointLights.push({
					position: [position.x, position.y, position.z],
					range: Math.max(0.001, (light as any).range ?? 1000),
					color: [
						sRGBToLinear((light.color.r ?? 255) / 255) * intensity,
						sRGBToLinear((light.color.g ?? 255) / 255) * intensity,
						sRGBToLinear((light.color.b ?? 255) / 255) * intensity,
					],
				});
				break;
			}
			case LightType.Spot: {
				if (state.spotLights.length >= WEBGL_MAX_SPOT_LIGHTS) {
					warn(
						"webgl-spot-light-limit",
						`WebGL forward shading supports at most ${WEBGL_MAX_SPOT_LIGHTS} spot lights; extra lights are ignored`
					);
					break;
				}
				const position = getSpotLightWorldPosition(light);
				const direction = getSpotLightWorldDirection(light);
				const outerCos = Math.cos((light as any).outerAngle ?? Math.PI / 4);
				const innerCos = Math.cos(getSpotLightInnerAngle(light as any));
				const intensity = light.intensity ?? 1;
				state.spotLights.push({
					position: [position.x, position.y, position.z],
					range: Math.max(0.001, (light as any).range ?? 1000),
					direction: [direction.x, direction.y, direction.z],
					outerCos,
					innerCos,
					color: [
						sRGBToLinear((light.color.r ?? 255) / 255) * intensity,
						sRGBToLinear((light.color.g ?? 255) / 255) * intensity,
						sRGBToLinear((light.color.b ?? 255) / 255) * intensity,
					],
				});
				state.spotShadows.push(
					resolveWebGLShadowData(
						enableShadows,
						shadowMaps?.get(light as ShadowCastingLight)
					)
				);
				break;
			}
			case LightType.LightProbe: {
				collectLightProbe(state, light as LightProbe, enableSH, warn);
				break;
			}
			case LightType.RectArea:
			default: {
				warn(
					`webgl-light-unsupported-${light.type}`,
					`WebGL v1 does not support ${light.type} lights yet; ignoring this light`
				);
				break;
			}
		}
	}

	return state;
}

function collectLightProbe(
	state: WebGLLightState,
	light: LightProbe,
	enableSH: boolean,
	warn: WarnFn
): void {
	if (!enableSH) {
		const dc = light.sh[0];
		if (dc) {
			const intensity = light.intensity ?? 1;
			state.ambientColor[0] +=
				(Math.max(0, dc.r * LIGHT_PROBE_DC_IRRADIANCE_SCALE) / 255) *
				intensity;
			state.ambientColor[1] +=
				(Math.max(0, dc.g * LIGHT_PROBE_DC_IRRADIANCE_SCALE) / 255) *
				intensity;
			state.ambientColor[2] +=
				(Math.max(0, dc.b * LIGHT_PROBE_DC_IRRADIANCE_SCALE) / 255) *
				intensity;
		}
	}

	if (state.envSpecularMap) {
		return;
	}

	const resolvedEnvSpecular = resolveEnvSpecularMap(light.prefilteredMap, warn);
	if (resolvedEnvSpecular) {
		state.envSpecularMap = resolvedEnvSpecular;
	}
}

function resolveEnvSpecularMap(
	texture: Texture | null,
	warn: WarnFn
): Texture | null {
	if (!texture) {
		return null;
	}
	if (texture.isLoadErrorFallback) {
		warn(
			"webgl-env-specular-load-error-fallback",
			"WebGL environment specular texture resolved to a load-error fallback; skipping IBL specular."
		);
		return null;
	}
	if (!isTextureReadyForEnvironment(texture)) {
		warn(
			"webgl-env-specular-texture-not-ready",
			"WebGL environment specular texture is not ready (missing pixels or invalid dimensions); skipping IBL specular."
		);
		return null;
	}
	return texture;
}

function isTextureReadyForEnvironment(texture: Texture): boolean {
	if (
		!isFinitePositiveNumber(texture.width) ||
		!isFinitePositiveNumber(texture.height)
	) {
		return false;
	}

	return !!texture.data || texture.mipmaps.length > 0;
}

function isFinitePositiveNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function resolveWebGLShadowData(
	enableShadows: boolean,
	shadowMap?: ShadowMap
): WebGLShadowData {
	if (!enableShadows || !shadowMap?.viewProjectionMatrix) {
		return {
			enabled: false,
			viewProjectionMatrix: null,
			depthBias: 0,
			slopeBias: 0,
			normalBias: 0,
			normalBiasMin: 0,
			pcfRadius: 0,
			shadowStrength: 0,
			shadowMapSize: 0,
			atlasTileSize: 0,
			shadowMap: null,
		};
	}

	const size = Math.max(1, shadowMap.size | 0);
	const texelBias = (shadowMap.params.shadowTexelBias ?? 1.0) * (1.0 / size);
	const maxBias = shadowMap.params.shadowMaxBias ?? 0.05;
	const depthBias = Math.min(
		maxBias,
		(shadowMap.params.shadowBias ?? 0.008) + texelBias
	);
	const pcfRadius =
		shadowMap.params.shadowRadius && shadowMap.params.shadowRadius > 0 ?
			shadowMap.params.shadowRadius
		:	Math.max(1, shadowMap.params.shadowPCF ?? 1);

	return {
		enabled: true,
		viewProjectionMatrix: shadowMap.viewProjectionMatrix,
		depthBias,
		slopeBias: Math.max(0, shadowMap.params.shadowSlopeBias ?? 0.03),
		normalBias: Math.max(0, shadowMap.params.shadowNormalBias ?? 1.0),
		normalBiasMin: Math.max(0, shadowMap.params.shadowNormalBiasMin ?? 0.05),
		pcfRadius: Math.max(1, pcfRadius),
		shadowStrength: clamp(shadowMap.params.shadowStrength ?? 1.0, 0, 1),
		shadowMapSize: size,
		atlasTileSize: size,
		shadowMap,
	};
}
