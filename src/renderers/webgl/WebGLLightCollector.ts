import { clamp, sRGBToLinear } from "../../maths/Common";
import type { Texture } from "../../core/Texture";
import { Logger } from "../../foundation/Logger";
import {
	LightType,
	type LightProbe,
	type ReflectionProbe,
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
	WEBGL_MAX_REFLECTION_PROBES,
	WEBGL_MAX_SPOT_LIGHTS,
} from "./constants";
import {
	collectReflectionProbeEnvironment,
	isTextureReadyForEnvironment,
} from "../../pipeline/reflectionProbeRuntime";

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
	clusteredLights: WebGLClusteredLight[];
	envSpecularMap: Texture | null;
	reflectionProbeCount: number;
	reflectionProbes: WebGLReflectionProbeUniform[];
}

export interface WebGLReflectionProbeUniform {
	id: string;
	worldToProbeMatrix: Matrix4;
	probeToWorldMatrix: Matrix4;
	invHalfExtents: [number, number, number];
	radiusInv: number;
	shape: 0 | 1;
	parallaxMode: 0 | 1 | 2;
	blendDistance: number;
	blendExponent: number;
	probeWorldPosition: [number, number, number];
	layer: number;
}

export interface WebGLClusteredLight {
	type: 0 | 1;
	position: [number, number, number];
	range: number;
	direction: [number, number, number];
	outerCos: number;
	innerCos: number;
	color: [number, number, number];
	castsShadow: boolean;
	shadowIndex: number;
}

const LIGHT_PROBE_DC_IRRADIANCE_SCALE = Math.PI * 0.282095;

export function collectWebGLLights(
	lights: SceneLight[],
	enableLighting: boolean,
	enableShadows = false,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowMap>,
	enableSH = false,
	skybox: Texture | null = null,
	enableClusteredLighting = false
): WebGLLightState {
	const state: WebGLLightState = {
		ambientColor: [0, 0, 0],
		directionalLights: [],
		directionalShadows: [],
		pointLights: [],
		spotLights: [],
		spotShadows: [],
		clusteredLights: [],
		envSpecularMap: null,
		reflectionProbeCount: 0,
		reflectionProbes: [],
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
					logWebGLLightCollectorWarning(
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
				const position = getPointLightWorldPosition(light);
				const intensity = light.intensity ?? 1;
				const range = Math.max(0.001, (light as any).range ?? 1000);
				const color: [number, number, number] = [
					sRGBToLinear((light.color.r ?? 255) / 255) * intensity,
					sRGBToLinear((light.color.g ?? 255) / 255) * intensity,
					sRGBToLinear((light.color.b ?? 255) / 255) * intensity,
				];
				if (enableClusteredLighting) {
					state.clusteredLights.push({
						type: 0,
						position: [position.x, position.y, position.z],
						range,
						direction: [0, 0, 0],
						outerCos: -2,
						innerCos: -2,
						color,
						castsShadow: false,
						shadowIndex: 0,
					});
				}

				if (state.pointLights.length >= WEBGL_MAX_POINT_LIGHTS) {
					if (!enableClusteredLighting) {
						logWebGLLightCollectorWarning(
							"webgl-point-light-limit",
							`WebGL forward shading supports at most ${WEBGL_MAX_POINT_LIGHTS} point lights; extra lights are ignored`
						);
					}
					break;
				}
				state.pointLights.push({
					position: [position.x, position.y, position.z],
					range,
					color,
				});
				break;
			}
			case LightType.Spot: {
				const position = getSpotLightWorldPosition(light);
				const direction = getSpotLightWorldDirection(light);
				const outerCos = Math.cos((light as any).outerAngle ?? Math.PI / 4);
				const innerCos = Math.cos(getSpotLightInnerAngle(light as any));
				const intensity = light.intensity ?? 1;
				const range = Math.max(0.001, (light as any).range ?? 1000);
				const color: [number, number, number] = [
					sRGBToLinear((light.color.r ?? 255) / 255) * intensity,
					sRGBToLinear((light.color.g ?? 255) / 255) * intensity,
					sRGBToLinear((light.color.b ?? 255) / 255) * intensity,
				];
				const resolvedShadow = resolveWebGLShadowData(
					enableShadows,
					shadowMaps?.get(light as ShadowCastingLight)
				);
				if (enableClusteredLighting) {
					const forwardShadowIndex = state.spotShadows.length;
					const clusteredShadowEnabled =
						resolvedShadow.enabled &&
						forwardShadowIndex >= 0 &&
						forwardShadowIndex < WEBGL_MAX_SPOT_LIGHTS;
					state.clusteredLights.push({
						type: 1,
						position: [position.x, position.y, position.z],
						range,
						direction: [direction.x, direction.y, direction.z],
						outerCos,
						innerCos,
						color,
						castsShadow: clusteredShadowEnabled,
						shadowIndex: clusteredShadowEnabled ? forwardShadowIndex : 0,
					});
				}

				if (state.spotLights.length >= WEBGL_MAX_SPOT_LIGHTS) {
					if (!enableClusteredLighting) {
						logWebGLLightCollectorWarning(
							"webgl-spot-light-limit",
							`WebGL forward shading supports at most ${WEBGL_MAX_SPOT_LIGHTS} spot lights; extra lights are ignored`
						);
					}
					break;
				}
				state.spotLights.push({
					position: [position.x, position.y, position.z],
					range,
					direction: [direction.x, direction.y, direction.z],
					outerCos,
					innerCos,
					color,
				});
				state.spotShadows.push(resolvedShadow);
				break;
			}
			case LightType.LightProbe: {
				collectLightProbe(state, light as LightProbe, enableSH);
				break;
			}
			case LightType.ReflectionProbe: {
				break;
			}
			case LightType.RectArea:
			default: {
				logWebGLLightCollectorWarning(
					`webgl-light-unsupported-${light.type}`,
					`WebGL backend does not support ${light.type} lights yet; ignoring this light`
				);
				break;
			}
		}
	}

	const reflectionEnvironment = collectReflectionProbeEnvironment(
		lights,
		WEBGL_MAX_REFLECTION_PROBES
	);
	if (reflectionEnvironment.probes.length > 0) {
		state.reflectionProbes = reflectionEnvironment.probes.map((probe, index) => {
			const cache = probe.getRuntimeCache();
			return {
				id: probe.id,
				worldToProbeMatrix: cache.worldToProbeMatrix.clone(),
				probeToWorldMatrix: cache.probeToWorldMatrix.clone(),
				invHalfExtents: [
					cache.invHalfExtents.x,
					cache.invHalfExtents.y,
					cache.invHalfExtents.z,
				],
				radiusInv: cache.radiusInv,
				shape: probe.shape === "box" ? 1 : 0,
				parallaxMode: mapParallaxModeCode(probe),
				blendDistance: cache.effectiveBlendDistance,
				blendExponent: cache.blendExponent,
				probeWorldPosition: [
					cache.probeWorldPosition.x,
					cache.probeWorldPosition.y,
					cache.probeWorldPosition.z,
				],
				layer: index,
			};
		});
		state.reflectionProbeCount = state.reflectionProbes.length;
		const atlas = resolveEnvSpecularMap(reflectionEnvironment.atlas);
		if (atlas) {
			state.envSpecularMap = atlas;
		}
	}

	if (!state.envSpecularMap) {
		state.envSpecularMap = resolveEnvironmentSkyboxMap(skybox);
		state.reflectionProbeCount = 0;
		state.reflectionProbes = [];
	}

	return state;
}

function collectLightProbe(
	state: WebGLLightState,
	light: LightProbe,
	enableSH: boolean
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
}

function resolveEnvSpecularMap(texture: Texture | null): Texture | null {
	if (!texture) {
		return null;
	}
	if (texture.isLoadErrorFallback) {
		logWebGLLightCollectorWarning(
			"webgl-env-specular-load-error-fallback",
			"WebGL environment specular texture resolved to a load-error fallback; skipping IBL specular."
		);
		return null;
	}
	if (!isTextureReadyForEnvironment(texture)) {
		logWebGLLightCollectorWarning(
			"webgl-env-specular-texture-not-ready",
			"WebGL environment specular texture is not ready (missing pixels or invalid dimensions); skipping IBL specular."
		);
		return null;
	}
	return texture;
}

function resolveEnvironmentSkyboxMap(texture: Texture | null): Texture | null {
	if (!texture) return null;
	if (texture.isLoadErrorFallback) {
		logWebGLLightCollectorWarning(
			"webgl-skybox-load-error-fallback",
			"WebGL skybox texture resolved to a load-error fallback; skipping skybox IBL fallback."
		);
		return null;
	}
	if (!isTextureReadyForEnvironment(texture)) {
		logWebGLLightCollectorWarning(
			"webgl-skybox-texture-not-ready",
			"WebGL skybox texture is not ready (missing pixels or invalid dimensions); skipping skybox IBL fallback."
		);
		return null;
	}
	return texture;
}

function logWebGLLightCollectorWarning(key: string, message: string): void {
	Logger.warn(`[${key}] ${message}`, {
		scope: "WebGLLightCollector",
		onceKey: key,
	});
}

function mapParallaxModeCode(probe: ReflectionProbe): 0 | 1 | 2 {
	const mode = probe.parallaxMode;
	if (mode === "box") return 1;
	if (mode === "sphere") return 2;
	return 0;
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
