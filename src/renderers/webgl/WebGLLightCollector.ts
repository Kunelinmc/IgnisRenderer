import { clamp, sRGBToLinear } from "../../maths/Common";
import { Texture } from "../../core/Texture";
import { Logger } from "../../foundation/Logger";
import {
	LightType,
	type LightProbe,
	type ReflectionProbe,
	type SceneLight,
	type ShadowCastingLight,
} from "../../lights";
import {
	type ShadowMap,
	type ShadowStrategyType,
	type ShadowRenderSet,
} from "../../lights/ShadowMapping";
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
import { collectReflectionProbeEnvironment } from "../../pipeline/reflectionProbeRuntime";
import {
	ensureEnvironmentTextureEquirect,
	isTextureReadyForEnvironment,
} from "../../pipeline/environmentMapRuntime";

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
	strategyType: ShadowStrategyType;
	cascadeCount: number;
	cascadeBlendRatio: number;
	cascadeViewProjectionMatrices: Array<Matrix4 | null>;
	cascadeSplits: Array<[number, number, number, number]>;
	viewProjectionMatrix: Matrix4 | null;
	depthBias: number;
	slopeBias: number;
	normalBias: number;
	normalBiasMin: number;
	pcfRadius: number;
	shadowStrength: number;
	shadowMapBaseSize: number;
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

type WebGLLightCollectorWarn = (key: string, message: string) => void;
type WebGLLightCollectorShadowMapLookup =
	ReadonlyMap<ShadowCastingLight, ShadowRenderSet>;

export function collectWebGLLights(
	lights: SceneLight[],
	enableLighting: boolean,
	enableShadows?: boolean,
	shadowMaps?: WebGLLightCollectorShadowMapLookup,
	enableSH?: boolean,
	skybox?: Texture | null,
	enableClusteredLighting?: boolean
): WebGLLightState;
export function collectWebGLLights(
	lights: SceneLight[],
	enableLighting: boolean,
	warn: WebGLLightCollectorWarn,
	enableShadows?: boolean,
	shadowMaps?: WebGLLightCollectorShadowMapLookup,
	enableSH?: boolean,
	skybox?: Texture | null,
	enableClusteredLighting?: boolean
): WebGLLightState;

export function collectWebGLLights(
	lights: SceneLight[],
	enableLighting: boolean,
	warnOrEnableShadows: WebGLLightCollectorWarn | boolean = false,
	enableShadowsOrShadowMaps:
		| boolean
		| WebGLLightCollectorShadowMapLookup = false,
	shadowMapsOrEnableSH?: WebGLLightCollectorShadowMapLookup | boolean,
	enableSHOrSkybox: boolean | Texture | null = false,
	skyboxOrEnableClusteredLighting: Texture | null | boolean = null,
	enableClusteredLightingMaybe = false
): WebGLLightState {
	let warn: WebGLLightCollectorWarn | undefined;
	let enableShadows = false;
	let shadowMaps: WebGLLightCollectorShadowMapLookup | undefined;
	let enableSH = false;
	let skybox: Texture | null = null;
	let enableClusteredLighting = false;
	if (typeof warnOrEnableShadows === "function") {
		warn = warnOrEnableShadows;
		enableShadows = enableShadowsOrShadowMaps === true;
		shadowMaps =
			isShadowMapLookup(shadowMapsOrEnableSH) ?
				shadowMapsOrEnableSH
			:	undefined;
		enableSH = typeof enableSHOrSkybox === "boolean" ? enableSHOrSkybox : false;
		skybox =
			isTextureOrNull(skyboxOrEnableClusteredLighting) ?
				skyboxOrEnableClusteredLighting
			:	null;
		enableClusteredLighting =
			typeof enableClusteredLightingMaybe === "boolean" ?
				enableClusteredLightingMaybe
			:	false;
	} else {
		enableShadows = warnOrEnableShadows === true;
		shadowMaps =
			isShadowMapLookup(enableShadowsOrShadowMaps) ?
				enableShadowsOrShadowMaps
			:	undefined;
		enableSH = typeof shadowMapsOrEnableSH === "boolean" ? shadowMapsOrEnableSH : false;
		skybox = isTextureOrNull(enableSHOrSkybox) ? enableSHOrSkybox : null;
		enableClusteredLighting =
			typeof skyboxOrEnableClusteredLighting === "boolean" ?
				skyboxOrEnableClusteredLighting
			:	false;
	}
	const emitWarning: WebGLLightCollectorWarn = (key, message) => {
		warn?.(key, message);
		logWebGLLightCollectorWarning(key, message);
	};
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
					emitWarning(
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
						emitWarning(
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
						emitWarning(
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
				emitWarning(
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
		const atlas = resolveEnvSpecularMap(reflectionEnvironment.atlas, emitWarning);
		if (atlas) {
			state.envSpecularMap = atlas;
		}
	}

	if (!state.envSpecularMap) {
		state.envSpecularMap = resolveEnvironmentSkyboxMap(skybox, emitWarning);
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

function resolveEnvSpecularMap(
	texture: Texture | null,
	warn: WebGLLightCollectorWarn
): Texture | null {
	const normalizedTexture = ensureEnvironmentTextureEquirect(texture);
	if (!normalizedTexture) {
		return null;
	}
	if (normalizedTexture.isLoadErrorFallback) {
		warn(
			"webgl-env-specular-load-error-fallback",
			"WebGL environment specular texture resolved to a load-error fallback; skipping IBL specular."
		);
		return null;
	}
	if (!isTextureReadyForEnvironment(normalizedTexture)) {
		warn(
			"webgl-env-specular-texture-not-ready",
			"WebGL environment specular texture is not ready (missing pixels or invalid dimensions); skipping IBL specular."
		);
		return null;
	}
	return normalizedTexture;
}

function resolveEnvironmentSkyboxMap(
	texture: Texture | null,
	warn: WebGLLightCollectorWarn
): Texture | null {
	const normalizedTexture = ensureEnvironmentTextureEquirect(texture);
	if (!normalizedTexture) return null;
	if (normalizedTexture.isLoadErrorFallback) {
		warn(
			"webgl-skybox-load-error-fallback",
			"WebGL skybox texture resolved to a load-error fallback; skipping skybox IBL fallback."
		);
		return null;
	}
	if (!isTextureReadyForEnvironment(normalizedTexture)) {
		warn(
			"webgl-skybox-texture-not-ready",
			"WebGL skybox texture is not ready (missing pixels or invalid dimensions); skipping skybox IBL fallback."
		);
		return null;
	}
	return normalizedTexture;
}

function logWebGLLightCollectorWarning(key: string, message: string): void {
	Logger.warn(`[${key}] ${message}`, {
		scope: "WebGLLightCollector",
		onceKey: key,
	});
}

function isShadowMapLookup(
	value: unknown
): value is WebGLLightCollectorShadowMapLookup {
	return (
		typeof value === "object" &&
		value !== null &&
		"get" in value &&
		typeof (value as { get?: unknown }).get === "function"
	);
}

function isTextureOrNull(value: unknown): value is Texture | null {
	return value === null || value instanceof Texture;
}

function mapParallaxModeCode(probe: ReflectionProbe): 0 | 1 | 2 {
	const mode = probe.parallaxMode;
	if (mode === "box") return 1;
	if (mode === "sphere") return 2;
	return 0;
}

function resolveWebGLShadowData(
	enableShadows: boolean,
	renderSetInput?: ShadowRenderSet | ShadowMap
): WebGLShadowData {
	const renderSet =
		renderSetInput &&
		typeof renderSetInput === "object" &&
		Array.isArray((renderSetInput as { slices?: unknown }).slices) ?
			(renderSetInput as ShadowRenderSet)
		:	null;
	const legacyShadowMap =
		!renderSet &&
		renderSetInput &&
		typeof renderSetInput === "object" &&
		"viewProjectionMatrix" in renderSetInput ?
			(renderSetInput as ShadowMap)
		:	null;
	const primarySlice = renderSet?.slices[0] ?? null;
	const shadowMap = primarySlice?.shadowMap ?? null;
	const resolvedShadowMap = shadowMap ?? legacyShadowMap;

	if (!enableShadows || !resolvedShadowMap?.viewProjectionMatrix) {
		return {
			enabled: false,
			strategyType: "single-map",
			cascadeCount: 1,
			cascadeBlendRatio: 0,
			cascadeViewProjectionMatrices: [null, null, null, null],
			cascadeSplits: [
				[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0],
			],
			viewProjectionMatrix: null,
			depthBias: 0,
			slopeBias: 0,
			normalBias: 0,
			normalBiasMin: 0,
			pcfRadius: 0,
			shadowStrength: 0,
			shadowMapBaseSize: 0,
			shadowMapSize: 0,
			atlasTileSize: 0,
			shadowMap: null,
		};
	}

	const size = Math.max(1, resolvedShadowMap.size | 0);
	const texelBias =
		(resolvedShadowMap.params.shadowTexelBias ?? 1.0) * (1.0 / size);
	const maxBias = resolvedShadowMap.params.shadowMaxBias ?? 0.05;
	const depthBias = Math.min(
		maxBias,
		(resolvedShadowMap.params.shadowBias ?? 0.008) + texelBias
	);
	const pcfRadius =
		resolvedShadowMap.params.shadowRadius &&
		resolvedShadowMap.params.shadowRadius > 0 ?
			resolvedShadowMap.params.shadowRadius
		:	Math.max(1, resolvedShadowMap.params.shadowPCF ?? 1);

	const cascadeViewProjectionMatrices: Array<Matrix4 | null> = [
		null,
		null,
		null,
		null,
	];
	const cascadeSplits: Array<[number, number, number, number]> = [
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],
	];
	if (renderSet) {
		for (let index = 0; index < Math.min(renderSet.slices.length, 4); index++) {
			const slice = renderSet.slices[index];
			cascadeViewProjectionMatrices[index] =
				slice.shadowMap.viewProjectionMatrix ?? null;
			const localTileX = index % 2;
			const localTileY = Math.floor(index / 2);
			cascadeSplits[index] = [
				Math.max(0, slice.splitNear),
				Math.max(0, slice.splitFar),
				localTileX,
				localTileY,
			];
		}
	} else {
		cascadeViewProjectionMatrices[0] = resolvedShadowMap.viewProjectionMatrix;
		cascadeSplits[0] = [0, 1, 0, 0];
	}

	const strategyType = renderSet?.effectiveStrategyType ?? "single-map";
	const availableCascadeCount = cascadeViewProjectionMatrices.reduce(
		(count, matrix) => (matrix ? count + 1 : count),
		0
	);
	const cascadeCount =
		strategyType === "csm" ?
			Math.max(1, Math.min(4, availableCascadeCount || 1))
		:	1;
	const cascadeBlendRatio =
		strategyType === "csm" &&
		cascadeCount > 1 &&
		renderSet &&
		renderSet.resolvedConfig.strategy === "csm" ?
			Math.max(0, Math.min(1, renderSet.resolvedConfig.blendRatio ?? 0.1))
		:	0;

	return {
		enabled: true,
		strategyType,
		cascadeCount,
		cascadeBlendRatio,
		cascadeViewProjectionMatrices,
		cascadeSplits,
		viewProjectionMatrix: resolvedShadowMap.viewProjectionMatrix,
		depthBias,
		slopeBias: Math.max(0, resolvedShadowMap.params.shadowSlopeBias ?? 0.03),
		normalBias: Math.max(0, resolvedShadowMap.params.shadowNormalBias ?? 1.0),
		normalBiasMin: Math.max(
			0,
			resolvedShadowMap.params.shadowNormalBiasMin ?? 0.05
		),
		pcfRadius: Math.max(1, pcfRadius),
		shadowStrength: clamp(resolvedShadowMap.params.shadowStrength ?? 1.0, 0, 1),
		shadowMapBaseSize: Math.max(1, renderSet?.size ?? resolvedShadowMap.size),
		shadowMapSize: size,
		atlasTileSize: 0,
		shadowMap: resolvedShadowMap,
	};
}
