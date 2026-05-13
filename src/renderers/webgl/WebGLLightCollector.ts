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
} from "../../lights/shadows/ShadowMapping";
import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3, SHCoefficients } from "../../maths/types";
import { collectActiveLocalizedLightProbes } from "../../pipeline/lightProbeRuntime";
import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_LOCAL_LIGHT_PROBES,
	WEBGL_MAX_POINT_LIGHTS,
	WEBGL_MAX_REFLECTION_PROBES,
	WEBGL_MAX_SPOT_LIGHTS,
} from "./constants";
import { collectReflectionProbeEnvironment } from "../../pipeline/reflectionProbeRuntime";
import {
	ensureEnvironmentTextureEquirect,
	isTextureReadyForEnvironment,
} from "../../pipeline/environmentMapRuntime";
import {
	accumulateAmbientLightColor,
	accumulateLightProbeFallbackAmbientColor,
	resolveShadowData as resolveSharedShadowData,
	toLinearLightColor,
} from "../../pipeline/lightingRuntime";

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
	pcssEnabled: boolean;
	pcssRadius: number;
	shadowSamples: number;
	shadowSearchSamples: number;
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
	envSpecularFallbackMap: Texture | null;
	localLightProbeCount: number;
	localLightProbes: WebGLLocalLightProbeUniform[];
	reflectionProbeCount: number;
	reflectionProbes: WebGLReflectionProbeUniform[];
}

export interface WebGLLocalLightProbeUniform {
	id: string;
	worldToProbeMatrix: Matrix4;
	invHalfExtents: [number, number, number];
	radiusInv: number;
	shape: 0 | 1;
	blendDistance: number;
	priority: number;
	sh: SHCoefficients;
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
	captureWorldPosition: [number, number, number];
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

type WebGLLightCollectorWarn = (key: string, message: string) => void;
type WebGLLightCollectorShadowMapLookup =
	ReadonlyMap<ShadowCastingLight, ShadowRenderSet>;

export function collectWebGLLights(
	lights: SceneLight[],
	enableLighting: boolean,
	warnOrEnableShadows: WebGLLightCollectorWarn | boolean = false,
	enableShadowsOrShadowMaps:
		| boolean
		| WebGLLightCollectorShadowMapLookup = false,
	shadowMapsOrEnableSH?: WebGLLightCollectorShadowMapLookup | boolean,
	enableSHOrEnvironment: boolean | Texture | null = false,
	environmentOrEnableClusteredLighting: Texture | null | boolean = null,
	enableClusteredLightingOrCameraWorldPositionMaybe:
		| boolean
		| IVector3
		| null = false,
	cameraWorldPositionMaybe: IVector3 | null = null
): WebGLLightState {
	let warn: WebGLLightCollectorWarn | undefined;
	let enableShadows = false;
	let shadowMaps: WebGLLightCollectorShadowMapLookup | undefined;
	let enableSH = false;
	let environmentTexture: Texture | null = null;
	let enableClusteredLighting = false;
	let cameraWorldPosition: IVector3 | null = null;
	if (typeof warnOrEnableShadows === "function") {
		warn = warnOrEnableShadows;
		enableShadows = enableShadowsOrShadowMaps === true;
		shadowMaps =
			isShadowMapLookup(shadowMapsOrEnableSH) ?
				shadowMapsOrEnableSH
			:	undefined;
		enableSH = typeof enableSHOrEnvironment === "boolean" ? enableSHOrEnvironment : false;
		environmentTexture =
			isTextureOrNull(environmentOrEnableClusteredLighting) ?
				environmentOrEnableClusteredLighting
			:	null;
		enableClusteredLighting =
			typeof enableClusteredLightingOrCameraWorldPositionMaybe === "boolean" ?
				enableClusteredLightingOrCameraWorldPositionMaybe
			:	false;
		cameraWorldPosition = isVector3Like(cameraWorldPositionMaybe) ? cameraWorldPositionMaybe : null;
	} else {
		enableShadows = warnOrEnableShadows === true;
		shadowMaps =
			isShadowMapLookup(enableShadowsOrShadowMaps) ?
				enableShadowsOrShadowMaps
			:	undefined;
		enableSH = typeof shadowMapsOrEnableSH === "boolean" ? shadowMapsOrEnableSH : false;
		environmentTexture = isTextureOrNull(enableSHOrEnvironment) ? enableSHOrEnvironment : null;
		enableClusteredLighting =
			typeof environmentOrEnableClusteredLighting === "boolean" ?
				environmentOrEnableClusteredLighting
			:	false;
		cameraWorldPosition =
			isVector3Like(enableClusteredLightingOrCameraWorldPositionMaybe) ?
				enableClusteredLightingOrCameraWorldPositionMaybe
			:	isVector3Like(cameraWorldPositionMaybe) ? cameraWorldPositionMaybe
			:	null;
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
		envSpecularFallbackMap: null,
		localLightProbeCount: 0,
		localLightProbes: [],
		reflectionProbeCount: 0,
		reflectionProbes: [],
	};
	if (!enableLighting) {
		return state;
	}

	for (const light of lights) {
		switch (light.type) {
			case LightType.Ambient: {
				accumulateAmbientLightColor(
					state.ambientColor,
					light.color,
					light.intensity ?? 1
				);
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
				const direction = light.getWorldLightDirection();
				state.directionalLights.push({
					direction: [-direction.x, -direction.y, -direction.z],
					color: toLinearLightColor(light.color, light.intensity ?? 1),
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
				const position = light.getWorldLightPosition();
				const range = Math.max(0.001, light.range);
				const color = toLinearLightColor(light.color, light.intensity ?? 1);
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
				const position = light.getWorldLightPosition();
				const direction = light.getWorldLightDirection();
				const outerCos = Math.cos(light.outerAngle);
				const innerCos = Math.cos(light.getInnerAngle());
				const range = Math.max(0.001, light.range);
				const color = toLinearLightColor(light.color, light.intensity ?? 1);
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

	if (enableSH) {
		state.localLightProbes = collectActiveLocalizedLightProbes(
			lights,
			WEBGL_MAX_LOCAL_LIGHT_PROBES,
			cameraWorldPosition
		).map((probe) => createWebGLLocalLightProbeUniform(probe));
		state.localLightProbeCount = state.localLightProbes.length;
	}

	const reflectionEnvironment = collectReflectionProbeEnvironment(
		lights,
		WEBGL_MAX_REFLECTION_PROBES,
		cameraWorldPosition
	);
	state.envSpecularFallbackMap = null;
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
				captureWorldPosition: [
					cache.captureWorldPosition.x,
					cache.captureWorldPosition.y,
					cache.captureWorldPosition.z,
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
		state.envSpecularMap = resolveEnvSpecularMap(environmentTexture, emitWarning);
	}

	return state;
}

function collectLightProbe(
	state: WebGLLightState,
	light: LightProbe,
	enableSH: boolean
): void {
	if (enableSH) return;
	accumulateLightProbeFallbackAmbientColor(
		state.ambientColor,
		light.sh[0],
		1
	);
}

function createWebGLLocalLightProbeUniform(
	probe: LightProbe
): WebGLLocalLightProbeUniform {
	const cache = probe.getRuntimeCache();
	return {
		id: probe.id,
		worldToProbeMatrix: cache.worldToProbeMatrix.clone(),
		invHalfExtents: [
			cache.invHalfExtents.x,
			cache.invHalfExtents.y,
			cache.invHalfExtents.z,
		],
		radiusInv: cache.radiusInv,
		shape: probe.shape === "box" ? 1 : 0,
		blendDistance: cache.effectiveBlendDistance,
		priority: cache.priority,
		sh: probe.sh.map((coefficient) => ({
			r: coefficient.r,
			g: coefficient.g,
			b: coefficient.b,
		})),
	};
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

function isVector3Like(value: unknown): value is IVector3 {
	return (
		typeof value === "object" &&
		value !== null &&
		"x" in value &&
		"y" in value &&
		"z" in value
	);
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
	return resolveSharedShadowData(enableShadows, renderSetInput);
}
