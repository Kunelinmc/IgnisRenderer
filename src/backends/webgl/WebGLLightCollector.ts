import { Texture } from "../../core/Texture";
import { Logger } from "../../foundation/Logger";
import {
	LightType,
	type IrradianceProbeGrid,
	type LightProbe,
	type ReflectionProbe,
	type SceneLight,
	type ShadowCastingLight,
} from "../../lights";
import type {
	PreparedShadowLight,
	ShadowFramePlan,
} from "../../lights/shadows/ShadowFramePlan";
import type { ResolvedShadowStrategy } from "../../lights/runtime/lightingRuntime";
import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3, SHCoefficients } from "../../maths/types";
import { collectActiveLocalizedLightProbes } from "../../lights/runtime/lightProbeRuntime";
import {
	collectIrradianceProbeGrids,
	selectActiveIrradianceProbeGrid,
} from "../../lights/runtime/irradianceProbeGridRuntime";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
} from "../constants";
import { collectReflectionProbeEnvironment } from "../../lights/runtime/reflectionProbeRuntime";
import {
	ensureEnvironmentTextureEquirect,
	isTextureReadyForEnvironment,
} from "../../lights/runtime/environmentMapRuntime";
import {
	accumulateAmbientLightColor,
	accumulateLightProbeFallbackAmbientColor,
	resolveShadowData as resolveSharedShadowData,
	toLinearLightColor,
} from "../../lights/runtime/lightingRuntime";

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
	strategyType: ResolvedShadowStrategy;
	cascadeCount: number;
	cascadeBlendRatio: number;
	cascadeViewProjectionMatrices: Array<Matrix4 | null>;
	cascadeSplits: Array<[number, number, number, number]>;
	depthProjectionParams: Array<readonly [number, number, number, number]>;
	viewProjectionMatrix: Matrix4 | null;
	depthBias: number;
	slopeBias: number;
	normalBias: number;
	normalBiasMin: number;
	filterMode: "pcf" | "pcss";
	samplingQuality: "low" | "medium" | "high";
	shadowStrength: number;
	shadowMapBaseSize: number;
	shadowMapSize: number;
	atlasTileSize: number;
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
	irradianceProbeGrid: WebGLIrradianceProbeGridUniform | null;
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

export interface WebGLIrradianceProbeGridUniform {
	id: string;
	worldToGridMatrix: Matrix4;
	dimensions: [number, number, number];
	invHalfExtents: [number, number, number];
	blendDistance: number;
	cellCount: number;
	textureRevision: number;
	sh: SHCoefficients[];
	validMask: Uint8Array;
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
/**
 * Configuration for collecting backend-ready WebGL lighting state.
 *
 * @internal WebGL backend frame preparation owns this contract.
 */
export interface WebGLLightCollectorOptions {
	enableLighting: boolean;
	warn?: WebGLLightCollectorWarn;
	enableShadows?: boolean;
	shadowPlan?: ShadowFramePlan;
	enableSH?: boolean;
	environmentTexture?: Texture | null;
	enableClusteredLighting?: boolean;
	cameraWorldPosition?: IVector3 | null;
}

/**
 * Collects scene lights into the bounded state consumed by WebGL shaders.
 *
 * @internal WebGL backend frame preparation owns light collection.
 */
export function collectWebGLLights(
	lights: SceneLight[],
	options: WebGLLightCollectorOptions
): WebGLLightState {
	const {
		enableLighting,
		warn,
		enableShadows = false,
		shadowPlan,
		enableSH = false,
		environmentTexture = null,
		enableClusteredLighting = false,
		cameraWorldPosition = null,
	} = options;
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
		irradianceProbeGrid: null,
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
				if (state.directionalLights.length >= MAX_DIRECTIONAL_LIGHTS) {
					emitWarning(
						"webgl-directional-light-limit",
						`WebGL forward shading supports at most ${MAX_DIRECTIONAL_LIGHTS} directional lights; extra lights are ignored`
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
						findPreparedShadow(shadowPlan, light as ShadowCastingLight)
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

				if (state.pointLights.length >= MAX_POINT_LIGHTS) {
					if (!enableClusteredLighting) {
						emitWarning(
							"webgl-point-light-limit",
							`WebGL forward shading supports at most ${MAX_POINT_LIGHTS} point lights; extra lights are ignored`
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
					findPreparedShadow(shadowPlan, light as ShadowCastingLight)
				);
				if (enableClusteredLighting) {
					const forwardShadowIndex = state.spotShadows.length;
					const clusteredShadowEnabled =
						resolvedShadow.enabled &&
						forwardShadowIndex >= 0 &&
						forwardShadowIndex < MAX_SPOT_LIGHTS;
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

				if (state.spotLights.length >= MAX_SPOT_LIGHTS) {
					if (!enableClusteredLighting) {
						emitWarning(
							"webgl-spot-light-limit",
							`WebGL forward shading supports at most ${MAX_SPOT_LIGHTS} spot lights; extra lights are ignored`
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
			case LightType.IrradianceProbeGrid: {
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
			MAX_LOCAL_LIGHT_PROBES,
			cameraWorldPosition
		).map((probe) => createWebGLLocalLightProbeUniform(probe));
		state.localLightProbeCount = state.localLightProbes.length;
		const irradianceProbeGridCount = collectIrradianceProbeGrids(lights).length;
		if (irradianceProbeGridCount > 1) {
			emitWarning(
				"webgl-irradiance-probe-grid-extra-ignored",
				"WebGL supports one active irradiance probe grid per frame; extra grids are ignored after priority selection."
			);
		}
		const activeGrid = selectActiveIrradianceProbeGrid(
			lights,
			cameraWorldPosition
		);
		state.irradianceProbeGrid = activeGrid ?
				createWebGLIrradianceProbeGridUniform(activeGrid)
			:	null;
	}

	const reflectionEnvironment = collectReflectionProbeEnvironment(
		lights,
		MAX_REFLECTION_PROBES,
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

function createWebGLIrradianceProbeGridUniform(
	grid: IrradianceProbeGrid
): WebGLIrradianceProbeGridUniform {
	const cache = grid.getRuntimeCache();
	return {
		id: grid.id,
		worldToGridMatrix: cache.worldToGridMatrix.clone(),
		dimensions: [
			cache.dimensions.x,
			cache.dimensions.y,
			cache.dimensions.z,
		],
		invHalfExtents: [
			cache.invHalfExtents.x,
			cache.invHalfExtents.y,
			cache.invHalfExtents.z,
		],
		blendDistance: cache.effectiveBlendDistance,
		cellCount: cache.cellCount,
		textureRevision: cache.textureRevision,
		sh: grid.sh,
		validMask: cache.validMask,
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

function mapParallaxModeCode(probe: ReflectionProbe): 0 | 1 | 2 {
	const mode = probe.parallaxMode;
	if (mode === "box") return 1;
	if (mode === "sphere") return 2;
	return 0;
}

function resolveWebGLShadowData(
	enableShadows: boolean,
	prepared?: PreparedShadowLight
): WebGLShadowData {
	return { ...resolveSharedShadowData(enableShadows, prepared), atlasTileSize: 0 };
}

function findPreparedShadow(
	plan: ShadowFramePlan | undefined,
	light: ShadowCastingLight
): PreparedShadowLight | undefined {
	return plan?.lights.find((candidate) => candidate.light === light);
}
