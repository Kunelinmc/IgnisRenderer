import { sRGBToLinear, clamp } from '../../../maths/Common'
import {
	LightType,
	type AmbientLight,
	type DirectionalLight,
	type PointLight,
	type SceneLight,
	type ShadowCastingLight,
	type SpotLight,
} from '../../../lights'
import {
	getDirectionalLightWorldDirection,
	getPointLightWorldPosition,
	getSpotLightInnerAngle,
	getSpotLightWorldDirection,
	getSpotLightWorldPosition,
} from '../../pipeline/LightTransforms'
import type { RGB } from '../../../utils/Color'
import type { ShadowMap } from '../../../utils/ShadowMapping'

import {
	WEBGPU_MAX_DIRECTIONAL_LIGHTS,
	WEBGPU_MAX_POINT_LIGHTS,
	WEBGPU_MAX_SPOT_LIGHTS,
} from './constants'
import type {
	WebGPULightingState,
	WebGPUShadowData,
	WebGPUVec3,
	WebGPUWarning,
} from './types'

export function collectWebGPULighting(
	lights: SceneLight[],
	enableLighting: boolean,
	enableShadows: boolean = false,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowMap>
): WebGPULightingState {
	const state = createEmptyWebGPULightingState()
	if (!enableLighting) return state

	for (const light of lights) {
		switch (light.type) {
			case LightType.Ambient:
				accumulateAmbientLight(state, light)
				break
			case LightType.Directional:
				collectDirectionalLight(state, light, enableShadows, shadowMaps)
				break
			case LightType.Point:
				collectPointLight(state, light)
				break
			case LightType.Spot:
				collectSpotLight(state, light, enableShadows, shadowMaps)
				break
			default:
				state.warnings.push(createUnsupportedLightWarning(light))
				break
		}
	}

	return state
}

function createEmptyWebGPULightingState(): WebGPULightingState {
	return {
		ambientColor: [0, 0, 0],
		directionalLights: [],
		directionalShadows: [],
		pointLights: [],
		spotLights: [],
		spotShadows: [],
		warnings: [],
	}
}

function accumulateAmbientLight(
	state: WebGPULightingState,
	light: AmbientLight
): void {
	state.ambientColor[0] += sRGBToLinear(light.color.r / 255) * light.intensity
	state.ambientColor[1] += sRGBToLinear(light.color.g / 255) * light.intensity
	state.ambientColor[2] += sRGBToLinear(light.color.b / 255) * light.intensity
}

function collectDirectionalLight(
	state: WebGPULightingState,
	light: DirectionalLight,
	enableShadows: boolean,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowMap>
): void {
	if (state.directionalLights.length >= WEBGPU_MAX_DIRECTIONAL_LIGHTS) {
		state.warnings.push(
			createLightLimitWarning('directional', WEBGPU_MAX_DIRECTIONAL_LIGHTS)
		)
		return
	}

	const direction = getDirectionalLightWorldDirection(light)

	state.directionalLights.push({
		direction: [-direction.x, -direction.y, -direction.z],
		color: toLinearLightColor(light.color, light.intensity),
	})
	state.directionalShadows.push(
		resolveWebGPUShadowData(
			enableShadows,
			shadowMaps?.get(light as ShadowCastingLight)
		)
	)
}

function collectPointLight(
	state: WebGPULightingState,
	light: PointLight
): void {
	if (state.pointLights.length >= WEBGPU_MAX_POINT_LIGHTS) {
		state.warnings.push(
			createLightLimitWarning('point', WEBGPU_MAX_POINT_LIGHTS)
		)
		return
	}

	const position = getPointLightWorldPosition(light)
	state.pointLights.push({
		position: [position.x, position.y, position.z],
		range: Math.max(light.range, 0.001),
		color: toLinearLightColor(light.color, light.intensity),
	})
}

function collectSpotLight(
	state: WebGPULightingState,
	light: SpotLight,
	enableShadows: boolean,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowMap>
): void {
	if (state.spotLights.length >= WEBGPU_MAX_SPOT_LIGHTS) {
		state.warnings.push(createLightLimitWarning('spot', WEBGPU_MAX_SPOT_LIGHTS))
		return
	}

	const position = getSpotLightWorldPosition(light)
	const direction = getSpotLightWorldDirection(light)
	const outerAngle = light.angle
	const innerAngle = getSpotLightInnerAngle(light)

	state.spotLights.push({
		position: [position.x, position.y, position.z],
		range: Math.max(light.range, 0.001),
		direction: [direction.x, direction.y, direction.z],
		outerCos: Math.cos(outerAngle),
		innerCos: Math.cos(innerAngle),
		color: toLinearLightColor(light.color, light.intensity),
	})
	state.spotShadows.push(
		resolveWebGPUShadowData(
			enableShadows,
			shadowMaps?.get(light as ShadowCastingLight)
		)
	)
}

function createLightLimitWarning(
	kind: string,
	maxCount: number
): WebGPUWarning {
	return {
		key: `webgpu-${kind}-limit`,
		message: `WebGPU backend supports at most ${maxCount} ${kind} lights; extra lights are ignored`,
	}
}

function createUnsupportedLightWarning(light: SceneLight): WebGPUWarning {
	return {
		key: `webgpu-light-${light.type}`,
		message: `WebGPU backend does not support ${light.type} lights yet; ignoring them for now`,
	}
}

function toLinearLightColor(color: RGB, intensity: number): WebGPUVec3 {
	return [
		sRGBToLinear(color.r / 255) * intensity,
		sRGBToLinear(color.g / 255) * intensity,
		sRGBToLinear(color.b / 255) * intensity,
	]
}

function resolveWebGPUShadowData(
	enableShadows: boolean,
	shadowMap?: ShadowMap
): WebGPUShadowData {
	if (!enableShadows || !shadowMap?.viewProjectionMatrix) {
		return {
			enabled: false,
			viewProjectionMatrix: null,
			depthBias: 0,
			normalBias: 0,
			normalBiasMin: 0,
			pcfRadius: 0,
			shadowStrength: 0,
			shadowMapSize: 0,
			atlasTileSize: 0,
			shadowMap: null,
		}
	}

	const size = Math.max(1, shadowMap.size | 0)
	const texelBias = (shadowMap.params.shadowTexelBias ?? 1.0) * (2.0 / size)
	const maxBias = shadowMap.params.shadowMaxBias ?? 0.05
	const depthBias =
		Math.min(maxBias, (shadowMap.params.shadowBias ?? 0.008) + texelBias) * 0.5
	const pcfRadius =
		shadowMap.params.shadowRadius && shadowMap.params.shadowRadius > 0
			? shadowMap.params.shadowRadius
			: Math.max(1, shadowMap.params.shadowPCF ?? 1)

	return {
		enabled: true,
		viewProjectionMatrix: shadowMap.viewProjectionMatrix,
		depthBias,
		normalBias: Math.max(0, shadowMap.params.shadowNormalBias ?? 1.0),
		normalBiasMin: Math.max(0, shadowMap.params.shadowNormalBiasMin ?? 0.05),
		pcfRadius: Math.max(1, pcfRadius),
		shadowStrength: clamp(shadowMap.params.shadowStrength ?? 1.0, 0, 1),
		shadowMapSize: size,
		atlasTileSize: size,
		shadowMap,
	}
}
