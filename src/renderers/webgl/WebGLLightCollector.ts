import { sRGBToLinear } from '../../maths/Common'
import { LightType, type SceneLight } from '../../lights'
import {
	getDirectionalLightWorldDirection,
	getPointLightWorldPosition,
	getSpotLightInnerAngle,
	getSpotLightWorldDirection,
	getSpotLightWorldPosition,
} from '../../pipeline/LightTransforms'
import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_POINT_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
} from './constants'

export interface WebGLDirectionalLight {
	direction: [number, number, number]
	color: [number, number, number]
}

export interface WebGLPointLight {
	position: [number, number, number]
	range: number
	color: [number, number, number]
}

export interface WebGLSpotLight {
	position: [number, number, number]
	range: number
	direction: [number, number, number]
	outerCos: number
	innerCos: number
	color: [number, number, number]
}

export interface WebGLLightState {
	ambientColor: [number, number, number]
	directionalLights: WebGLDirectionalLight[]
	pointLights: WebGLPointLight[]
	spotLights: WebGLSpotLight[]
}

type WarnFn = (key: string, message: string) => void

export function collectWebGLLights(
	lights: SceneLight[],
	enableLighting: boolean,
	warn: WarnFn
): WebGLLightState {
	const state: WebGLLightState = {
		ambientColor: [0, 0, 0],
		directionalLights: [],
		pointLights: [],
		spotLights: [],
	}
	if (!enableLighting) {
		return state
	}

	for (const light of lights) {
		switch (light.type) {
			case LightType.Ambient: {
				state.ambientColor[0] +=
					sRGBToLinear((light.color.r ?? 255) / 255) * (light.intensity ?? 1)
				state.ambientColor[1] +=
					sRGBToLinear((light.color.g ?? 255) / 255) * (light.intensity ?? 1)
				state.ambientColor[2] +=
					sRGBToLinear((light.color.b ?? 255) / 255) * (light.intensity ?? 1)
				break
			}
			case LightType.Directional: {
				if (state.directionalLights.length >= WEBGL_MAX_DIRECTIONAL_LIGHTS) {
					warn(
						'webgl-directional-light-limit',
						`WebGL forward shading supports at most ${WEBGL_MAX_DIRECTIONAL_LIGHTS} directional lights; extra lights are ignored`
					)
					break
				}
				const direction = getDirectionalLightWorldDirection(light)
				const intensity = light.intensity ?? 1
				state.directionalLights.push({
					direction: [-direction.x, -direction.y, -direction.z],
					color: [
						sRGBToLinear((light.color.r ?? 255) / 255) * intensity,
						sRGBToLinear((light.color.g ?? 255) / 255) * intensity,
						sRGBToLinear((light.color.b ?? 255) / 255) * intensity,
					],
				})
				break
			}
			case LightType.Point: {
				if (state.pointLights.length >= WEBGL_MAX_POINT_LIGHTS) {
					warn(
						'webgl-point-light-limit',
						`WebGL forward shading supports at most ${WEBGL_MAX_POINT_LIGHTS} point lights; extra lights are ignored`
					)
					break
				}
				const position = getPointLightWorldPosition(light)
				const intensity = light.intensity ?? 1
				state.pointLights.push({
					position: [position.x, position.y, position.z],
					range: Math.max(0.001, (light as any).range ?? 1000),
					color: [
						sRGBToLinear((light.color.r ?? 255) / 255) * intensity,
						sRGBToLinear((light.color.g ?? 255) / 255) * intensity,
						sRGBToLinear((light.color.b ?? 255) / 255) * intensity,
					],
				})
				break
			}
			case LightType.Spot: {
				if (state.spotLights.length >= WEBGL_MAX_SPOT_LIGHTS) {
					warn(
						'webgl-spot-light-limit',
						`WebGL forward shading supports at most ${WEBGL_MAX_SPOT_LIGHTS} spot lights; extra lights are ignored`
					)
					break
				}
				const position = getSpotLightWorldPosition(light)
				const direction = getSpotLightWorldDirection(light)
				const outerCos = Math.cos((light as any).angle ?? Math.PI / 4)
				const innerCos = Math.cos(getSpotLightInnerAngle(light as any))
				const intensity = light.intensity ?? 1
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
				})
				break
			}
			case LightType.LightProbe:
			case LightType.RectArea:
			default: {
				warn(
					`webgl-light-unsupported-${light.type}`,
					`WebGL v1 does not support ${light.type} lights yet; ignoring this light`
				)
				break
			}
		}
	}

	return state
}
