import {
	LightType,
	type AmbientLight,
	type AreaLight,
	type DirectionalLight,
	type LightProbe,
	type PointLight,
	type SceneLight,
	type ShadowCastingLight,
	type SpotLight,
} from "../../lights";
import type { ShadowRenderSet } from "../../lights/shadows/ShadowMapping";
import {
	accumulateAmbientLightColor,
	accumulateLightProbeFallbackAmbientColor,
	resolveShadowData as resolveSharedShadowData,
	toLinearLightColor,
} from "../../lights/runtime/lightingRuntime";

import {
	WEBGPU_CLUSTERED_LIGHT_TYPE_AREA,
	WEBGPU_CLUSTERED_LIGHT_TYPE_POINT,
	WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT,
	WEBGPU_MAX_AREA_LIGHTS,
	WEBGPU_MAX_DIRECTIONAL_LIGHTS,
	WEBGPU_MAX_POINT_LIGHTS,
	WEBGPU_MAX_SPOT_LIGHTS,
} from "./constants";
import type {
	WebGPUAreaLightUniform,
	WebGPUClusteredLightUniform,
	WebGPULightingState,
	WebGPUShadowData,
	WebGPUVolumetricLightUniform,
	WebGPUVec3,
	WebGPUWarning,
} from "./types";

export function collectWebGPULighting(
	lights: SceneLight[],
	enableLighting: boolean,
	enableSH: boolean,
	enableShadows: boolean = false,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>,
	enableClusteredLighting: boolean = false
): WebGPULightingState {
	const state = createEmptyWebGPULightingState();
	if (!enableLighting) return state;

	for (const light of lights) {
		switch (light.type) {
			case LightType.Ambient:
				accumulateAmbientLight(state, light);
				break;
			case LightType.Directional:
				collectDirectionalLight(
					state,
					light,
					enableShadows,
					shadowMaps
				);
				break;
			case LightType.Point:
				collectPointLight(state, light, enableClusteredLighting);
				break;
			case LightType.Spot:
				collectSpotLight(
					state,
					light,
					enableShadows,
					shadowMaps,
					enableClusteredLighting
				);
				break;
			case LightType.RectArea:
				collectAreaLight(state, light, enableClusteredLighting);
				break;
			case LightType.LightProbe:
				accumulateLightProbeFallbackAmbient(state, light, enableSH);
				break;
			case LightType.IrradianceProbeGrid:
				break;
			case LightType.ReflectionProbe:
				break;
			default:
				state.warnings.push(createUnsupportedLightWarning(light));
				break;
		}
	}

	return state;
}

function createEmptyWebGPULightingState(): WebGPULightingState {
	return {
		ambientColor: [0, 0, 0],
		directionalLights: [],
		directionalShadows: [],
		pointLights: [],
		spotLights: [],
		spotShadows: [],
		areaLights: [],
		clusteredLights: [],
		volumetricLights: [],
		warnings: [],
	};
}

function accumulateAmbientLight(
	state: WebGPULightingState,
	light: AmbientLight
): void {
	accumulateAmbientLightColor(state.ambientColor, light.color, light.intensity);
}

function accumulateLightProbeFallbackAmbient(
	state: WebGPULightingState,
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

function collectDirectionalLight(
	state: WebGPULightingState,
	light: DirectionalLight,
	enableShadows: boolean,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>
): void {
	const direction = light.getWorldLightDirection();
	const color = toLinearLightColor(light.color, light.intensity);
	pushVolumetricDirectionalLight(state, direction, color);

	if (state.directionalLights.length >= WEBGPU_MAX_DIRECTIONAL_LIGHTS) {
		state.warnings.push(
			createLightLimitWarning("directional", WEBGPU_MAX_DIRECTIONAL_LIGHTS)
		);
		return;
	}

	state.directionalLights.push({
		direction: [-direction.x, -direction.y, -direction.z],
		color,
	});
	state.directionalShadows.push(
		resolveWebGPUShadowData(
			enableShadows,
			shadowMaps?.get(light as ShadowCastingLight)
		)
	);
}

function collectPointLight(
	state: WebGPULightingState,
	light: PointLight,
	enableClusteredLighting: boolean
): void {
	const position = light.getWorldLightPosition();
	const color = toLinearLightColor(light.color, light.intensity);
	const range = Math.max(light.range, 0.001);
	pushVolumetricPointLight(state, position, range, color);
	pushClusteredPointLight(state, position, range, color, enableClusteredLighting);

	if (state.pointLights.length >= WEBGPU_MAX_POINT_LIGHTS) {
		if (!enableClusteredLighting) {
			state.warnings.push(
				createLightLimitWarning("point", WEBGPU_MAX_POINT_LIGHTS)
			);
		}
		return;
	}

	state.pointLights.push({
		position: [position.x, position.y, position.z],
		range,
		color,
	});
}

function collectSpotLight(
	state: WebGPULightingState,
	light: SpotLight,
	enableShadows: boolean,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>,
	enableClusteredLighting: boolean = false
): void {
	const position = light.getWorldLightPosition();
	const direction = light.getWorldLightDirection();
	const outerAngle = light.outerAngle;
	const innerAngle = light.getInnerAngle();
	const range = Math.max(light.range, 0.001);
	const color = toLinearLightColor(light.color, light.intensity);
	pushVolumetricSpotLight(
		state,
		position,
		range,
		direction,
		Math.cos(outerAngle),
		Math.cos(innerAngle),
		color
	);

	const shadowData = resolveWebGPUShadowData(
		enableShadows,
		shadowMaps?.get(light as ShadowCastingLight)
	);
	const shadowIndex = state.spotShadows.length;
	const clusteredCastsShadow =
		shadowData.enabled && shadowIndex < WEBGPU_MAX_SPOT_LIGHTS;
	if (enableClusteredLighting && shadowData.enabled && !clusteredCastsShadow) {
		pushWarningOnce(state, {
			key: "webgpu-clustered-spot-shadow-budget",
			message:
				"WebGPU clustered lighting keeps spot lights beyond the spot " +
				"shadow budget, but disables shadows for the extra spot lights.",
		});
	}
	pushClusteredSpotLight(
		state,
		position,
		range,
		direction,
		Math.cos(outerAngle),
		Math.cos(innerAngle),
		color,
		clusteredCastsShadow,
		shadowIndex,
		enableClusteredLighting
	);

	if (state.spotLights.length >= WEBGPU_MAX_SPOT_LIGHTS) {
		if (!enableClusteredLighting) {
			state.warnings.push(
				createLightLimitWarning("spot", WEBGPU_MAX_SPOT_LIGHTS)
			);
		}
		return;
	}

	state.spotLights.push({
		position: [position.x, position.y, position.z],
		range,
		direction: [direction.x, direction.y, direction.z],
		outerCos: Math.cos(outerAngle),
		innerCos: Math.cos(innerAngle),
		color,
	});
	state.spotShadows.push(shadowData);
}

function collectAreaLight(
	state: WebGPULightingState,
	light: AreaLight,
	enableClusteredLighting: boolean
): void {
	const width = Math.max(light.width, 0);
	const height = Math.max(light.height, 0);
	const range = Math.max(light.range, 0);
	if (width <= 0 || height <= 0 || range <= 0) {
		return;
	}

	const matrix = light.worldMatrix.elements;
	const right = normalizeVector3(
		matrix[0][0],
		matrix[1][0],
		matrix[2][0],
		[1, 0, 0]
	);
	const up = normalizeVector3(
		matrix[0][2],
		matrix[1][2],
		matrix[2][2],
		[0, 0, 1]
	);
	const normal = normalizeVector3(
		matrix[0][1],
		matrix[1][1],
		matrix[2][1],
		[0, 1, 0]
	);

	const areaLight: WebGPUAreaLightUniform = {
		position: [matrix[0][3], matrix[1][3], matrix[2][3]],
		range,
		right,
		width,
		up,
		height,
		normal,
		areaScale: width * height,
		color: toLinearLightColor(light.color, light.intensity),
	};

	pushClusteredAreaLight(state, areaLight, enableClusteredLighting);

	if (state.areaLights.length >= WEBGPU_MAX_AREA_LIGHTS) {
		if (!enableClusteredLighting) {
			state.warnings.push(
				createLightLimitWarning("area", WEBGPU_MAX_AREA_LIGHTS)
			);
		}
		return;
	}

	state.areaLights.push(areaLight);
}

function createLightLimitWarning(
	kind: string,
	maxCount: number
): WebGPUWarning {
	return {
		key: `webgpu-${kind}-limit`,
		message: `WebGPU forward shading supports at most ${maxCount} ${kind} lights; extra lights are skipped in main shading`,
	};
}

function createUnsupportedLightWarning(light: SceneLight): WebGPUWarning {
	return {
		key: `webgpu-light-${light.type}`,
		message: `WebGPU backend does not support ${light.type} lights yet; ignoring them for now`,
	};
}

function pushWarningOnce(
	state: WebGPULightingState,
	warning: WebGPUWarning
): void {
	if (state.warnings.some((existing) => existing.key === warning.key)) {
		return;
	}
	state.warnings.push(warning);
}

function pushVolumetricDirectionalLight(
	state: WebGPULightingState,
	direction: { x: number; y: number; z: number },
	color: WebGPUVec3
): void {
	const light: WebGPUVolumetricLightUniform = {
		type: 0,
		position: [0, 0, 0],
		range: -1,
		direction: [-direction.x, -direction.y, -direction.z],
		outerCos: 0,
		innerCos: 0,
		color,
	};
	state.volumetricLights.push(light);
}

function pushVolumetricPointLight(
	state: WebGPULightingState,
	position: { x: number; y: number; z: number },
	range: number,
	color: WebGPUVec3
): void {
	const light: WebGPUVolumetricLightUniform = {
		type: 1,
		position: [position.x, position.y, position.z],
		range,
		direction: [0, 0, 0],
		outerCos: -2,
		innerCos: -2,
		color,
	};
	state.volumetricLights.push(light);
}

function pushVolumetricSpotLight(
	state: WebGPULightingState,
	position: { x: number; y: number; z: number },
	range: number,
	direction: { x: number; y: number; z: number },
	outerCos: number,
	innerCos: number,
	color: WebGPUVec3
): void {
	const light: WebGPUVolumetricLightUniform = {
		type: 2,
		position: [position.x, position.y, position.z],
		range,
		direction: [direction.x, direction.y, direction.z],
		outerCos,
		innerCos,
		color,
	};
	state.volumetricLights.push(light);
}

function pushClusteredPointLight(
	state: WebGPULightingState,
	position: { x: number; y: number; z: number },
	range: number,
	color: WebGPUVec3,
	enableClusteredLighting: boolean
): void {
	if (!enableClusteredLighting) {
		return;
	}
	const light: WebGPUClusteredLightUniform = {
		type: WEBGPU_CLUSTERED_LIGHT_TYPE_POINT,
		position: [position.x, position.y, position.z],
		range,
		direction: [0, 0, 0],
		outerCos: -2,
		innerCos: -2,
		right: [0, 0, 0],
		width: 0,
		up: [0, 0, 0],
		height: 0,
		normal: [0, 1, 0],
		areaScale: 0,
		color,
		castsShadow: false,
		affectsVolumetric: true,
		shadowIndex: 0,
	};
	state.clusteredLights.push(light);
}

function pushClusteredSpotLight(
	state: WebGPULightingState,
	position: { x: number; y: number; z: number },
	range: number,
	direction: { x: number; y: number; z: number },
	outerCos: number,
	innerCos: number,
	color: WebGPUVec3,
	castsShadow: boolean,
	shadowIndex: number,
	enableClusteredLighting: boolean
): void {
	if (!enableClusteredLighting) {
		return;
	}
	const light: WebGPUClusteredLightUniform = {
		type: WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT,
		position: [position.x, position.y, position.z],
		range,
		direction: [direction.x, direction.y, direction.z],
		outerCos,
		innerCos,
		right: [0, 0, 0],
		width: 0,
		up: [0, 0, 0],
		height: 0,
		normal: [0, 1, 0],
		areaScale: 0,
		color,
		castsShadow,
		affectsVolumetric: true,
		shadowIndex: Math.max(0, shadowIndex | 0),
	};
	state.clusteredLights.push(light);
}

function pushClusteredAreaLight(
	state: WebGPULightingState,
	areaLight: WebGPUAreaLightUniform,
	enableClusteredLighting: boolean
): void {
	if (!enableClusteredLighting) {
		return;
	}
	const light: WebGPUClusteredLightUniform = {
		type: WEBGPU_CLUSTERED_LIGHT_TYPE_AREA,
		position: areaLight.position,
		range: areaLight.range,
		direction: [0, 0, 0],
		outerCos: -2,
		innerCos: -2,
		right: areaLight.right,
		width: areaLight.width,
		up: areaLight.up,
		height: areaLight.height,
		normal: areaLight.normal,
		areaScale: areaLight.areaScale,
		color: areaLight.color,
		castsShadow: false,
		affectsVolumetric: false,
		shadowIndex: 0,
	};
	state.clusteredLights.push(light);
}

function resolveWebGPUShadowData(
	enableShadows: boolean,
	renderSetInput?: ShadowRenderSet
): WebGPUShadowData {
	return resolveSharedShadowData(enableShadows, renderSetInput, {
		keepShadowMapWhenDisabled: true,
	});
}

function normalizeVector3(
	x: number,
	y: number,
	z: number,
	fallback: WebGPUVec3
): WebGPUVec3 {
	const length = Math.hypot(x, y, z);
	if (length <= 1e-6) {
		return fallback;
	}
	const invLength = 1 / length;
	return [x * invLength, y * invLength, z * invLength];
}
