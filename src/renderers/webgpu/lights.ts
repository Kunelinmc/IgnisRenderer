import { sRGBToLinear, clamp } from "../../maths/Common";
import {
	LightType,
	type AmbientLight,
	type DirectionalLight,
	type LightProbe,
	type PointLight,
	type SceneLight,
	type ShadowCastingLight,
	type SpotLight,
} from "../../lights";
import type { ShadowMap, ShadowRenderSet } from "../../lights/ShadowMapping";
import {
	getDirectionalLightWorldDirection,
	getPointLightWorldPosition,
	getSpotLightInnerAngle,
	getSpotLightWorldDirection,
	getSpotLightWorldPosition,
} from "../../pipeline/LightTransforms";
import type { RGB } from "../../foundation/Color";
import type { Matrix4 } from "../../maths/Matrix4";

import {
	WEBGPU_MAX_DIRECTIONAL_LIGHTS,
	WEBGPU_MAX_POINT_LIGHTS,
	WEBGPU_MAX_SPOT_LIGHTS,
} from "./constants";
import type {
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
			case LightType.LightProbe:
				accumulateLightProbeFallbackAmbient(state, light, enableSH);
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
		clusteredLights: [],
		volumetricLights: [],
		warnings: [],
	};
}

function accumulateAmbientLight(
	state: WebGPULightingState,
	light: AmbientLight
): void {
	state.ambientColor[0] += sRGBToLinear(light.color.r / 255) * light.intensity;
	state.ambientColor[1] += sRGBToLinear(light.color.g / 255) * light.intensity;
	state.ambientColor[2] += sRGBToLinear(light.color.b / 255) * light.intensity;
}

function accumulateLightProbeFallbackAmbient(
	state: WebGPULightingState,
	light: LightProbe,
	enableSH: boolean
): void {
	if (enableSH) return;

	const dc = light.sh[0];
	if (!dc) return;

	const irradianceScale = Math.PI * 0.282095;
	state.ambientColor[0] +=
		(Math.max(0, dc.r * irradianceScale) / 255) * light.intensity;
	state.ambientColor[1] +=
		(Math.max(0, dc.g * irradianceScale) / 255) * light.intensity;
	state.ambientColor[2] +=
		(Math.max(0, dc.b * irradianceScale) / 255) * light.intensity;
}

function collectDirectionalLight(
	state: WebGPULightingState,
	light: DirectionalLight,
	enableShadows: boolean,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>
): void {
	const direction = getDirectionalLightWorldDirection(light);
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
	const position = getPointLightWorldPosition(light);
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
	const position = getSpotLightWorldPosition(light);
	const direction = getSpotLightWorldDirection(light);
	const outerAngle = light.outerAngle;
	const innerAngle = getSpotLightInnerAngle(light);
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
	pushClusteredSpotLight(
		state,
		position,
		range,
		direction,
		Math.cos(outerAngle),
		Math.cos(innerAngle),
		color,
		shadowData.enabled,
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

function toLinearLightColor(color: RGB, intensity: number): WebGPUVec3 {
	return [
		sRGBToLinear(color.r / 255) * intensity,
		sRGBToLinear(color.g / 255) * intensity,
		sRGBToLinear(color.b / 255) * intensity,
	];
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
		type: 0,
		position: [position.x, position.y, position.z],
		range,
		direction: [0, 0, 0],
		outerCos: -2,
		innerCos: -2,
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
		type: 1,
		position: [position.x, position.y, position.z],
		range,
		direction: [direction.x, direction.y, direction.z],
		outerCos,
		innerCos,
		color,
		castsShadow,
		affectsVolumetric: true,
		shadowIndex: Math.max(0, shadowIndex | 0),
	};
	state.clusteredLights.push(light);
}

function resolveWebGPUShadowData(
	enableShadows: boolean,
	renderSetInput?: ShadowRenderSet | ShadowMap
): WebGPUShadowData {
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
			shadowMap: resolvedShadowMap,
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
	const cascadeCount =
		strategyType === "csm" ?
			Math.max(1, Math.min(4, renderSet?.slices.length ?? 1))
		: 	1;
	const cascadeBlendRatio =
		strategyType === "csm" &&
		renderSet &&
		renderSet.resolvedConfig.strategy === "csm" ?
			Math.max(0, Math.min(1, renderSet.resolvedConfig.blendRatio ?? 0.1))
		: 	0;

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
