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
} from "./constants";
import {
	MAX_AREA_LIGHTS,
	MAX_DIRECTIONAL_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../constants";
import type {
	Vec3Tuple,
	WebGPUAreaLightUniform,
	WebGPUClusteredLightUniform,
	WebGPUClusteredLightingData,
	WebGPUDirectionalLightUniform,
	WebGPULightingCatalog,
	WebGPULightingCatalogLight,
	WebGPULightingState,
	WebGPUPointLightUniform,
	WebGPUShadowData,
	WebGPUSpotLightUniform,
	WebGPUSurfaceLightingView,
	WebGPUVolumetricLightUniform,
	WebGPUVolumetricLightingData,
	WebGPUWarning,
} from "./types";

export function collectWebGPULightingCatalog(
	lights: SceneLight[],
	enableLighting: boolean,
	enableSH: boolean,
	enableShadows: boolean = false,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>
): WebGPULightingCatalog {
	const catalog: WebGPULightingCatalog = {
		ambientColor: [0, 0, 0],
		lights: [],
		warnings: [],
	};
	if (!enableLighting) return catalog;

	let spotShadowIndex = 0;
	for (const light of lights) {
		switch (light.type) {
			case LightType.Ambient:
				accumulateAmbientLight(catalog, light);
				break;
			case LightType.Directional:
				collectDirectionalLight(catalog, light, enableShadows, shadowMaps);
				break;
			case LightType.Point:
				collectPointLight(catalog, light);
				break;
			case LightType.Spot:
				collectSpotLight(
					catalog,
					light,
					spotShadowIndex++,
					enableShadows,
					shadowMaps
				);
				break;
			case LightType.RectArea:
				collectAreaLight(catalog, light);
				break;
			case LightType.LightProbe:
				accumulateLightProbeFallbackAmbient(catalog, light, enableSH);
				break;
			case LightType.IrradianceProbeGrid:
				break;
			case LightType.ReflectionProbe:
				break;
			default:
				catalog.warnings.push(createUnsupportedLightWarning(light));
				break;
		}
	}

	return catalog;
}

export function collectWebGPULighting(
	lights: SceneLight[],
	enableLighting: boolean,
	enableSH: boolean,
	enableShadows: boolean = false,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>,
	enableClusteredLighting: boolean = false
): WebGPULightingState {
	return createWebGPULightingState(
		collectWebGPULightingCatalog(
			lights,
			enableLighting,
			enableSH,
			enableShadows,
			shadowMaps
		),
		enableClusteredLighting
	);
}

export function createWebGPULightingState(
	catalog: WebGPULightingCatalog,
	enableClusteredLighting: boolean = false
): WebGPULightingState {
	const state: WebGPULightingState = {
		ambientColor: catalog.ambientColor,
		directionalLights: [],
		directionalShadows: [],
		pointLights: [],
		spotLights: [],
		spotShadows: [],
		areaLights: [],
		warnings: catalog.warnings.slice(),
	};

	for (const light of catalog.lights) {
		switch (light.type) {
			case "directional":
				if (state.directionalLights.length >= MAX_DIRECTIONAL_LIGHTS) {
					state.warnings.push(
						createLightLimitWarning("directional", MAX_DIRECTIONAL_LIGHTS)
					);
					continue;
				}
				state.directionalLights.push({
					direction: light.direction,
					color: light.color,
				});
				state.directionalShadows.push(light.shadow ?? createDisabledShadowData());
				break;
			case "point":
				if (state.pointLights.length >= MAX_POINT_LIGHTS) {
					if (!enableClusteredLighting) {
						state.warnings.push(
							createLightLimitWarning("point", MAX_POINT_LIGHTS)
						);
					}
					continue;
				}
				state.pointLights.push({
					position: light.position,
					range: light.range,
					color: light.color,
				});
				break;
			case "spot":
				if (state.spotLights.length >= MAX_SPOT_LIGHTS) {
					if (!enableClusteredLighting) {
						state.warnings.push(
							createLightLimitWarning("spot", MAX_SPOT_LIGHTS)
						);
					}
					continue;
				}
				state.spotLights.push({
					position: light.position,
					range: light.range,
					direction: light.direction,
					outerCos: light.outerCos,
					innerCos: light.innerCos,
					color: light.color,
				});
				state.spotShadows.push(light.shadow ?? createDisabledShadowData());
				break;
			case "area":
				if (state.areaLights.length >= MAX_AREA_LIGHTS) {
					if (!enableClusteredLighting) {
						state.warnings.push(
							createLightLimitWarning("area", MAX_AREA_LIGHTS)
						);
					}
					continue;
				}
				state.areaLights.push({
					position: light.position,
					range: light.range,
					right: light.right,
					width: light.width,
					up: light.up,
					height: light.height,
					normal: light.normal,
					areaScale: light.areaScale,
					color: light.color,
				});
				break;
		}
	}

	return state;
}

export function createWebGPUClusteredLightingData(
	catalog: WebGPULightingCatalog,
	maxLights: number
): WebGPUClusteredLightingData {
	const lights: WebGPUClusteredLightUniform[] = [];
	const warnings: WebGPUWarning[] = [];
	for (const light of catalog.lights) {
		const clustered = createClusteredLight(light);
		if (!clustered) {
			continue;
		}
		if (lights.length >= maxLights) {
			continue;
		}
		lights.push(clustered);
		if (
			light.type === "spot" &&
			light.shadow?.enabled &&
			light.shadowIndex >= MAX_SPOT_LIGHTS
		) {
			pushWarningOnce(warnings, {
				key: "webgpu-clustered-spot-shadow-budget",
				message:
					"WebGPU clustered lighting keeps spot lights beyond the spot " +
					"shadow budget, but disables shadows for the extra spot lights.",
			});
		}
	}
	if (catalog.lights.some((light) => isClusteredLightType(light)) && lights.length >= maxLights) {
		const sourceCount = catalog.lights.filter(isClusteredLightType).length;
		if (sourceCount > maxLights) {
			warnings.push({
				key: "webgpu-clustered-light-budget",
				message:
					`WebGPU clustered lighting clamps lights to ${maxLights}; ` +
					"extra lights are skipped",
			});
		}
	}
	return { lights, warnings };
}

export function createWebGPUSurfaceLightingView(
	lightingState: WebGPULightingState,
	clusteredLighting: WebGPUClusteredLightingData | null
): WebGPUSurfaceLightingView {
	return {
		directionalLights: lightingState.directionalLights,
		pointLights: clusteredLighting ? [] : lightingState.pointLights,
		spotLights: clusteredLighting ? [] : lightingState.spotLights,
		areaLights: clusteredLighting ? [] : lightingState.areaLights,
		clusteredLights: clusteredLighting?.lights ?? [],
	};
}

export function createWebGPUVolumetricLightingData(
	surface: WebGPUSurfaceLightingView
): WebGPUVolumetricLightingData {
	const lights: WebGPUVolumetricLightUniform[] = [];
	for (const light of surface.directionalLights) {
		lights.push(createVolumetricDirectionalLight(light));
	}
	for (const light of surface.pointLights) {
		lights.push(createVolumetricPointLight(light));
	}
	for (const light of surface.spotLights) {
		lights.push(createVolumetricSpotLight(light));
	}
	for (const light of surface.clusteredLights) {
		const volumetric = createVolumetricClusteredLight(light);
		if (volumetric) {
			lights.push(volumetric);
		}
	}
	return { lights, warnings: [] };
}

function accumulateAmbientLight(
	catalog: WebGPULightingCatalog,
	light: AmbientLight
): void {
	accumulateAmbientLightColor(catalog.ambientColor, light.color, light.intensity);
}

function accumulateLightProbeFallbackAmbient(
	catalog: WebGPULightingCatalog,
	light: LightProbe,
	enableSH: boolean
): void {
	if (enableSH) return;

	accumulateLightProbeFallbackAmbientColor(
		catalog.ambientColor,
		light.sh[0],
		1
	);
}

function collectDirectionalLight(
	catalog: WebGPULightingCatalog,
	light: DirectionalLight,
	enableShadows: boolean,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>
): void {
	const direction = light.getWorldLightDirection();
	catalog.lights.push({
		type: "directional",
		source: light,
		position: [0, 0, 0],
		range: -1,
		direction: [-direction.x, -direction.y, -direction.z],
		outerCos: 0,
		innerCos: 0,
		right: [0, 0, 0],
		width: 0,
		up: [0, 0, 0],
		height: 0,
		normal: [0, 1, 0],
		areaScale: 0,
		color: toLinearLightColor(light.color, light.intensity),
		shadow: resolveWebGPUShadowData(
			enableShadows,
			shadowMaps?.get(light as ShadowCastingLight)
		),
		shadowIndex: -1,
	});
}

function collectPointLight(
	catalog: WebGPULightingCatalog,
	light: PointLight
): void {
	const position = light.getWorldLightPosition();
	catalog.lights.push({
		type: "point",
		source: light,
		position: [position.x, position.y, position.z],
		range: Math.max(light.range, 0.001),
		direction: [0, 0, 0],
		outerCos: -2,
		innerCos: -2,
		right: [0, 0, 0],
		width: 0,
		up: [0, 0, 0],
		height: 0,
		normal: [0, 1, 0],
		areaScale: 0,
		color: toLinearLightColor(light.color, light.intensity),
		shadow: null,
		shadowIndex: -1,
	});
}

function collectSpotLight(
	catalog: WebGPULightingCatalog,
	light: SpotLight,
	shadowIndex: number,
	enableShadows: boolean,
	shadowMaps?: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>
): void {
	const position = light.getWorldLightPosition();
	const direction = light.getWorldLightDirection();
	catalog.lights.push({
		type: "spot",
		source: light,
		position: [position.x, position.y, position.z],
		range: Math.max(light.range, 0.001),
		direction: [direction.x, direction.y, direction.z],
		outerCos: Math.cos(light.outerAngle),
		innerCos: Math.cos(light.getInnerAngle()),
		right: [0, 0, 0],
		width: 0,
		up: [0, 0, 0],
		height: 0,
		normal: [0, 1, 0],
		areaScale: 0,
		color: toLinearLightColor(light.color, light.intensity),
		shadow: resolveWebGPUShadowData(
			enableShadows,
			shadowMaps?.get(light as ShadowCastingLight)
		),
		shadowIndex,
	});
}

function collectAreaLight(
	catalog: WebGPULightingCatalog,
	light: AreaLight
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

	catalog.lights.push({
		type: "area",
		source: light,
		position: [matrix[0][3], matrix[1][3], matrix[2][3]],
		range,
		direction: [0, 0, 0],
		outerCos: -2,
		innerCos: -2,
		right,
		width,
		up,
		height,
		normal,
		areaScale: width * height,
		color: toLinearLightColor(light.color, light.intensity),
		shadow: null,
		shadowIndex: -1,
	});
}

function createClusteredLight(
	light: WebGPULightingCatalogLight
): WebGPUClusteredLightUniform | null {
	if (light.type === "directional") {
		return null;
	}
	const type =
		light.type === "area" ? WEBGPU_CLUSTERED_LIGHT_TYPE_AREA
		: light.type === "spot" ? WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT
		: WEBGPU_CLUSTERED_LIGHT_TYPE_POINT;
	return {
		type,
		position: light.position,
		range: light.range,
		direction: light.direction,
		outerCos: light.outerCos,
		innerCos: light.innerCos,
		right: light.right,
		width: light.width,
		up: light.up,
		height: light.height,
		normal: light.normal,
		areaScale: light.areaScale,
		color: light.color,
		castsShadow:
			light.type === "spot" &&
			!!light.shadow?.enabled &&
			light.shadowIndex < MAX_SPOT_LIGHTS,
		affectsVolumetric: light.type === "point" || light.type === "spot",
		shadowIndex: Math.max(0, light.shadowIndex | 0),
	};
}

function isClusteredLightType(light: WebGPULightingCatalogLight): boolean {
	return light.type === "point" || light.type === "spot" || light.type === "area";
}

function createVolumetricDirectionalLight(
	light: WebGPUDirectionalLightUniform
): WebGPUVolumetricLightUniform {
	return {
		type: 0,
		position: [0, 0, 0],
		range: -1,
		direction: light.direction,
		outerCos: 0,
		innerCos: 0,
		color: light.color,
	};
}

function createVolumetricPointLight(
	light: WebGPUPointLightUniform
): WebGPUVolumetricLightUniform {
	return {
		type: 1,
		position: light.position,
		range: light.range,
		direction: [0, 0, 0],
		outerCos: -2,
		innerCos: -2,
		color: light.color,
	};
}

function createVolumetricSpotLight(
	light: WebGPUSpotLightUniform
): WebGPUVolumetricLightUniform {
	return {
		type: 2,
		position: light.position,
		range: light.range,
		direction: light.direction,
		outerCos: light.outerCos,
		innerCos: light.innerCos,
		color: light.color,
	};
}

function createVolumetricClusteredLight(
	light: WebGPUClusteredLightUniform
): WebGPUVolumetricLightUniform | null {
	if (!light.affectsVolumetric) {
		return null;
	}
	if (light.type === WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT) {
		return createVolumetricSpotLight(light);
	}
	if (light.type === WEBGPU_CLUSTERED_LIGHT_TYPE_POINT) {
		return createVolumetricPointLight(light);
	}
	return null;
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
	warnings: WebGPUWarning[],
	warning: WebGPUWarning
): void {
	if (warnings.some((existing) => existing.key === warning.key)) {
		return;
	}
	warnings.push(warning);
}

function resolveWebGPUShadowData(
	enableShadows: boolean,
	renderSetInput?: ShadowRenderSet
): WebGPUShadowData {
	return resolveSharedShadowData(enableShadows, renderSetInput, {
		keepShadowMapWhenDisabled: true,
	});
}

function createDisabledShadowData(): WebGPUShadowData {
	return resolveWebGPUShadowData(false);
}

function normalizeVector3(
	x: number,
	y: number,
	z: number,
	fallback: Vec3Tuple
): Vec3Tuple {
	const length = Math.hypot(x, y, z);
	if (length <= 1e-6) {
		return fallback;
	}
	const invLength = 1 / length;
	return [x * invLength, y * invLength, z * invLength];
}
