import { CameraType } from "../../cameras/Camera";
import { WEBGPU_CLUSTERED_MAX_LIGHTS } from "./constants";
import {
	defineWebGPUFrameFeatureKey,
	WebGPUFrameFeatureRegistry,
	type WebGPUFrameFeatureContext,
	type WebGPUFrameFeatureModule,
} from "./FrameFeatures";
import {
	createWebGPUClusteredLightingData,
	createWebGPUSurfaceLightingView,
	createWebGPUVolumetricLightingData,
} from "./lights";
import type {
	WebGPUClusteredLightingData,
	WebGPUSurfaceLightingView,
	WebGPUVolumetricLightingData,
} from "./types";

export const WEBGPU_CLUSTERED_LIGHTING_DATA =
	defineWebGPUFrameFeatureKey<WebGPUClusteredLightingData>(
		"webgpu:clustered-lighting"
	);

export const WEBGPU_SURFACE_LIGHTING_VIEW =
	defineWebGPUFrameFeatureKey<WebGPUSurfaceLightingView>(
		"webgpu:surface-lighting-view"
	);

export const WEBGPU_VOLUMETRIC_LIGHTING_DATA =
	defineWebGPUFrameFeatureKey<WebGPUVolumetricLightingData>(
		"webgpu:volumetric-lighting"
	);

const CLUSTERED_PARAMS_DEFAULTS = {
	maxLights: 256,
} as const;
const WEBGPU_VOLUMETRIC_LIGHTING_PASS_ID = "volumetric";

const WebGPUClusteredLightingFeature: WebGPUFrameFeatureModule<
	WebGPUClusteredLightingData
> = {
	id: "webgpu-clustered-lighting",
	key: WEBGPU_CLUSTERED_LIGHTING_DATA,
	isEnabled: (context) => canPrepareClusteredLighting(context),
	prepare: (context) => {
		const requested = context.featureState.clusteredLightingOptions ?? {};
		const requestedMaxLights = Math.max(
			1,
			finiteInteger(requested.maxLights, CLUSTERED_PARAMS_DEFAULTS.maxLights)
		);
		const maxLights = Math.min(requestedMaxLights, WEBGPU_CLUSTERED_MAX_LIGHTS);
		return createWebGPUClusteredLightingData(context.lightingCatalog, maxLights);
	},
};

const WebGPUSurfaceLightingViewFeature: WebGPUFrameFeatureModule<
	WebGPUSurfaceLightingView
> = {
	id: "webgpu-surface-lighting-view",
	key: WEBGPU_SURFACE_LIGHTING_VIEW,
	isEnabled: () => true,
	prepare: (context) =>
		createWebGPUSurfaceLightingView(
			context.lightingState,
			context.dataStore.get(WEBGPU_CLUSTERED_LIGHTING_DATA) ?? null
		),
};

const WebGPUVolumetricLightingFeature: WebGPUFrameFeatureModule<
	WebGPUVolumetricLightingData
> = {
	id: "webgpu-volumetric-lighting",
	key: WEBGPU_VOLUMETRIC_LIGHTING_DATA,
	isEnabled: (context) =>
		context.featureState.postProcess.isEnabled(WEBGPU_VOLUMETRIC_LIGHTING_PASS_ID),
	prepare: (context) => {
		const surface = context.dataStore.get(WEBGPU_SURFACE_LIGHTING_VIEW);
		return createWebGPUVolumetricLightingData(
			surface ?? {
				directionalLights: [],
				pointLights: [],
				spotLights: [],
				areaLights: [],
				clusteredLights: [],
			}
		);
	},
};

export function createWebGPUFrameFeatureRegistry(): WebGPUFrameFeatureRegistry {
	return new WebGPUFrameFeatureRegistry([
		WebGPUClusteredLightingFeature,
		WebGPUSurfaceLightingViewFeature,
		WebGPUVolumetricLightingFeature,
	]);
}

export function canPrepareClusteredLighting(
	context: Pick<WebGPUFrameFeatureContext, "scene" | "featureState">
): boolean {
	if (!context.featureState.enableClusteredLighting || !context.featureState.enableLighting) {
		return false;
	}
	if (context.scene.camera.type !== CameraType.Perspective) {
		return false;
	}
	const near = Math.max(0.05, context.scene.camera.near ?? 0.1);
	const far = Math.max(near + 1e-3, context.scene.camera.far ?? near + 1);
	return Math.log(far) - Math.log(near) > 1e-6;
}

function finiteInteger(value: unknown, fallback: number): number {
	return Number.isFinite(value) ? Math.trunc(Number(value)) : fallback;
}
