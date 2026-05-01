import type { BackendCapabilities } from "../renderers/IRenderBackend";
import type { RendererFeatureRequest, ResolvedFeatureState } from "./types";
import {
	DEFAULT_BLOOM_OPTIONS,
	DEFAULT_COLOR_FILTER_OPTIONS,
	DEFAULT_CLUSTERED_LIGHTING_OPTIONS,
	DEFAULT_DOF_OPTIONS,
	DEFAULT_FOG_OPTIONS,
	DEFAULT_MOTION_BLUR_OPTIONS,
	DEFAULT_SSAO_OPTIONS,
	DEFAULT_SSGI_OPTIONS,
	DEFAULT_SSR_OPTIONS,
	DEFAULT_TAA_OPTIONS,
	DEFAULT_VOLUMETRIC_OPTIONS,
} from "./types";

const FEATURE_WARNING_KEYS: Record<keyof BackendCapabilities, string> = {
	sh: "feature-sh",
	shadows: "feature-shadows",
	reflection: "feature-reflection",
	skybox: "feature-skybox",
	oit: "feature-oit",
	ssao: "feature-ssao",
	ssgi: "feature-ssgi",
	taa: "feature-taa",
	ssr: "feature-ssr",
	volumetric: "feature-volumetric",
	fog: "feature-fog",
	motionBlur: "feature-motion-blur",
	dof: "feature-dof",
	bloom: "feature-bloom",
	colorFilter: "feature-color-filter",
	clusteredLighting: "feature-clustered-lighting",
};

const FEATURE_WARNING_LABELS: Record<keyof BackendCapabilities, string> = {
	sh: "spherical harmonics",
	shadows: "shadows",
	reflection: "planar reflections",
	skybox: "skybox rendering",
	oit: "order-independent transparency",
	ssao: "SSAO",
	ssgi: "SSGI",
	taa: "TAA",
	ssr: "SSR",
	volumetric: "volumetric effects",
	fog: "fog",
	motionBlur: "motion blur",
	dof: "depth of field",
	bloom: "bloom",
	colorFilter: "color filter",
	clusteredLighting: "clustered lighting",
};

export function resolveFeatureState(
	request: RendererFeatureRequest,
	capabilities: BackendCapabilities,
	backendType: string
): ResolvedFeatureState {
	const warnings: ResolvedFeatureState["warnings"] = [];

	return {
		enableLighting: request.enableLighting !== false,
		enableGamma: request.enableGamma !== false,
		enableToneMapping: request.enableToneMapping !== false,
		enableFXAA: request.enableFXAA === true,
		ssrOptions: { ...DEFAULT_SSR_OPTIONS, ...(request.ssrOptions ?? {}) },
		ssaoOptions: { ...DEFAULT_SSAO_OPTIONS, ...(request.ssaoOptions ?? {}) },
		ssgiOptions: { ...DEFAULT_SSGI_OPTIONS, ...(request.ssgiOptions ?? {}) },
		taaOptions: { ...DEFAULT_TAA_OPTIONS, ...(request.taaOptions ?? {}) },
		volumetricOptions: {
			...DEFAULT_VOLUMETRIC_OPTIONS,
			...(request.volumetricOptions ?? {}),
		},
		fogOptions: {
			...DEFAULT_FOG_OPTIONS,
			...(request.fogOptions ?? {}),
		},
		bloomOptions: {
			...DEFAULT_BLOOM_OPTIONS,
			...(request.bloomOptions ?? {}),
		},
		motionBlurOptions: {
			...DEFAULT_MOTION_BLUR_OPTIONS,
			...(request.motionBlurOptions ?? {}),
		},
		dofOptions: {
			...DEFAULT_DOF_OPTIONS,
			...(request.dofOptions ?? {}),
		},
		colorFilterOptions: {
			...DEFAULT_COLOR_FILTER_OPTIONS,
			...(request.colorFilterOptions ?? {}),
		},
		clusteredLightingOptions: {
			...DEFAULT_CLUSTERED_LIGHTING_OPTIONS,
			...(request.clusteredLightingOptions ?? {}),
		},
		enableSH: resolveBooleanFeature(
			request.enableSH,
			capabilities.sh,
			"sh",
			backendType,
			warnings
		),
		enableShadows: resolveBooleanFeature(
			request.enableShadows,
			capabilities.shadows,
			"shadows",
			backendType,
			warnings
		),
		enableReflection: resolveBooleanFeature(
			request.enableReflection,
			capabilities.reflection,
			"reflection",
			backendType,
			warnings
		),
		enableSkybox: resolveBooleanFeature(
			request.enableSkybox,
			capabilities.skybox,
			"skybox",
			backendType,
			warnings
		),
		enableOIT: resolveBooleanFeature(
			request.enableOIT,
			capabilities.oit,
			"oit",
			backendType,
			warnings
		),
		enableSSAO: resolveBooleanFeature(
			request.enableSSAO,
			capabilities.ssao,
			"ssao",
			backendType,
			warnings
		),
		enableSSGI: resolveBooleanFeature(
			request.enableSSGI,
			capabilities.ssgi,
			"ssgi",
			backendType,
			warnings
		),
		enableTAA: resolveBooleanFeature(
			request.enableTAA,
			capabilities.taa,
			"taa",
			backendType,
			warnings
		),
		enableSSR: resolveBooleanFeature(
			request.enableSSR,
			capabilities.ssr,
			"ssr",
			backendType,
			warnings
		),
		enableVolumetric: resolveBooleanFeature(
			request.enableVolumetric,
			capabilities.volumetric,
			"volumetric",
			backendType,
			warnings
		),
		enableFog: resolveBooleanFeature(
			request.enableFog,
			capabilities.fog,
			"fog",
			backendType,
			warnings
		),
		enableMotionBlur: resolveBooleanFeature(
			request.enableMotionBlur,
			capabilities.motionBlur,
			"motionBlur",
			backendType,
			warnings
		),
		enableDOF: resolveBooleanFeature(
			request.enableDOF,
			capabilities.dof,
			"dof",
			backendType,
			warnings
		),
		enableBloom: resolveBooleanFeature(
			request.enableBloom,
			capabilities.bloom,
			"bloom",
			backendType,
			warnings
		),
		enableColorFilter: resolveBooleanFeature(
			request.enableColorFilter,
			capabilities.colorFilter,
			"colorFilter",
			backendType,
			warnings
		),
		enableClusteredLighting: resolveBooleanFeature(
			request.enableClusteredLighting,
			capabilities.clusteredLighting,
			"clusteredLighting",
			backendType,
			warnings
		),
		warnings,
	};
}

function resolveBooleanFeature(
	requested: boolean | undefined,
	supported: boolean,
	feature: keyof BackendCapabilities,
	backendType: string,
	warnings: ResolvedFeatureState["warnings"]
): boolean {
	const enabled = requested === true;

	if (enabled && !supported) {
		warnings.push({
			key: `${backendType}-${FEATURE_WARNING_KEYS[feature]}`,
			message: `${backendType} backend does not support ${FEATURE_WARNING_LABELS[feature]} yet; disabling it`,
		});
	}

	return enabled && supported;
}
