import type { BackendCapabilities } from "../renderers/IRenderBackend";
import type { RendererFeatureRequest, ResolvedFeatureState } from "./types";
import {
	DEFAULT_SSAO_OPTIONS,
	DEFAULT_SSR_OPTIONS,
	DEFAULT_TAA_OPTIONS,
	DEFAULT_VOLUMETRIC_OPTIONS,
} from "./types";

const FEATURE_WARNING_KEYS: Record<keyof BackendCapabilities, string> = {
	sh: "feature-sh",
	shadows: "feature-shadows",
	reflection: "feature-reflection",
	skybox: "feature-skybox",
	ssao: "feature-ssao",
	taa: "feature-taa",
	ssr: "feature-ssr",
	volumetric: "feature-volumetric",
};

const FEATURE_WARNING_LABELS: Record<keyof BackendCapabilities, string> = {
	sh: "spherical harmonics",
	shadows: "shadows",
	reflection: "planar reflections",
	skybox: "skybox rendering",
	ssao: "SSAO",
	taa: "TAA",
	ssr: "SSR",
	volumetric: "volumetric effects",
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
		enableFXAA: request.enableFXAA === true,
		ssrOptions: { ...DEFAULT_SSR_OPTIONS, ...(request.ssrOptions ?? {}) },
		ssaoOptions: { ...DEFAULT_SSAO_OPTIONS, ...(request.ssaoOptions ?? {}) },
		taaOptions: { ...DEFAULT_TAA_OPTIONS, ...(request.taaOptions ?? {}) },
		volumetricOptions: {
			...DEFAULT_VOLUMETRIC_OPTIONS,
			...(request.volumetricOptions ?? {}),
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
		enableSSAO: resolveBooleanFeature(
			request.enableSSAO,
			capabilities.ssao,
			"ssao",
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
