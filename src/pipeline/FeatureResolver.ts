import type { BackendCapabilities } from "../renderers/IRenderBackend";
import type { RendererFeatureRequest, ResolvedFeatureState } from "./types";
import {
	DEFAULT_CLUSTERED_LIGHTING_OPTIONS,
} from "./types";

const FEATURE_WARNING_KEYS: Record<keyof BackendCapabilities, string> = {
	sh: "feature-sh",
	shadows: "feature-shadows",
	reflection: "feature-reflection",
	environment: "feature-environment",
	oit: "feature-oit",
	clusteredLighting: "feature-clustered-lighting",
};

const FEATURE_WARNING_LABELS: Record<keyof BackendCapabilities, string> = {
	sh: "spherical harmonics",
	shadows: "shadows",
	reflection: "planar reflections",
	environment: "environment rendering",
	oit: "order-independent transparency",
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
		enableEnvironment: resolveBooleanFeature(
			request.enableEnvironment,
			capabilities.environment,
			"environment",
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
