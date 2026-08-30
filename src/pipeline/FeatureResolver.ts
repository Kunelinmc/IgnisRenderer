import type {
	BackendCapabilities,
	RenderBackendType,
} from "../backends/IRenderBackend";
import type { RendererFeatureRequest, ResolvedFeatureState } from "./types";
import {
	DEFAULT_CLUSTERED_LIGHTING_OPTIONS,
	DEFAULT_OCCLUSION_CULLING_OPTIONS,
} from "./types";
import { normalizeOcclusionCullingOptions } from "./OcclusionCulling";

type RendererFeatureCapability =
	| "sh"
	| "reflection"
	| "environment"
	| "oit"
	| "clusteredLighting"
	| "occlusionCulling";

const FEATURE_WARNING_KEYS: Record<RendererFeatureCapability, string> = {
	sh: "feature-sh",
	reflection: "feature-reflection",
	environment: "feature-environment",
	oit: "feature-oit",
	clusteredLighting: "feature-clustered-lighting",
	occlusionCulling: "feature-occlusion-culling",
};

const FEATURE_WARNING_LABELS: Record<RendererFeatureCapability, string> = {
	sh: "spherical harmonics",
	reflection: "planar reflections",
	environment: "environment rendering",
	oit: "order-independent transparency",
	clusteredLighting: "clustered lighting",
	occlusionCulling: "occlusion culling",
};

export function resolveFeatureState(
	request: RendererFeatureRequest,
	capabilities: BackendCapabilities,
	backendType: RenderBackendType
): ResolvedFeatureState {
	const warnings: ResolvedFeatureState["warnings"] = [];

	return {
		enableLighting: request.enableLighting !== false,
		clusteredLightingOptions: {
			...DEFAULT_CLUSTERED_LIGHTING_OPTIONS,
			...(request.clusteredLightingOptions ?? {}),
		},
		occlusionCullingOptions: normalizeOcclusionCullingOptions({
			...DEFAULT_OCCLUSION_CULLING_OPTIONS,
			...(request.occlusionCullingOptions ?? {}),
		}),
		enableSH: resolveBooleanFeature(
			request.enableSH,
			capabilities.sh,
			"sh",
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
		enableOcclusionCulling: resolveBooleanFeature(
			request.enableOcclusionCulling,
			capabilities.occlusionCulling,
			"occlusionCulling",
			backendType,
			warnings
		),
		warnings,
	};
}

function resolveBooleanFeature(
	requested: boolean | undefined,
	supported: boolean,
	feature: RendererFeatureCapability,
	backendType: RenderBackendType,
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
