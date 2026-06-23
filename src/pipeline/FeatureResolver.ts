import type { BackendCapabilities } from "../renderers/IRenderBackend";
import type { RendererFeatureRequest, ResolvedFeatureState } from "./types";
import {
	DEFAULT_CLUSTERED_LIGHTING_OPTIONS,
	DEFAULT_OCCLUSION_CULLING_OPTIONS,
} from "./types";
import { normalizeOcclusionCullingOptions } from "./OcclusionCulling";

const FEATURE_WARNING_KEYS: Record<keyof BackendCapabilities, string> = {
	sh: "feature-sh",
	shadows: "feature-shadows",
	reflection: "feature-reflection",
	environment: "feature-environment",
	postProcess: "feature-postprocess",
	oit: "feature-oit",
	clusteredLighting: "feature-clustered-lighting",
	occlusionCulling: "feature-occlusion-culling",
	customRenderTargets: "feature-custom-render-targets",
	customRenderPasses: "feature-custom-render-passes",
	renderTargetReadback: "feature-render-target-readback",
};

const FEATURE_WARNING_LABELS: Record<keyof BackendCapabilities, string> = {
	sh: "spherical harmonics",
	shadows: "shadows",
	reflection: "planar reflections",
	environment: "environment rendering",
	postProcess: "post-processing",
	oit: "order-independent transparency",
	clusteredLighting: "clustered lighting",
	occlusionCulling: "occlusion culling",
	customRenderTargets: "custom render targets",
	customRenderPasses: "custom render passes",
	renderTargetReadback: "render target readback",
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
