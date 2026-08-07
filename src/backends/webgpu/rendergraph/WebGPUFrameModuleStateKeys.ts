import { defineWebGPUFrameModuleStateKey } from "./WebGPUFrameGraphModule";
import type {
	WebGPUDeferredFeatureAnalysis,
	WebGPUPostProcessFeatureAnalysis,
	WebGPUReflectionFeatureAnalysis,
	WebGPUTransparencyAnalysis,
	WebGPUVisibilityFeatureAnalysis,
} from "./WebGPUFrameFeatureAnalyzer";

export const WEBGPU_DEFERRED_FEATURE_ANALYSIS =
	defineWebGPUFrameModuleStateKey<WebGPUDeferredFeatureAnalysis>(
		"deferred",
		"webgpu:deferred-analysis",
	);

export const WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS =
	defineWebGPUFrameModuleStateKey<WebGPUTransparencyAnalysis>(
		"transparency",
		"webgpu:transparency-analysis",
	);

export const WEBGPU_REFLECTION_FEATURE_ANALYSIS =
	defineWebGPUFrameModuleStateKey<WebGPUReflectionFeatureAnalysis>(
		"reflection",
		"webgpu:reflection-analysis",
	);

export const WEBGPU_VISIBILITY_FEATURE_ANALYSIS =
	defineWebGPUFrameModuleStateKey<WebGPUVisibilityFeatureAnalysis>(
		"visibility",
		"webgpu:visibility-analysis",
	);

export const WEBGPU_POST_PROCESS_FEATURE_ANALYSIS =
	defineWebGPUFrameModuleStateKey<WebGPUPostProcessFeatureAnalysis>(
		"post-process",
		"webgpu:post-process-analysis",
	);
