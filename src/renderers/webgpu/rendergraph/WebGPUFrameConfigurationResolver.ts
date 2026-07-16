import type { WebGPUFrameFeatureAnalysis } from "./WebGPUFrameFeatureAnalyzer";
import {
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_TARGET_COUNT,
	WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT,
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
} from "../constants";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import type { WebGPUFrameTargetRequirements } from "./WebGPUFrameTargetManager";

export interface WebGPUFrameCapabilitySnapshot {
	readonly maxColorAttachments: number;
	readonly maxColorAttachmentBytesPerSample: number;
	readonly maxStorageTexturesPerShaderStage: number;
}

export interface WebGPUFrameConfigurationOptions {
	readonly enableDeferredLighting: boolean;
	readonly enableEarlyZPrepass: boolean;
	readonly sampleCount: number;
	readonly supportsInFrameTextureCopy: boolean;
	readonly forceDeferredFallback?: boolean;
	readonly forceForwardMrt?: boolean;
}

export interface WebGPUFrameDiagnostic {
	readonly code: string;
	readonly message: string;
}

export interface WebGPUFrameConfiguration {
	readonly mrtSupported: boolean;
	readonly deferredSupported: boolean;
	readonly deferredActive: boolean;
	readonly oitActive: boolean;
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly targetRequirements: WebGPUFrameTargetRequirements | null;
	readonly needsHiZBuild: boolean;
	readonly needsOcclusionTest: boolean;
	readonly enableEarlyZPrepass: boolean;
	readonly diagnostics: readonly WebGPUFrameDiagnostic[];
}

/** Resolves WebGPU frame feature policy without allocating resources or recording commands. */
export class WebGPUFrameConfigurationResolver {
	public resolve(
		analysis: WebGPUFrameFeatureAnalysis,
		capabilities: WebGPUFrameCapabilitySnapshot,
		options: WebGPUFrameConfigurationOptions,
	): WebGPUFrameConfiguration {
		const diagnostics: WebGPUFrameDiagnostic[] = [];
		const mrtSupported =
			capabilities.maxColorAttachments >= WEBGPU_MRT_COLOR_TARGET_COUNT &&
			capabilities.maxColorAttachmentBytesPerSample >= WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE;
		if (!mrtSupported) {
			if (capabilities.maxColorAttachments < WEBGPU_MRT_COLOR_TARGET_COUNT) {
				diagnostics.push({
					code: "webgpu-mrt-disabled-attachments",
					message:
						`WebGPU device maxColorAttachments is ${capabilities.maxColorAttachments}, ` +
						`requires ${WEBGPU_MRT_COLOR_TARGET_COUNT}; disabling MRT/GBuffer post-process pipeline`,
				});
			}
			if (
				capabilities.maxColorAttachmentBytesPerSample <
				WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE
			) {
				diagnostics.push({
					code: "webgpu-mrt-disabled-bytes",
					message:
						`WebGPU device maxColorAttachmentBytesPerSample is ` +
						`${capabilities.maxColorAttachmentBytesPerSample}, requires ` +
						`${WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE}; disabling MRT/GBuffer post-process pipeline`,
				});
			}
		}

		const deferredSupported =
			mrtSupported &&
			options.sampleCount === 1 &&
			capabilities.maxColorAttachments >= WEBGPU_DEFERRED_COLOR_TARGET_COUNT &&
			capabilities.maxColorAttachmentBytesPerSample >=
				WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE &&
			capabilities.maxStorageTexturesPerShaderStage >=
				WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT;
		const wantsDeferred =
			options.enableDeferredLighting &&
			!options.forceDeferredFallback &&
			analysis.hasDeferredLightingWork;
		if (options.enableDeferredLighting && !options.forceDeferredFallback && !deferredSupported) {
			diagnostics.push({
				code: !mrtSupported ? "webgpu-deferred-disabled-mrt" : "webgpu-deferred-disabled-runtime",
				message: !mrtSupported
					? "WebGPU deferred lighting requires MRT scene targets; using the non-deferred fallback path."
					: "WebGPU deferred lighting requirements are unavailable; using the legacy MRT forward path.",
			});
		}
		const deferredActive = wantsDeferred && deferredSupported;
		const oitActive =
			mrtSupported &&
			options.sampleCount === 1 &&
			options.supportsInFrameTextureCopy &&
			analysis.oitRequested &&
			analysis.hasOITWork;
		if (analysis.oitRequested && analysis.hasOITWork && !oitActive) {
			diagnostics.push({
				code:
					options.sampleCount > 1
						? "webgpu-oit-disabled-msaa"
						: !mrtSupported
							? "webgpu-oit-disabled-mrt-unavailable"
							: "webgpu-oit-disabled-runtime",
				message:
					options.sampleCount > 1
						? "WebGPU OIT v1 only supports sampleCount=1; falling back to legacy transparent rendering."
						: !mrtSupported
							? "WebGPU OIT requires MRT scene targets; falling back to legacy transparent rendering."
							: "WebGPU OIT requires in-frame texture-copy support; falling back to legacy transparent rendering.",
			});
		}
		const hasOffscreenWork =
			deferredActive ||
			analysis.needsPostProcessTargets ||
			analysis.needsPlanarReflection ||
			oitActive ||
			analysis.needsTransmissionTargets ||
			analysis.needsOcclusionTargets ||
			analysis.needsHiZTarget ||
			options.forceForwardMrt === true;
		const sceneTargetMode: WebGPUSceneTargetMode = !mrtSupported || !hasOffscreenWork
			? "single"
			: deferredActive
				? "gbuffer"
				: analysis.needsPostProcessGBuffer ||
						analysis.needsOcclusionTargets || analysis.needsHiZTarget
					? "mrt"
					: "color";
		return {
			mrtSupported,
			deferredSupported,
			deferredActive,
			oitActive,
			sceneTargetMode,
			targetRequirements:
				sceneTargetMode === "single"
					? null
					: {
							sceneTargetMode,
							needsPostProcessTargets: analysis.needsPostProcessTargets,
							needsOITTargets: oitActive,
							needsTransmissionTargets: analysis.needsTransmissionTargets,
							needsPlanarReflectionMask: analysis.needsPlanarReflectionMask,
							needsHiZTarget: analysis.needsHiZTarget,
						},
			needsHiZBuild: analysis.needsHiZTarget,
			needsOcclusionTest: analysis.needsOcclusionTargets,
			enableEarlyZPrepass: options.enableEarlyZPrepass,
			diagnostics,
		};
	}

}
