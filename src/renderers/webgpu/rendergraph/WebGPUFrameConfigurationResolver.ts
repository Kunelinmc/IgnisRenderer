import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import {
	SCREEN_SPACE_REFLECTIONS_PASS_ID,
	SCREEN_SPACE_REFRACTIONS_PASS_ID,
	resolvePostProcessExecutionOrder,
} from "../../../postprocess";
import type { ResolvedPostProcessPass } from "../../../postprocess";
import { materialUsesTransmission } from "../../../materials/transparency";
import { materialSupportsWebGPUDeferredLighting } from "../material";
import {
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_TARGET_COUNT,
	WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT,
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
} from "../constants";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import type { WebGPUFrameTargetRequirements } from "./WebGPUFrameTargetManager";

const SHARED_HIZ_POSTPROCESS_PASS_IDS = new Set([
	SCREEN_SPACE_REFLECTIONS_PASS_ID,
	SCREEN_SPACE_REFRACTIONS_PASS_ID,
	"volumetric",
]);

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
	readonly particleOpaquePackets?: readonly DrawPacket[];
	readonly particleTransparentPackets?: readonly DrawPacket[];
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
		context: FrameContext,
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
			this._frameHasDeferredLightingWork(context, options.particleOpaquePackets ?? []);
		if (options.enableDeferredLighting && !options.forceDeferredFallback && !deferredSupported) {
			diagnostics.push({
				code: !mrtSupported ? "webgpu-deferred-disabled-mrt" : "webgpu-deferred-disabled-runtime",
				message: !mrtSupported
					? "WebGPU deferred lighting requires MRT scene targets; using the non-deferred fallback path."
					: "WebGPU deferred lighting requirements are unavailable; using the legacy MRT forward path.",
			});
		}
		const deferredActive = wantsDeferred && deferredSupported;
		const postProcessPasses = resolvePostProcessExecutionOrder(context.postProcess, {
			backend: "webgpu",
			frameContext: context,
		});
		const needsPostProcessTargets = postProcessPasses.length > 0;
		const needsPostProcessGBuffer = this._postProcessNeedsGBuffer(context, postProcessPasses);
		const needsPlanarReflection =
			context.features.enableReflection && context.scene.reflectivePackets.length > 0;
		const needsPlanarReflectionMask =
			needsPlanarReflection ||
			postProcessPasses.some((resolved) => resolved.id === SCREEN_SPACE_REFLECTIONS_PASS_ID);
		const needsOITWork = this._frameHasOITWork(context);
		const oitActive =
			mrtSupported &&
			options.sampleCount === 1 &&
			options.supportsInFrameTextureCopy &&
			context.features.enableOIT === true &&
			needsOITWork;
		if (context.features.enableOIT === true && needsOITWork && !oitActive) {
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
		const needsTransmissionTargets =
			postProcessPasses.some((resolved) => resolved.id === SCREEN_SPACE_REFRACTIONS_PASS_ID) &&
			this._frameHasTransmissionWork(context, options.particleTransparentPackets ?? []);
		const needsOcclusionTargets =
			context.features.enableOcclusionCulling === true &&
			(context.scene.occlusion?.eligibleCandidateCount ?? 0) > 0;
		const needsHiZTarget =
			needsOcclusionTargets ||
			postProcessPasses.some((resolved) => SHARED_HIZ_POSTPROCESS_PASS_IDS.has(resolved.id));
		const hasOffscreenWork =
			deferredActive ||
			needsPostProcessTargets ||
			needsPlanarReflection ||
			oitActive ||
			needsTransmissionTargets ||
			needsOcclusionTargets ||
			needsHiZTarget ||
			options.forceForwardMrt === true;
		const sceneTargetMode: WebGPUSceneTargetMode = !mrtSupported || !hasOffscreenWork
			? "single"
			: deferredActive
				? "gbuffer"
				: needsPostProcessGBuffer || needsOcclusionTargets || needsHiZTarget
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
							needsPostProcessTargets,
							needsOITTargets: oitActive,
							needsTransmissionTargets,
							needsPlanarReflectionMask,
							needsHiZTarget,
						},
			needsHiZBuild: needsHiZTarget,
			needsOcclusionTest: needsOcclusionTargets,
			enableEarlyZPrepass: options.enableEarlyZPrepass,
			diagnostics,
		};
	}

	private _postProcessNeedsGBuffer(
		context: FrameContext,
		passes: readonly ResolvedPostProcessPass[],
	): boolean {
		return passes.some((resolved) => {
			if (!resolved.pass.builtIn) return true;
			return (resolved.pass.getRequirements({
				frameContext: context,
				postProcess: context.postProcess,
				backend: "webgpu",
				options: resolved.options,
			}).gBuffer?.length ?? 0) > 0;
		});
	}

	private _frameHasDeferredLightingWork(
		context: FrameContext,
		particlePackets: readonly DrawPacket[],
	): boolean {
		return particlePackets.concat(context.scene.opaquePackets)
			.some((packet) => materialSupportsWebGPUDeferredLighting(packet.material));
	}

	private _frameHasOITWork(context: FrameContext): boolean {
		return context.scene.transparentPackets.length > 0 ||
			(context.scene.particleSystems?.length ?? 0) > 0;
	}

	private _frameHasTransmissionWork(
		context: FrameContext,
		particlePackets: readonly DrawPacket[],
	): boolean {
		return particlePackets.concat(context.scene.transparentPackets)
			.some((packet) => materialUsesTransmission(packet.material));
	}
}
