import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import {
	SCREEN_SPACE_REFLECTIONS_PASS_ID,
	SCREEN_SPACE_REFRACTIONS_PASS_ID,
	resolvePostProcessExecutionOrder,
	type ResolvedPostProcessPass,
} from "../../../postprocess";
import { materialUsesTransmission } from "../../../materials/transparency";
import { materialSupportsWebGPUDeferredLighting } from "../material";

const SHARED_HIZ_POSTPROCESS_PASS_IDS = new Set([
	SCREEN_SPACE_REFLECTIONS_PASS_ID,
	SCREEN_SPACE_REFRACTIONS_PASS_ID,
	"volumetric",
]);

export interface WebGPUFrameFeatureAnalysis {
	readonly particleOpaquePackets: readonly DrawPacket[];
	readonly particleTransparentPackets: readonly DrawPacket[];
	readonly postProcessPasses: readonly ResolvedPostProcessPass[];
	readonly hasDeferredLightingWork: boolean;
	readonly oitRequested: boolean;
	readonly hasOITWork: boolean;
	readonly needsPostProcessTargets: boolean;
	readonly needsPostProcessGBuffer: boolean;
	readonly needsPlanarReflection: boolean;
	readonly needsPlanarReflectionMask: boolean;
	readonly needsTransmissionTargets: boolean;
	readonly needsOcclusionTargets: boolean;
	readonly needsHiZTarget: boolean;
}

export interface WebGPUFrameFeatureAnalysisOptions {
	readonly particleOpaquePackets?: readonly DrawPacket[];
	readonly particleTransparentPackets?: readonly DrawPacket[];
}

/** Scans desired WebGPU frame work without applying capability policy. */
export class WebGPUFrameFeatureAnalyzer {
	public analyze(
		context: FrameContext,
		options: WebGPUFrameFeatureAnalysisOptions = {},
	): WebGPUFrameFeatureAnalysis {
		const particleOpaquePackets = options.particleOpaquePackets ?? [];
		const particleTransparentPackets = options.particleTransparentPackets ?? [];
		const postProcessPasses = resolvePostProcessExecutionOrder(context.postProcess, {
			backend: "webgpu",
			frameContext: context,
		});
		const needsPostProcessGBuffer = postProcessPasses.some((resolved) => {
			if (!resolved.pass.builtIn) return true;
			return (resolved.pass.getRequirements({
				frameContext: context,
				postProcess: context.postProcess,
				backend: "webgpu",
				options: resolved.options,
			}).gBuffer?.length ?? 0) > 0;
		});
		const needsPlanarReflection =
			context.features.enableReflection && context.scene.reflectivePackets.length > 0;
		const needsOcclusionTargets =
			context.features.enableOcclusionCulling === true &&
			(context.scene.occlusion?.eligibleCandidateCount ?? 0) > 0;
		const needsHiZTarget =
			needsOcclusionTargets ||
			postProcessPasses.some((resolved) =>
				SHARED_HIZ_POSTPROCESS_PASS_IDS.has(resolved.id),
			);
		return {
			particleOpaquePackets,
			particleTransparentPackets,
			postProcessPasses,
			hasDeferredLightingWork: particleOpaquePackets
				.concat(context.scene.opaquePackets)
				.some((packet) => materialSupportsWebGPUDeferredLighting(packet.material)),
			oitRequested: context.features.enableOIT === true,
			hasOITWork:
				context.scene.transparentPackets.length > 0 ||
				(context.scene.particleSystems?.length ?? 0) > 0,
			needsPostProcessTargets: postProcessPasses.length > 0,
			needsPostProcessGBuffer,
			needsPlanarReflection,
			needsPlanarReflectionMask:
				needsPlanarReflection ||
				postProcessPasses.some((resolved) =>
					resolved.id === SCREEN_SPACE_REFLECTIONS_PASS_ID,
				),
			needsTransmissionTargets:
				postProcessPasses.some((resolved) =>
					resolved.id === SCREEN_SPACE_REFRACTIONS_PASS_ID,
				) &&
				particleTransparentPackets
					.concat(context.scene.transparentPackets)
					.some((packet) => materialUsesTransmission(packet.material)),
			needsOcclusionTargets,
			needsHiZTarget,
		};
	}
}
