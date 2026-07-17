import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import {
	SCREEN_SPACE_REFLECTIONS_PASS_ID,
	SCREEN_SPACE_REFRACTIONS_PASS_ID,
	VOLUMETRIC_LIGHTING_PASS_ID,
	resolvePostProcessExecutionOrder,
	type ResolvedPostProcessPass,
} from "../../../postprocess";
import { materialUsesTransmission } from "../../../materials/transparency";
import { ParticleBlendMode } from "../../../particles";
import { materialSupportsWebGPUDeferredLighting } from "../material";

const SHARED_HIZ_POSTPROCESS_PASS_IDS = new Set([
	SCREEN_SPACE_REFLECTIONS_PASS_ID,
	SCREEN_SPACE_REFRACTIONS_PASS_ID,
	VOLUMETRIC_LIGHTING_PASS_ID,
]);

export interface WebGPUFrameFeatureAnalysis {
	readonly particleOpaquePackets: readonly DrawPacket[];
	readonly particleTransparentPackets: readonly DrawPacket[];
	readonly postProcessPasses: readonly ResolvedPostProcessPass[];
	readonly hasDeferredLightingWork: boolean;
	readonly oitRequested: boolean;
	readonly hasOITWork: boolean;
	readonly transparency: WebGPUTransparencyAnalysis;
	readonly needsPostProcessTargets: boolean;
	readonly needsPostProcessGBuffer: boolean;
	readonly needsPlanarReflection: boolean;
	readonly needsPlanarReflectionMask: boolean;
	readonly needsTransmissionTargets: boolean;
	readonly needsOcclusionTargets: boolean;
	readonly needsHiZTarget: boolean;
}

/**
 * Prepared transparency work for one WebGPU frame.
 *
 * @internal Owned by the WebGPU frame feature analyzer. Transparency runtimes
 * must consume this result instead of re-classifying scene packets.
 */
export interface WebGPUTransparencyAnalysis {
	readonly oitPackets: readonly DrawPacket[];
	readonly transmissionPackets: readonly DrawPacket[];
	readonly legacyTransparentPackets: readonly DrawPacket[];
	readonly hasAlphaBillboardParticles: boolean;
	readonly hasAdditiveBillboardParticles: boolean;
	readonly hasOITContributors: boolean;
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
		const transparency = this._analyzeTransparency(
			context,
			particleTransparentPackets,
		);
		return {
			particleOpaquePackets,
			particleTransparentPackets,
			postProcessPasses,
			hasDeferredLightingWork: particleOpaquePackets
				.concat(context.scene.opaquePackets)
				.some((packet) => materialSupportsWebGPUDeferredLighting(packet.material)),
			oitRequested: context.features.enableOIT === true,
			hasOITWork: transparency.hasOITContributors,
			transparency,
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

	private _analyzeTransparency(
		context: FrameContext,
		particleTransparentPackets: readonly DrawPacket[],
	): WebGPUTransparencyAnalysis {
		const oitPackets: DrawPacket[] = [];
		const transmissionPackets: DrawPacket[] = [];
		for (const packet of context.scene.transparentPackets) {
			if (materialUsesTransmission(packet.material)) {
				transmissionPackets.push(packet);
			} else {
				oitPackets.push(packet);
			}
		}
		for (const packet of particleTransparentPackets) {
			if (materialUsesTransmission(packet.material)) {
				transmissionPackets.push(packet);
			} else {
				oitPackets.push(packet);
			}
		}
		let hasAlphaBillboardParticles = false;
		let hasAdditiveBillboardParticles = false;
		for (const system of context.scene.particleSystems ?? []) {
			if (system.visible === false) continue;
			const templates = system.templates;
			if (!templates) {
				// Test and compatibility frame contexts may provide a lightweight
				// particle-system descriptor. Preserve the historical particle pass
				// behavior until a concrete template classification is available.
				hasAlphaBillboardParticles = true;
				hasAdditiveBillboardParticles = true;
				continue;
			}
			for (const template of templates) {
				if (template.shape.kind !== "billboard") continue;
				if (template.shape.blendMode === ParticleBlendMode.Additive) {
					hasAdditiveBillboardParticles = true;
				} else {
					hasAlphaBillboardParticles = true;
				}
			}
		}
		return {
			oitPackets,
			transmissionPackets,
			legacyTransparentPackets: oitPackets,
			hasAlphaBillboardParticles,
			hasAdditiveBillboardParticles,
			hasOITContributors:
				oitPackets.length > 0 || hasAlphaBillboardParticles,
		};
	}
}
