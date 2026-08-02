import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import type { PreparedFramePacketSet } from "../../../pipeline/FramePacketContributorRegistry";
import type { PlannedPostProcessPass } from "../../../postprocess";
import { materialUsesTransmission } from "../../../materials/transparency";
import { ParticleBlendMode } from "../../../particles";
import { materialSupportsWebGPUDeferredLighting } from "../material";
import {
	getWebGPUPostProcessSharedResourceDescriptor,
	type WebGPUPostProcessAllocationGroup,
} from "./WebGPUPostProcessSharedResourceCatalog";

export interface WebGPUFrameFeatureAnalysis {
	readonly framePackets: PreparedFramePacketSet;
	readonly postProcessPasses: readonly PlannedPostProcessPass[];
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
	readonly framePackets: PreparedFramePacketSet;
	readonly postProcessPasses: readonly PlannedPostProcessPass[];
}

/** Scans desired WebGPU frame work without applying capability policy. */
export class WebGPUFrameFeatureAnalyzer {
	public analyze(
		context: FrameContext,
		options: WebGPUFrameFeatureAnalysisOptions,
	): WebGPUFrameFeatureAnalysis {
		const framePackets = options.framePackets;
		const postProcessPasses = options.postProcessPasses;
		// Declarations are retained before target allocation, so discovery can
		// remain data-driven without describing an implementation twice.
		const needsPostProcessGBuffer = postProcessPasses.length > 0;
		const needsPlanarReflection =
			context.features.enableReflection && context.scene.reflectivePackets.length > 0;
		const needsOcclusionTargets =
			context.features.enableOcclusionCulling === true &&
			(context.scene.occlusion?.eligibleCandidateCount ?? 0) > 0;
		const sharedAllocationGroups = this._collectSharedAllocationGroups(postProcessPasses);
		const needsHiZTarget =
			needsOcclusionTargets ||
			sharedAllocationGroups.has("hiz");
		const transparency = this._analyzeTransparency(
			context,
			framePackets.transparent,
		);
		return {
			framePackets,
			postProcessPasses,
			hasDeferredLightingWork: framePackets.opaque
				.some((packet) => materialSupportsWebGPUDeferredLighting(packet.material)),
			oitRequested: context.features.enableOIT === true,
			hasOITWork: transparency.hasOITContributors,
			transparency,
			needsPostProcessTargets: postProcessPasses.length > 0,
			needsPostProcessGBuffer,
			needsPlanarReflection,
			needsPlanarReflectionMask:
				needsPlanarReflection ||
				sharedAllocationGroups.has("planar-reflection-mask"),
			needsTransmissionTargets:
				sharedAllocationGroups.has("transmission") &&
				framePackets.transparent
					.some((packet) => materialUsesTransmission(packet.material)),
			needsOcclusionTargets,
			needsHiZTarget,
		};
	}

	private _collectSharedAllocationGroups(
		passes: readonly PlannedPostProcessPass[],
	): ReadonlySet<WebGPUPostProcessAllocationGroup> {
		const groups = new Set<WebGPUPostProcessAllocationGroup>();
		for (const pass of passes) {
			for (const resource of pass.declaration.shared ?? []) {
				const descriptor = getWebGPUPostProcessSharedResourceDescriptor(resource.id);
				if (
					descriptor &&
					(resource.optional !== true || descriptor.allocateWhenOptional)
				) {
					groups.add(descriptor.allocationGroup);
				}
			}
		}
		return groups;
	}

	private _analyzeTransparency(
		context: FrameContext,
		transparentPackets: readonly DrawPacket[],
	): WebGPUTransparencyAnalysis {
		const oitPackets: DrawPacket[] = [];
		const transmissionPackets: DrawPacket[] = [];
		for (const packet of transparentPackets) {
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
