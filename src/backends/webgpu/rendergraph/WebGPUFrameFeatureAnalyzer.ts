import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import type { PreparedFramePacketSet } from "../../../pipeline/FramePacketContributorRegistry";
import type { PlannedPostProcessPass } from "../../../postprocess";
import { materialUsesTransmission } from "../../../materials/transparency";
import { ParticleBlendMode } from "../../../particles";
import {
	materialRequiresExtendedWebGPUGBuffer,
	materialSupportsWebGPUDeferredLighting,
} from "../material";
import type { WebGPUDeferredGBufferLayout } from "../constants";
import {
	getWebGPUPostProcessSharedResourceDescriptor,
	type WebGPUPostProcessAllocationGroup,
} from "./WebGPUPostProcessSharedResourceCatalog";

export interface WebGPUFrameFeatureAnalysis {
	readonly framePackets: PreparedFramePacketSet;
	readonly postProcessPasses: readonly PlannedPostProcessPass[];
	readonly hasDeferredLightingWork: boolean;
	readonly deferredGBufferLayout: WebGPUDeferredGBufferLayout;
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

export interface WebGPUDeferredFeatureAnalysis {
	readonly hasDeferredLightingWork: boolean;
	readonly deferredGBufferLayout: WebGPUDeferredGBufferLayout;
}

export interface WebGPUPostProcessFeatureAnalysis {
	readonly postProcessPasses: readonly PlannedPostProcessPass[];
	readonly needsPostProcessTargets: boolean;
	readonly needsPostProcessGBuffer: boolean;
	readonly needsPlanarReflectionMask: boolean;
	readonly needsTransmissionTargets: boolean;
	readonly needsHiZTarget: boolean;
}

export interface WebGPUReflectionFeatureAnalysis {
	readonly needsPlanarReflection: boolean;
}

export interface WebGPUVisibilityFeatureAnalysis {
	readonly needsOcclusionTargets: boolean;
}

/** Scans desired WebGPU frame work without applying capability policy. */
export class WebGPUFrameFeatureAnalyzer {
	public analyze(
		context: FrameContext,
		options: WebGPUFrameFeatureAnalysisOptions,
	): WebGPUFrameFeatureAnalysis {
		const framePackets = options.framePackets;
		const postProcess = analyzeWebGPUPostProcessFeatures(options.postProcessPasses);
		const reflection = analyzeWebGPUReflectionFeatures(context);
		const visibility = analyzeWebGPUVisibilityFeatures(context);
		const transparency = analyzeWebGPUTransparency(context, framePackets.transparent);
		const deferred = analyzeWebGPUDeferredFeatures(context, framePackets);
		return {
			framePackets,
			postProcessPasses: postProcess.postProcessPasses,
			hasDeferredLightingWork: deferred.hasDeferredLightingWork,
			deferredGBufferLayout: deferred.deferredGBufferLayout,
			oitRequested: context.features.enableOIT === true,
			hasOITWork: transparency.hasOITContributors,
			transparency,
			needsPostProcessTargets: postProcess.needsPostProcessTargets,
			needsPostProcessGBuffer: postProcess.needsPostProcessGBuffer,
			needsPlanarReflection: reflection.needsPlanarReflection,
			needsPlanarReflectionMask:
				reflection.needsPlanarReflection || postProcess.needsPlanarReflectionMask,
			needsTransmissionTargets: postProcess.needsTransmissionTargets &&
				framePackets.transparent.some((packet) => materialUsesTransmission(packet.material)),
			needsOcclusionTargets: visibility.needsOcclusionTargets,
			needsHiZTarget:
				visibility.needsOcclusionTargets || postProcess.needsHiZTarget,
		};
	}
}

export function analyzeWebGPUDeferredFeatures(
	context: FrameContext,
	framePackets: PreparedFramePacketSet,
): WebGPUDeferredFeatureAnalysis {
	return {
		hasDeferredLightingWork: framePackets.opaque.some((packet) =>
			materialSupportsWebGPUDeferredLighting(packet.material)),
		deferredGBufferLayout:
			(context.scene.decalPackets?.length ?? 0) > 0 ||
			framePackets.opaque.some((packet) =>
				materialRequiresExtendedWebGPUGBuffer(packet.material))
				? "extended"
				: "base",
	};
}

export function analyzeWebGPUPostProcessFeatures(
	passes: readonly PlannedPostProcessPass[],
): WebGPUPostProcessFeatureAnalysis {
	const groups = collectSharedAllocationGroups(passes);
	return {
		postProcessPasses: passes,
		needsPostProcessTargets: passes.length > 0,
		needsPostProcessGBuffer: passes.length > 0,
		needsPlanarReflectionMask: groups.has("planar-reflection-mask"),
		needsTransmissionTargets: groups.has("transmission"),
		needsHiZTarget: groups.has("hiz"),
	};
}

export function analyzeWebGPUReflectionFeatures(
	context: FrameContext,
): WebGPUReflectionFeatureAnalysis {
	return {
		needsPlanarReflection:
			context.features.enableReflection && context.scene.reflectivePackets.length > 0,
	};
}

export function analyzeWebGPUVisibilityFeatures(
	context: FrameContext,
): WebGPUVisibilityFeatureAnalysis {
	return {
		needsOcclusionTargets:
			context.features.enableOcclusionCulling === true &&
			(context.scene.occlusion?.eligibleCandidateCount ?? 0) > 0,
	};
}

export function analyzeWebGPUTransparency(
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
			// Lightweight compatibility contexts have no template classification.
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

function collectSharedAllocationGroups(
	passes: readonly PlannedPostProcessPass[],
): ReadonlySet<WebGPUPostProcessAllocationGroup> {
	const groups = new Set<WebGPUPostProcessAllocationGroup>();
	for (const pass of passes) {
		for (const resource of pass.declaration.shared ?? []) {
			const descriptor = getWebGPUPostProcessSharedResourceDescriptor(resource.id);
			if (descriptor && (resource.optional !== true || descriptor.allocateWhenOptional)) {
				groups.add(descriptor.allocationGroup);
			}
		}
	}
	return groups;
}
