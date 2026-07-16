import type { WebGPUFrameTargets } from "../WebGPUPostProcessContracts";
import type { WebGPUFrameMSAATargets } from "./WebGPUFrameTargetManager";

export const WEBGPU_FRAME_GRAPH_RESOURCES = {
	canvasColor: "canvas:scene-color-main",
	canvasDepth: "canvas:depth",
	frameColor: "frame:scene-color-main",
	frameDepth: "frame:depth",
	frameHiZ: "frame:hiz",
	postPing: "post:ping",
	postPong: "post:pong",
	gbufferAlbedoAlpha: "gbuffer:albedo-alpha",
	gbufferNormalRoughMetal: "gbuffer:normal-rough-metal",
	gbufferEmissiveOcclusion: "gbuffer:emissive-occlusion",
	gbufferMotionDepth: "gbuffer:motion-depth",
	gbufferSpecular: "gbuffer:specular",
	gbufferCoatSheen: "gbuffer:coat-sheen",
	gbufferSheenReflectance: "gbuffer:sheen-reflectance",
	gbufferMaterialExt0: "gbuffer:material-ext0",
	gbufferMaterialExt1: "gbuffer:material-ext1",
	gbufferMaterialExt2: "gbuffer:material-ext2",
	gbufferMaterialExt3: "gbuffer:material-ext3",
	msaaColor: "msaa:scene-color-main",
	msaaDepth: "msaa:depth",
	msaaGBufferAlbedoAlpha: "msaa:gbuffer:albedo-alpha",
	msaaGBufferNormalRoughMetal: "msaa:gbuffer:normal-rough-metal",
	msaaGBufferEmissiveOcclusion: "msaa:gbuffer:emissive-occlusion",
	msaaGBufferMotionDepth: "msaa:gbuffer:motion-depth",
	msaaPlanarReflectionMask: "msaa:planar-reflection:mask",
	oitAccum: "oit:accum",
	oitReveal: "oit:reveal",
	oitSceneColorCopy: "oit:scene-color-copy",
	transmissionSceneColorCopy: "transmission:scene-color-copy",
	transmissionLighting: "transmission:lighting",
	transmissionSurface0: "transmission:surface0",
	transmissionSurface1: "transmission:surface1",
	transmissionSurface2: "transmission:surface2",
	transmissionDepth: "transmission:depth",
	planarReflectionCapture: "planar-reflection:capture",
	planarReflectionMask: "planar-reflection:mask",
	shadowAtlas: "shadow-atlas",
	pagedShadowFeedbackFlags: "paged-shadow:feedback-flags",
	pagedShadowPageRequestFlags: "paged-shadow:page-request-flags",
	pagedShadowPageRequests: "paged-shadow:page-requests",
	pagedShadowCounters: "paged-shadow:counters",
	pagedShadowPageTable: "paged-shadow:page-table",
	pagedShadowPageMetadata: "paged-shadow:page-metadata",
	pagedShadowResidencyState: "paged-shadow:residency-state",
	pagedShadowFreeList: "paged-shadow:free-list",
	pagedShadowDirtyPhysicalPages: "paged-shadow:dirty-physical-pages",
	pagedShadowPageTableTexture: "paged-shadow:page-table-texture",
	pagedShadowDrawInstances: "paged-shadow:draw-instances",
	pagedShadowDrawIndirectArgs: "paged-shadow:draw-indirect-args",
	pagedShadowClearDrawIndirectArgs: "paged-shadow:clear-draw-indirect-args",
	pagedShadowPhysicalDepth: "paged-shadow:physical-depth",
	pagedShadowNextFeedbackFlags: "paged-shadow:next-feedback-flags",
	occlusionResults: "occlusion:results",
} as const;

export type WebGPUFrameGraphResourceId =
	(typeof WEBGPU_FRAME_GRAPH_RESOURCES)[keyof typeof WEBGPU_FRAME_GRAPH_RESOURCES];

export function collectActiveWebGPUFrameGraphResources(
	targets: WebGPUFrameTargets | null,
	msaaTargets: WebGPUFrameMSAATargets | null,
): WebGPUFrameGraphResourceId[] {
	const r = WEBGPU_FRAME_GRAPH_RESOURCES;
	const active = new Set<WebGPUFrameGraphResourceId>([r.canvasColor, r.canvasDepth]);
	if (targets) {
		active.add(r.frameColor);
		active.add(r.frameDepth);
		if (targets.postPing) active.add(r.postPing);
		if (targets.postPong) active.add(r.postPong);
		if (targets.hiZ) active.add(r.frameHiZ);
		if (targets.gAlbedoAlpha) active.add(r.gbufferAlbedoAlpha);
		if (targets.gNormalRoughMetal) active.add(r.gbufferNormalRoughMetal);
		if (targets.gEmissiveOcclusion) active.add(r.gbufferEmissiveOcclusion);
		if (targets.gMotionDepth) active.add(r.gbufferMotionDepth);
		if (targets.gSpecular) active.add(r.gbufferSpecular);
		if (targets.gCoatSheen) active.add(r.gbufferCoatSheen);
		if (targets.gSheenReflectance) active.add(r.gbufferSheenReflectance);
		if (targets.gMaterialExt0) active.add(r.gbufferMaterialExt0);
		if (targets.gMaterialExt1) active.add(r.gbufferMaterialExt1);
		if (targets.gMaterialExt2) active.add(r.gbufferMaterialExt2);
		if (targets.gMaterialExt3) active.add(r.gbufferMaterialExt3);
		if (targets.oitAccum) active.add(r.oitAccum);
		if (targets.oitReveal) active.add(r.oitReveal);
		if (targets.oitSceneColorCopy) active.add(r.oitSceneColorCopy);
		if (targets.transmissionSceneColorCopy) active.add(r.transmissionSceneColorCopy);
		if (targets.transmissionLighting) active.add(r.transmissionLighting);
		if (targets.gTransmissionSurface0) active.add(r.transmissionSurface0);
		if (targets.gTransmissionSurface1) active.add(r.transmissionSurface1);
		if (targets.gTransmissionSurface2) active.add(r.transmissionSurface2);
		if (targets.transmissionDepth) active.add(r.transmissionDepth);
		if (targets.planarReflectionMask) active.add(r.planarReflectionMask);
	}
	if (msaaTargets) {
		active.add(r.msaaColor);
		active.add(r.msaaDepth);
		if (msaaTargets.gAlbedoAlpha) active.add(r.msaaGBufferAlbedoAlpha);
		if (msaaTargets.gNormalRoughMetal) active.add(r.msaaGBufferNormalRoughMetal);
		if (msaaTargets.gEmissiveOcclusion) active.add(r.msaaGBufferEmissiveOcclusion);
		if (msaaTargets.gMotionDepth) active.add(r.msaaGBufferMotionDepth);
		if (msaaTargets.planarReflectionMask) active.add(r.msaaPlanarReflectionMask);
	}
	return Array.from(active);
}
