import type { WebGPUFrameTargets } from "../WebGPUPostProcessContracts";
import type { WebGPUFrameMSAATargets } from "./WebGPUFrameTargetManager";
import type { IRenderTexture } from "../../types";
import type {
	RenderGraphPhysicalBinding,
	RenderGraphResourceDescriptor,
} from "../../../rendergraph/types";

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

export type WebGPUFrameGraphResourceId = string;

export interface WebGPUFrameGraphResourceCatalogSnapshot {
	readonly resources: readonly RenderGraphResourceDescriptor[];
	readonly bindings: readonly RenderGraphPhysicalBinding[];
}

export function collectWebGPUFrameGraphResourceCatalog(
	targets: WebGPUFrameTargets | null,
	msaaTargets: WebGPUFrameMSAATargets | null,
	width: number,
	height: number,
	msaaSampleCount: number,
	physicalResolver?: Map<string, IRenderTexture>,
	includeShadowResources = true,
): WebGPUFrameGraphResourceCatalogSnapshot {
	const textures = collectTextureBindings(targets, msaaTargets);
	physicalResolver?.clear();
	const resources: RenderGraphResourceDescriptor[] = [];
	const bindings: RenderGraphPhysicalBinding[] = [];
	for (const id of Object.values(WEBGPU_FRAME_GRAPH_RESOURCES)) {
		if (!includeShadowResources && isShadowResource(id)) continue;
		const texture = textures.get(id);
		if (texture) {
			resources.push({
				id,
				origin: "imported",
				kind: "texture",
				residency: "frame",
				initialContent: "unknown",
				format: texture.format ?? texture.requestedFormat,
				width: texture.width,
				height: texture.height,
				depthOrArrayLayers: 1,
				dimension: "2d",
				sampleCount: id.startsWith("msaa:") ? Math.max(1, msaaSampleCount) : 1,
				mipLevelCount: id === WEBGPU_FRAME_GRAPH_RESOURCES.frameHiZ
					? Math.floor(Math.log2(Math.max(1, texture.width, texture.height))) + 1
					: 1,
			});
			const physicalId = `webgpu:${id}`;
			bindings.push({ resourceId: id, physicalId, kind: "texture" });
			physicalResolver?.set(physicalId, texture);
			continue;
		}
		const canvas = id === WEBGPU_FRAME_GRAPH_RESOURCES.canvasColor ||
			id === WEBGPU_FRAME_GRAPH_RESOURCES.canvasDepth;
		resources.push(canvas ? {
			id,
			origin: "imported",
			kind: "texture",
			residency: "external",
			initialContent: "unknown",
			width: Math.max(1, width),
			height: Math.max(1, height),
			depthOrArrayLayers: 1,
			dimension: "2d",
			sampleCount: 1,
			mipLevelCount: 1,
		} : {
			id,
			origin: "imported",
			kind: "external",
			residency: "external",
			initialContent: "unknown",
		});
		bindings.push({
			resourceId: id,
			physicalId: `webgpu:${id}`,
			kind: canvas ? "texture" : "external",
		});
	}
	return Object.freeze({
		resources: Object.freeze(resources),
		bindings: Object.freeze(bindings),
	});
}

function isShadowResource(id: WebGPUFrameGraphResourceId): boolean {
	return id === WEBGPU_FRAME_GRAPH_RESOURCES.shadowAtlas ||
		id.startsWith("paged-shadow:");
}

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

function collectTextureBindings(
	targets: WebGPUFrameTargets | null,
	msaaTargets: WebGPUFrameMSAATargets | null,
): Map<WebGPUFrameGraphResourceId, IRenderTexture> {
	const r = WEBGPU_FRAME_GRAPH_RESOURCES;
	const textures = new Map<WebGPUFrameGraphResourceId, IRenderTexture>();
	if (targets) {
		const entries: readonly [WebGPUFrameGraphResourceId, IRenderTexture | null | undefined][] = [
			[r.frameColor, targets.sceneColorMain], [r.frameDepth, targets.depth],
			[r.postPing, targets.postPing], [r.postPong, targets.postPong],
			[r.frameHiZ, targets.hiZ], [r.gbufferAlbedoAlpha, targets.gAlbedoAlpha],
			[r.gbufferNormalRoughMetal, targets.gNormalRoughMetal],
			[r.gbufferEmissiveOcclusion, targets.gEmissiveOcclusion],
			[r.gbufferMotionDepth, targets.gMotionDepth], [r.gbufferSpecular, targets.gSpecular],
			[r.gbufferCoatSheen, targets.gCoatSheen],
			[r.gbufferSheenReflectance, targets.gSheenReflectance],
			[r.gbufferMaterialExt0, targets.gMaterialExt0], [r.gbufferMaterialExt1, targets.gMaterialExt1],
			[r.gbufferMaterialExt2, targets.gMaterialExt2], [r.gbufferMaterialExt3, targets.gMaterialExt3],
			[r.oitAccum, targets.oitAccum], [r.oitReveal, targets.oitReveal],
			[r.oitSceneColorCopy, targets.oitSceneColorCopy],
			[r.transmissionSceneColorCopy, targets.transmissionSceneColorCopy],
			[r.transmissionLighting, targets.transmissionLighting],
			[r.transmissionSurface0, targets.gTransmissionSurface0],
			[r.transmissionSurface1, targets.gTransmissionSurface1],
			[r.transmissionSurface2, targets.gTransmissionSurface2],
			[r.transmissionDepth, targets.transmissionDepth],
			[r.planarReflectionMask, targets.planarReflectionMask],
		];
		for (const [id, texture] of entries) if (texture) textures.set(id, texture);
	}
	if (msaaTargets) {
		const entries: readonly [WebGPUFrameGraphResourceId, IRenderTexture | null | undefined][] = [
			[r.msaaColor, msaaTargets.sceneColorMain], [r.msaaDepth, msaaTargets.depth],
			[r.msaaGBufferAlbedoAlpha, msaaTargets.gAlbedoAlpha],
			[r.msaaGBufferNormalRoughMetal, msaaTargets.gNormalRoughMetal],
			[r.msaaGBufferEmissiveOcclusion, msaaTargets.gEmissiveOcclusion],
			[r.msaaGBufferMotionDepth, msaaTargets.gMotionDepth],
			[r.msaaPlanarReflectionMask, msaaTargets.planarReflectionMask],
		];
		for (const [id, texture] of entries) if (texture) textures.set(id, texture);
	}
	return textures;
}
