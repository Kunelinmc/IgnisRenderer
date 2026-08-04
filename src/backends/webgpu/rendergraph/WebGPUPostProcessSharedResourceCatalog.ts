import {
	POST_PROCESS_SHARED_RESOURCE_IDS,
	type BuiltInPostProcessSharedResourceId,
} from "../../../postprocess/executionDeclarations";
import type { IRenderTexture } from "../../types";
import type { WebGPUFrameTargets } from "../WebGPUFrameTargetContracts";

import {
	WEBGPU_FRAME_GRAPH_RESOURCES,
	type WebGPUFrameGraphResourceId,
} from "./WebGPUFrameGraphResourceCatalog";

export type WebGPUPostProcessAllocationGroup =
	| "hiz"
	| "planar-reflection-mask"
	| "transmission";

export interface WebGPUPostProcessSharedResourceResolveContext {
	readonly targets: WebGPUFrameTargets | null;
	readonly isHiZReady: boolean;
}

export interface WebGPUPostProcessSharedResourceDescriptor {
	readonly id: BuiltInPostProcessSharedResourceId;
	readonly graphResourceId: WebGPUFrameGraphResourceId;
	readonly allocationGroup: WebGPUPostProcessAllocationGroup;
	readonly allocateWhenOptional: boolean;
	isAllocated(targets: WebGPUFrameTargets | null): boolean;
	resolveTexture(
		context: WebGPUPostProcessSharedResourceResolveContext,
	): IRenderTexture | null;
}

const IDS = POST_PROCESS_SHARED_RESOURCE_IDS;
const GRAPH = WEBGPU_FRAME_GRAPH_RESOURCES;

const DESCRIPTORS = Object.freeze([
	createDescriptor(
		IDS.frameHiZ,
		GRAPH.frameHiZ,
		"hiz",
		(targets) => targets?.hiZ ?? null,
		(context) => context.isHiZReady,
	),
	createDescriptor(
		IDS.planarReflectionMask,
		GRAPH.planarReflectionMask,
		"planar-reflection-mask",
		(targets) => targets?.planarReflectionMask ?? null,
		undefined,
		true,
	),
	createDescriptor(
		IDS.transmissionSceneColor,
		GRAPH.transmissionSceneColorCopy,
		"transmission",
		(targets) => targets?.transmissionSceneColorCopy ?? null,
	),
	createDescriptor(
		IDS.transmissionLighting,
		GRAPH.transmissionLighting,
		"transmission",
		(targets) => targets?.transmissionLighting ?? null,
	),
	createDescriptor(
		IDS.transmissionSurface1,
		GRAPH.transmissionSurface1,
		"transmission",
		(targets) => targets?.gTransmissionSurface1 ?? null,
	),
	createDescriptor(
		IDS.transmissionSurface2,
		GRAPH.transmissionSurface2,
		"transmission",
		(targets) => targets?.gTransmissionSurface2 ?? null,
	),
] as const);

const DESCRIPTOR_BY_ID = new Map<string, WebGPUPostProcessSharedResourceDescriptor>(
	DESCRIPTORS.map((descriptor) => [descriptor.id, descriptor]),
);

/** @internal Returns one built-in WebGPU shared-resource descriptor. */
export function getWebGPUPostProcessSharedResourceDescriptor(
	id: string,
): WebGPUPostProcessSharedResourceDescriptor | null {
	return DESCRIPTOR_BY_ID.get(id) ?? null;
}

/** @internal Lists the immutable built-in WebGPU shared-resource catalog. */
export function listWebGPUPostProcessSharedResourceDescriptors(): readonly WebGPUPostProcessSharedResourceDescriptor[] {
	return DESCRIPTORS;
}

function createDescriptor(
	id: BuiltInPostProcessSharedResourceId,
	graphResourceId: WebGPUFrameGraphResourceId,
	allocationGroup: WebGPUPostProcessAllocationGroup,
	getTexture: (targets: WebGPUFrameTargets | null) => IRenderTexture | null,
	isReady: (context: WebGPUPostProcessSharedResourceResolveContext) => boolean = () => true,
	allocateWhenOptional = false,
): WebGPUPostProcessSharedResourceDescriptor {
	return Object.freeze({
		id,
		graphResourceId,
		allocationGroup,
		allocateWhenOptional,
		isAllocated: (targets) => getTexture(targets) !== null,
		resolveTexture: (context) =>
			isReady(context) ? getTexture(context.targets) : null,
	});
}
