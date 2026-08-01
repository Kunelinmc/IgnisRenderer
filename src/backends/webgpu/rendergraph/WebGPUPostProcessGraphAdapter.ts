import type {
	PostProcessRenderGraphFrame,
} from "../../../postprocess/BackendPostProcessRuntime";
import type {
	RenderGraphDefinition,
	RenderGraphNode,
	RenderGraphResourceDescriptor,
} from "../../../rendergraph/types";

import { WEBGPU_FRAME_GRAPH_RESOURCES as r } from "./WebGPUFrameGraphResourceCatalog";
import type { WebGPUFrameGraphNode, WebGPUFrameGraphNodeKind } from "./types";
import { getWebGPUPostProcessSharedResourceDescriptor } from "./WebGPUPostProcessSharedResourceCatalog";

export interface WebGPUPostProcessGraphComposition {
	readonly definition: RenderGraphDefinition<WebGPUFrameGraphNode, WebGPUFrameGraphNodeKind>;
	readonly inputs: Readonly<Record<string, string>>;
	readonly importResources: readonly RenderGraphResourceDescriptor[];
	readonly outputColor: string;
}

/** @internal Converts the shared post-process subgraph to WebGPU frame payloads. */
export function createWebGPUPostProcessGraphComposition(
	frame: PostProcessRenderGraphFrame,
): WebGPUPostProcessGraphComposition {
	const definition: RenderGraphDefinition<WebGPUFrameGraphNode, WebGPUFrameGraphNodeKind> = {
		...frame.subgraph,
		nodes: Object.freeze(frame.subgraph.nodes.map((node) => toWebGPUNode(node))),
	};
	const inputs = resolveInputs(frame);
	return Object.freeze({
		definition: Object.freeze(definition),
		inputs: Object.freeze(inputs),
		importResources: Object.freeze(resolveImportResources(frame, inputs)),
		outputColor: `postprocess:${frame.subgraph.outputColor}`,
	});
}

function toWebGPUNode(
	node: RenderGraphNode<import("../../../postprocess/PostProcessSubgraphBuilder").PostProcessSubgraphNodePayload>,
): RenderGraphNode<WebGPUFrameGraphNode, WebGPUFrameGraphNodeKind> {
	return {
		...node,
		kind: "post-process-pass",
		internalAccesses: "ordered",
		payload: {
			id: node.id,
			stage: node.stage,
			kind: "post-process-pass",
			label: node.label,
			domain: "compute",
			retention: "always",
			postProcess: node.payload,
		},
	};
}

function resolveInputs(frame: PostProcessRenderGraphFrame): Record<string, string> {
	const inputs: Record<string, string> = {};
	for (const port of frame.subgraph.imports ?? []) {
		const resource = resolveInputResource(port.name);
		if (resource) inputs[port.name] = resource;
	}
	return inputs;
}

function resolveInputResource(port: string): string | null {
	const sharedResource = getWebGPUPostProcessSharedResourceDescriptor(port);
	if (sharedResource) return sharedResource.graphResourceId;
	switch (port) {
		case "scene-color": return r.frameColor;
		case "gbuffer:depth":
		case "gbuffer:motion": return r.gbufferMotionDepth;
		case "gbuffer:normal":
		case "gbuffer:roughness":
		case "gbuffer:metallic": return r.gbufferNormalRoughMetal;
		case "gbuffer:albedo": return r.gbufferAlbedoAlpha;
		case "gbuffer:specular": return r.gbufferSpecular;
		case "gbuffer:transmission": return r.transmissionSurface0;
		case "gbuffer:emissive":
		case "gbuffer:occlusion": return r.gbufferEmissiveOcclusion;
		default: return port.startsWith("history:") ? `postprocess-import:${port}` : null;
	}
}

function resolveImportResources(
	frame: PostProcessRenderGraphFrame,
	inputs: Readonly<Record<string, string>>,
): RenderGraphResourceDescriptor[] {
	const resources: RenderGraphResourceDescriptor[] = [];
	for (const port of frame.subgraph.imports ?? []) {
		const id = inputs[port.name];
		if (!id?.startsWith("postprocess-import:")) continue;
		const descriptor = frame.subgraph.resources.find(
			(resource) => resource.id === port.resource,
		);
		if (descriptor) resources.push({ ...descriptor, id });
	}
	return resources;
}
