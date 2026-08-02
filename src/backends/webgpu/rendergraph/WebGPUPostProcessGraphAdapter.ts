import type {
	PostProcessRenderGraphFrame,
} from "../../../postprocess/BackendPostProcessRuntime";
import type {
	RenderGraphDefinition,
	RenderGraphNode,
	RenderGraphResourceDescriptor,
	RenderGraphResourceId,
} from "../../../rendergraph/types";
import { renderGraphResourceId } from "../../../rendergraph/types";

import { WEBGPU_FRAME_GRAPH_RESOURCES as r } from "./WebGPUFrameGraphResourceCatalog";
import type { WebGPUFrameGraphNode, WebGPUFrameGraphNodeKind } from "./types";
import { getWebGPUPostProcessSharedResourceDescriptor } from "./WebGPUPostProcessSharedResourceCatalog";

export interface WebGPUPostProcessGraphComposition {
	readonly definition: RenderGraphDefinition<WebGPUFrameGraphNode, WebGPUFrameGraphNodeKind>;
	readonly inputs: Readonly<Record<string, RenderGraphResourceId>>;
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

function resolveInputs(
	frame: PostProcessRenderGraphFrame,
): Record<string, RenderGraphResourceId> {
	const inputs: Record<string, RenderGraphResourceId> = {};
	for (const port of frame.subgraph.imports ?? []) {
		const resource = resolveInputResource(port.name);
		if (resource) inputs[port.name] = resource;
	}
	return inputs;
}

function resolveInputResource(port: string): RenderGraphResourceId | null {
	const sharedResource = getWebGPUPostProcessSharedResourceDescriptor(port);
	if (sharedResource) return renderGraphResourceId(sharedResource.graphResourceId);
	switch (port) {
		case "scene-color": return renderGraphResourceId(r.frameColor);
		case "gbuffer:depth":
		case "gbuffer:motion": return renderGraphResourceId(r.gbufferMotionDepth);
		case "gbuffer:normal":
		case "gbuffer:roughness":
		case "gbuffer:metallic": return renderGraphResourceId(r.gbufferNormalRoughMetal);
		case "gbuffer:albedo": return renderGraphResourceId(r.gbufferAlbedoAlpha);
		case "gbuffer:specular": return renderGraphResourceId(r.gbufferSpecular);
		case "gbuffer:transmission": return renderGraphResourceId(r.transmissionSurface0);
		case "gbuffer:emissive":
		case "gbuffer:occlusion": return renderGraphResourceId(r.gbufferEmissiveOcclusion);
		default: return port.startsWith("history:")
			? renderGraphResourceId(`postprocess-import:${port}`)
			: null;
	}
}

function resolveImportResources(
	frame: PostProcessRenderGraphFrame,
	inputs: Readonly<Record<string, RenderGraphResourceId>>,
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
