import type {
	PostProcessRenderGraphFrame,
} from "../../../postprocess/BackendPostProcessRuntime";
import type {
	RenderGraphDefinition,
	RenderGraphNode,
	RenderGraphResourceDescriptor,
} from "../../../rendergraph/types";

import type { WebGLFrameGraphNode, WebGLFrameGraphNodeKind } from "./types";

export interface WebGLPostProcessGraphComposition {
	readonly definition: RenderGraphDefinition<WebGLFrameGraphNode, WebGLFrameGraphNodeKind>;
	readonly inputs: Readonly<Record<string, string>>;
	readonly importResources: readonly RenderGraphResourceDescriptor[];
	readonly outputColor: string;
}

/** @internal Converts the shared post-process subgraph to WebGL frame payloads. */
export function createWebGLPostProcessGraphComposition(
	frame: PostProcessRenderGraphFrame,
): WebGLPostProcessGraphComposition {
	const definition: RenderGraphDefinition<WebGLFrameGraphNode, WebGLFrameGraphNodeKind> = {
		...frame.subgraph,
		nodes: Object.freeze(frame.subgraph.nodes.map((node) => toWebGLNode(node))),
	};
	const inputs = resolveInputs(frame);
	return Object.freeze({
		definition: Object.freeze(definition),
		inputs: Object.freeze(inputs),
		importResources: Object.freeze(resolveImportResources(frame, inputs)),
		outputColor: `postprocess:${frame.subgraph.outputColor}`,
	});
}

function toWebGLNode(
	node: RenderGraphNode<import("../../../postprocess/PostProcessSubgraphBuilder").PostProcessSubgraphNodePayload>,
): RenderGraphNode<WebGLFrameGraphNode, WebGLFrameGraphNodeKind> {
	return {
		...node,
		kind: "post-process-pass",
		payload: {
			id: node.id,
			stage: node.stage,
			kind: "post-process-pass",
			label: node.label,
			domain: "graphics",
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
	switch (port) {
		case "scene-color": return "frame:scene-color";
		case "gbuffer:depth":
		case "gbuffer:motion": return "frame:motion-depth";
		case "gbuffer:normal":
		case "gbuffer:roughness":
		case "gbuffer:metallic": return "frame:normal";
		case "gbuffer:albedo": return "frame:albedo";
		case "gbuffer:specular": return "frame:specular";
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
