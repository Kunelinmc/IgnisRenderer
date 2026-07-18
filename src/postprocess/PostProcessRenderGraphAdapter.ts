import type {
	RenderGraphNode,
	RenderGraphResourceDescriptor,
	RenderGraphResourceRef,
} from "../rendergraph/types";
import type { CompiledPostProcessGraph } from "./PostProcessGraphCompiler";
import type { PostProcessColorFlow, PostProcessGraphMetadata } from "./types";

export interface PostProcessRenderGraphNodePayload {
	readonly passId: string;
	readonly color: PostProcessColorFlow;
	readonly outputValidation: "strict" | "compatibility";
	readonly compatibilityOpaque: boolean;
}

export interface PostProcessRenderGraphSubgraph {
	readonly resources: readonly RenderGraphResourceDescriptor[];
	readonly nodes: readonly RenderGraphNode<PostProcessRenderGraphNodePayload>[];
	readonly plannedOutputColor: string;
}

/** @internal Builds a logical post-process subgraph without native resources. */
export class PostProcessRenderGraphAdapter {
	public build(graph: CompiledPostProcessGraph): PostProcessRenderGraphSubgraph {
		const resources: RenderGraphResourceDescriptor[] = [{
			id: "frame:scene-color",
			origin: "imported",
			format: graph.gBuffer.channels.color?.format,
			width: graph.width,
			height: graph.height,
		}];
		for (const [semantic, channel] of Object.entries(graph.gBuffer.channels)) {
			if (!channel || semantic === "color") continue;
			resources.push({ id: `frame:gbuffer:${semantic}`, origin: "imported", format: channel.format, width: channel.width, height: channel.height });
		}
		for (const descriptor of graph.historyDescriptors) {
			resources.push({ id: `postprocess:history:${descriptor.id}:read`, origin: "imported", format: descriptor.format, width: graph.width, height: graph.height });
			resources.push({ id: `postprocess:history:${descriptor.id}:write`, origin: "imported", format: descriptor.format, width: graph.width, height: graph.height });
		}
		for (const descriptor of graph.transientDescriptors) {
			resources.push({ id: `postprocess:transient:${descriptor.id}`, origin: "imported", format: descriptor.format, width: graph.width, height: graph.height, mipMode: descriptor.mipMode ?? "single" });
		}

		const nodes: RenderGraphNode<PostProcessRenderGraphNodePayload>[] = [];
		let currentColor = "frame:scene-color";
		let previousNodeId: string | null = null;
		for (let index = 0; index < graph.passes.length; index++) {
			const pass = graph.passes[index];
			const metadata = pass.implementation?.metadata?.graph;
			const compatibilityOpaque = !metadata;
			const color = metadata?.color ?? this._defaultColorFlow(graph.backend);
			const outputValidation = metadata?.outputValidation ??
				(pass.pass.builtIn ? "strict" : "compatibility");
			const nodeId = `postprocess:${pass.id}`;
			const refs: RenderGraphResourceRef[] = [];
			if (color.access !== "none") {
				refs.push({ resource: currentColor, access: color.access, usage: graph.backend === "software" ? "cpu-write" : "sampled" });
			}
			for (const semantic of pass.pass.getRequirements({
				frameContext: graph.frameContext,
				postProcess: graph.postProcess,
				backend: graph.backend,
				gBuffer: graph.gBuffer,
				width: graph.width,
				height: graph.height,
				options: pass.options,
			}).gBuffer ?? []) {
				refs.push({
					resource: semantic === "color" ? "frame:scene-color" :
						`frame:gbuffer:${semantic}`,
					access: "read",
					usage: "sampled",
				});
			}
			this._appendPoolRefs(refs, pass.historyIds, "postprocess:history", metadata?.histories);
			const transientIds = graph.transientDescriptors.map((descriptor) => descriptor.id);
			this._appendPoolRefs(refs, transientIds, "postprocess:transient", metadata?.transients);
			for (const id of metadata?.backendShared ?? []) refs.push({ resource: id, access: "read", usage: "sampled", optional: true });

			const creates: string[] = [];
			if (color.output === "new-version") {
				const output = `postprocess:color:${index}`;
				resources.push({ id: output, origin: "graph", format: graph.gBuffer.channels.color?.format, width: graph.width, height: graph.height });
				creates.push(output);
				refs.push({ resource: output, access: "write", usage: graph.backend === "webgl" ? "color-attachment" : "storage" });
				currentColor = output;
			}
			nodes.push({ id: nodeId, stage: "postprocess", kind: "postprocess-pass", label: `PostProcess:${pass.id}`, dependsOn: previousNodeId ? [previousNodeId] : [], creates, resources: refs, payload: { passId: pass.id, color, outputValidation, compatibilityOpaque } });
			previousNodeId = nodeId;
		}
		return { resources, nodes, plannedOutputColor: currentColor };
	}

	private _defaultColorFlow(backend: CompiledPostProcessGraph["backend"]): PostProcessColorFlow {
		return backend === "software" ? { access: "read-write", output: "preserve" } : { access: "read", output: "new-version" };
	}

	private _appendPoolRefs(
		refs: RenderGraphResourceRef[],
		ids: readonly string[],
		prefix: string,
		declared: PostProcessGraphMetadata["histories"] | PostProcessGraphMetadata["transients"] | undefined,
	): void {
		for (const id of ids) {
			const access = declared?.[id] ?? "read-write";
			if (prefix.endsWith("history")) {
				refs.push({ resource: `${prefix}:${id}:read`, access: access === "write" ? "read" : access, usage: "sampled" });
				refs.push({ resource: `${prefix}:${id}:write`, access: access === "read" ? "write" : access, usage: "storage" });
			} else refs.push({ resource: `${prefix}:${id}`, access, usage: "storage" });
		}
	}
}
