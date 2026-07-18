import type {
	RenderGraphNode,
	RenderGraphResourceDescriptor,
	RenderGraphResourceRef,
	RenderGraphUsage,
} from "../rendergraph/types";
import type { CompiledPostProcessGraph } from "./PostProcessGraphCompiler";
import type {
	PostProcessColorFlow,
	PostProcessGraphMetadata,
	PostProcessGraphResourceUse,
} from "./types";

export type PostProcessLogicalResourceRole =
	| "scene-color"
	| "gbuffer"
	| "history-read"
	| "history-write"
	| "transient"
	| "backend-shared"
	| "color-version";

export interface PostProcessRenderGraphNodePayload {
	readonly passId: string;
	readonly color: PostProcessColorFlow;
	readonly inputColor: string | null;
	readonly plannedOutputColor: string | null;
	readonly outputValidation: "strict" | "compatibility";
	readonly compatibilityOpaque: boolean;
}

export interface PostProcessRenderGraphSubgraph {
	readonly resources: readonly RenderGraphResourceDescriptor[];
	readonly nodes: readonly RenderGraphNode<PostProcessRenderGraphNodePayload>[];
	readonly outputColor: string;
	readonly resourceRoles: Readonly<Record<string, PostProcessLogicalResourceRole>>;
}

/** @internal Builds a logical post-process subgraph without native resources. */
export class PostProcessRenderGraphAdapter {
	public build(graph: CompiledPostProcessGraph): PostProcessRenderGraphSubgraph {
		const resources: RenderGraphResourceDescriptor[] = [];
		const roles: Record<string, PostProcessLogicalResourceRole> = {};
		const addResource = (
			resource: RenderGraphResourceDescriptor,
			role: PostProcessLogicalResourceRole
		): void => {
			if (roles[resource.id]) return;
			resources.push(resource);
			roles[resource.id] = role;
		};
		addResource({
			id: "frame:scene-color",
			origin: "imported",
			format: graph.gBuffer.channels.color?.format,
			width: graph.width,
			height: graph.height,
		}, "scene-color");
		for (const [semantic, channel] of Object.entries(graph.gBuffer.channels)) {
			if (!channel || semantic === "color") continue;
			addResource({
				id: `frame:gbuffer:${semantic}`,
				origin: "imported",
				format: channel.format,
				width: channel.width,
				height: channel.height,
			}, "gbuffer");
		}
		for (const descriptor of graph.historyDescriptors) {
			addResource({ id: `postprocess:history:${descriptor.id}:read`, origin: "imported", format: descriptor.format, width: graph.width, height: graph.height }, "history-read");
			addResource({ id: `postprocess:history:${descriptor.id}:write`, origin: "imported", format: descriptor.format, width: graph.width, height: graph.height }, "history-write");
		}
		for (const descriptor of graph.transientDescriptors) {
			addResource({ id: `postprocess:transient:${descriptor.id}`, origin: "imported", format: descriptor.format, width: graph.width, height: graph.height, mipMode: descriptor.mipMode ?? "single" }, "transient");
		}

		const nodes: RenderGraphNode<PostProcessRenderGraphNodePayload>[] = [];
		let currentColor = "frame:scene-color";
		let previousNodeId: string | null = null;
		for (let index = 0; index < graph.passes.length; index++) {
			const pass = graph.passes[index];
			const metadata = pass.graphMetadata;
			const color = metadata?.color ?? this._defaultColorFlow(graph.backend);
			const compatibilityOpaque = pass.compatibilityOpaque;
			const outputValidation = metadata?.outputValidation ??
				(pass.pass.builtIn ? "strict" : "compatibility");
			const nodeId = `postprocess:${pass.id}`;
			const refs: RenderGraphResourceRef[] = [];
			const inputColor = color.access === "none" ? null : currentColor;
			if (inputColor) {
				refs.push({
					resource: inputColor,
					access: color.access === "read-write" ? "read-write" : "read",
					usage: this._colorInputUsage(graph.backend),
				});
			}
			for (const semantic of pass.requirements.gBuffer ?? []) {
				refs.push({
					resource: semantic === "color" ? "frame:scene-color" : `frame:gbuffer:${semantic}`,
					access: "read",
					usage: "sampled",
				});
			}
			this._appendHistoryRefs(refs, pass.historyIds, metadata, graph.backend);
			this._appendTransientRefs(refs, pass.transientIds, metadata, graph.backend);
			for (const shared of metadata?.backendShared ?? []) {
				addResource({ id: shared.id, origin: "imported", optional: shared.optional }, "backend-shared");
				refs.push({ resource: shared.id, access: shared.access, usage: shared.usage, optional: shared.optional });
			}

			const creates: string[] = [];
			let plannedOutputColor: string | null = null;
			if (color.output === "new-version") {
				plannedOutputColor = `postprocess:color:${index}`;
				addResource({ id: plannedOutputColor, origin: "graph", format: graph.gBuffer.channels.color?.format, width: graph.width, height: graph.height }, "color-version");
				creates.push(plannedOutputColor);
				refs.push({ resource: plannedOutputColor, access: "write", usage: this._colorOutputUsage(graph.backend) });
				currentColor = plannedOutputColor;
			}
			nodes.push({
				id: nodeId,
				stage: "postprocess",
				kind: "postprocess-pass",
				label: `PostProcess:${pass.id}`,
				dependsOn: previousNodeId ? [previousNodeId] : [],
				creates,
				resources: refs,
				payload: { passId: pass.id, color, inputColor, plannedOutputColor, outputValidation, compatibilityOpaque },
			});
			previousNodeId = nodeId;
		}
		return Object.freeze({
			resources: Object.freeze(resources.slice()),
			nodes: Object.freeze(nodes.slice()),
			outputColor: currentColor,
			resourceRoles: Object.freeze({ ...roles }),
		});
	}

	private _defaultColorFlow(backend: CompiledPostProcessGraph["backend"]): PostProcessColorFlow {
		return backend === "software" ? { access: "read-write", output: "preserve" } : { access: "read", output: "new-version" };
	}

	private _colorInputUsage(backend: CompiledPostProcessGraph["backend"]): RenderGraphUsage {
		return backend === "software" ? "cpu-read" : "sampled";
	}

	private _colorOutputUsage(backend: CompiledPostProcessGraph["backend"]): RenderGraphUsage {
		return backend === "webgl" ? "color-attachment" : backend === "software" ? "cpu-write" : "storage";
	}

	private _appendHistoryRefs(
		refs: RenderGraphResourceRef[],
		ids: readonly string[],
		metadata: PostProcessGraphMetadata | null,
		backend: CompiledPostProcessGraph["backend"]
	): void {
		for (const id of ids) {
			const declaration = metadata?.histories?.[id];
			const read = declaration?.read ?? [{ access: "read", usage: backend === "software" ? "cpu-read" : "sampled" }];
			const write = declaration?.write ?? [{ access: "write", usage: backend === "webgl" ? "color-attachment" : backend === "software" ? "cpu-write" : "storage" }];
			this._appendUses(refs, `postprocess:history:${id}:read`, read);
			this._appendUses(refs, `postprocess:history:${id}:write`, write);
		}
	}

	private _appendTransientRefs(
		refs: RenderGraphResourceRef[],
		ids: readonly string[],
		metadata: PostProcessGraphMetadata | null,
		backend: CompiledPostProcessGraph["backend"]
	): void {
		for (const id of ids) {
			const uses = metadata?.transients?.[id] ?? [{ access: "read-write", usage: backend === "webgl" ? "color-attachment" : backend === "software" ? "cpu-write" : "storage" }];
			this._appendUses(refs, `postprocess:transient:${id}`, uses);
		}
	}

	private _appendUses(
		refs: RenderGraphResourceRef[],
		resource: string,
		uses: readonly PostProcessGraphResourceUse[]
	): void {
		for (const use of uses) refs.push({ resource, access: use.access, usage: use.usage, optional: use.optional });
	}
}
