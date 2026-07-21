import type {
	RenderGraphNode,
	RenderGraphDefinition,
	RenderGraphResourceDescriptor,
	RenderGraphResourceRef,
	RenderGraphUsage,
} from "../rendergraph/types";
import type { PostProcessPlan } from "./PostProcessPlanner";
import { resolvePostProcessResourceDescriptor } from "./resourceDescriptors";
import type {
	PostProcessColorDeclaration,
	PostProcessExecutionDeclaration,
	PostProcessExecutionResourceUse,
} from "./types";

export type PostProcessLogicalResourceRole =
	| "scene-color"
	| "gbuffer"
	| "history-read"
	| "history-write"
	| "transient"
	| "backend-shared"
	| "color-version";

export interface PostProcessSubgraphNodePayload {
	readonly passId: string;
	readonly color: PostProcessColorDeclaration;
	readonly inputColor: string | null;
	readonly plannedOutputColor: string | null;
}

export interface PostProcessSubgraph
	extends RenderGraphDefinition<PostProcessSubgraphNodePayload> {
	readonly resources: readonly RenderGraphResourceDescriptor[];
	readonly nodes: readonly RenderGraphNode<PostProcessSubgraphNodePayload>[];
	readonly outputColor: string;
	readonly resourceRoles: Readonly<Record<string, PostProcessLogicalResourceRole>>;
}

/** @internal Builds a logical post-process subgraph without native resources. */
export class PostProcessSubgraphBuilder {
	public build(graph: PostProcessPlan): PostProcessSubgraph {
		const resources: RenderGraphResourceDescriptor[] = [];
		const roles: Record<string, PostProcessLogicalResourceRole> = {};
		const colorFormat = graph.gBuffer.channels.color?.format ??
			(graph.backend === "webgpu" ? "rgba16float" : "rgba8unorm");
		const addResource = (
			resource: RenderGraphResourceDescriptor,
			role: PostProcessLogicalResourceRole
		): void => {
			if (roles[resource.id]) return;
			resources.push(resource);
			roles[resource.id] = role;
		};
		addResource({
			id: "scene-color",
			origin: "imported",
			kind: "texture",
			residency: "frame",
			initialContent: "valid",
			format: colorFormat,
			width: graph.width,
			height: graph.height,
		}, "scene-color");
		for (const [semantic, channel] of Object.entries(graph.gBuffer.channels)) {
			if (!channel || semantic === "color") continue;
			const resourceSemantic = semantic === "world-position" ? "depth" : semantic;
			addResource({
				id: `gbuffer:${resourceSemantic}`,
				origin: "imported",
				kind: "texture",
				residency: "frame",
				initialContent: "valid",
				format: channel.format,
				width: channel.width,
				height: channel.height,
			}, "gbuffer");
		}
		for (const descriptor of graph.historyDescriptors) {
			const resolved = resolvePostProcessResourceDescriptor(
				descriptor,
				graph.width,
				graph.height,
			);
			addResource({
				id: `history:${descriptor.id}:read`,
				origin: "imported",
				kind: "texture",
				residency: "history",
				initialContent: "unknown",
				format: resolved.format,
				width: resolved.width,
				height: resolved.height,
			}, "history-read");
			addResource({
				id: `history:${descriptor.id}:write`,
				origin: "imported",
				kind: "texture",
				residency: "history",
				initialContent: "undefined",
				format: resolved.format,
				width: resolved.width,
				height: resolved.height,
			}, "history-write");
		}
		for (const descriptor of graph.transientDescriptors) {
			const resolved = resolvePostProcessResourceDescriptor(
				descriptor,
				graph.width,
				graph.height,
				{ includeMipMode: true },
			);
			addResource({
				id: `transient:${descriptor.id}`,
				origin: "graph",
				kind: "texture",
				residency: "transient",
				initialContent: "undefined",
				format: resolved.format,
				width: resolved.width,
				height: resolved.height,
				mipMode: resolved.mipMode,
			}, "transient");
		}

		const nodes: RenderGraphNode<PostProcessSubgraphNodePayload>[] = [];
		let currentColor = "scene-color";
		let previousNodeId: string | null = null;
		for (let index = 0; index < graph.passes.length; index++) {
			const pass = graph.passes[index];
			const declaration = pass.declaration;
			const color = declaration.color;
			const nodeId = `pass:${pass.id}`;
			const refs: RenderGraphResourceRef[] = [];
			const inputColor = color.access === "none" ? null : currentColor;
			if (inputColor) {
				refs.push({
					resource: inputColor,
					access: color.access === "read-write" ? "read-write" : "read",
					usage: this._colorInputUsage(graph.backend),
				});
			}
			for (const gBufferUse of declaration.gBuffer ?? []) {
				const semantic = gBufferUse.semantic;
				const resourceSemantic = semantic === "world-position" ? "depth" : semantic;
				refs.push({
					resource: semantic === "color" ? "scene-color" : `gbuffer:${resourceSemantic}`,
					access: gBufferUse.access,
					usage: gBufferUse.usage,
					optional: gBufferUse.optional,
				});
			}
			this._appendHistoryRefs(refs, declaration.histories ?? []);
			this._appendTransientRefs(refs, declaration.transients ?? []);
			for (const shared of declaration.shared ?? []) {
				addResource({
					id: shared.id,
					origin: "imported",
					kind: "external",
					residency: "external",
					initialContent: "unknown",
					optional: shared.optional,
				}, "backend-shared");
				refs.push({ resource: shared.id, access: shared.access, usage: shared.usage, optional: shared.optional });
			}

			const creates: string[] = [];
			let plannedOutputColor: string | null = null;
			if (color.output === "new-version") {
				plannedOutputColor = `color:${index}`;
				addResource({
					id: plannedOutputColor,
					origin: "graph",
					kind: "texture",
					residency: "transient",
					initialContent: "undefined",
					format: colorFormat,
					width: graph.width,
					height: graph.height,
				}, "color-version");
				creates.push(plannedOutputColor);
				refs.push({
					resource: plannedOutputColor,
					access: "write",
					usage: this._colorOutputUsage(graph.backend),
				});
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
				payload: {
					passId: pass.id,
					color,
					inputColor,
					plannedOutputColor,
				},
			});
			previousNodeId = nodeId;
		}
		return Object.freeze({
			resources: Object.freeze(resources.slice()),
			nodes: Object.freeze(nodes.slice()),
			imports: Object.freeze(resources
				.filter((resource) => resource.origin === "imported")
				.map((resource) => Object.freeze({
					name: resource.id,
					resource: resource.id,
					optional: resource.optional,
				}))),
			outputPorts: Object.freeze([Object.freeze({
				name: "color",
				resource: currentColor,
			})]),
			exports: Object.freeze([Object.freeze({
				name: "color",
				resource: currentColor,
			})]),
			outputColor: currentColor,
			resourceRoles: Object.freeze({ ...roles }),
		});
	}

	private _colorInputUsage(backend: PostProcessPlan["backend"]): RenderGraphUsage {
		return backend === "software" ? "cpu-read" : "sampled";
	}

	private _colorOutputUsage(backend: PostProcessPlan["backend"]): RenderGraphUsage {
		return backend === "webgl" ? "color-attachment" : backend === "software" ? "cpu-write" : "storage";
	}

	private _appendHistoryRefs(
		refs: RenderGraphResourceRef[],
		declarations: NonNullable<PostProcessExecutionDeclaration["histories"]>
	): void {
		for (const declaration of declarations) {
			const id = declaration.descriptor.id;
			this._appendUses(refs, `history:${id}:read`, declaration.read);
			this._appendUses(refs, `history:${id}:write`, declaration.write);
		}
	}

	private _appendTransientRefs(
		refs: RenderGraphResourceRef[],
		declarations: NonNullable<PostProcessExecutionDeclaration["transients"]>
	): void {
		for (const declaration of declarations) {
			this._appendUses(
				refs,
				`transient:${declaration.descriptor.id}`,
				declaration.uses,
			);
		}
	}

	private _appendUses(
		refs: RenderGraphResourceRef[],
		resource: string,
		uses: readonly PostProcessExecutionResourceUse[]
	): void {
		for (const use of uses) {
			refs.push({
				resource,
				access: use.access,
				usage: use.usage,
				optional: use.optional,
			});
		}
	}
}
