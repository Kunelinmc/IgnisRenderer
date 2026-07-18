import type {
	CompiledRenderGraph,
	RenderGraphDiagnostic,
	RenderGraphNode,
	RenderGraphResourceDescriptor,
	RenderGraphResourceLifetime,
	RenderGraphTransition,
} from "./types";

/** @internal Validates and stably orders backend-private logical render graphs. */
export class RenderGraphCompiler {
	public compile<TPayload>(request: {
		readonly nodes: readonly RenderGraphNode<TPayload>[];
		readonly resources: readonly RenderGraphResourceDescriptor[];
	}): CompiledRenderGraph<TPayload> {
		const diagnostics: RenderGraphDiagnostic[] = [];
		const resources = new Map<string, RenderGraphResourceDescriptor>();
		for (const resource of request.resources) {
			if (resources.has(resource.id)) {
				diagnostics.push({
					severity: "error",
					code: "duplicate-resource",
					resourceId: resource.id,
					message: `Render graph declares resource "${resource.id}" more than once.`,
				});
				continue;
			}
			resources.set(resource.id, resource);
		}

		const byId = new Map<string, RenderGraphNode<TPayload>>();
		for (const node of request.nodes) {
			if (byId.has(node.id)) {
				diagnostics.push({
					severity: "error",
					code: "duplicate-node",
					nodeId: node.id,
					message: `Render graph declares node "${node.id}" more than once.`,
				});
				continue;
			}
			byId.set(node.id, node);
		}

		const order: RenderGraphNode<TPayload>[] = [];
		const state = new Map<string, "visiting" | "done">();
		const visit = (node: RenderGraphNode<TPayload>): void => {
			const current = state.get(node.id);
			if (current === "done") return;
			if (current === "visiting") {
				diagnostics.push({
					severity: "error",
					code: "cycle",
					nodeId: node.id,
					message: `Render graph dependency cycle includes node "${node.id}".`,
				});
				return;
			}
			state.set(node.id, "visiting");
			for (const dependencyId of node.dependsOn ?? []) {
				const dependency = byId.get(dependencyId);
				if (!dependency) {
					diagnostics.push({
						severity: "error",
						code: "missing-dependency",
						nodeId: node.id,
						message: `Render graph node "${node.id}" depends on missing node "${dependencyId}".`,
					});
					continue;
				}
				visit(dependency);
			}
			state.set(node.id, "done");
			order.push(node);
		};
		for (const node of request.nodes) visit(node);

		const active = new Set<string>();
		const transitions: RenderGraphTransition[] = [];
		const previousAccess = new Map<string, "read" | "write" | "read-write">();
		const lifetimes = new Map<string, RenderGraphResourceLifetime>();
		for (const resource of resources.values()) {
			if (resource.origin === "imported") active.add(resource.id);
		}
		for (const node of order) {
			for (const id of node.creates ?? []) {
				if (active.has(id)) {
					diagnostics.push({ severity: "error", code: "duplicate-create", nodeId: node.id, resourceId: id, message: `Render graph node "${node.id}" creates active resource "${id}".` });
				}
				active.add(id);
			}
			for (const ref of node.resources ?? []) {
				const previous = previousAccess.get(ref.resource);
				transitions.push({
					nodeId: node.id,
					resourceId: ref.resource,
					previousAccess: previous,
					access: ref.access,
					usage: ref.usage,
				});
				previousAccess.set(ref.resource, ref.access);
				const lifetime = lifetimes.get(ref.resource);
				lifetimes.set(ref.resource, lifetime ? {
					...lifetime,
					lastNodeId: node.id,
				} : {
					resourceId: ref.resource,
					firstNodeId: node.id,
					lastNodeId: node.id,
				});
				if (!resources.has(ref.resource) && !ref.optional) {
					diagnostics.push({ severity: "error", code: "read-before-create", nodeId: node.id, resourceId: ref.resource, message: `Render graph node "${node.id}" references undeclared resource "${ref.resource}".` });
					continue;
				}
				if ((ref.access === "read" || ref.access === "read-write") && !active.has(ref.resource) && !ref.optional) {
					diagnostics.push({ severity: "error", code: "read-before-create", nodeId: node.id, resourceId: ref.resource, message: `Render graph node "${node.id}" reads inactive resource "${ref.resource}".` });
				}
				if (ref.access === "write" || ref.access === "read-write") active.add(ref.resource);
			}
		}
		return Object.freeze({
			nodes: Object.freeze(order.slice()),
			resources: Object.freeze(Array.from(resources.values())),
			diagnostics: Object.freeze(diagnostics),
			transitions: Object.freeze(transitions),
			lifetimes: Object.freeze(Array.from(lifetimes.values())),
		});
	}
}
