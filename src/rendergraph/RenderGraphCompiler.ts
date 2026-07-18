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

		const nodes = new Map<string, RenderGraphNode<TPayload>>();
		const declarationIndex = new Map<string, number>();
		for (let index = 0; index < request.nodes.length; index++) {
			const node = request.nodes[index];
			if (nodes.has(node.id)) {
				diagnostics.push({
					severity: "error",
					code: "duplicate-node",
					nodeId: node.id,
					message: `Render graph declares node "${node.id}" more than once.`,
				});
				continue;
			}
			nodes.set(node.id, node);
			declarationIndex.set(node.id, index);
		}

		const dependents = new Map<string, string[]>();
		const indegree = new Map<string, number>();
		for (const node of nodes.values()) indegree.set(node.id, 0);
		for (const node of nodes.values()) {
			for (const dependencyId of node.dependsOn ?? []) {
				if (!nodes.has(dependencyId)) {
					diagnostics.push({
						severity: "error",
						code: "missing-dependency",
						nodeId: node.id,
						message: `Render graph node "${node.id}" depends on missing node "${dependencyId}".`,
					});
					continue;
				}
				indegree.set(node.id, (indegree.get(node.id) ?? 0) + 1);
				const entries = dependents.get(dependencyId) ?? [];
				entries.push(node.id);
				dependents.set(dependencyId, entries);
			}
		}

		const ready = Array.from(nodes.values()).filter((node) =>
			(indegree.get(node.id) ?? 0) === 0
		);
		ready.sort((left, right) =>
			(declarationIndex.get(left.id) ?? 0) - (declarationIndex.get(right.id) ?? 0)
		);
		const order: RenderGraphNode<TPayload>[] = [];
		while (ready.length > 0) {
			const node = ready.shift()!;
			order.push(node);
			for (const dependentId of dependents.get(node.id) ?? []) {
				const next = (indegree.get(dependentId) ?? 0) - 1;
				indegree.set(dependentId, next);
				if (next === 0) {
					ready.push(nodes.get(dependentId)!);
					ready.sort((left, right) =>
						(declarationIndex.get(left.id) ?? 0) -
						(declarationIndex.get(right.id) ?? 0)
					);
				}
			}
		}
		if (order.length !== nodes.size) {
			for (const node of nodes.values()) {
				if ((indegree.get(node.id) ?? 0) <= 0) continue;
				diagnostics.push({
					severity: "error",
					code: "cycle",
					nodeId: node.id,
					message: `Render graph dependency cycle includes node "${node.id}".`,
				});
				order.push(node);
			}
		}

		const active = new Set<string>();
		for (const resource of resources.values()) {
			if (resource.origin === "imported") active.add(resource.id);
		}
		const transitions: RenderGraphTransition[] = [];
		const previousAccess = new Map<string, "read" | "write" | "read-write">();
		const lifetimes = new Map<string, RenderGraphResourceLifetime>();
		for (const node of order) {
			for (const id of node.creates ?? []) {
				if (active.has(id)) {
					diagnostics.push({
						severity: "error",
						code: "duplicate-create",
						nodeId: node.id,
						resourceId: id,
						message: `Render graph node "${node.id}" creates active resource "${id}".`,
					});
				}
				active.add(id);
			}
			for (const ref of node.resources ?? []) {
				const previous = previousAccess.get(ref.resource);
				const hazard = this._resolveHazard(previous, ref.access);
				transitions.push({
					nodeId: node.id,
					resourceId: ref.resource,
					previousAccess: previous,
					access: ref.access,
					usage: ref.usage,
					hazard,
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
					diagnostics.push({
						severity: "error",
						code: "read-before-create",
						nodeId: node.id,
						resourceId: ref.resource,
						message: `Render graph node "${node.id}" references undeclared resource "${ref.resource}".`,
					});
					continue;
				}
				if ((ref.access === "read" || ref.access === "read-write") && !active.has(ref.resource) && !ref.optional) {
					diagnostics.push({
						severity: "error",
						code: "read-before-create",
						nodeId: node.id,
						resourceId: ref.resource,
						message: `Render graph node "${node.id}" reads inactive resource "${ref.resource}".`,
					});
				}
				if (ref.access === "write" || ref.access === "read-write") active.add(ref.resource);
			}
			for (const id of node.destroys ?? []) {
				if (!active.has(id)) {
					diagnostics.push({
						severity: "error",
						code: "destroy-before-create",
						nodeId: node.id,
						resourceId: id,
						message: `Render graph node "${node.id}" destroys inactive resource "${id}".`,
					});
					continue;
				}
				active.delete(id);
			}
		}

		return this._freeze({
			nodes: order,
			resources: Array.from(resources.values()),
			diagnostics,
			transitions,
			lifetimes: Array.from(lifetimes.values()),
		});
	}

	private _resolveHazard(
		previous: "read" | "write" | "read-write" | undefined,
		next: "read" | "write" | "read-write"
	): RenderGraphTransition["hazard"] {
		if (!previous) return undefined;
		const previousWrites = previous !== "read";
		const nextWrites = next !== "read";
		if (previousWrites && !nextWrites) return "read-after-write";
		if (!previousWrites && nextWrites) return "write-after-read";
		if (previousWrites && nextWrites) return "write-after-write";
		return undefined;
	}

	private _freeze<TPayload>(graph: {
		nodes: readonly RenderGraphNode<TPayload>[];
		resources: readonly RenderGraphResourceDescriptor[];
		diagnostics: readonly RenderGraphDiagnostic[];
		transitions: readonly RenderGraphTransition[];
		lifetimes: readonly RenderGraphResourceLifetime[];
	}): CompiledRenderGraph<TPayload> {
		const freezeItems = <T extends object>(items: readonly T[]): readonly T[] =>
			Object.freeze(items.map((item) => Object.freeze({ ...item })));
		return Object.freeze({
			nodes: freezeItems(graph.nodes),
			resources: freezeItems(graph.resources),
			diagnostics: freezeItems(graph.diagnostics),
			transitions: freezeItems(graph.transitions),
			lifetimes: freezeItems(graph.lifetimes),
		});
	}
}
