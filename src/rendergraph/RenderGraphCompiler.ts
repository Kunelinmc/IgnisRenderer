import { RenderGraphAnalyzer } from "./RenderGraphAnalyzer";
import type {
	CompiledRenderGraph,
	RenderGraphDiagnostic,
	RenderGraphNode,
	RenderGraphResourceDescriptor,
	RenderGraphResourceMutation,
	RenderGraphTransition,
} from "./types";

/** @internal Pure validation and stable ordering for backend-private graphs. */
export class RenderGraphCompiler {
	public compile<TPayload, TKind extends string = string>(request: {
		readonly nodes: readonly RenderGraphNode<TPayload, TKind>[];
		readonly resources: readonly RenderGraphResourceDescriptor[];
	}): CompiledRenderGraph<TPayload, TKind> {
		const diagnostics: RenderGraphDiagnostic[] = [];
		const resources = new Map<string, RenderGraphResourceDescriptor>();
		for (const resource of request.resources) {
			if (resources.has(resource.id)) {
				diagnostics.push({
					phase: "compile",
					enforcement: "enforced",
					severity: "error",
					code: "duplicate-resource",
					resourceId: resource.id,
					message: `Render graph declares resource "${resource.id}" more than once.`,
				});
				continue;
			}
			resources.set(resource.id, resource);
		}

		const nodes = new Map<string, RenderGraphNode<TPayload, TKind>>();
		const declarationIndex = new Map<string, number>();
		for (let index = 0; index < request.nodes.length; index++) {
			const node = request.nodes[index];
			if (nodes.has(node.id)) {
				diagnostics.push({
					phase: "compile",
					enforcement: "enforced",
					severity: "error",
					code: "duplicate-node",
					stage: node.stage,
					nodeId: node.id,
					message: `Render graph declares node "${node.id}" more than once.`,
				});
				continue;
			}
			nodes.set(node.id, node);
			declarationIndex.set(node.id, index);
		}

		const order = this._stableOrder(nodes, declarationIndex, diagnostics);
		const analyzer = new RenderGraphAnalyzer<TPayload, TKind>();
		analyzer.reset(Array.from(resources.values()));
		analyzer.analyzeNodes(order);
		const liveRanges = analyzer.getLiveRanges();
		return Object.freeze({
			nodes: freezeNodes(order),
			resources: freezeItems(Array.from(resources.values())),
			diagnostics: freezeItems([
				...diagnostics,
				...analyzer.getDiagnostics(),
			]),
			shadowDiagnostics: analyzer.getShadowDiagnostics(),
			transitions: freezeTransitions(analyzer.getTransitions()),
			lifetimes: liveRanges,
			liveRanges,
		});
	}

	private _stableOrder<TPayload, TKind extends string>(
		nodes: ReadonlyMap<string, RenderGraphNode<TPayload, TKind>>,
		declarationIndex: ReadonlyMap<string, number>,
		diagnostics: RenderGraphDiagnostic[],
	): RenderGraphNode<TPayload, TKind>[] {
		const dependents = new Map<string, string[]>();
		const indegree = new Map<string, number>();
		for (const node of nodes.values()) indegree.set(node.id, 0);
		for (const node of nodes.values()) {
			for (const dependencyId of node.dependsOn ?? []) {
				if (!nodes.has(dependencyId)) {
					diagnostics.push({
						phase: "compile",
						enforcement: "enforced",
						severity: "error",
						code: "missing-dependency",
						stage: node.stage,
						nodeId: node.id,
						message:
							`Render graph node "${node.id}" depends on missing ` +
							`node "${dependencyId}".`,
					});
					continue;
				}
				indegree.set(node.id, (indegree.get(node.id) ?? 0) + 1);
				const entries = dependents.get(dependencyId) ?? [];
				entries.push(node.id);
				dependents.set(dependencyId, entries);
			}
		}

		const ready = Array.from(nodes.values()).filter(
			(node) => (indegree.get(node.id) ?? 0) === 0,
		);
		const compare = (
			left: RenderGraphNode<TPayload, TKind>,
			right: RenderGraphNode<TPayload, TKind>,
		): number =>
			(declarationIndex.get(left.id) ?? 0) -
			(declarationIndex.get(right.id) ?? 0);
		ready.sort(compare);
		const order: RenderGraphNode<TPayload, TKind>[] = [];
		while (ready.length > 0) {
			const node = ready.shift()!;
			order.push(node);
			for (const dependentId of dependents.get(node.id) ?? []) {
				const next = (indegree.get(dependentId) ?? 0) - 1;
				indegree.set(dependentId, next);
				if (next === 0) {
					ready.push(nodes.get(dependentId)!);
					ready.sort(compare);
				}
			}
		}
		if (order.length !== nodes.size) {
			for (const node of nodes.values()) {
				if ((indegree.get(node.id) ?? 0) <= 0) continue;
				diagnostics.push({
					phase: "compile",
					enforcement: "enforced",
					severity: "error",
					code: "cycle",
					stage: node.stage,
					nodeId: node.id,
					message: `Render graph dependency cycle includes node "${node.id}".`,
				});
				order.push(node);
			}
		}
		return order;
	}
}

function freezeItems<T extends object>(items: readonly T[]): readonly T[] {
	return Object.freeze(items.map((item) => Object.freeze({ ...item })));
}

function freezeNodes<TPayload, TKind extends string>(
	nodes: readonly RenderGraphNode<TPayload, TKind>[],
): readonly RenderGraphNode<TPayload, TKind>[] {
	return Object.freeze(nodes.map((node) => Object.freeze({
		...node,
		dependsOn: node.dependsOn ? Object.freeze(node.dependsOn.slice()) : undefined,
		requires: node.requires ? freezeItems(node.requires) : undefined,
		creates: node.creates ? freezeMutations(node.creates) : undefined,
		destroys: node.destroys ? freezeMutations(node.destroys) : undefined,
		resources: node.resources ? Object.freeze(node.resources.map((ref) =>
			Object.freeze({
				...ref,
				subresource: ref.subresource ? Object.freeze({ ...ref.subresource }) : undefined,
			}),
		)) : undefined,
	})));
}

function freezeMutations(
	mutations: readonly RenderGraphResourceMutation[],
): readonly RenderGraphResourceMutation[] {
	return Object.freeze(mutations.map((mutation) =>
		typeof mutation === "string" ? mutation : Object.freeze({ ...mutation }),
	));
}

function freezeTransitions(
	transitions: readonly RenderGraphTransition[],
): readonly RenderGraphTransition[] {
	return Object.freeze(transitions.map((transition) => Object.freeze({
		...transition,
		subresource: transition.subresource ?
			Object.freeze({ ...transition.subresource }) : undefined,
	})));
}
