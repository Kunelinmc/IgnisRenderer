import {
	renderGraphNodeId,
	renderGraphResourceId,
} from "./types";
import type {
	RenderGraphDefinition,
	RenderGraphDiagnostic,
	RenderGraphExport,
	RenderGraphNode,
	RenderGraphNodeId,
	RenderGraphPhysicalBinding,
	RenderGraphPortResolution,
	RenderGraphResourceDescriptor,
	RenderGraphResourceId,
	RenderGraphResourceMutation,
} from "./types";

/** @internal Port mappings used when composing one logical subgraph. */
export interface RenderGraphSubgraphComposition {
	readonly namespace: string;
	readonly inputs?: Readonly<Record<string, RenderGraphResourceId>>;
	readonly outputs?: Readonly<Record<string, RenderGraphResourceId>>;
	readonly dependsOn?: readonly RenderGraphNodeId[];
}

/** @internal Result of composing one namespaced subgraph. */
export interface RenderGraphSubgraphCompositionResult {
	readonly outputs: Readonly<Record<string, RenderGraphResourceId>>;
	readonly resources: Readonly<Record<string, RenderGraphResourceId>>;
	readonly nodes: Readonly<Record<string, RenderGraphNodeId>>;
}

/** @internal Mutable construction helper that emits an immutable definition. */
export class RenderGraphBuilder<TPayload = unknown, TKind extends string = string> {
	private readonly _resources: RenderGraphResourceDescriptor[] = [];
	private readonly _bindings: RenderGraphPhysicalBinding[] = [];
	private readonly _nodes: RenderGraphNode<TPayload, TKind>[] = [];
	private readonly _exports: RenderGraphExport[] = [];
	private readonly _buildDiagnostics: RenderGraphDiagnostic[] = [];
	private readonly _shadowDiagnostics: RenderGraphDiagnostic[] = [];
	private readonly _portResolutions: RenderGraphPortResolution[] = [];
	private _completeness: RenderGraphDefinition["completeness"] = "complete";

	public addResource(descriptor: RenderGraphResourceDescriptor): this {
		this._resources.push(descriptor);
		return this;
	}

	public addBinding(binding: RenderGraphPhysicalBinding): this {
		this._bindings.push(binding);
		return this;
	}

	public addNode(node: RenderGraphNode<TPayload, TKind>): this {
		this._nodes.push(node);
		return this;
	}

	public addExport(entry: RenderGraphExport): this {
		this._exports.push(entry);
		return this;
	}

	public addShadowDiagnostic(diagnostic: RenderGraphDiagnostic): this {
		this._shadowDiagnostics.push(diagnostic);
		return this;
	}

	public markCompleteness(
		completeness: NonNullable<RenderGraphDefinition["completeness"]>,
	): this {
		if (completenessRank(completeness) > completenessRank(this._completeness ?? "complete")) {
			this._completeness = completeness;
		}
		return this;
	}

	public addSubgraph(
		subgraph: RenderGraphDefinition<TPayload, TKind>,
		composition: RenderGraphSubgraphComposition,
	): RenderGraphSubgraphCompositionResult {
		const namespace = composition.namespace.trim();
		if (!namespace) {
			this._buildDiagnostics.push(buildError(
				"missing-subgraph-port",
				"Render graph subgraph composition requires a non-empty namespace.",
			));
			return emptyCompositionResult();
		}
		const resourceMap = new Map<RenderGraphResourceId, RenderGraphResourceId>();
		const omittedOptionalImports = new Set<RenderGraphResourceId>();
		for (const port of subgraph.imports ?? []) {
			const parentId = composition.inputs?.[port.name];
			if (!parentId) {
				if (!port.optional) {
					this._buildDiagnostics.push(buildError(
						"missing-subgraph-port",
						`Subgraph "${namespace}" requires input port "${port.name}".`,
						port.resource,
					));
				}
				continue;
			}
			resourceMap.set(port.resource, parentId);
			this._portResolutions.push({
				namespace,
				direction: "import",
				port: port.name,
				resource: parentId,
			});
		}
		const resolvedOutputs: Record<string, RenderGraphResourceId> = {};
		for (const port of subgraph.outputPorts ?? []) {
			const parentId = composition.outputs?.[port.name] ??
				namespaceResourceId(namespace, port.resource);
			resourceMap.set(port.resource, parentId);
			resolvedOutputs[port.name] = parentId;
			this._portResolutions.push({
				namespace,
				direction: "export",
				port: port.name,
				resource: parentId,
			});
		}
		for (const descriptor of subgraph.resources) {
			const importPort = (subgraph.imports ?? []).find(
				(port) => port.resource === descriptor.id,
			);
			const mappedId = resourceMap.get(descriptor.id);
			const omittedOptionalImport = importPort?.optional === true && mappedId === undefined;
			const remappedId = mappedId ?? namespaceResourceId(namespace, descriptor.id);
			resourceMap.set(descriptor.id, remappedId);
			if (omittedOptionalImport) {
				omittedOptionalImports.add(descriptor.id);
				continue;
			}
			const existing = this._resources.find((entry) => entry.id === remappedId);
			const isImport = importPort !== undefined;
			const isMappedOutput = (subgraph.outputPorts ?? []).some(
				(port) => port.resource === descriptor.id && !!composition.outputs?.[port.name],
			);
			if (existing) {
				if (!areDescriptorsCompatible(existing, descriptor)) {
					this._buildDiagnostics.push(buildError(
						"incompatible-subgraph-port",
						`Subgraph "${namespace}" maps incompatible resource ` +
							`"${descriptor.id}" to "${remappedId}".`,
						remappedId,
					));
				} else if (!isImport && !isMappedOutput) {
					this._buildDiagnostics.push(buildError(
						"duplicate-resource",
						`Subgraph "${namespace}" resource ID "${remappedId}" collides ` +
							"with a parent resource.",
						remappedId,
					));
				}
				continue;
			}
			if (isImport) {
				this._buildDiagnostics.push(buildError(
					"missing-resource",
					`Subgraph "${namespace}" input maps to undeclared parent resource ` +
						`"${remappedId}".`,
					remappedId,
				));
				continue;
			}
			this._resources.push({ ...descriptor, id: remappedId });
		}
		const nodeMap = new Map(
			subgraph.nodes.map((node) => [node.id, namespaceNodeId(namespace, node.id)]),
		);
		for (const binding of subgraph.bindings ?? []) {
			if (omittedOptionalImports.has(binding.resourceId)) continue;
			this._bindings.push({
				...binding,
				resourceId: resourceMap.get(binding.resourceId) ??
					namespaceResourceId(namespace, binding.resourceId),
			});
		}
		for (const node of subgraph.nodes) {
			const remappedNodeId = nodeMap.get(node.id)!;
			if (this._nodes.some((entry) => entry.id === remappedNodeId)) {
				this._buildDiagnostics.push(buildError(
					"duplicate-node",
					`Subgraph "${namespace}" node ID "${remappedNodeId}" collides ` +
						"with a parent node.",
				));
				continue;
			}
			const externalDependencies = (node.dependsOn?.length ?? 0) === 0
				? composition.dependsOn ?? []
				: [];
			this._nodes.push({
				...node,
				id: remappedNodeId,
				dependsOn: [
					...(node.dependsOn?.map((id) =>
						nodeMap.get(id) ?? namespaceNodeId(namespace, id)) ?? []),
					...externalDependencies,
				],
				requires: node.requires?.map((requirement) => ({
					...requirement,
					resource: remapResource(resourceMap, namespace, requirement.resource),
				})),
				creates: remapMutations(node.creates, resourceMap, namespace),
				destroys: remapMutations(node.destroys, resourceMap, namespace),
				resources: node.resources?.map((ref) => ({
					...ref,
					resource: remapResource(resourceMap, namespace, ref.resource),
				})),
			});
		}
		for (const entry of subgraph.exports ?? []) {
			this._exports.push({
				...entry,
				name: entry.name ? `${namespace}:${entry.name}` : undefined,
				resource: remapResource(resourceMap, namespace, entry.resource),
			});
		}
		this._buildDiagnostics.push(...(subgraph.buildDiagnostics ?? []));
		this._shadowDiagnostics.push(...(subgraph.shadowDiagnostics ?? []));
		this.markCompleteness(subgraph.completeness ?? "complete");
		return {
			outputs: Object.freeze(resolvedOutputs),
			resources: Object.freeze(Object.fromEntries(resourceMap)),
			nodes: Object.freeze(Object.fromEntries(nodeMap)),
		};
	}

	public build(): RenderGraphDefinition<TPayload, TKind> {
		return Object.freeze({
			resources: Object.freeze(this._resources.slice()),
			bindings: Object.freeze(this._bindings.slice()),
			nodes: Object.freeze(this._nodes.slice()),
			exports: Object.freeze(this._exports.slice()),
			portResolutions: Object.freeze(this._portResolutions.slice()),
			completeness: this._completeness,
			buildDiagnostics: Object.freeze(this._buildDiagnostics.slice()),
			shadowDiagnostics: Object.freeze(this._shadowDiagnostics.slice()),
		});
	}
}

function emptyCompositionResult(): RenderGraphSubgraphCompositionResult {
	return {
		outputs: Object.freeze({}),
		resources: Object.freeze({}),
		nodes: Object.freeze({}),
	};
}

function areDescriptorsCompatible(
	parent: RenderGraphResourceDescriptor,
	child: RenderGraphResourceDescriptor,
): boolean {
	if (parent.kind === "external" || child.kind === "external") return true;
	if (parent.kind !== child.kind) return false;
	if (parent.kind === "buffer" && child.kind === "buffer") {
		return parent.size === undefined || child.size === undefined || parent.size === child.size;
	}
	if (parent.kind !== "texture" || child.kind !== "texture") return false;
	const fields = [
		"format",
		"width",
		"height",
		"depthOrArrayLayers",
		"dimension",
		"sampleCount",
		"mipLevelCount",
	] as const;
	return fields.every((field) =>
		parent[field] === undefined || child[field] === undefined || parent[field] === child[field],
	);
}

function namespaceResourceId(
	namespace: string,
	id: RenderGraphResourceId,
): RenderGraphResourceId {
	return renderGraphResourceId(`${namespace}:${id}`);
}

function namespaceNodeId(namespace: string, id: RenderGraphNodeId): RenderGraphNodeId {
	return renderGraphNodeId(`${namespace}:${id}`);
}

function remapResource(
	resourceMap: ReadonlyMap<RenderGraphResourceId, RenderGraphResourceId>,
	namespace: string,
	resourceId: RenderGraphResourceId,
): RenderGraphResourceId {
	return resourceMap.get(resourceId) ?? namespaceResourceId(namespace, resourceId);
}

function remapMutations(
	mutations: readonly RenderGraphResourceMutation[] | undefined,
	resourceMap: ReadonlyMap<RenderGraphResourceId, RenderGraphResourceId>,
	namespace: string,
): readonly RenderGraphResourceMutation[] | undefined {
	return mutations?.map((mutation) =>
		typeof mutation === "string"
			? remapResource(resourceMap, namespace, mutation)
			: {
					...mutation,
					resource: remapResource(resourceMap, namespace, mutation.resource),
				},
	);
}

function buildError(
	code: string,
	message: string,
	resourceId?: RenderGraphResourceId,
): RenderGraphDiagnostic {
	return {
		phase: "build",
		enforcement: "enforced",
		severity: "error",
		code,
		resourceId,
		message,
	};
}

function completenessRank(value: NonNullable<RenderGraphDefinition["completeness"]>): number {
	if (value === "opaque") return 2;
	if (value === "coarse") return 1;
	return 0;
}
