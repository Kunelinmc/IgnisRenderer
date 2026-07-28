import { RenderGraphAnalyzer } from "./RenderGraphAnalyzer";
import {
	createRenderGraphCompatibilityKey,
	normalizeRenderGraphSubresource,
	renderGraphSubresourcesOverlap,
} from "./subresources";
import type {
	CompiledRenderGraph,
	CompiledRenderGraphStage,
	RenderGraphAllocationRequest,
	RenderGraphDefinition,
	RenderGraphDependency,
	RenderGraphDiagnostic,
	RenderGraphExport,
	RenderGraphLiveRange,
	RenderGraphNode,
	RenderGraphPhysicalBinding,
	RenderGraphResourceDescriptor,
	RenderGraphResourceMutation,
	RenderGraphSubresourceLiveRange,
	RenderGraphTransition,
} from "./types";

/** @internal Pure whole-frame validation, analysis, and opt-in optimization. */
export class RenderGraphCompiler {
	public compile<TPayload, TKind extends string = string>(
		request: RenderGraphDefinition<TPayload, TKind>,
	): CompiledRenderGraph<TPayload, TKind> {
		const diagnostics: RenderGraphDiagnostic[] = [...(request.buildDiagnostics ?? [])];
		const shadowDiagnostics: RenderGraphDiagnostic[] = [
			...(request.shadowDiagnostics ?? []),
		];
		const resources = collectResources(request.resources, diagnostics, shadowDiagnostics);
		const bindings = collectBindings(request.bindings ?? [], resources, diagnostics);
		const nodes = collectNodes(request.nodes, diagnostics);
		const exports = collectExports(request.exports ?? [], resources, diagnostics);
		const portResolutions = collectPortResolutions(request.portResolutions ?? [
			...(request.imports ?? []).map((port) => ({
				direction: "import" as const,
				port: port.name,
				resource: port.resource,
			})),
			...(request.outputPorts ?? []).map((port) => ({
				direction: "export" as const,
				port: port.name,
				resource: port.resource,
			})),
		], resources, diagnostics);
		const declarationIndex = new Map<string, number>();
		for (let index = 0; index < nodes.length; index++) {
			declarationIndex.set(nodes[index].id, index);
		}
		const explicitDependencies: RenderGraphDependency[] = [];
		const explicitOrder = stableOrder(
			nodes,
			declarationIndex,
			diagnostics,
			explicitDependencies,
		);
		const fullAnalyzer = new RenderGraphAnalyzer<TPayload, TKind>();
		fullAnalyzer.reset(resources, bindings);
		fullAnalyzer.analyzeNodes(explicitOrder);
		diagnostics.push(...fullAnalyzer.getDiagnostics());
		shadowDiagnostics.push(...fullAnalyzer.getShadowDiagnostics());
		const inferredDependencies = inferDependencies(fullAnalyzer.getTransitions());
		const dependencies = deduplicateDependencies([
			...explicitDependencies,
			...inferredDependencies,
		]);
		appendPhysicalFeedbackDiagnostics(
			explicitOrder,
			resources,
			fullAnalyzer.getTransitions(),
			diagnostics,
		);
		appendUsageDiagnostics(explicitOrder, resources, diagnostics);
		appendMissingBindingDiagnostics(
			explicitOrder,
			resources,
			bindings,
			shadowDiagnostics,
		);
		appendMetadataShadowDiagnostics(explicitOrder, resources, shadowDiagnostics);

		const hasErrors = diagnostics.some(
			(diagnostic) => diagnostic.enforcement === "enforced" && diagnostic.severity === "error",
		);
		const retainedIds = hasErrors
			? new Set(explicitOrder.map((node) => node.id))
			: collectReachableNodeIds(explicitOrder, dependencies, exports, resources);
		const retainedNodes = explicitOrder.filter((node) => retainedIds.has(node.id));
		const culledNodeIds = explicitOrder
			.filter((node) => !retainedIds.has(node.id))
			.map((node) => node.id);
		const retainedAnalyzer = new RenderGraphAnalyzer<TPayload, TKind>();
		retainedAnalyzer.reset(resources, bindings);
		retainedAnalyzer.analyzeNodes(retainedNodes);
		const retainedDiagnostics = retainedAnalyzer.getDiagnostics();
		for (const diagnostic of retainedDiagnostics) {
			if (!hasMatchingDiagnostic(diagnostics, diagnostic)) diagnostics.push(diagnostic);
		}
		for (const diagnostic of retainedAnalyzer.getShadowDiagnostics()) {
			if (!hasMatchingDiagnostic(shadowDiagnostics, diagnostic)) shadowDiagnostics.push(diagnostic);
		}
		const liveRanges = retainedAnalyzer.getLiveRanges();
		const transitions = retainedAnalyzer.getTransitions();
		const completeness = resolveCompleteness(
			request.completeness ?? "complete",
			retainedNodes,
			shadowDiagnostics,
		);
		const subresourceLiveRanges = createSubresourceLiveRanges(transitions, liveRanges);
		const allocationRequests = createAllocationRequests(resources, bindings, liveRanges);
		return freezeCompiledGraph({
			declaredNodes: nodes,
			nodes: retainedNodes,
			culledNodeIds,
			stages: groupStages(retainedNodes),
			resources,
			bindings,
			exports,
			portResolutions,
			dependencies,
			diagnostics,
			shadowDiagnostics,
			transitions,
			liveRanges,
			subresourceLiveRanges,
			allocationRequests,
			completeness,
		});
	}
}

function collectResources(
	input: readonly RenderGraphResourceDescriptor[],
	diagnostics: RenderGraphDiagnostic[],
	shadowDiagnostics: RenderGraphDiagnostic[],
): RenderGraphResourceDescriptor[] {
	const resources = new Map<string, RenderGraphResourceDescriptor>();
	for (const descriptor of input) {
		if (resources.has(descriptor.id)) {
			diagnostics.push(errorDiagnostic(
				"duplicate-resource",
				`Render graph declares resource "${descriptor.id}" more than once.`,
				{ resourceId: descriptor.id },
			));
			continue;
		}
		resources.set(descriptor.id, descriptor);
		const issue = validateDescriptor(descriptor);
		if (!issue) continue;
		const target = descriptor.origin === "graph" ? diagnostics : shadowDiagnostics;
		target.push({
			phase: "compile",
			enforcement: descriptor.origin === "graph" ? "enforced" : "shadow",
			severity: descriptor.origin === "graph" ? "error" : "warning",
			code: "invalid-resource-descriptor",
			resourceId: descriptor.id,
			message: issue,
		});
	}
	return Array.from(resources.values());
}

function validateDescriptor(descriptor: RenderGraphResourceDescriptor): string | null {
	if (!("kind" in descriptor) || !descriptor.kind) return null;
	if (descriptor.kind === "external") {
		return descriptor.origin === "graph"
			? `Graph-owned external resource "${descriptor.id}" is not allocatable.`
			: null;
	}
	if (descriptor.kind === "buffer") {
		if (descriptor.size !== undefined && (!Number.isSafeInteger(descriptor.size) || descriptor.size <= 0)) {
			return `Buffer resource "${descriptor.id}" has an invalid byte size.`;
		}
		if (descriptor.origin === "graph" && descriptor.size === undefined) {
			return `Graph-owned buffer resource "${descriptor.id}" requires a byte size.`;
		}
		return null;
	}
	const numericValues = [
		descriptor.width,
		descriptor.height,
		descriptor.depthOrArrayLayers,
		descriptor.sampleCount,
		descriptor.mipLevelCount,
	].filter((value): value is number => value !== undefined);
	if (numericValues.some((value) => !Number.isSafeInteger(value) || value <= 0)) {
		return `Texture resource "${descriptor.id}" has invalid dimensions or counts.`;
	}
	if (
		descriptor.origin === "graph" &&
		(!descriptor.format || descriptor.width === undefined || descriptor.height === undefined)
	) {
		return `Graph-owned texture resource "${descriptor.id}" requires format, width, and height.`;
	}
	return null;
}

function collectBindings(
	input: readonly RenderGraphPhysicalBinding[],
	resources: readonly RenderGraphResourceDescriptor[],
	diagnostics: RenderGraphDiagnostic[],
): RenderGraphPhysicalBinding[] {
	const descriptors = new Map(resources.map((resource) => [resource.id, resource]));
	const bindings = new Map<string, RenderGraphPhysicalBinding>();
	const descriptorsByPhysical = new Map<string, RenderGraphResourceDescriptor>();
	for (const binding of input) {
		if (
			!binding.physicalId ||
			(binding.generation !== undefined &&
				(!Number.isSafeInteger(binding.generation) || binding.generation < 0))
		) {
			diagnostics.push(errorDiagnostic(
				"invalid-physical-binding",
				`Render graph physical binding for "${binding.resourceId}" is invalid.`,
				{ resourceId: binding.resourceId },
			));
			continue;
		}
		const key = `${binding.resourceId}\u0000${binding.generation ?? "*"}`;
		if (bindings.has(key)) {
			diagnostics.push(errorDiagnostic(
				"duplicate-binding",
				`Render graph declares physical binding for "${binding.resourceId}" more than once.`,
				{ resourceId: binding.resourceId },
			));
			continue;
		}
		const descriptor = descriptors.get(binding.resourceId);
		if (!descriptor) {
			diagnostics.push(errorDiagnostic(
				"missing-resource",
				`Physical binding references undeclared resource "${binding.resourceId}".`,
				{ resourceId: binding.resourceId },
			));
			continue;
		}
		if (descriptor.kind !== binding.kind) {
			diagnostics.push(errorDiagnostic(
				"binding-kind-mismatch",
				`Physical binding kind "${binding.kind}" does not match logical resource ` +
					`"${binding.resourceId}" kind "${descriptor.kind}".`,
				{ resourceId: binding.resourceId },
			));
			continue;
		}
		const physicalDescriptor = descriptorsByPhysical.get(binding.physicalId);
		if (
			physicalDescriptor &&
			!arePhysicalDescriptorsCompatible(physicalDescriptor, descriptor)
		) {
			diagnostics.push(errorDiagnostic(
				"physical-descriptor-conflict",
				`Physical resource "${binding.physicalId}" is bound to incompatible ` +
					`logical descriptors "${physicalDescriptor.id}" and "${descriptor.id}".`,
				{ resourceId: binding.resourceId },
			));
			continue;
		}
		descriptorsByPhysical.set(binding.physicalId, physicalDescriptor ?? descriptor);
		bindings.set(key, binding);
	}
	return Array.from(bindings.values());
}

function arePhysicalDescriptorsCompatible(
	left: RenderGraphResourceDescriptor,
	right: RenderGraphResourceDescriptor,
): boolean {
	if (left.kind !== right.kind) return false;
	if (left.kind === "external" || right.kind === "external") return true;
	if (left.kind === "buffer" && right.kind === "buffer") {
		return left.size === undefined || right.size === undefined || left.size === right.size;
	}
	if (left.kind !== "texture" || right.kind !== "texture") return false;
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
		left[field] === undefined || right[field] === undefined || left[field] === right[field],
	);
}

function collectNodes<TPayload, TKind extends string>(
	input: readonly RenderGraphNode<TPayload, TKind>[],
	diagnostics: RenderGraphDiagnostic[],
): RenderGraphNode<TPayload, TKind>[] {
	const nodes = new Map<string, RenderGraphNode<TPayload, TKind>>();
	for (const node of input) {
		if (nodes.has(node.id)) {
			diagnostics.push(errorDiagnostic(
				"duplicate-node",
				`Render graph declares node "${node.id}" more than once.`,
				{ stage: node.stage, nodeId: node.id },
			));
			continue;
		}
		nodes.set(node.id, node);
	}
	return Array.from(nodes.values());
}

function collectExports(
	input: readonly RenderGraphExport[],
	resources: readonly RenderGraphResourceDescriptor[],
	diagnostics: RenderGraphDiagnostic[],
): RenderGraphExport[] {
	const descriptors = new Map(resources.map((resource) => [resource.id, resource]));
	const names = new Set<string>();
	const exports: RenderGraphExport[] = [];
	for (const entry of input) {
		const descriptor = descriptors.get(entry.resource);
		if (!descriptor) {
			diagnostics.push(errorDiagnostic(
				"missing-resource",
				`Render graph export references undeclared resource "${entry.resource}".`,
				{ resourceId: entry.resource },
			));
			continue;
		}
		const normalized = normalizeRenderGraphSubresource(descriptor, entry.subresource, {});
		if (normalized.diagnostic) diagnostics.push(normalized.diagnostic);
		if (entry.name && names.has(entry.name)) {
			diagnostics.push(errorDiagnostic(
				"duplicate-subgraph-port",
				`Render graph declares export port "${entry.name}" more than once.`,
				{ resourceId: entry.resource },
			));
			continue;
		}
		if (entry.name) names.add(entry.name);
		exports.push(entry);
	}
	return exports;
}

function collectPortResolutions(
	input: readonly {
		readonly namespace?: string;
		readonly direction: "import" | "export";
		readonly port: string;
		readonly resource: string;
	}[],
	resources: readonly RenderGraphResourceDescriptor[],
	diagnostics: RenderGraphDiagnostic[],
) {
	const resourceIds = new Set(resources.map((resource) => resource.id));
	const keys = new Set<string>();
	return input.filter((resolution) => {
		const key = `${resolution.namespace ?? ""}\u0000${resolution.direction}\u0000${resolution.port}`;
		if (keys.has(key)) {
			diagnostics.push(errorDiagnostic(
				"duplicate-subgraph-port",
				`Render graph resolves port "${resolution.port}" more than once.`,
				{ resourceId: resolution.resource },
			));
			return false;
		}
		keys.add(key);
		if (!resourceIds.has(resolution.resource)) {
			diagnostics.push(errorDiagnostic(
				"missing-subgraph-port",
				`Render graph port "${resolution.port}" maps to undeclared resource ` +
					`"${resolution.resource}".`,
				{ resourceId: resolution.resource },
			));
			return false;
		}
		return true;
	});
}

function stableOrder<TPayload, TKind extends string>(
	nodes: readonly RenderGraphNode<TPayload, TKind>[],
	declarationIndex: ReadonlyMap<string, number>,
	diagnostics: RenderGraphDiagnostic[],
	dependencies: RenderGraphDependency[],
): RenderGraphNode<TPayload, TKind>[] {
	const byId = new Map(nodes.map((node) => [node.id, node]));
	const dependents = new Map<string, string[]>();
	const indegree = new Map(nodes.map((node) => [node.id, 0]));
	for (const node of nodes) {
		for (const dependencyId of node.dependsOn ?? []) {
			if (!byId.has(dependencyId)) {
				diagnostics.push(errorDiagnostic(
					"missing-dependency",
					`Render graph node "${node.id}" depends on missing node "${dependencyId}".`,
					{ stage: node.stage, nodeId: node.id },
				));
				continue;
			}
			indegree.set(node.id, (indegree.get(node.id) ?? 0) + 1);
			const entries = dependents.get(dependencyId) ?? [];
			entries.push(node.id);
			dependents.set(dependencyId, entries);
			dependencies.push({
				fromNodeId: dependencyId,
				toNodeId: node.id,
				kind: "explicit",
			});
		}
	}
	const compare = (
		left: RenderGraphNode<TPayload, TKind>,
		right: RenderGraphNode<TPayload, TKind>,
	): number => (declarationIndex.get(left.id) ?? 0) - (declarationIndex.get(right.id) ?? 0);
	const ready = nodes.filter((node) => (indegree.get(node.id) ?? 0) === 0).sort(compare);
	const order: RenderGraphNode<TPayload, TKind>[] = [];
	while (ready.length > 0) {
		const node = ready.shift()!;
		order.push(node);
		for (const dependentId of dependents.get(node.id) ?? []) {
			const next = (indegree.get(dependentId) ?? 0) - 1;
			indegree.set(dependentId, next);
			if (next === 0) {
				ready.push(byId.get(dependentId)!);
				ready.sort(compare);
			}
		}
	}
	if (order.length !== nodes.length) {
		for (const node of nodes) {
			if ((indegree.get(node.id) ?? 0) <= 0) continue;
			diagnostics.push(errorDiagnostic(
				"cycle",
				`Render graph dependency cycle includes node "${node.id}".`,
				{ stage: node.stage, nodeId: node.id },
			));
			order.push(node);
		}
	}
	return order;
}

function inferDependencies(
	transitions: readonly RenderGraphTransition[],
): RenderGraphDependency[] {
	const dependencies: RenderGraphDependency[] = [];
	const history = new Map<string, RenderGraphTransition[]>();
	for (const transition of transitions) {
		const key = transition.physicalId ??
			`logical:${transition.resourceId}\u0000${transition.generation}`;
		const previous = history.get(key) ?? [];
		if (
			transition.reason === "usage-transition" &&
			transition.fromNodeId && transition.fromNodeId !== transition.nodeId
		) {
			dependencies.push(createInferredDependency(
				transition.fromNodeId,
				transition,
				"usage-transition",
			));
		}
		const currentWrites = transition.access !== "read";
		const currentReads = transition.access !== "write";
		if (!currentWrites) {
			for (let index = previous.length - 1; index >= 0; index--) {
				const candidate = previous[index];
				if (!renderGraphSubresourcesOverlap(candidate.subresource, transition.subresource)) {
					continue;
				}
				if (candidate.access === "read") continue;
				if (candidate.nodeId !== transition.nodeId) {
					dependencies.push(createInferredDependency(
						candidate.nodeId,
						transition,
						"read-after-write",
					));
				}
				break;
			}
		} else {
			for (let index = previous.length - 1; index >= 0; index--) {
				const candidate = previous[index];
				if (!renderGraphSubresourcesOverlap(candidate.subresource, transition.subresource)) {
					continue;
				}
				const candidateWrites = candidate.access !== "read";
				if (candidate.nodeId !== transition.nodeId) {
					if (!candidateWrites) {
						dependencies.push(createInferredDependency(
							candidate.nodeId,
							transition,
							"write-after-read",
						));
					} else {
						dependencies.push(createInferredDependency(
							candidate.nodeId,
							transition,
							"write-after-write",
						));
					}
				}
				if (candidateWrites) break;
			}
			if (currentReads) {
				for (let index = previous.length - 1; index >= 0; index--) {
					const candidate = previous[index];
					if (
						candidate.access === "read" ||
						!renderGraphSubresourcesOverlap(candidate.subresource, transition.subresource)
					) {
						continue;
					}
					if (candidate.nodeId !== transition.nodeId) {
						dependencies.push(createInferredDependency(
							candidate.nodeId,
							transition,
							"read-after-write",
						));
					}
					break;
				}
			}
		}
		previous.push(transition);
		history.set(key, previous);
	}
	return dependencies;
}

function createInferredDependency(
	fromNodeId: string,
	transition: RenderGraphTransition,
	kind: RenderGraphDependency["kind"],
): RenderGraphDependency {
	return {
		fromNodeId,
		toNodeId: transition.nodeId,
		kind,
		resourceId: transition.resourceId,
		physicalId: transition.physicalId,
		subresource: transition.subresource,
	};
}

function deduplicateDependencies(
	input: readonly RenderGraphDependency[],
): RenderGraphDependency[] {
	const seen = new Set<string>();
	return input.filter((dependency) => {
		const key = `${dependency.fromNodeId}\u0000${dependency.toNodeId}\u0000${dependency.kind}` +
			`\u0000${dependency.physicalId ?? dependency.resourceId ?? ""}` +
			`\u0000${dependency.subresource ? JSON.stringify(dependency.subresource) : "full"}`;
		if (seen.has(key)) return false;
		seen.add(key);
		return true;
	});
}

function collectReachableNodeIds<TPayload, TKind extends string>(
	nodes: readonly RenderGraphNode<TPayload, TKind>[],
	dependencies: readonly RenderGraphDependency[],
	exports: readonly RenderGraphExport[],
	resources: readonly RenderGraphResourceDescriptor[],
): Set<string> {
	const reachable = new Set<string>();
	const pending: string[] = [];
	for (const node of nodes) {
		if (mustAlwaysRetain(node, resources)) {
			reachable.add(node.id);
			pending.push(node.id);
		}
	}
	for (const exported of exports) {
		const descriptor = resources.find((resource) => resource.id === exported.resource);
		const normalizedExport = descriptor
			? normalizeRenderGraphSubresource(descriptor, exported.subresource, {}).range
			: undefined;
		for (let index = nodes.length - 1; index >= 0; index--) {
			const writes = (nodes[index].resources ?? []).some((ref) => {
				if (ref.resource !== exported.resource || ref.access === "read") return false;
				const normalizedRef = descriptor
					? normalizeRenderGraphSubresource(descriptor, ref.subresource, {}).range
					: undefined;
				return renderGraphSubresourcesOverlap(normalizedRef, normalizedExport);
			});
			if (!writes) continue;
			if (!reachable.has(nodes[index].id)) {
				reachable.add(nodes[index].id);
				pending.push(nodes[index].id);
			}
			break;
		}
	}
	const predecessors = new Map<string, string[]>();
	for (const dependency of dependencies) {
		const entries = predecessors.get(dependency.toNodeId) ?? [];
		entries.push(dependency.fromNodeId);
		predecessors.set(dependency.toNodeId, entries);
	}
	while (pending.length > 0) {
		const nodeId = pending.pop()!;
		for (const predecessor of predecessors.get(nodeId) ?? []) {
			if (reachable.has(predecessor)) continue;
			reachable.add(predecessor);
			pending.push(predecessor);
		}
	}
	return reachable;
}

function mustAlwaysRetain<TPayload, TKind extends string>(
	node: RenderGraphNode<TPayload, TKind>,
	resources: readonly RenderGraphResourceDescriptor[],
): boolean {
	if ((node.retention ?? "always") === "always" || node.opaque || node.domain === "cpu") {
		return true;
	}
	const descriptors = new Map(resources.map((resource) => [resource.id, resource]));
	return (node.resources ?? []).some((reference) =>
		reference.usage === "present" ||
		(reference.access !== "read" &&
			(descriptors.get(reference.resource)?.residency === "history" ||
				descriptors.get(reference.resource)?.residency === "external")),
	);
}

function appendPhysicalFeedbackDiagnostics<TPayload, TKind extends string>(
	nodes: readonly RenderGraphNode<TPayload, TKind>[],
	resources: readonly RenderGraphResourceDescriptor[],
	transitions: readonly RenderGraphTransition[],
	diagnostics: RenderGraphDiagnostic[],
): void {
	const descriptorById = new Map(resources.map((resource) => [resource.id, resource]));
	const transitionsByNode = new Map<string, RenderGraphTransition[]>();
	for (const transition of transitions) {
		const entries = transitionsByNode.get(transition.nodeId) ?? [];
		entries.push(transition);
		transitionsByNode.set(transition.nodeId, entries);
	}
	for (const node of nodes) {
		if (node.internalAccesses === "ordered") continue;
		const sampled = new Map<string, string>();
		for (const transition of transitionsByNode.get(node.id) ?? []) {
			if (transition.access === "write" || transition.usage !== "sampled") continue;
			const physicalId = transition.physicalId ??
				`logical:${transition.resourceId}:${transition.generation}`;
			sampled.set(physicalId, transition.resourceId);
		}
		for (const transition of transitionsByNode.get(node.id) ?? []) {
			if (transition.access === "read" || transition.usage === "copy-target") continue;
			const physicalId = transition.physicalId ??
				`logical:${transition.resourceId}:${transition.generation}`;
			if (!sampled.has(physicalId)) continue;
			if (descriptorById.get(transition.resourceId)?.kind !== "texture") continue;
			diagnostics.push({
				phase: "lower",
				enforcement: "enforced",
				severity: "error",
				code: "physical-feedback-loop",
				stage: node.stage,
				nodeId: node.id,
				resourceId: transition.resourceId,
				message:
					`Render graph node "${node.id}" samples and writes physical texture ` +
					`"${physicalId}" in one node.`,
			});
		}
	}
}

function appendUsageDiagnostics<TPayload, TKind extends string>(
	nodes: readonly RenderGraphNode<TPayload, TKind>[],
	resources: readonly RenderGraphResourceDescriptor[],
	diagnostics: RenderGraphDiagnostic[],
): void {
	const descriptors = new Map(resources.map((resource) => [resource.id, resource]));
	for (const node of nodes) {
		for (const ref of node.resources ?? []) {
			const descriptor = descriptors.get(ref.resource);
			if (!descriptor?.allowedUsages || descriptor.allowedUsages.includes(ref.usage)) continue;
			diagnostics.push(errorDiagnostic(
				"unsupported-node-resource",
				`Render graph node "${node.id}" uses resource "${ref.resource}" as ` +
					`"${ref.usage}", which is not declared by its descriptor.`,
				{ stage: node.stage, nodeId: node.id, resourceId: ref.resource },
			));
		}
	}
}

function appendMissingBindingDiagnostics<TPayload, TKind extends string>(
	nodes: readonly RenderGraphNode<TPayload, TKind>[],
	resources: readonly RenderGraphResourceDescriptor[],
	bindings: readonly RenderGraphPhysicalBinding[],
	shadowDiagnostics: RenderGraphDiagnostic[],
): void {
	const bound = new Set(bindings.map((binding) => binding.resourceId));
	const referenced = new Set(nodes.flatMap((node) =>
		(node.resources ?? []).filter((ref) => !ref.optional).map((ref) => ref.resource),
	));
	for (const descriptor of resources) {
		if (
			!("kind" in descriptor) || !descriptor.kind ||
			descriptor.origin !== "imported" || bound.has(descriptor.id) ||
			!referenced.has(descriptor.id)
		) {
			continue;
		}
		shadowDiagnostics.push({
			phase: "compile",
			enforcement: "shadow",
			severity: "warning",
			code: "missing-binding",
			resourceId: descriptor.id,
			message: `Imported resource "${descriptor.id}" has no stable physical binding.`,
		});
	}
}

function appendMetadataShadowDiagnostics<TPayload, TKind extends string>(
	nodes: readonly RenderGraphNode<TPayload, TKind>[],
	resources: readonly RenderGraphResourceDescriptor[],
	shadowDiagnostics: RenderGraphDiagnostic[],
): void {
	const referenced = new Set(nodes.flatMap((node) =>
		(node.resources ?? []).map((reference) => reference.resource),
	));
	for (const descriptor of resources) {
		if (
			descriptor.kind !== "external" ||
			descriptor.residency !== "external" ||
			!referenced.has(descriptor.id)
		) {
			continue;
		}
		shadowDiagnostics.push({
			phase: "compile",
			enforcement: "shadow",
			severity: "warning",
			code: "external-metadata-unknown",
			resourceId: descriptor.id,
			message: `External resource "${descriptor.id}" has backend-owned metadata ` +
				"that cannot be validated by the logical graph.",
		});
	}
}

function createSubresourceLiveRanges(
	transitions: readonly RenderGraphTransition[],
	liveRanges: readonly RenderGraphLiveRange[],
): RenderGraphSubresourceLiveRange[] {
	const lifetimeByKey = new Map(
		liveRanges.map((range) => [`${range.resourceId}\u0000${range.generation}`, range]),
	);
	const ranges = new Map<string, RenderGraphSubresourceLiveRange>();
	for (const transition of transitions) {
		if (!transition.subresource) continue;
		const serialized = JSON.stringify(transition.subresource);
		const key = `${transition.resourceId}\u0000${transition.generation}\u0000${serialized}`;
		const existing = ranges.get(key);
		if (existing) {
			ranges.set(key, { ...existing, lastNodeId: transition.nodeId, lastUseNodeId: transition.nodeId });
			continue;
		}
		const lifetime = lifetimeByKey.get(`${transition.resourceId}\u0000${transition.generation}`);
		ranges.set(key, {
			resourceId: transition.resourceId,
			generation: transition.generation,
			firstNodeId: transition.nodeId,
			lastNodeId: transition.nodeId,
			createdByNodeId: lifetime?.createdByNodeId,
			firstUseNodeId: transition.nodeId,
			lastUseNodeId: transition.nodeId,
			destroyedByNodeId: lifetime?.destroyedByNodeId,
			subresource: transition.subresource,
		});
	}
	return Array.from(ranges.values());
}

function createAllocationRequests(
	resources: readonly RenderGraphResourceDescriptor[],
	bindings: readonly RenderGraphPhysicalBinding[],
	liveRanges: readonly RenderGraphLiveRange[],
): RenderGraphAllocationRequest[] {
	const descriptors = new Map(resources.map((resource) => [resource.id, resource]));
	const bound = new Set(bindings.map((binding) =>
		`${binding.resourceId}\u0000${binding.generation ?? "*"}`,
	));
	const requests: RenderGraphAllocationRequest[] = [];
	for (const range of liveRanges) {
		const descriptor = descriptors.get(range.resourceId);
		if (!descriptor || descriptor.origin !== "graph") continue;
		if (descriptor.residency !== "frame" && descriptor.residency !== "transient") continue;
		if (
			bound.has(`${range.resourceId}\u0000${range.generation}`) ||
			bound.has(`${range.resourceId}\u0000*`)
		) {
			continue;
		}
		requests.push({
			resourceId: range.resourceId,
			generation: range.generation,
			compatibilityKey: createRenderGraphCompatibilityKey(descriptor),
			allocateBeforeNodeId: range.createdByNodeId ?? range.firstNodeId,
			releaseAfterNodeId: range.destroyedByNodeId ?? range.lastNodeId,
		});
	}
	return requests;
}

function groupStages<TPayload, TKind extends string>(
	nodes: readonly RenderGraphNode<TPayload, TKind>[],
): CompiledRenderGraphStage<TPayload, TKind>[] {
	const stages: CompiledRenderGraphStage<TPayload, TKind>[] = [];
	const byStage = new Map<string, RenderGraphNode<TPayload, TKind>[]>();
	for (const node of nodes) {
		let entries = byStage.get(node.stage);
		if (!entries) {
			entries = [];
			byStage.set(node.stage, entries);
			stages.push({ stage: node.stage, nodes: entries });
		}
		entries.push(node);
	}
	return stages;
}

function resolveCompleteness<TPayload, TKind extends string>(
	initial: CompiledRenderGraph<TPayload, TKind>["completeness"],
	nodes: readonly RenderGraphNode<TPayload, TKind>[],
	shadowDiagnostics: readonly RenderGraphDiagnostic[],
): CompiledRenderGraph<TPayload, TKind>["completeness"] {
	if (nodes.some((node) => node.opaque) || shadowDiagnostics.some(
		(diagnostic) => diagnostic.code === "opaque-stage-effects",
	)) {
		return "opaque";
	}
	if (initial === "opaque") return "opaque";
	if (
		initial === "coarse" ||
		shadowDiagnostics.some((diagnostic) =>
			diagnostic.code === "missing-binding" ||
			diagnostic.code === "invalid-resource-descriptor" ||
			diagnostic.code === "external-metadata-unknown" ||
			diagnostic.code === "read-content-unknown",
		)
	) {
		return "coarse";
	}
	return "complete";
}

function errorDiagnostic(
	code: string,
	message: string,
	context: Partial<Pick<RenderGraphDiagnostic, "stage" | "nodeId" | "resourceId">>,
): RenderGraphDiagnostic {
	return {
		phase: "compile",
		enforcement: "enforced",
		severity: "error",
		code,
		message,
		...context,
	};
}

function hasMatchingDiagnostic(
	diagnostics: readonly RenderGraphDiagnostic[],
	candidate: RenderGraphDiagnostic,
): boolean {
	return diagnostics.some((diagnostic) =>
		diagnostic.code === candidate.code &&
		diagnostic.nodeId === candidate.nodeId &&
		diagnostic.resourceId === candidate.resourceId,
	);
}

function freezeCompiledGraph<TPayload, TKind extends string>(
	graph: CompiledRenderGraph<TPayload, TKind>,
): CompiledRenderGraph<TPayload, TKind> {
	return Object.freeze({
		...graph,
		declaredNodes: freezeNodes(graph.declaredNodes),
		nodes: freezeNodes(graph.nodes),
		culledNodeIds: Object.freeze(graph.culledNodeIds.slice()),
		stages: Object.freeze(graph.stages.map((stage) => Object.freeze({
			stage: stage.stage,
			nodes: freezeNodes(stage.nodes),
		}))),
		resources: freezeItems(graph.resources),
		bindings: freezeItems(graph.bindings),
		exports: Object.freeze(graph.exports.map((entry) => Object.freeze({
			...entry,
			subresource: entry.subresource ? Object.freeze({ ...entry.subresource }) : undefined,
		}))),
		portResolutions: freezeItems(graph.portResolutions),
		dependencies: freezeItems(graph.dependencies),
		diagnostics: freezeItems(graph.diagnostics),
		shadowDiagnostics: freezeItems(graph.shadowDiagnostics),
		transitions: freezeItems(graph.transitions),
		liveRanges: freezeItems(graph.liveRanges),
		subresourceLiveRanges: freezeItems(graph.subresourceLiveRanges),
		allocationRequests: freezeItems(graph.allocationRequests),
	});
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
		resources: node.resources ? Object.freeze(node.resources.map((ref) => Object.freeze({
			...ref,
			subresource: ref.subresource ? Object.freeze({ ...ref.subresource }) : undefined,
		}))) : undefined,
	})));
}

function freezeMutations(
	mutations: readonly RenderGraphResourceMutation[],
): readonly RenderGraphResourceMutation[] {
	return Object.freeze(mutations.map((mutation) =>
		typeof mutation === "string" ? mutation : Object.freeze({ ...mutation }),
	));
}
