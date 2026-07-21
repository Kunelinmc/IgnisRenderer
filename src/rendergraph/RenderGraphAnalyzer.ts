import type {
	RenderGraphAccess,
	RenderGraphAnalysisCompleteness,
	RenderGraphAnalysisSnapshot,
	RenderGraphAnalyzedStage,
	RenderGraphContentState,
	RenderGraphDiagnostic,
	RenderGraphLiveRange,
	RenderGraphNode,
	RenderGraphNormalizedSubresourceRange,
	RenderGraphPhysicalBinding,
	RenderGraphResourceDebugState,
	RenderGraphResourceDescriptor,
	RenderGraphResourceId,
	RenderGraphResourceMutation,
	RenderGraphResourceRef,
	RenderGraphTransition,
	RenderGraphUsage,
	RenderGraphValidationRule,
} from "./types";
import {
	normalizeRenderGraphSubresource,
	renderGraphSubresourcesOverlap,
} from "./subresources";

interface MutableRenderGraphLiveRange {
	resourceId: RenderGraphResourceId;
	generation: number;
	firstNodeId: string;
	lastNodeId: string;
	createdByNodeId?: string;
	firstUseNodeId?: string;
	lastUseNodeId?: string;
	destroyedByNodeId?: string;
}

interface RenderGraphAnalyzerResourceState {
	descriptor: RenderGraphResourceDescriptor;
	active: boolean;
	generation: number;
	everActivated: boolean;
	content: RenderGraphContentState;
	lastNodeId: string | null;
	lastEvent: RenderGraphResourceDebugState["lastAccess"];
	lastAccess: RenderGraphAccess | null;
	lastUsage: RenderGraphUsage | null;
	wasDestroyed: boolean;
	unknownReadReported: boolean;
	undefinedReadReported: boolean;
	implicitCreateReported: boolean;
}

interface RenderGraphAnalyzerAccessState {
	readonly nodeId: string;
	readonly resourceId: RenderGraphResourceId;
	readonly generation: number;
	readonly access: RenderGraphAccess;
	readonly usage: RenderGraphUsage;
	readonly subresource: RenderGraphNormalizedSubresourceRange | undefined;
}

export interface RenderGraphAnalyzerOptions<TPayload = unknown, TKind extends string = string> {
	readonly allowImplicitResources?: boolean;
	readonly validateStreamingDependencies?: boolean;
	readonly rules?: readonly RenderGraphValidationRule<TPayload, TKind>[];
}

/** @internal Shared logical lifetime, transition, and diagnostic state machine. */
export class RenderGraphAnalyzer<TPayload = unknown, TKind extends string = string> {
	private readonly _allowImplicitResources: boolean;
	private readonly _validateStreamingDependencies: boolean;
	private readonly _rules: readonly RenderGraphValidationRule<TPayload, TKind>[];
	private readonly _resources = new Map<
		RenderGraphResourceId,
		RenderGraphAnalyzerResourceState
	>();
	private readonly _transitions: RenderGraphTransition[] = [];
	private readonly _diagnostics: RenderGraphDiagnostic[] = [];
	private readonly _liveRanges: MutableRenderGraphLiveRange[] = [];
	private readonly _liveRangeByGeneration = new Map<string, MutableRenderGraphLiveRange>();
	private readonly _nodeIds = new Set<string>();
	private readonly _orderedNodeIds: string[] = [];
	private readonly _bindings = new Map<string, RenderGraphPhysicalBinding>();
	private readonly _accessHistory = new Map<string, RenderGraphAnalyzerAccessState[]>();
	private _completeness: RenderGraphAnalysisCompleteness = "complete";

	constructor(options: RenderGraphAnalyzerOptions<TPayload, TKind> = {}) {
		this._allowImplicitResources = options.allowImplicitResources === true;
		this._validateStreamingDependencies = options.validateStreamingDependencies === true;
		this._rules = options.rules ?? [];
	}

	public reset(
		resources: readonly RenderGraphResourceDescriptor[],
		bindings: readonly RenderGraphPhysicalBinding[] = [],
	): void {
		this._resources.clear();
		this._transitions.length = 0;
		this._diagnostics.length = 0;
		this._liveRanges.length = 0;
		this._liveRangeByGeneration.clear();
		this._nodeIds.clear();
		this._orderedNodeIds.length = 0;
		this._bindings.clear();
		this._accessHistory.clear();
		this._completeness = "complete";
		for (const binding of bindings) {
			this._bindings.set(bindingKey(binding.resourceId, binding.generation), binding);
		}
		for (const descriptor of resources) {
			if (this._resources.has(descriptor.id)) continue;
			const active = descriptor.origin === "imported";
			this._resources.set(descriptor.id, {
				descriptor,
				active,
				generation: 0,
				everActivated: active,
				content:
					descriptor.initialContent ??
					(descriptor.origin === "imported" ? "valid" : "undefined"),
				lastNodeId: null,
				lastEvent: active ? "create" : null,
				lastAccess: null,
				lastUsage: null,
				wasDestroyed: false,
				unknownReadReported: false,
				undefinedReadReported: false,
				implicitCreateReported: false,
			});
		}
	}

	public analyzeNodes(
		nodes: readonly RenderGraphNode<TPayload, TKind>[],
	): RenderGraphAnalyzedStage<TPayload, TKind> {
		const transitionStart = this._transitions.length;
		const diagnosticStart = this._diagnostics.length;
		for (const node of nodes) this._analyzeNode(node);
		const diagnostics = this._diagnostics.slice(diagnosticStart);
		return freezeAnalyzedStage({
			nodes,
			diagnostics: diagnostics.filter((diagnostic) => diagnostic.enforcement === "enforced"),
			shadowDiagnostics: diagnostics.filter(
				(diagnostic) => diagnostic.enforcement === "shadow",
			),
			transitions: this._transitions.slice(transitionStart),
		});
	}

	public markCompleteness(completeness: RenderGraphAnalysisCompleteness): void {
		if (completenessRank(completeness) > completenessRank(this._completeness)) {
			this._completeness = completeness;
		}
	}

	public recordOpaqueStage(stage: string, message: string): void {
		this.markCompleteness("opaque");
		this._diagnostics.push({
			phase: "compile",
			enforcement: "shadow",
			severity: "warning",
			code: "opaque-stage-effects",
			stage,
			message,
		});
	}

	public getDiagnostics(): readonly RenderGraphDiagnostic[] {
		return freezeItems(
			this._diagnostics.filter((diagnostic) => diagnostic.enforcement === "enforced"),
		);
	}

	public getShadowDiagnostics(): readonly RenderGraphDiagnostic[] {
		return freezeItems(
			this._diagnostics.filter((diagnostic) => diagnostic.enforcement === "shadow"),
		);
	}

	public getAllDiagnostics(): readonly RenderGraphDiagnostic[] {
		return freezeItems(this._diagnostics);
	}

	public getTransitions(): readonly RenderGraphTransition[] {
		return freezeItems(this._transitions);
	}

	public getLiveRanges(): readonly RenderGraphLiveRange[] {
		return freezeItems(this._liveRanges);
	}

	public getResourceDebugState(): readonly RenderGraphResourceDebugState[] {
		return Object.freeze(
			Array.from(this._resources.values())
				.sort((left, right) => left.descriptor.id.localeCompare(right.descriptor.id))
				.map((state) =>
					Object.freeze({
						id: state.descriptor.id,
						origin: state.descriptor.origin,
						active: state.active,
						generation: state.generation,
						content: state.content,
						physicalId: this._resolveBinding(state)?.physicalId,
						lastNodeId: state.lastNodeId,
						lastAccess: state.lastEvent,
						lastUsage: state.lastUsage,
					}),
				),
		);
	}

	public createSnapshot(
		state: RenderGraphAnalysisSnapshot["state"],
	): RenderGraphAnalysisSnapshot {
		return Object.freeze({
			state,
			completeness: this._completeness,
			nodeIds: Object.freeze(this._orderedNodeIds.slice()),
			resources: this.getResourceDebugState(),
			transitions: this.getTransitions(),
			liveRanges: this.getLiveRanges(),
			executionOverlay: Object.freeze({
				skippedNodeIds: Object.freeze([]),
				resourceAliases: Object.freeze([]),
			}),
			diagnostics: this.getDiagnostics(),
			shadowDiagnostics: this.getShadowDiagnostics(),
		});
	}

	private _analyzeNode(node: RenderGraphNode<TPayload, TKind>): void {
		if (this._nodeIds.has(node.id)) {
			this._diagnostics.push({
				phase: "compile",
				enforcement: "shadow",
				severity: "error",
				code: "duplicate-node",
				stage: node.stage,
				nodeId: node.id,
				message: `Render graph declares node "${node.id}" more than once.`,
			});
		}
		if (this._validateStreamingDependencies) {
			for (const dependencyId of node.dependsOn ?? []) {
				if (this._nodeIds.has(dependencyId)) continue;
				this._diagnostics.push({
					phase: "compile",
					enforcement: "shadow",
					severity: "error",
					code: "missing-dependency",
					stage: node.stage,
					nodeId: node.id,
					message:
						`Streaming render graph node "${node.id}" depends on ` +
						`unavailable prior node "${dependencyId}".`,
				});
			}
		}
		for (const rule of this._rules) {
			this._diagnostics.push(
				...rule.validateNode(node, {
					isResourceActive: (resourceId) =>
						this._resources.get(resourceId)?.active === true,
				}),
			);
		}
		for (const requirement of node.requires ?? []) {
			const state = this._resolveState(
				requirement.resource,
				requirement.optional === true,
				node,
			);
			if (!state) continue;
			if (!state.active && !requirement.optional) {
				this._diagnostics.push({
					phase: "compile",
					enforcement: "enforced",
					severity: "error",
					code: "missing-resource",
					stage: node.stage,
					nodeId: node.id,
					resourceId: requirement.resource,
					message:
						`Render graph node "${node.id}" requires inactive ` +
						`resource "${requirement.resource}".`,
				});
			}
			if (state.active) this._touchLiveRange(state, node.id, { use: true });
		}
		for (const mutation of node.creates ?? []) this._recordCreate(node, mutation);
		for (const ref of node.resources ?? []) this._recordAccess(node, ref);
		for (const mutation of node.destroys ?? []) this._recordDestroy(node, mutation);
		this._nodeIds.add(node.id);
		this._orderedNodeIds.push(node.id);
	}

	private _recordCreate(
		node: RenderGraphNode<TPayload, TKind>,
		mutation: RenderGraphResourceMutation,
	): void {
		const resolved = resolveMutation(mutation);
		const state = this._resolveState(resolved.resource, resolved.optional === true, node);
		if (!state) return;
		if (state.active && !resolved.optional) {
			this._diagnostics.push({
				phase: "compile",
				enforcement: "enforced",
				severity: "error",
				code: "duplicate-create",
				stage: node.stage,
				nodeId: node.id,
				resourceId: resolved.resource,
				message:
					`Render graph node "${node.id}" creates already active ` +
					`resource "${resolved.resource}".`,
			});
		}
		if (!state.active) state.generation++;
		state.active = true;
		state.everActivated = true;
		state.content = "undefined";
		state.lastNodeId = node.id;
		state.lastEvent = "create";
		state.lastAccess = null;
		state.lastUsage = resolved.usage ?? null;
		state.wasDestroyed = false;
		state.unknownReadReported = false;
		state.undefinedReadReported = false;
		state.implicitCreateReported = false;
		this._clearLogicalHazardHistory(state);
		this._touchLiveRange(state, node.id, { createdByNodeId: node.id });
	}

	private _recordAccess(
		node: RenderGraphNode<TPayload, TKind>,
		ref: RenderGraphResourceRef,
	): void {
		if (!ref) return;
		const state = this._resolveState(ref.resource, ref.optional === true, node);
		if (!state) return;
		if (!state.active && ref.optional) return;
		const reads = ref.access !== "write";
		const writes = ref.access !== "read";
		const normalized = normalizeRenderGraphSubresource(state.descriptor, ref.subresource, {
			nodeId: node.id,
			stage: node.stage,
		});
		if (normalized.diagnostic) {
			this._diagnostics.push(normalized.diagnostic);
			return;
		}
		const wasInactive = !state.active;
		if (reads && wasInactive) {
			this._diagnostics.push({
				phase: "compile",
				enforcement: "enforced",
				severity: "error",
				code: "read-before-create",
				stage: node.stage,
				nodeId: node.id,
				resourceId: ref.resource,
				message:
					`Render graph node "${node.id}" reads inactive ` +
					`resource "${ref.resource}".`,
			});
			if (state.wasDestroyed) {
				this._diagnostics.push({
					phase: "compile",
					enforcement: "shadow",
					severity: "error",
					code: "use-after-destroy",
					stage: node.stage,
					nodeId: node.id,
					resourceId: ref.resource,
					message:
						`Render graph node "${node.id}" uses destroyed ` +
						`resource "${ref.resource}".`,
				});
			}
		}
		if (writes && wasInactive) this._activateImplicitWrite(state, node);
		if (reads && !wasInactive) this._recordContentDiagnostic(state, node);
		const physicalId = this._resolveBinding(state)?.physicalId;
		const hazardKey = physicalId
			? physicalId
			: `${state.descriptor.id}\u0000${state.generation}`;
		const history = this._accessHistory.get(hazardKey) ?? [];
		const previous = findLastOverlappingAccess(history, normalized.range);
		const previousAccess = previous?.access;
		const previousUsage = previous?.usage;
		const hazard = resolveHazard(previousAccess, ref.access);
		const reason = resolveTransitionReason(
			previousAccess,
			previousUsage,
			ref.access,
			ref.usage,
			hazard,
		);
		this._transitions.push({
			nodeId: node.id,
			fromNodeId: previous?.nodeId,
			resourceId: ref.resource,
			physicalId,
			generation: state.generation,
			previousAccess,
			previousUsage,
			access: ref.access,
			usage: ref.usage,
			subresource: normalized.range,
			scope: !previousAccess
				? "initial"
				: previous?.nodeId === node.id
					? "intra-node"
					: "inter-node",
			hazard,
			reason,
		});
		state.lastNodeId = node.id;
		state.lastEvent = ref.access;
		state.lastAccess = ref.access;
		state.lastUsage = ref.usage;
		history.push({
			nodeId: node.id,
			resourceId: state.descriptor.id,
			generation: state.generation,
			access: ref.access,
			usage: ref.usage,
			subresource: normalized.range,
		});
		this._accessHistory.set(hazardKey, history);
		if (writes) state.content = "valid";
		this._touchLiveRange(state, node.id, { use: true });
	}

	private _recordDestroy(
		node: RenderGraphNode<TPayload, TKind>,
		mutation: RenderGraphResourceMutation,
	): void {
		const resolved = resolveMutation(mutation);
		const state = this._resolveState(resolved.resource, resolved.optional === true, node);
		if (!state) return;
		if (!state.active && !resolved.optional) {
			this._diagnostics.push({
				phase: "compile",
				enforcement: "enforced",
				severity: "error",
				code: "destroy-before-create",
				stage: node.stage,
				nodeId: node.id,
				resourceId: resolved.resource,
				message:
					`Render graph node "${node.id}" destroys inactive ` +
					`resource "${resolved.resource}".`,
			});
		}
		if (state.active) {
			const liveRange = this._touchLiveRange(state, node.id);
			liveRange.destroyedByNodeId = node.id;
		}
		state.active = false;
		state.content = "undefined";
		state.lastNodeId = node.id;
		state.lastEvent = "destroy";
		state.lastAccess = null;
		state.lastUsage = resolved.usage ?? state.lastUsage;
		state.wasDestroyed = true;
		this._clearLogicalHazardHistory(state);
	}

	private _activateImplicitWrite(
		state: RenderGraphAnalyzerResourceState,
		node: RenderGraphNode<TPayload, TKind>,
	): void {
		state.generation++;
		state.active = true;
		state.everActivated = true;
		state.content = "undefined";
		state.lastAccess = null;
		state.lastUsage = null;
		state.wasDestroyed = false;
		state.unknownReadReported = false;
		state.undefinedReadReported = false;
		this._clearLogicalHazardHistory(state);
		if (!state.implicitCreateReported) {
			this._diagnostics.push({
				phase: "compile",
				enforcement: "shadow",
				severity: "warning",
				code: "implicit-create",
				stage: node.stage,
				nodeId: node.id,
				resourceId: state.descriptor.id,
				message:
					`Render graph node "${node.id}" implicitly activates ` +
					`resource "${state.descriptor.id}" through a write.`,
			});
			state.implicitCreateReported = true;
		}
	}

	private _recordContentDiagnostic(
		state: RenderGraphAnalyzerResourceState,
		node: RenderGraphNode<TPayload, TKind>,
	): void {
		if (state.content === "unknown" && !state.unknownReadReported) {
			this._diagnostics.push({
				phase: "compile",
				enforcement: "shadow",
				severity: "warning",
				code: "read-content-unknown",
				stage: node.stage,
				nodeId: node.id,
				resourceId: state.descriptor.id,
				message:
					`Render graph node "${node.id}" reads resource ` +
					`"${state.descriptor.id}" with unknown initial contents.`,
			});
			state.unknownReadReported = true;
		}
		if (state.content === "undefined" && !state.undefinedReadReported) {
			this._diagnostics.push({
				phase: "compile",
				enforcement: "shadow",
				severity: "error",
				code: "read-before-initialize",
				stage: node.stage,
				nodeId: node.id,
				resourceId: state.descriptor.id,
				message:
					`Render graph node "${node.id}" reads resource ` +
					`"${state.descriptor.id}" before its contents are initialized.`,
			});
			state.undefinedReadReported = true;
		}
	}

	private _resolveState(
		resourceId: RenderGraphResourceId,
		optional: boolean,
		node: RenderGraphNode<TPayload, TKind>,
	): RenderGraphAnalyzerResourceState | null {
		const existing = this._resources.get(resourceId);
		if (existing) return existing;
		if (optional) return null;
		if (!this._allowImplicitResources) {
			this._diagnostics.push({
				phase: "compile",
				enforcement: "enforced",
				severity: "error",
				code: "read-before-create",
				stage: node.stage,
				nodeId: node.id,
				resourceId,
				message:
					`Render graph node "${node.id}" references undeclared ` +
					`resource "${resourceId}".`,
			});
			return null;
		}
		const descriptor: RenderGraphResourceDescriptor = {
			id: resourceId,
			origin: "graph",
			kind: "external",
			residency: "frame",
			initialContent: "undefined",
		};
		const state: RenderGraphAnalyzerResourceState = {
			descriptor,
			active: false,
			generation: 0,
			everActivated: false,
			content: "undefined",
			lastNodeId: null,
			lastEvent: null,
			lastAccess: null,
			lastUsage: null,
			wasDestroyed: false,
			unknownReadReported: false,
			undefinedReadReported: false,
			implicitCreateReported: false,
		};
		this._resources.set(resourceId, state);
		this._diagnostics.push({
			phase: "compile",
			enforcement: "shadow",
			severity: "warning",
			code: "implicit-resource-declaration",
			stage: node.stage,
			nodeId: node.id,
			resourceId,
			message:
				`Render graph implicitly declares resource "${resourceId}" ` +
				`for node "${node.id}".`,
		});
		return state;
	}

	private _touchLiveRange(
		state: RenderGraphAnalyzerResourceState,
		nodeId: string,
		options: {
			readonly createdByNodeId?: string;
			readonly use?: boolean;
		} = {},
	): MutableRenderGraphLiveRange {
		const key = `${state.descriptor.id}\u0000${state.generation}`;
		let liveRange = this._liveRangeByGeneration.get(key);
		if (!liveRange) {
			liveRange = {
				resourceId: state.descriptor.id,
				generation: state.generation,
				firstNodeId: nodeId,
				lastNodeId: nodeId,
				createdByNodeId: options.createdByNodeId,
				firstUseNodeId: options.use ? nodeId : undefined,
				lastUseNodeId: options.use ? nodeId : undefined,
			};
			this._liveRangeByGeneration.set(key, liveRange);
			this._liveRanges.push(liveRange);
		} else {
			liveRange.lastNodeId = nodeId;
			liveRange.createdByNodeId ??= options.createdByNodeId;
		}
		if (options.use) {
			liveRange.firstUseNodeId ??= nodeId;
			liveRange.lastUseNodeId = nodeId;
		}
		return liveRange;
	}

	private _resolveBinding(
		state: RenderGraphAnalyzerResourceState,
	): RenderGraphPhysicalBinding | undefined {
		return this._bindings.get(bindingKey(state.descriptor.id, state.generation)) ??
			this._bindings.get(bindingKey(state.descriptor.id, undefined));
	}

	private _clearLogicalHazardHistory(state: RenderGraphAnalyzerResourceState): void {
		const binding = this._resolveBinding(state);
		if (!binding?.physicalId) {
			this._accessHistory.delete(`${state.descriptor.id}\u0000${state.generation}`);
			return;
		}
		const history = this._accessHistory.get(binding.physicalId);
		if (!history) return;
		const retained = history.filter(
			(access) => access.resourceId !== state.descriptor.id,
		);
		if (retained.length > 0) {
			this._accessHistory.set(binding.physicalId, retained);
		} else {
			this._accessHistory.delete(binding.physicalId);
		}
	}
}

function bindingKey(resourceId: string, generation: number | undefined): string {
	return `${resourceId}\u0000${generation ?? "*"}`;
}

function findLastOverlappingAccess(
	history: readonly RenderGraphAnalyzerAccessState[],
	subresource: RenderGraphNormalizedSubresourceRange | undefined,
): RenderGraphAnalyzerAccessState | undefined {
	for (let index = history.length - 1; index >= 0; index--) {
		if (renderGraphSubresourcesOverlap(history[index].subresource, subresource)) {
			return history[index];
		}
	}
	return undefined;
}

function resolveMutation(mutation: RenderGraphResourceMutation): {
	resource: RenderGraphResourceId;
	usage?: RenderGraphUsage;
	optional?: boolean;
} {
	return typeof mutation === "string" ? { resource: mutation } : mutation;
}

function resolveHazard(
	previous: RenderGraphAccess | undefined,
	next: RenderGraphAccess,
): RenderGraphTransition["hazard"] {
	if (!previous) return undefined;
	const previousWrites = previous !== "read";
	const nextWrites = next !== "read";
	if (previousWrites && !nextWrites) return "read-after-write";
	if (!previousWrites && nextWrites) return "write-after-read";
	if (previousWrites && nextWrites) return "write-after-write";
	return undefined;
}

function resolveTransitionReason(
	previousAccess: RenderGraphAccess | undefined,
	previousUsage: RenderGraphUsage | undefined,
	nextAccess: RenderGraphAccess,
	nextUsage: RenderGraphUsage,
	hazard: RenderGraphTransition["hazard"],
): RenderGraphTransition["reason"] {
	if (!previousAccess) return undefined;
	if (hazard) return hazard;
	if (previousAccess !== nextAccess || previousUsage !== nextUsage) {
		return "usage-transition";
	}
	return undefined;
}

function completenessRank(value: RenderGraphAnalysisCompleteness): number {
	if (value === "opaque") return 2;
	if (value === "coarse") return 1;
	return 0;
}

function freezeItems<T extends object>(items: readonly T[]): readonly T[] {
	return Object.freeze(items.map((item) => Object.freeze({ ...item })));
}

function freezeAnalyzedStage<TPayload, TKind extends string>(stage: {
	readonly nodes: readonly RenderGraphNode<TPayload, TKind>[];
	readonly diagnostics: readonly RenderGraphDiagnostic[];
	readonly shadowDiagnostics: readonly RenderGraphDiagnostic[];
	readonly transitions: readonly RenderGraphTransition[];
}): RenderGraphAnalyzedStage<TPayload, TKind> {
	return Object.freeze({
		nodes: Object.freeze(stage.nodes.slice()),
		diagnostics: freezeItems(stage.diagnostics),
		shadowDiagnostics: freezeItems(stage.shadowDiagnostics),
		transitions: freezeItems(stage.transitions),
	});
}
