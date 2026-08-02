import type {
	CompiledRenderGraph,
	RenderGraphAnalysisSnapshot,
	RenderGraphNodeId,
	RenderGraphPhysicalResourceId,
	RenderGraphResourceAlias,
	RenderGraphResourceDebugState,
	RenderGraphResourceId,
	RenderGraphTrackerDebugState,
} from "./types";

/** @internal Tracks transactional outcomes for precompiled whole-frame graphs. */
export class RenderGraphAttemptTracker {
	private _state: RenderGraphTrackerDebugState["state"] = "idle";
	private _current: RenderGraphAnalysisSnapshot | null = null;
	private _lastAttempt: RenderGraphAnalysisSnapshot | null = null;
	private _lastSuccessful: RenderGraphAnalysisSnapshot | null = null;

	public begin(graph: CompiledRenderGraph): void {
		if (this._state === "active" || this._state === "sealed") {
			this.abort();
		}
		this._current = createSnapshot(graph, "active");
		this._state = "active";
	}

	public seal(): void {
		if (this._state !== "active" || !this._current) {
			throw new Error(`Render graph attempt cannot seal in state "${this._state}".`);
		}
		this._current = Object.freeze({ ...this._current, state: "sealed" });
		this._state = "sealed";
	}

	public recordSkippedNode(
		nodeId: RenderGraphNodeId,
		aliases: readonly RenderGraphResourceAlias[] = [],
	): void {
		if (this._state !== "active" || !this._current) {
			throw new Error(
				`Render graph execution overlay cannot change in state "${this._state}".`,
			);
		}
		if (!this._current.nodeIds.includes(nodeId)) {
			throw new Error(`Render graph execution overlay has no node "${nodeId}".`);
		}
		const skippedNodeIds = this._current.executionOverlay.skippedNodeIds.includes(nodeId)
			? this._current.executionOverlay.skippedNodeIds
			: Object.freeze([
				...this._current.executionOverlay.skippedNodeIds,
				nodeId,
			]);
		const aliasesByResource = new Map(
			this._current.executionOverlay.resourceAliases.map((alias) => [alias.resourceId, alias]),
		);
		for (const alias of aliases) aliasesByResource.set(alias.resourceId, Object.freeze({ ...alias }));
		this._current = Object.freeze({
			...this._current,
			executionOverlay: Object.freeze({
				skippedNodeIds,
				resourceAliases: Object.freeze(Array.from(aliasesByResource.values())),
			}),
		});
	}

	public commit(): void {
		if ((this._state !== "active" && this._state !== "sealed") || !this._current) return;
		const snapshot = Object.freeze({ ...this._current, state: "committed" }) as RenderGraphAnalysisSnapshot;
		this._lastAttempt = snapshot;
		this._lastSuccessful = snapshot;
		this._current = null;
		this._state = "committed";
	}

	public abort(_error?: unknown): void {
		if ((this._state !== "active" && this._state !== "sealed") || !this._current) return;
		this._lastAttempt = Object.freeze({
			...this._current,
			state: "aborted",
		}) as RenderGraphAnalysisSnapshot;
		this._current = null;
		this._state = "aborted";
	}

	public getDebugState(): RenderGraphTrackerDebugState {
		return Object.freeze({
			state: this._state,
			current: this._current,
			lastAttempt: this._lastAttempt,
			lastSuccessful: this._lastSuccessful,
		});
	}
}

function createSnapshot(
	graph: CompiledRenderGraph,
	state: RenderGraphAnalysisSnapshot["state"],
): RenderGraphAnalysisSnapshot {
	return Object.freeze({
		state,
		completeness: graph.completeness,
		nodeIds: Object.freeze(graph.nodes.map((node) => node.id)),
		declaredNodeIds: Object.freeze(graph.declaredNodes.map((node) => node.id)),
		culledNodeIds: graph.culledNodeIds,
		resources: createResourceStates(graph),
		transitions: graph.transitions,
		dependencies: graph.dependencies,
		liveRanges: graph.liveRanges,
		subresourceLiveRanges: graph.subresourceLiveRanges,
		allocationRequests: graph.allocationRequests,
		executionOverlay: Object.freeze({
			skippedNodeIds: Object.freeze([]),
			resourceAliases: Object.freeze([]),
		}),
		diagnostics: graph.diagnostics,
		shadowDiagnostics: graph.shadowDiagnostics,
	});
}

function createResourceStates(
	graph: CompiledRenderGraph,
): readonly RenderGraphResourceDebugState[] {
	const lastTransitionByResource = new Map(
		graph.transitions.map((transition) => [transition.resourceId, transition]),
	);
	return Object.freeze(graph.resources.map((resource) => {
		const transition = lastTransitionByResource.get(resource.id);
		const lifetimes = graph.liveRanges.filter((range) => range.resourceId === resource.id);
		const lifetime = transition
			? lifetimes.find((range) => range.generation === transition.generation)
			: lifetimes.reduce((latest, range) =>
				!latest || range.generation > latest.generation ? range : latest, undefined);
		const initialized = lifetime?.destroyedByNodeId ? false
			: resource.origin === "imported" ||
				!!lifetime?.createdByNodeId ||
				transition?.access === "write" ||
				transition?.access === "read-write";
		return Object.freeze({
			id: resource.id,
			origin: resource.origin,
			active: initialized,
			generation: lifetime?.generation ?? 0,
			content: transition && transition.access !== "read"
				? "valid"
				: resource.initialContent ?? (resource.origin === "imported" ? "unknown" : "undefined"),
			physicalId: resolvePhysicalId(
				graph,
				resource.id,
				lifetime?.generation ?? transition?.generation ?? 0,
			),
			lastNodeId: transition?.nodeId ?? lifetime?.lastNodeId ?? null,
			lastAccess: transition?.access ?? (lifetime?.createdByNodeId ? "create" : null),
			lastUsage: transition?.usage ?? null,
		});
	}));
}

function resolvePhysicalId(
	graph: CompiledRenderGraph,
	resourceId: RenderGraphResourceId,
	generation: number,
): RenderGraphPhysicalResourceId | undefined {
	return graph.bindings.find(
		(binding) => binding.resourceId === resourceId && binding.generation === generation,
	)?.physicalId ?? graph.bindings.find(
		(binding) => binding.resourceId === resourceId && binding.generation === undefined,
	)?.physicalId;
}
