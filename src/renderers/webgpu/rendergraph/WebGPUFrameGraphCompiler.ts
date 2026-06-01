import type {
	WebGPUCompiledFrameGraphStage,
	WebGPUFrameGraphBarrier,
	WebGPUFrameGraphDiagnostic,
	WebGPUFrameGraphNode,
	WebGPUFrameGraphResourceDebugState,
	WebGPUFrameGraphResourceId,
	WebGPUFrameGraphResourceRef,
	WebGPUFrameGraphResourceUsage,
	WebGPUFrameGraphStagePlan,
} from "./types";

interface WebGPUFrameGraphResourceState {
	id: WebGPUFrameGraphResourceId;
	initialized: boolean;
	lastNodeId: string | null;
	lastAccess: "create" | "read" | "write" | "destroy" | null;
	lastUsage: WebGPUFrameGraphResourceUsage | null;
}

/**
 * Tracks WebGPU internal frame graph resource lifetimes and usage transitions.
 */
export class WebGPUFrameGraphCompiler {
	private readonly _resources = new Map<
		WebGPUFrameGraphResourceId,
		WebGPUFrameGraphResourceState
	>();
	private readonly _compiledStages: WebGPUCompiledFrameGraphStage[] = [];
	private readonly _barriers: WebGPUFrameGraphBarrier[] = [];
	private readonly _diagnostics: WebGPUFrameGraphDiagnostic[] = [];

	/**
	 * Resets per-frame graph state.
	 *
	 * @param initialResources Resources treated as externally available.
	 * @sideEffects Clears prior compiled graph debug state.
	 */
	public beginFrame(initialResources: readonly WebGPUFrameGraphResourceId[]): void {
		this._resources.clear();
		this._compiledStages.length = 0;
		this._barriers.length = 0;
		this._diagnostics.length = 0;
		for (const id of initialResources) {
			this._resources.set(id, {
				id,
				initialized: true,
				lastNodeId: null,
				lastAccess: "create",
				lastUsage: "external",
			});
		}
	}

	/**
	 * Records and validates one planned WebGPU internal stage.
	 *
	 * @param plan Stage plan produced by `WebGPUFrameGraphPlanner`.
	 * @returns Compiled stage with barriers and diagnostics produced by it.
	 * @sideEffects Updates resource state for later stages in the same frame.
	 */
	public compileStage(
		plan: WebGPUFrameGraphStagePlan
	): WebGPUCompiledFrameGraphStage {
		const barriersStart = this._barriers.length;
		const diagnosticsStart = this._diagnostics.length;
		for (const node of plan.nodes) {
			this._recordNode(node);
		}
		const compiled: WebGPUCompiledFrameGraphStage = {
			pass: plan.pass,
			nodes: plan.nodes.slice(),
			barriers: this._barriers.slice(barriersStart),
			diagnostics: this._diagnostics.slice(diagnosticsStart),
		};
		this._compiledStages.push(compiled);
		return compiled;
	}

	public getCompiledStages(): readonly WebGPUCompiledFrameGraphStage[] {
		return this._compiledStages.slice();
	}

	public getBarriers(): readonly WebGPUFrameGraphBarrier[] {
		return this._barriers.slice();
	}

	public getDiagnostics(): readonly WebGPUFrameGraphDiagnostic[] {
		return this._diagnostics.slice();
	}

	public getResourceDebugState(): readonly WebGPUFrameGraphResourceDebugState[] {
		return Array.from(this._resources.values())
			.sort((a, b) => a.id.localeCompare(b.id))
			.map((state) => ({
				id: state.id,
				initialized: state.initialized,
				lastNodeId: state.lastNodeId,
				lastAccess: state.lastAccess,
				lastUsage: state.lastUsage,
			}));
	}

	private _recordNode(node: WebGPUFrameGraphNode): void {
		for (const mutation of node.creates ?? []) {
			const state = this._getOrCreateResourceState(mutation.id);
			if (state.initialized && !mutation.optional) {
				this._diagnostics.push({
					severity: "error",
					nodeId: node.id,
					resource: mutation.id,
					code: "duplicate-create",
					message:
						`Frame graph node "${node.id}" creates already active ` +
						`resource "${mutation.id}".`,
				});
			}
			state.initialized = true;
			state.lastNodeId = node.id;
			state.lastAccess = "create";
			state.lastUsage = mutation.usage ?? state.lastUsage ?? "external";
		}
		for (const read of node.reads ?? []) {
			this._recordRead(node, read);
		}
		for (const write of node.writes ?? []) {
			this._recordWrite(node, write);
		}
		for (const mutation of node.destroys ?? []) {
			const state = this._getOrCreateResourceState(mutation.id);
			if (!state.initialized && !mutation.optional) {
				this._diagnostics.push({
					severity: "error",
					nodeId: node.id,
					resource: mutation.id,
					code: "destroy-before-create",
					message:
						`Frame graph node "${node.id}" destroys inactive ` +
						`resource "${mutation.id}".`,
				});
			}
			state.initialized = false;
			state.lastNodeId = node.id;
			state.lastAccess = "destroy";
			state.lastUsage = mutation.usage ?? state.lastUsage;
		}
	}

	private _recordRead(
		node: WebGPUFrameGraphNode,
		read: WebGPUFrameGraphResourceRef
	): void {
		const state = this._getOrCreateResourceState(read.id);
		if (!state.initialized) {
			if (!read.optional) {
				this._diagnostics.push({
					severity: "error",
					nodeId: node.id,
					resource: read.id,
					code: "read-before-create",
					message:
						`Frame graph node "${node.id}" reads inactive resource ` +
						`"${read.id}".`,
				});
			} else {
				return;
			}
		}
		this._recordUsageTransition(node, state, read.usage, "read");
		state.lastNodeId = node.id;
		state.lastAccess = "read";
		state.lastUsage = read.usage;
	}

	private _recordWrite(
		node: WebGPUFrameGraphNode,
		write: WebGPUFrameGraphResourceRef
	): void {
		const state = this._getOrCreateResourceState(write.id);
		this._recordUsageTransition(node, state, write.usage, "write");
		state.initialized = true;
		state.lastNodeId = node.id;
		state.lastAccess = "write";
		state.lastUsage = write.usage;
	}

	private _recordUsageTransition(
		node: WebGPUFrameGraphNode,
		state: WebGPUFrameGraphResourceState,
		nextUsage: WebGPUFrameGraphResourceUsage,
		nextAccess: "read" | "write"
	): void {
		if (!state.lastUsage || !state.lastAccess || state.lastAccess === "create") {
			return;
		}
		if (state.lastUsage === nextUsage && state.lastAccess === nextAccess) {
			return;
		}
		this._barriers.push({
			resource: state.id,
			beforeNodeId: state.lastNodeId,
			nodeId: node.id,
			fromUsage: state.lastUsage,
			toUsage: nextUsage,
			reason: this._resolveBarrierReason(state.lastAccess, nextAccess),
		});
	}

	private _resolveBarrierReason(
		lastAccess: WebGPUFrameGraphResourceState["lastAccess"],
		nextAccess: "read" | "write"
	): WebGPUFrameGraphBarrier["reason"] {
		if (lastAccess === "write" && nextAccess === "read") {
			return "read-after-write";
		}
		if (lastAccess === "read" && nextAccess === "write") {
			return "write-after-read";
		}
		if (lastAccess === "write" && nextAccess === "write") {
			return "write-after-write";
		}
		return "usage-transition";
	}

	private _getOrCreateResourceState(
		id: WebGPUFrameGraphResourceId
	): WebGPUFrameGraphResourceState {
		let state = this._resources.get(id);
		if (!state) {
			state = {
				id,
				initialized: false,
				lastNodeId: null,
				lastAccess: null,
				lastUsage: null,
			};
			this._resources.set(id, state);
		}
		return state;
	}
}
