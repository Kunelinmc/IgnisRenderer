import type {
	WebGLCompiledFrameGraphStage,
	WebGLFrameGraphBarrier,
	WebGLFrameGraphDiagnostic,
	WebGLFrameGraphNode,
	WebGLFrameGraphResourceDebugState,
	WebGLFrameGraphResourceId,
	WebGLFrameGraphResourceRef,
	WebGLFrameGraphResourceUsage,
	WebGLFrameGraphStagePlan,
} from "./types";

const SUPPORTED_WEBGL_RESOURCE_USAGES = new Set<WebGLFrameGraphResourceUsage>([
	"external",
	"framebuffer-color",
	"framebuffer-depth",
	"texture-sampling",
	"copy-source",
	"copy-target",
	"present",
]);

interface WebGLFrameGraphResourceState {
	id: WebGLFrameGraphResourceId;
	initialized: boolean;
	lastNodeId: string | null;
	lastAccess: "create" | "read" | "write" | "destroy" | null;
	lastUsage: WebGLFrameGraphResourceUsage | null;
}

/**
 * Tracks WebGL internal frame graph resource lifetimes and unsafe texture use.
 */
export class WebGLFrameGraphCompiler {
	private readonly _resources = new Map<
		WebGLFrameGraphResourceId,
		WebGLFrameGraphResourceState
	>();
	private readonly _compiledStages: WebGLCompiledFrameGraphStage[] = [];
	private readonly _barriers: WebGLFrameGraphBarrier[] = [];
	private readonly _diagnostics: WebGLFrameGraphDiagnostic[] = [];

	/**
	 * Resets per-frame WebGL graph state.
	 *
	 * @internal WebGL frame graph lifecycle hook.
	 * @param initialResources Resources already owned by the active frame.
	 * @returns Nothing.
	 * @sideEffects Clears prior graph debug state.
	 */
	public beginFrame(initialResources: readonly WebGLFrameGraphResourceId[]): void {
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
	 * Records and validates one planned WebGL graph stage.
	 *
	 * @internal Owned by `WebGLFrameGraphRuntime`.
	 * @param plan Stage plan from `WebGLFrameGraphPlanner`.
	 * @returns Compiled stage diagnostics and barriers.
	 * @sideEffects Updates resource state for later stages.
	 */
	public compileStage(
		plan: WebGLFrameGraphStagePlan
	): WebGLCompiledFrameGraphStage {
		const barriersStart = this._barriers.length;
		const diagnosticsStart = this._diagnostics.length;
		for (const node of plan.nodes) {
			this._recordNode(node);
		}
		const compiled: WebGLCompiledFrameGraphStage = {
			pass: plan.pass,
			nodes: plan.nodes.slice(),
			barriers: this._barriers.slice(barriersStart),
			diagnostics: this._diagnostics.slice(diagnosticsStart),
		};
		this._compiledStages.push(compiled);
		return compiled;
	}

	public getCompiledStages(): readonly WebGLCompiledFrameGraphStage[] {
		return this._compiledStages.slice();
	}

	public getBarriers(): readonly WebGLFrameGraphBarrier[] {
		return this._barriers.slice();
	}

	public getDiagnostics(): readonly WebGLFrameGraphDiagnostic[] {
		return this._diagnostics.slice();
	}

	public getResourceDebugState(): readonly WebGLFrameGraphResourceDebugState[] {
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

	private _recordNode(node: WebGLFrameGraphNode): void {
		this._validateNodeResourceUsages(node);
		this._validateFeedbackLoop(node);
		for (const mutation of node.creates ?? []) {
			const state = this._getOrCreateResourceState(mutation.id);
			if (state.initialized && !mutation.optional) {
				this._diagnostics.push({
					severity: "error",
					nodeId: node.id,
					resource: mutation.id,
					code: "duplicate-create",
					message:
						`WebGL frame graph node "${node.id}" creates already ` +
						`active resource "${mutation.id}".`,
				});
			}
			state.initialized = true;
			state.lastNodeId = node.id;
			state.lastAccess = "create";
			state.lastUsage = mutation.usage ?? state.lastUsage ?? "external";
		}
		for (const mutation of node.requires ?? []) {
			const state = this._getOrCreateResourceState(mutation.id);
			if (!state.initialized && !mutation.optional) {
				this._diagnostics.push({
					severity: "error",
					nodeId: node.id,
					resource: mutation.id,
					code: "missing-resource",
					message:
						`WebGL frame graph node "${node.id}" requires missing ` +
						`resource "${mutation.id}".`,
				});
			}
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
						`WebGL frame graph node "${node.id}" destroys inactive ` +
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
		node: WebGLFrameGraphNode,
		read: WebGLFrameGraphResourceRef
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
						`WebGL frame graph node "${node.id}" reads inactive ` +
						`resource "${read.id}".`,
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
		node: WebGLFrameGraphNode,
		write: WebGLFrameGraphResourceRef
	): void {
		const state = this._getOrCreateResourceState(write.id);
		this._recordUsageTransition(node, state, write.usage, "write");
		state.initialized = true;
		state.lastNodeId = node.id;
		state.lastAccess = "write";
		state.lastUsage = write.usage;
	}

	private _recordUsageTransition(
		node: WebGLFrameGraphNode,
		state: WebGLFrameGraphResourceState,
		nextUsage: WebGLFrameGraphResourceUsage,
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

	private _validateNodeResourceUsages(node: WebGLFrameGraphNode): void {
		for (const ref of [
			...(node.reads ?? []),
			...(node.writes ?? []),
		]) {
			if (!SUPPORTED_WEBGL_RESOURCE_USAGES.has(ref.usage)) {
				this._diagnostics.push({
					severity: "error",
					nodeId: node.id,
					resource: ref.id,
					code: "unsupported-node-resource",
					message:
						`WebGL frame graph node "${node.id}" references ` +
						`unsupported usage "${String(ref.usage)}" for resource ` +
						`"${ref.id}".`,
				});
			}
		}
	}

	private _validateFeedbackLoop(node: WebGLFrameGraphNode): void {
		const sampled = new Set(
			(node.reads ?? [])
				.filter((read) => read.usage === "texture-sampling")
				.map((read) => read.id)
		);
		for (const write of node.writes ?? []) {
			if (
				sampled.has(write.id) &&
				(write.usage === "framebuffer-color" ||
					write.usage === "framebuffer-depth")
			) {
				this._diagnostics.push({
					severity: "error",
					nodeId: node.id,
					resource: write.id,
					code: "texture-feedback-loop",
					message:
						`WebGL frame graph node "${node.id}" samples and writes ` +
						`resource "${write.id}" in the same framebuffer pass.`,
				});
			}
		}
	}

	private _resolveBarrierReason(
		lastAccess: WebGLFrameGraphResourceState["lastAccess"],
		nextAccess: "read" | "write"
	): WebGLFrameGraphBarrier["reason"] {
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
		id: WebGLFrameGraphResourceId
	): WebGLFrameGraphResourceState {
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
