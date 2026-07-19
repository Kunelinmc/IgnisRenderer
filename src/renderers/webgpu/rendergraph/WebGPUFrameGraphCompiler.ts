import { RenderGraphStateTracker } from "../../../rendergraph/RenderGraphStateTracker";
import type {
	RenderGraphDiagnostic,
	RenderGraphNode,
	RenderGraphResourceDescriptor,
	RenderGraphTrackerDebugState,
	RenderGraphTransition,
	RenderGraphUsage,
} from "../../../rendergraph/types";
import type {
	WebGPUCompiledFrameGraphStage,
	WebGPUFrameGraphBarrier,
	WebGPUFrameGraphDiagnostic,
	WebGPUFrameGraphNode,
	WebGPUFrameGraphResourceDebugState,
	WebGPUFrameGraphResourceId,
	WebGPUFrameGraphResourceUsage,
	WebGPUFrameGraphStagePlan,
} from "./types";

type SharedWebGPUNode = RenderGraphNode<
	WebGPUFrameGraphNode,
	WebGPUFrameGraphNode["kind"]
>;

/** Tracks WebGPU frame graph state through the shared logical analyzer. */
export class WebGPUFrameGraphCompiler {
	private readonly _tracker = new RenderGraphStateTracker<
		WebGPUFrameGraphNode,
		WebGPUFrameGraphNode["kind"]
	>({ allowImplicitResources: true });
	private readonly _compiledStages: WebGPUCompiledFrameGraphStage[] = [];
	private readonly _barriers: WebGPUFrameGraphBarrier[] = [];
	private readonly _diagnostics: WebGPUFrameGraphDiagnostic[] = [];

	public beginFrame(initialResources: readonly WebGPUFrameGraphResourceId[]): void {
		const state = this._tracker.getDebugState().state;
		if (state === "active" || state === "sealed") {
			this._tracker.abort(
				new Error("A new WebGPU frame superseded an uncommitted graph attempt."),
			);
		}
		this._compiledStages.length = 0;
		this._barriers.length = 0;
		this._diagnostics.length = 0;
		this._tracker.beginFrame(initialResources.map(toImportedDescriptor));
	}

	public compileStage(
		plan: WebGPUFrameGraphStagePlan,
	): WebGPUCompiledFrameGraphStage {
		const analyzed = this._tracker.appendStage({
			nodes: plan.nodes.map(toSharedNode),
		});
		if (plan.pass.stage === "postprocess") {
			this._tracker.markCompleteness("coarse");
		}
		const barriers = analyzed.transitions
			.filter((transition) => transition.reason !== undefined)
			.map(toWebGPUBarrier);
		const diagnostics = analyzed.diagnostics
			.map(toWebGPUDiagnostic)
			.filter((diagnostic): diagnostic is WebGPUFrameGraphDiagnostic => !!diagnostic);
		this._barriers.push(...barriers);
		this._diagnostics.push(...diagnostics);
		const compiled: WebGPUCompiledFrameGraphStage = {
			pass: plan.pass,
			nodes: plan.nodes.slice(),
			barriers,
			diagnostics,
		};
		this._compiledStages.push(compiled);
		return compiled;
	}

	public seal(): void {
		this._tracker.seal();
	}

	public commit(): void {
		this._tracker.commit();
	}

	public abort(error?: unknown): void {
		this._tracker.abort(error);
	}

	public recordOpaqueStage(stage: string, message: string): void {
		this._tracker.recordOpaqueStage(stage, message);
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
		return this._tracker.getResourceDebugState().map((state) => ({
			id: state.id as WebGPUFrameGraphResourceId,
			initialized: state.active,
			lastNodeId: state.lastNodeId,
			lastAccess: toLegacyAccess(state.lastAccess),
			lastUsage:
				state.lastUsage ? toWebGPUUsage(state.lastUsage)
				: state.lastAccess === "create" ? "external"
				: null,
		}));
	}

	public getGraphAnalysis(): RenderGraphTrackerDebugState {
		return this._tracker.getDebugState();
	}
}

function toImportedDescriptor(
	id: WebGPUFrameGraphResourceId,
): RenderGraphResourceDescriptor {
	return {
		id,
		origin: "imported",
		kind: "external",
		residency: "frame",
		initialContent: "unknown",
	};
}

function toSharedNode(node: WebGPUFrameGraphNode): SharedWebGPUNode {
	return {
		id: node.id,
		stage: node.stage,
		kind: node.kind,
		label: node.label,
		creates: node.creates?.map((mutation) => ({
			resource: mutation.id,
			usage: mutation.usage ? toSharedUsage(mutation.usage) : undefined,
			optional: mutation.optional,
		})),
		resources: [
			...(node.reads ?? []).map((ref) => ({
				resource: ref.id,
				access: "read" as const,
				usage: toSharedUsage(ref.usage),
				optional: ref.optional,
			})),
			...(node.writes ?? []).map((ref) => ({
				resource: ref.id,
				access: "write" as const,
				usage: toSharedUsage(ref.usage),
				optional: ref.optional,
			})),
		],
		destroys: node.destroys?.map((mutation) => ({
			resource: mutation.id,
			usage: mutation.usage ? toSharedUsage(mutation.usage) : undefined,
			optional: mutation.optional,
		})),
		payload: node,
	};
}

function toSharedUsage(usage: WebGPUFrameGraphResourceUsage): RenderGraphUsage {
	switch (usage) {
		case "render-attachment":
			return "color-attachment";
		case "texture-binding":
			return "sampled";
		case "storage-binding":
			return "storage";
		case "copy-src":
			return "copy-source";
		case "copy-dst":
			return "copy-target";
		case "present":
			return "present";
		case "depth-attachment":
			return "depth-attachment";
		case "external":
			return "sampled";
	}
}

function toWebGPUUsage(usage: RenderGraphUsage): WebGPUFrameGraphResourceUsage {
	switch (usage) {
		case "color-attachment":
			return "render-attachment";
		case "sampled":
			return "texture-binding";
		case "storage":
			return "storage-binding";
		case "copy-source":
			return "copy-src";
		case "copy-target":
			return "copy-dst";
		case "present":
			return "present";
		case "depth-attachment":
			return "depth-attachment";
		default:
			return "external";
	}
}

function toWebGPUBarrier(
	transition: RenderGraphTransition,
): WebGPUFrameGraphBarrier {
	return {
		resource: transition.resourceId as WebGPUFrameGraphResourceId,
		beforeNodeId: transition.fromNodeId ?? null,
		nodeId: transition.nodeId,
		fromUsage:
			transition.previousUsage ? toWebGPUUsage(transition.previousUsage) : null,
		toUsage: toWebGPUUsage(transition.usage),
		reason: transition.reason ?? "usage-transition",
	};
}

function toWebGPUDiagnostic(
	diagnostic: RenderGraphDiagnostic,
): WebGPUFrameGraphDiagnostic | null {
	if (
		diagnostic.code !== "read-before-create" &&
		diagnostic.code !== "destroy-before-create" &&
		diagnostic.code !== "duplicate-create"
	) {
		return null;
	}
	return {
		severity: diagnostic.severity,
		nodeId: diagnostic.nodeId ?? "unknown",
		resource: (diagnostic.resourceId ?? "unknown") as WebGPUFrameGraphResourceId,
		code: diagnostic.code,
		message:
			diagnostic.code === "duplicate-create" ?
				`Frame graph node "${diagnostic.nodeId}" creates already active ` +
				`resource "${diagnostic.resourceId}".`
				: diagnostic.message.replace("Render graph node", "Frame graph node"),
	};
}

function toLegacyAccess(
	access: "create" | "read" | "write" | "read-write" | "destroy" | null,
): "create" | "read" | "write" | "destroy" | null {
	return access === "read-write" ? "write" : access;
}
