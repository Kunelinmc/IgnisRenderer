import { RenderGraphAttemptTracker } from "../../../rendergraph/RenderGraphAttemptTracker";
import { RenderGraphCompiler } from "../../../rendergraph/RenderGraphCompiler";
import type {
	CompiledRenderGraph,
	RenderGraphDefinition,
	RenderGraphDiagnostic,
	RenderGraphNode,
	RenderGraphResourceDescriptor,
	RenderGraphResourceRef,
	RenderGraphTrackerDebugState,
	RenderGraphTransition,
	RenderGraphUsage,
} from "../../../rendergraph/types";
import type {
	WebGPUCompiledFrameGraph,
	WebGPUCompiledFrameGraphStage,
	WebGPUFrameGraphBarrier,
	WebGPUFrameGraphDiagnostic,
	WebGPUFrameGraphFramePlan,
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

/** Compiles WebGPU frame plans through the shared whole-frame Render Graph. */
export class WebGPUFrameGraphCompiler {
	private readonly _compiler = new RenderGraphCompiler();
	private readonly _attempts = new RenderGraphAttemptTracker();
	private _compiled: WebGPUCompiledFrameGraph | null = null;
	private _lastFramePlan: WebGPUFrameGraphFramePlan | null = null;
	private _legacyResources: RenderGraphResourceDescriptor[] = [];
	private _legacyStages: WebGPUFrameGraphStagePlan[] = [];

	public compileFrame(plan: WebGPUFrameGraphFramePlan): WebGPUCompiledFrameGraph {
		const definition = createDefinition(plan);
		const graph = this._compiler.compile(definition);
		const stages = plan.stages.map((stage) => projectStage(stage, graph));
		this._compiled = Object.freeze({ graph, stages: Object.freeze(stages) });
		this._lastFramePlan = plan;
		this._attempts.begin(graph);
		return this._compiled;
	}

	/** @internal Legacy test adapter; production backends must use `compileFrame`. */
	public beginFrame(initialResources: readonly WebGPUFrameGraphResourceId[]): void {
		this._legacyResources = initialResources.map(toImportedDescriptor);
		this._legacyStages = [];
		this.compileFrame({
			resources: this._legacyResources,
			bindings: this._legacyResources.map((resource) => ({
				resourceId: resource.id,
				physicalId: `webgpu:${resource.id}`,
				kind: resource.kind,
			})),
			stages: [],
		});
	}

	/** @internal Legacy test adapter; production backends must use `compileFrame`. */
	public compileStage(plan: WebGPUFrameGraphStagePlan): WebGPUCompiledFrameGraphStage {
		this._legacyStages.push(plan);
		const knownResources = new Set(this._legacyResources.map((resource) => resource.id));
		for (const node of plan.nodes) {
			const referenced = [
				...(node.creates ?? []).map((mutation) => mutation.id),
				...(node.reads ?? []).map((reference) => reference.id),
				...(node.writes ?? []).map((reference) => reference.id),
				...(node.destroys ?? []).map((mutation) => mutation.id),
			];
			for (const id of referenced) {
				if (knownResources.has(id)) continue;
				knownResources.add(id);
				this._legacyResources.push({
					id,
					origin: "graph",
					kind: "external",
					residency: "frame",
					initialContent: "undefined",
				});
			}
		}
		const bindings = this._legacyResources.map((resource) => ({
			resourceId: resource.id,
			physicalId: `webgpu:${resource.id}`,
			kind: resource.kind,
		}));
		const compiled = this.compileFrame({
			resources: this._legacyResources,
			bindings,
			stages: this._legacyStages,
		});
		return compiled.stages[compiled.stages.length - 1];
	}

	public seal(): void {
		this._attempts.seal();
	}

	public commit(): void {
		this._attempts.commit();
	}

	public abort(error?: unknown): void {
		this._attempts.abort(error);
	}

	public recordOpaqueStage(stage: string, message: string): void {
		if (!this._lastFramePlan) return;
		const diagnostic: RenderGraphDiagnostic = {
			phase: "compile",
			enforcement: "shadow",
			severity: "warning",
			code: "opaque-stage-effects",
			stage,
			message,
		};
		this.compileFrame({
			...this._lastFramePlan,
			completeness: "opaque",
			shadowDiagnostics: [
				...(this._lastFramePlan.shadowDiagnostics ?? []),
				diagnostic,
			],
		});
	}

	public getCompiledFrame(): WebGPUCompiledFrameGraph | null {
		return this._compiled;
	}

	public getCompiledStages(): readonly WebGPUCompiledFrameGraphStage[] {
		return this._compiled?.stages ?? [];
	}

	public getBarriers(): readonly WebGPUFrameGraphBarrier[] {
		return this._compiled?.stages.flatMap((stage) => stage.barriers) ?? [];
	}

	public getDiagnostics(): readonly WebGPUFrameGraphDiagnostic[] {
		return this._compiled?.stages.flatMap((stage) => stage.diagnostics) ?? [];
	}

	public getResourceDebugState(): readonly WebGPUFrameGraphResourceDebugState[] {
		const debug = this._attempts.getDebugState();
		const snapshot = debug.current ?? debug.lastAttempt;
		return snapshot?.resources.map((state) => ({
			id: state.id as WebGPUFrameGraphResourceId,
			initialized: state.active,
			lastNodeId: state.lastNodeId,
			lastAccess: toLegacyAccess(state.lastAccess),
			lastUsage:
				state.lastUsage ? toWebGPUUsage(state.lastUsage)
				: state.lastAccess === "create" ? "external"
				: null,
		})) ?? [];
	}

	public getGraphAnalysis(): RenderGraphTrackerDebugState {
		return this._attempts.getDebugState();
	}
}

function createDefinition(
	plan: WebGPUFrameGraphFramePlan,
): RenderGraphDefinition<WebGPUFrameGraphNode, WebGPUFrameGraphNode["kind"]> {
	const nodes: SharedWebGPUNode[] = [];
	const terminalByStage = new Map<string, string>();
	for (const stage of plan.stages) {
		let previousNodeId: string | null = null;
		for (let index = 0; index < stage.nodes.length; index++) {
			const node = stage.nodes[index];
			const dependsOn = new Set(node.dependsOn ?? []);
			if (previousNodeId) dependsOn.add(previousNodeId);
			if (index === 0) {
				for (const dependencyStage of stage.pass.dependsOn ?? []) {
					const dependencyNode = terminalByStage.get(dependencyStage);
					if (dependencyNode) dependsOn.add(dependencyNode);
				}
			}
			nodes.push(toSharedNode(node, Array.from(dependsOn)));
			previousNodeId = node.id;
		}
		if (previousNodeId) terminalByStage.set(stage.pass.stage, previousNodeId);
	}
	return {
		resources: plan.resources,
		bindings: plan.bindings,
		nodes,
		exports: plan.exports,
		completeness: plan.completeness,
		shadowDiagnostics: plan.shadowDiagnostics,
	};
}

function projectStage(
	stage: WebGPUFrameGraphStagePlan,
	graph: CompiledRenderGraph<WebGPUFrameGraphNode, WebGPUFrameGraphNode["kind"]>,
): WebGPUCompiledFrameGraphStage {
	const declaredIds = new Set(stage.nodes.map((node) => node.id));
	const nodes = graph.nodes
		.filter((node) => declaredIds.has(node.id))
		.map((node) => node.payload!);
	const retainedIds = new Set(nodes.map((node) => node.id));
	const barriers = graph.transitions
		.filter((transition) => retainedIds.has(transition.nodeId) && transition.reason !== undefined)
		.map(toWebGPUBarrier);
	const diagnostics = graph.diagnostics
		.filter((diagnostic) => diagnostic.nodeId && declaredIds.has(diagnostic.nodeId))
		.map(toWebGPUDiagnostic)
		.filter((diagnostic): diagnostic is WebGPUFrameGraphDiagnostic => diagnostic !== null);
	return Object.freeze({
		pass: stage.pass,
		nodes: Object.freeze(nodes),
		barriers: Object.freeze(barriers),
		diagnostics: Object.freeze(diagnostics),
	});
}

function toImportedDescriptor(id: WebGPUFrameGraphResourceId): RenderGraphResourceDescriptor {
	return {
		id,
		origin: "imported",
		kind: "external",
		residency: "frame",
		initialContent: "unknown",
	};
}

function toSharedNode(node: WebGPUFrameGraphNode, dependsOn: readonly string[]): SharedWebGPUNode {
	const resources: RenderGraphResourceRef[] = [];
	for (const ref of node.reads ?? []) {
		resources.push({
			resource: ref.id,
			access: "read",
			usage: toSharedUsage(ref.usage),
			optional: ref.optional,
		});
	}
	for (const ref of node.writes ?? []) {
		resources.push({
			resource: ref.id,
			access: "write",
			usage: toSharedUsage(ref.usage),
			optional: ref.optional,
		});
	}
	return {
		id: node.id,
		stage: node.stage,
		kind: node.kind,
		label: node.label,
		domain: node.domain,
		retention: node.retention,
		opaque: node.opaque,
		dependsOn,
		creates: node.creates?.map((mutation) => ({
			resource: mutation.id,
			usage: mutation.usage ? toSharedUsage(mutation.usage) : undefined,
			optional: mutation.optional,
		})),
		resources,
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
		case "render-attachment": return "color-attachment";
		case "texture-binding": return "sampled";
		case "storage-binding": return "storage";
		case "copy-src": return "copy-source";
		case "copy-dst": return "copy-target";
		case "present": return "present";
		case "depth-attachment": return "depth-attachment";
		case "external": return "sampled";
	}
}

function toWebGPUUsage(usage: RenderGraphUsage): WebGPUFrameGraphResourceUsage {
	switch (usage) {
		case "color-attachment": return "render-attachment";
		case "sampled": return "texture-binding";
		case "storage": return "storage-binding";
		case "copy-source": return "copy-src";
		case "copy-target": return "copy-dst";
		case "present": return "present";
		case "depth-attachment": return "depth-attachment";
		default: return "external";
	}
}

function toWebGPUBarrier(transition: RenderGraphTransition): WebGPUFrameGraphBarrier {
	return {
		resource: transition.resourceId as WebGPUFrameGraphResourceId,
		beforeNodeId: transition.fromNodeId ?? null,
		nodeId: transition.nodeId,
		fromUsage: transition.previousUsage ? toWebGPUUsage(transition.previousUsage) : null,
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
		diagnostic.code !== "duplicate-create" &&
		diagnostic.code !== "physical-feedback-loop"
	) {
		return null;
	}
	return {
		severity: diagnostic.severity,
		nodeId: diagnostic.nodeId ?? "unknown",
		resource: (diagnostic.resourceId ?? "unknown") as WebGPUFrameGraphResourceId,
		code: diagnostic.code as WebGPUFrameGraphDiagnostic["code"],
		message: diagnostic.message.replace("Render graph node", "Frame graph node"),
	};
}

function toLegacyAccess(
	access: "create" | "read" | "write" | "read-write" | "destroy" | null,
): "create" | "read" | "write" | "destroy" | null {
	return access === "read-write" ? "write" : access;
}
