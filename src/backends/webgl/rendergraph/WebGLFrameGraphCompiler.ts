import { RenderGraphAttemptTracker } from "../../../rendergraph/RenderGraphAttemptTracker";
import { RenderGraphBuilder } from "../../../rendergraph/RenderGraphBuilder";
import { RenderGraphCompiler } from "../../../rendergraph/RenderGraphCompiler";
import {
	renderGraphNodeId,
	renderGraphPhysicalResourceId,
	renderGraphResourceId,
} from "../../../rendergraph/types";
import type {
	CompiledRenderGraph,
	RenderGraphDefinition,
	RenderGraphDiagnostic,
	RenderGraphNode,
	RenderGraphNodeId,
	RenderGraphResourceDescriptor,
	RenderGraphResourceId,
	RenderGraphResourceRef,
	RenderGraphTrackerDebugState,
	RenderGraphTransition,
	RenderGraphUsage,
} from "../../../rendergraph/types";
import type {
	WebGLCompiledFrameGraph,
	WebGLCompiledFrameGraphStage,
	WebGLFrameGraphBarrier,
	WebGLFrameGraphDiagnostic,
	WebGLFrameGraphFramePlan,
	WebGLFrameGraphNode,
	WebGLFrameGraphResourceDebugState,
	WebGLFrameGraphResourceId,
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

type SharedWebGLNode = RenderGraphNode<
	WebGLFrameGraphNode,
	WebGLFrameGraphNode["kind"]
>;

/** Compiles WebGL plans through the shared immutable whole-frame graph. */
export class WebGLFrameGraphCompiler {
	private readonly _compiler = new RenderGraphCompiler();
	private readonly _attempts = new RenderGraphAttemptTracker();
	private _compiled: WebGLCompiledFrameGraph | null = null;
	private _lastFramePlan: WebGLFrameGraphFramePlan | null = null;
	private _legacyResources: RenderGraphResourceDescriptor[] = [];
	private _legacyStages: WebGLFrameGraphStagePlan[] = [];

	public compileFrame(plan: WebGLFrameGraphFramePlan): WebGLCompiledFrameGraph {
		const built = createDefinition(plan);
		const graph = this._compiler.compile(built.definition);
		const stages = plan.stages.map((stage) =>
			projectStage(stage, built.stageNodeIds.get(stage) ?? [], graph));
		this._compiled = Object.freeze({ graph, stages: Object.freeze(stages) });
		this._lastFramePlan = plan;
		this._attempts.begin(graph);
		return this._compiled;
	}

	/** @internal Legacy test adapter; production backends must use `compileFrame`. */
	public beginFrame(initialResources: readonly WebGLFrameGraphResourceId[]): void {
		this._legacyResources = initialResources.map(toImportedDescriptor);
		this._legacyStages = [];
		this.compileFrame({
			resources: this._legacyResources,
			bindings: this._legacyResources.map((resource) => ({
				resourceId: resource.id,
				physicalId: renderGraphPhysicalResourceId(`webgl:${resource.id}`),
				kind: resource.kind,
			})),
			stages: [],
		});
	}

	/** @internal Legacy test adapter; production backends must use `compileFrame`. */
	public compileStage(plan: WebGLFrameGraphStagePlan): WebGLCompiledFrameGraphStage {
		this._legacyStages.push(plan);
		const knownResources = new Set<string>(
			this._legacyResources.map((resource) => resource.id),
		);
		for (const node of plan.nodes) {
			const referenced = [
				...(node.creates ?? []).map((mutation) => mutation.id),
				...(node.requires ?? []).map((requirement) => requirement.id),
				...(node.reads ?? []).map((reference) => reference.id),
				...(node.writes ?? []).map((reference) => reference.id),
				...(node.destroys ?? []).map((mutation) => mutation.id),
			];
			for (const id of referenced) {
				if (knownResources.has(id)) continue;
				knownResources.add(id);
				this._legacyResources.push({
					id: renderGraphResourceId(id),
					origin: "graph",
					kind: "external",
					residency: "frame",
					initialContent: "undefined",
				});
			}
		}
		const compiled = this.compileFrame({
			resources: this._legacyResources,
			bindings: this._legacyResources.map((resource) => ({
				resourceId: resource.id,
				physicalId: renderGraphPhysicalResourceId(`webgl:${resource.id}`),
				kind: resource.kind,
			})),
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

	public recordSkippedNode(
		nodeId: string,
		resourceId?: string,
		resolvedResourceId?: string,
	): void {
		this._attempts.recordSkippedNode(
			renderGraphNodeId(nodeId),
			resourceId && resolvedResourceId ? [{
				resourceId: renderGraphResourceId(resourceId),
				resolvedResourceId: renderGraphResourceId(resolvedResourceId),
			}] : [],
		);
	}

	public recordOpaqueStage(stage: string, message: string): void {
		if (!this._lastFramePlan) return;
		this.compileFrame({
			...this._lastFramePlan,
			completeness: "opaque",
			shadowDiagnostics: [
				...(this._lastFramePlan.shadowDiagnostics ?? []),
				{
					phase: "compile",
					enforcement: "shadow",
					severity: "warning",
					code: "opaque-stage-effects",
					stage,
					message,
				},
			],
		});
	}

	public getCompiledFrame(): WebGLCompiledFrameGraph | null {
		return this._compiled;
	}

	public getCompiledStages(): readonly WebGLCompiledFrameGraphStage[] {
		return this._compiled?.stages ?? [];
	}

	public getBarriers(): readonly WebGLFrameGraphBarrier[] {
		return this._compiled?.stages.flatMap((stage) => stage.barriers) ?? [];
	}

	public getDiagnostics(): readonly WebGLFrameGraphDiagnostic[] {
		return this._compiled?.stages.flatMap((stage) => stage.diagnostics) ?? [];
	}

	public getResourceDebugState(): readonly WebGLFrameGraphResourceDebugState[] {
		const debug = this._attempts.getDebugState();
		const snapshot = debug.current ?? debug.lastAttempt;
		return snapshot?.resources.map((state) => ({
			id: state.id,
			initialized: state.active,
			lastNodeId: state.lastNodeId,
			lastAccess: toLegacyAccess(state.lastAccess),
			lastUsage: state.lastUsage ? toWebGLUsage(state.lastUsage)
				: state.lastAccess === "create" ? "external"
				: null,
		})) ?? [];
	}

	public getGraphAnalysis(): RenderGraphTrackerDebugState {
		return this._attempts.getDebugState();
	}
}

interface BuiltWebGLFrameDefinition {
	readonly definition: RenderGraphDefinition<WebGLFrameGraphNode, WebGLFrameGraphNode["kind"]>;
	readonly stageNodeIds: ReadonlyMap<WebGLFrameGraphStagePlan, readonly RenderGraphNodeId[]>;
}

function createDefinition(
	plan: WebGLFrameGraphFramePlan,
): BuiltWebGLFrameDefinition {
	const builder = new RenderGraphBuilder<WebGLFrameGraphNode, WebGLFrameGraphNode["kind"]>();
	for (const resource of plan.resources) builder.addResource(resource);
	for (const binding of plan.bindings) builder.addBinding(binding);
	for (const diagnostic of plan.shadowDiagnostics ?? []) builder.addShadowDiagnostic(diagnostic);
	builder.markCompleteness(plan.completeness ?? "complete");
	const stageNodeIds = new Map<WebGLFrameGraphStagePlan, readonly RenderGraphNodeId[]>();
	const terminalByStage = new Map<string, RenderGraphNodeId>();
	const buildDiagnostics: RenderGraphDiagnostic[] = [];
	const resourceById = new Map(plan.resources.map((resource) => [resource.id, resource]));
	for (const stage of plan.stages) {
		const dependencyNodes = (stage.pass.dependsOn ?? [])
			.map((dependencyStage) => terminalByStage.get(dependencyStage))
			.filter((nodeId): nodeId is RenderGraphNodeId => !!nodeId);
		if (stage.composition) {
			const composed = builder.addSubgraph(stage.composition.definition, {
				namespace: stage.composition.namespace,
				inputs: stage.composition.inputs,
				dependsOn: dependencyNodes,
			});
			const ids = stage.composition.definition.nodes
				.map((node) => composed.nodes[node.id])
				.filter((nodeId): nodeId is RenderGraphNodeId => !!nodeId);
			stageNodeIds.set(stage, Object.freeze(ids));
			if (ids.length > 0) terminalByStage.set(stage.pass.stage, ids[ids.length - 1]);
			continue;
		}
		let previousNodeId: RenderGraphNodeId | null = null;
		const ids: RenderGraphNodeId[] = [];
		for (let index = 0; index < stage.nodes.length; index++) {
			const node = stage.nodes[index];
			const sharedNodeId = renderGraphNodeId(node.id);
			const dependsOn = new Set(
				(node.dependsOn ?? []).map(renderGraphNodeId),
			);
			if (previousNodeId) dependsOn.add(previousNodeId);
			if (index === 0) for (const dependencyNode of dependencyNodes) dependsOn.add(dependencyNode);
			appendUnsupportedUsageDiagnostics(node, node.reads, buildDiagnostics);
			appendUnsupportedUsageDiagnostics(node, node.writes, buildDiagnostics);
			appendLegacyFeedbackDiagnostics(node, resourceById, buildDiagnostics);
			builder.addNode(toSharedNode(node, Array.from(dependsOn)));
			ids.push(sharedNodeId);
			previousNodeId = sharedNodeId;
		}
		stageNodeIds.set(stage, Object.freeze(ids));
		if (previousNodeId) terminalByStage.set(stage.pass.stage, previousNodeId);
	}
	for (const exported of plan.exports ?? []) builder.addExport(exported);
	const definition = builder.build();
	return {
		definition: Object.freeze({
			...definition,
			buildDiagnostics: Object.freeze([
				...(definition.buildDiagnostics ?? []),
				...buildDiagnostics,
			]),
		}),
		stageNodeIds,
	};
}

function appendLegacyFeedbackDiagnostics(
	node: WebGLFrameGraphNode,
	resources: ReadonlyMap<string, RenderGraphResourceDescriptor>,
	diagnostics: RenderGraphDiagnostic[],
): void {
	const sampled = new Set(
		(node.reads ?? [])
			.filter((reference) => reference.usage === "texture-sampling")
			.map((reference) => reference.id),
	);
	for (const write of node.writes ?? []) {
		if (
			!sampled.has(write.id) ||
			(write.usage !== "framebuffer-color" && write.usage !== "framebuffer-depth") ||
			resources.get(write.id)?.kind === "texture"
		) {
			continue;
		}
		diagnostics.push({
			phase: "lower",
			enforcement: "enforced",
			severity: "error",
			code: "texture-feedback-loop",
			backend: "webgl",
			stage: node.stage,
			nodeId: renderGraphNodeId(node.id),
			resourceId: renderGraphResourceId(write.id),
			message:
				`WebGL frame graph node "${node.id}" samples and writes resource ` +
				`"${write.id}" in the same framebuffer pass.`,
		});
	}
}

function projectStage(
	stage: WebGLFrameGraphStagePlan,
	stageNodeIds: readonly RenderGraphNodeId[],
	graph: CompiledRenderGraph<WebGLFrameGraphNode, WebGLFrameGraphNode["kind"]>,
): WebGLCompiledFrameGraphStage {
	const declaredIds = new Set(stageNodeIds);
	const nodes = graph.nodes
		.filter((node) => declaredIds.has(node.id))
		.map((node) => remapComposedNode(stage, node.id, node.payload!));
	const retainedIds = new Set(nodes.map((node) => node.id));
	const barriers = graph.transitions
		.filter((transition) => retainedIds.has(transition.nodeId) && transition.reason !== undefined)
		.map(toWebGLBarrier);
	const diagnostics = graph.diagnostics
		.filter((diagnostic) => diagnostic.nodeId && declaredIds.has(diagnostic.nodeId))
		.map(toWebGLDiagnostic)
		.filter((diagnostic): diagnostic is WebGLFrameGraphDiagnostic => diagnostic !== null);
	return Object.freeze({
		pass: stage.pass,
		nodes: Object.freeze(nodes),
		barriers: Object.freeze(barriers),
		diagnostics: Object.freeze(diagnostics),
	});
}

function remapComposedNode(
	stage: WebGLFrameGraphStagePlan,
	nodeId: RenderGraphNodeId,
	node: WebGLFrameGraphNode,
): WebGLFrameGraphNode {
	if (!stage.composition || !node.postProcess) return node.id === nodeId ? node : { ...node, id: nodeId };
	const remapResource = (
		resourceId: RenderGraphResourceId | null,
	): RenderGraphResourceId | null => {
		if (!resourceId) return null;
		const port = stage.composition!.definition.imports?.find(
			(candidate) => candidate.resource === resourceId,
		);
		return (port && stage.composition!.inputs[port.name]) ?? renderGraphResourceId(
			`${stage.composition!.namespace}:${resourceId}`,
		);
	};
	return {
		...node,
		id: nodeId,
		postProcess: {
			...node.postProcess,
			inputColor: remapResource(node.postProcess.inputColor),
			plannedOutputColor: remapResource(node.postProcess.plannedOutputColor),
		},
	};
}

function appendUnsupportedUsageDiagnostics(
	node: WebGLFrameGraphNode,
	refs: WebGLFrameGraphNode["reads"] | WebGLFrameGraphNode["writes"],
	diagnostics: RenderGraphDiagnostic[],
): void {
	for (const ref of refs ?? []) {
		if (SUPPORTED_WEBGL_RESOURCE_USAGES.has(ref.usage)) continue;
		diagnostics.push({
			phase: "lower",
			enforcement: "enforced",
			severity: "error",
			code: "unsupported-node-resource",
			backend: "webgl",
			stage: node.stage,
			nodeId: renderGraphNodeId(node.id),
			resourceId: renderGraphResourceId(ref.id),
			message:
				`WebGL frame graph node "${node.id}" references unsupported usage ` +
				`"${String(ref.usage)}" for resource "${ref.id}".`,
		});
	}
}

function toImportedDescriptor(id: string): RenderGraphResourceDescriptor {
	return {
		id: renderGraphResourceId(id),
		origin: "imported",
		kind: "external",
		residency: "frame",
		initialContent: "unknown",
	};
}

function toSharedNode(
	node: WebGLFrameGraphNode,
	dependsOn: readonly RenderGraphNodeId[],
): SharedWebGLNode {
	const resources: RenderGraphResourceRef[] = [];
	for (const ref of node.reads ?? []) {
		resources.push({
			resource: renderGraphResourceId(ref.id),
			access: "read",
			usage: toSharedUsage(ref.usage),
			optional: ref.optional,
		});
	}
	for (const ref of node.writes ?? []) {
		resources.push({
			resource: renderGraphResourceId(ref.id),
			access: "write",
			usage: toSharedUsage(ref.usage),
			optional: ref.optional,
		});
	}
	return {
		id: renderGraphNodeId(node.id),
		stage: node.stage,
		kind: node.kind,
		label: node.label,
		domain: node.domain,
		retention: node.retention,
		opaque: node.opaque,
		dependsOn,
		requires: node.requires?.map((requirement) => ({
			resource: renderGraphResourceId(requirement.id),
			optional: requirement.optional,
		})),
		creates: node.creates?.map((mutation) => ({
			resource: renderGraphResourceId(mutation.id),
			usage: mutation.usage ? toSharedUsage(mutation.usage) : undefined,
			optional: mutation.optional,
		})),
		resources,
		destroys: node.destroys?.map((mutation) => ({
			resource: renderGraphResourceId(mutation.id),
			usage: mutation.usage ? toSharedUsage(mutation.usage) : undefined,
			optional: mutation.optional,
		})),
		payload: node,
	};
}

function toSharedUsage(usage: WebGLFrameGraphResourceUsage): RenderGraphUsage {
	switch (usage) {
		case "framebuffer-color": return "color-attachment";
		case "framebuffer-depth": return "depth-attachment";
		case "texture-sampling": return "sampled";
		case "copy-source": return "copy-source";
		case "copy-target": return "copy-target";
		case "present": return "present";
		case "external": return "sampled";
		default: return "storage";
	}
}

function toWebGLUsage(usage: RenderGraphUsage): WebGLFrameGraphResourceUsage {
	switch (usage) {
		case "color-attachment": return "framebuffer-color";
		case "depth-attachment": return "framebuffer-depth";
		case "sampled": return "texture-sampling";
		case "copy-source": return "copy-source";
		case "copy-target": return "copy-target";
		case "present": return "present";
		default: return "external";
	}
}

function toWebGLBarrier(transition: RenderGraphTransition): WebGLFrameGraphBarrier {
	return {
		resource: transition.resourceId,
		beforeNodeId: transition.fromNodeId ?? null,
		nodeId: transition.nodeId,
		fromUsage: transition.previousUsage ? toWebGLUsage(transition.previousUsage) : null,
		toUsage: toWebGLUsage(transition.usage),
		reason: transition.reason ?? "usage-transition",
	};
}

function toWebGLDiagnostic(
	diagnostic: RenderGraphDiagnostic,
): WebGLFrameGraphDiagnostic | null {
	const code = diagnostic.code === "physical-feedback-loop"
		? "texture-feedback-loop"
		: diagnostic.code;
	if (
		code !== "read-before-create" &&
		code !== "duplicate-create" &&
		code !== "missing-resource" &&
		code !== "texture-feedback-loop" &&
		code !== "unsupported-node-resource" &&
		code !== "destroy-before-create"
	) {
		return null;
	}
	return {
		severity: diagnostic.severity,
		nodeId: diagnostic.nodeId ?? "unknown",
		resource: diagnostic.resourceId ?? "unknown",
		code: code as WebGLFrameGraphDiagnostic["code"],
		message: toWebGLDiagnosticMessage(diagnostic),
	};
}

function toWebGLDiagnosticMessage(diagnostic: RenderGraphDiagnostic): string {
	if (diagnostic.code === "duplicate-create") {
		return `WebGL frame graph node "${diagnostic.nodeId}" creates already ` +
			`active resource "${diagnostic.resourceId}".`;
	}
	if (diagnostic.code === "missing-resource") {
		return `WebGL frame graph node "${diagnostic.nodeId}" requires missing ` +
			`resource "${diagnostic.resourceId}".`;
	}
	return diagnostic.message.replace("Render graph node", "WebGL frame graph node");
}

function toLegacyAccess(
	access: "create" | "read" | "write" | "read-write" | "destroy" | null,
): "create" | "read" | "write" | "destroy" | null {
	return access === "read-write" ? "write" : access;
}
