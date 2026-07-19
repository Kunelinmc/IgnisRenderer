import { RenderGraphStateTracker } from "../../../rendergraph/RenderGraphStateTracker";
import type {
	RenderGraphDiagnostic,
	RenderGraphNode,
	RenderGraphResourceDescriptor,
	RenderGraphTrackerDebugState,
	RenderGraphTransition,
	RenderGraphUsage,
	RenderGraphValidationRule,
} from "../../../rendergraph/types";
import type {
	WebGLCompiledFrameGraphStage,
	WebGLFrameGraphBarrier,
	WebGLFrameGraphDiagnostic,
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

/** Tracks WebGL frame graph state through the shared logical analyzer. */
export class WebGLFrameGraphCompiler {
	private readonly _tracker: RenderGraphStateTracker<
		WebGLFrameGraphNode,
		WebGLFrameGraphNode["kind"]
	>;
	private readonly _compiledStages: WebGLCompiledFrameGraphStage[] = [];
	private readonly _barriers: WebGLFrameGraphBarrier[] = [];
	private readonly _diagnostics: WebGLFrameGraphDiagnostic[] = [];

	public constructor() {
		this._tracker = new RenderGraphStateTracker({
			allowImplicitResources: true,
			rules: [createWebGLValidationRule()],
		});
	}

	public beginFrame(initialResources: readonly WebGLFrameGraphResourceId[]): void {
		const state = this._tracker.getDebugState().state;
		if (state === "active" || state === "sealed") {
			this._tracker.abort(
				new Error("A new WebGL frame superseded an uncommitted graph attempt."),
			);
		}
		this._compiledStages.length = 0;
		this._barriers.length = 0;
		this._diagnostics.length = 0;
		this._tracker.beginFrame(initialResources.map(toImportedDescriptor));
	}

	public compileStage(plan: WebGLFrameGraphStagePlan): WebGLCompiledFrameGraphStage {
		const analyzed = this._tracker.appendStage({
			nodes: plan.nodes.map(toSharedNode),
		});
		if (plan.pass.stage === "postprocess") {
			this._tracker.markCompleteness("coarse");
		}
		const barriers = analyzed.transitions
			.filter((transition) => transition.reason !== undefined)
			.map(toWebGLBarrier);
		const diagnostics = analyzed.diagnostics
			.map(toWebGLDiagnostic)
			.filter((diagnostic): diagnostic is WebGLFrameGraphDiagnostic => !!diagnostic);
		this._barriers.push(...barriers);
		this._diagnostics.push(...diagnostics);
		const compiled: WebGLCompiledFrameGraphStage = {
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
		return this._tracker.getResourceDebugState().map((state) => ({
			id: state.id,
			initialized: state.active,
			lastNodeId: state.lastNodeId,
			lastAccess: toLegacyAccess(state.lastAccess),
			lastUsage:
				state.lastUsage ? toWebGLUsage(state.lastUsage)
				: state.lastAccess === "create" ? "external"
				: null,
		}));
	}

	public getGraphAnalysis(): RenderGraphTrackerDebugState {
		return this._tracker.getDebugState();
	}
}

function createWebGLValidationRule(): RenderGraphValidationRule<
	WebGLFrameGraphNode,
	WebGLFrameGraphNode["kind"]
> {
	return {
		validateNode(node) {
			const source = node.payload;
			if (!source) return [];
			const diagnostics: RenderGraphDiagnostic[] = [];
			for (const ref of [...(source.reads ?? []), ...(source.writes ?? [])]) {
				if (SUPPORTED_WEBGL_RESOURCE_USAGES.has(ref.usage)) continue;
				diagnostics.push({
					phase: "lower",
					enforcement: "enforced",
					severity: "error",
					code: "unsupported-node-resource",
					backend: "webgl",
					stage: source.stage,
					nodeId: source.id,
					resourceId: ref.id,
					message:
						`WebGL frame graph node "${source.id}" references ` +
						`unsupported usage "${String(ref.usage)}" for resource ` +
						`"${ref.id}".`,
				});
			}
			const sampled = new Set(
				(source.reads ?? [])
					.filter((read) => read.usage === "texture-sampling")
					.map((read) => read.id),
			);
			for (const write of source.writes ?? []) {
				if (
					!sampled.has(write.id) ||
					(write.usage !== "framebuffer-color" &&
						write.usage !== "framebuffer-depth")
				) {
					continue;
				}
				diagnostics.push({
					phase: "lower",
					enforcement: "enforced",
					severity: "error",
					code: "texture-feedback-loop",
					backend: "webgl",
					stage: source.stage,
					nodeId: source.id,
					resourceId: write.id,
					message:
						`WebGL frame graph node "${source.id}" samples and writes ` +
						`resource "${write.id}" in the same framebuffer pass.`,
				});
			}
			return diagnostics;
		},
	};
}

function toImportedDescriptor(id: string): RenderGraphResourceDescriptor {
	return {
		id,
		origin: "imported",
		kind: "external",
		residency: "frame",
		initialContent: "unknown",
	};
}

function toSharedNode(node: WebGLFrameGraphNode): SharedWebGLNode {
	return {
		id: node.id,
		stage: node.stage,
		kind: node.kind,
		label: node.label,
		requires: node.requires?.map((requirement) => ({
			resource: requirement.id,
			optional: requirement.optional,
		})),
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

function toSharedUsage(usage: WebGLFrameGraphResourceUsage): RenderGraphUsage {
	switch (usage) {
		case "framebuffer-color":
			return "color-attachment";
		case "framebuffer-depth":
			return "depth-attachment";
		case "texture-sampling":
			return "sampled";
		case "copy-source":
			return "copy-source";
		case "copy-target":
			return "copy-target";
		case "present":
			return "present";
		case "external":
			return "sampled";
		default:
			return "storage";
	}
}

function toWebGLUsage(usage: RenderGraphUsage): WebGLFrameGraphResourceUsage {
	switch (usage) {
		case "color-attachment":
			return "framebuffer-color";
		case "depth-attachment":
			return "framebuffer-depth";
		case "sampled":
			return "texture-sampling";
		case "copy-source":
			return "copy-source";
		case "copy-target":
			return "copy-target";
		case "present":
			return "present";
		default:
			return "external";
	}
}

function toWebGLBarrier(transition: RenderGraphTransition): WebGLFrameGraphBarrier {
	return {
		resource: transition.resourceId,
		beforeNodeId: transition.fromNodeId ?? null,
		nodeId: transition.nodeId,
		fromUsage:
			transition.previousUsage ? toWebGLUsage(transition.previousUsage) : null,
		toUsage: toWebGLUsage(transition.usage),
		reason: transition.reason ?? "usage-transition",
	};
}

function toWebGLDiagnostic(
	diagnostic: RenderGraphDiagnostic,
): WebGLFrameGraphDiagnostic | null {
	if (
		diagnostic.code !== "read-before-create" &&
		diagnostic.code !== "duplicate-create" &&
		diagnostic.code !== "missing-resource" &&
		diagnostic.code !== "texture-feedback-loop" &&
		diagnostic.code !== "unsupported-node-resource" &&
		diagnostic.code !== "destroy-before-create"
	) {
		return null;
	}
	return {
		severity: diagnostic.severity,
		nodeId: diagnostic.nodeId ?? "unknown",
		resource: diagnostic.resourceId ?? "unknown",
		code: diagnostic.code,
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
