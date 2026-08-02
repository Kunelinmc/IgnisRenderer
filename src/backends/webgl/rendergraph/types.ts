import type {
	FramePass,
	FramePassStage,
} from "../../../pipeline/types";
import type { PostProcessGraphDebugState } from "../../../postprocess/BackendPostProcessRuntime";
import type { PostProcessSubgraphNodePayload } from "../../../postprocess/PostProcessSubgraphBuilder";
import type {
	CompiledRenderGraph,
	RenderGraphAnalysisCompleteness,
	RenderGraphDiagnostic,
	RenderGraphExecutionDomain,
	RenderGraphExport,
	RenderGraphPhysicalBinding,
	RenderGraphResourceDescriptor,
	RenderGraphResourceId,
	RenderGraphTrackerDebugState,
} from "../../../rendergraph/types";

export type WebGLFrameGraphNodeKind =
	| "frame-setup"
	| "opaque-external"
	| "shadow"
	| "scene-clear"
	| "environment"
	| "opaque-depth-prepass"
	| "opaque-scene"
	| "transparent-legacy"
	| "oit-clear"
	| "oit-accum"
	| "oit-reveal"
	| "oit-copy-scene-color"
	| "oit-resolve"
	| "particles"
	| "post-process-pass"
	| "present";

export const WEBGL_FRAME_GRAPH_NODE_KINDS = [
	"frame-setup",
	"opaque-external",
	"shadow",
	"scene-clear",
	"environment",
	"opaque-depth-prepass",
	"opaque-scene",
	"transparent-legacy",
	"oit-clear",
	"oit-accum",
	"oit-reveal",
	"oit-copy-scene-color",
	"oit-resolve",
	"particles",
	"post-process-pass",
	"present",
] as const satisfies readonly WebGLFrameGraphNodeKind[];

export type WebGLFrameGraphResourceId = string;

export type WebGLFrameGraphResourceUsage =
	| "external"
	| "framebuffer-color"
	| "framebuffer-depth"
	| "texture-sampling"
	| "copy-source"
	| "copy-target"
	| "present";

export interface WebGLFrameGraphResourceRef {
	readonly id: WebGLFrameGraphResourceId;
	readonly usage: WebGLFrameGraphResourceUsage;
	readonly optional?: boolean;
}

export interface WebGLFrameGraphResourceMutation {
	readonly id: WebGLFrameGraphResourceId;
	readonly usage?: WebGLFrameGraphResourceUsage;
	readonly optional?: boolean;
}

export type WebGLFrameGraphNodeScope = "frame" | "transparent" | "particles";

export interface WebGLFrameGraphNode {
	readonly id: string;
	readonly stage: FramePassStage;
	readonly kind: WebGLFrameGraphNodeKind;
	readonly label: string;
	readonly scope?: WebGLFrameGraphNodeScope;
	readonly domain?: RenderGraphExecutionDomain;
	readonly retention?: "always" | "if-reachable";
	readonly opaque?: boolean;
	readonly dependsOn?: readonly string[];
	readonly creates?: readonly WebGLFrameGraphResourceMutation[];
	readonly requires?: readonly WebGLFrameGraphResourceMutation[];
	readonly reads?: readonly WebGLFrameGraphResourceRef[];
	readonly writes?: readonly WebGLFrameGraphResourceRef[];
	readonly destroys?: readonly WebGLFrameGraphResourceMutation[];
	readonly postProcess?: PostProcessSubgraphNodePayload;
}

export interface WebGLComposedFrameGraphStage {
	readonly namespace: string;
	readonly definition: import("../../../rendergraph/types").RenderGraphDefinition<
		WebGLFrameGraphNode,
		WebGLFrameGraphNodeKind
	>;
	readonly inputs: Readonly<Record<string, RenderGraphResourceId>>;
}

export interface WebGLFrameGraphPlannerState {
	readonly oitActive: boolean;
	readonly hasParticleSystems: boolean;
	readonly hasEnvironmentBackground: boolean;
}

export interface WebGLFrameGraphStagePlan {
	readonly pass: FramePass;
	readonly nodes: readonly WebGLFrameGraphNode[];
	readonly composition?: WebGLComposedFrameGraphStage;
}

export interface WebGLFrameGraphFramePlan {
	readonly resources: readonly RenderGraphResourceDescriptor[];
	readonly bindings: readonly RenderGraphPhysicalBinding[];
	readonly stages: readonly WebGLFrameGraphStagePlan[];
	readonly exports?: readonly RenderGraphExport[];
	readonly completeness?: RenderGraphAnalysisCompleteness;
	readonly shadowDiagnostics?: readonly RenderGraphDiagnostic[];
}

export interface WebGLFrameGraphResourceCatalogSnapshot {
	readonly resources: readonly RenderGraphResourceDescriptor[];
	readonly bindings: readonly RenderGraphPhysicalBinding[];
}

export interface WebGLFrameGraphBarrier {
	readonly resource: WebGLFrameGraphResourceId;
	readonly beforeNodeId: string | null;
	readonly nodeId: string;
	readonly fromUsage: WebGLFrameGraphResourceUsage | null;
	readonly toUsage: WebGLFrameGraphResourceUsage;
	readonly reason:
		| "read-after-write"
		| "write-after-read"
		| "write-after-write"
		| "usage-transition";
}

export interface WebGLFrameGraphDiagnostic {
	readonly severity: "error" | "warning";
	readonly nodeId: string;
	readonly resource: WebGLFrameGraphResourceId;
	readonly code:
		| "read-before-create"
		| "duplicate-create"
		| "missing-resource"
		| "texture-feedback-loop"
		| "unsupported-node-resource"
		| "missing-node-executor"
		| "destroy-before-create";
	readonly message: string;
}

export interface WebGLCompiledFrameGraphStage {
	readonly pass: FramePass;
	readonly nodes: readonly WebGLFrameGraphNode[];
	readonly barriers: readonly WebGLFrameGraphBarrier[];
	readonly diagnostics: readonly WebGLFrameGraphDiagnostic[];
}

export interface WebGLCompiledFrameGraph {
	readonly graph: CompiledRenderGraph<WebGLFrameGraphNode, WebGLFrameGraphNodeKind>;
	readonly stages: readonly WebGLCompiledFrameGraphStage[];
}

export interface WebGLFrameGraphResourceDebugState {
	readonly id: WebGLFrameGraphResourceId;
	readonly initialized: boolean;
	readonly lastNodeId: string | null;
	readonly lastAccess: "create" | "read" | "write" | "destroy" | null;
	readonly lastUsage: WebGLFrameGraphResourceUsage | null;
}

export interface WebGLFrameGraphDebugState {
	readonly active: boolean;
	readonly oitActive: boolean;
	readonly hasPresentedInFrame: boolean;
	readonly lastPlannedNodeIds: readonly string[];
	readonly lastExecutedNodeIds: readonly string[];
	readonly compiledStages: readonly WebGLCompiledFrameGraphStage[];
	readonly compiledGraph: CompiledRenderGraph<WebGLFrameGraphNode, WebGLFrameGraphNodeKind> | null;
	readonly graphResources: readonly WebGLFrameGraphResourceDebugState[];
	readonly graphBarriers: readonly WebGLFrameGraphBarrier[];
	readonly graphDiagnostics: readonly WebGLFrameGraphDiagnostic[];
	readonly graphAnalysis: RenderGraphTrackerDebugState;
	readonly postProcess: PostProcessGraphDebugState;
}
