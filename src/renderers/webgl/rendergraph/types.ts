import type {
	FramePass,
	FramePassStage,
} from "../../../pipeline/types";
import type { PostProcessGraphDebugState } from "../../../postprocess/BackendPostProcessRuntime";

export type WebGLFrameGraphNodeKind =
	| "shadow"
	| "scene-clear"
	| "environment"
	| "opaque-depth-prepass"
	| "opaque-scene"
	| "transparent-legacy"
	| "oit-clear"
	| "oit-accum"
	| "oit-reveal"
	| "oit-resolve"
	| "particles"
	| "postprocess"
	| "present";

export const WEBGL_FRAME_GRAPH_NODE_KINDS = [
	"shadow",
	"scene-clear",
	"environment",
	"opaque-depth-prepass",
	"opaque-scene",
	"transparent-legacy",
	"oit-clear",
	"oit-accum",
	"oit-reveal",
	"oit-resolve",
	"particles",
	"postprocess",
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
	readonly creates?: readonly WebGLFrameGraphResourceMutation[];
	readonly requires?: readonly WebGLFrameGraphResourceMutation[];
	readonly reads?: readonly WebGLFrameGraphResourceRef[];
	readonly writes?: readonly WebGLFrameGraphResourceRef[];
	readonly destroys?: readonly WebGLFrameGraphResourceMutation[];
}

export interface WebGLFrameGraphPlannerState {
	readonly oitActive: boolean;
	readonly hasParticleSystems: boolean;
	readonly hasEnvironmentBackground: boolean;
}

export interface WebGLFrameGraphStagePlan {
	readonly pass: FramePass;
	readonly nodes: readonly WebGLFrameGraphNode[];
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
	readonly graphResources: readonly WebGLFrameGraphResourceDebugState[];
	readonly graphBarriers: readonly WebGLFrameGraphBarrier[];
	readonly graphDiagnostics: readonly WebGLFrameGraphDiagnostic[];
	readonly postProcess: PostProcessGraphDebugState;
}
