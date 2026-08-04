import type {
	FramePass,
	FramePassStage,
} from "../../../pipeline/types";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import type { WebGPUFrameGraphResourceId } from "./WebGPUFrameGraphResourceCatalog";
import type { WebGPUFrameCommitDebugState } from "./WebGPUFrameCommitter";
import type { PostProcessGraphDebugState } from "../../../postprocess/BackendPostProcessRuntime";
import type { PostProcessSubgraphNodePayload } from "../../../postprocess/PostProcessSubgraphBuilder";
import type { RenderGraphTrackerDebugState } from "../../../rendergraph/types";
import type {
	CompiledRenderGraph,
	RenderGraphAnalysisCompleteness,
	RenderGraphDiagnostic,
	RenderGraphExecutionDomain,
	RenderGraphExport,
	RenderGraphPhysicalBinding,
	RenderGraphResourceDescriptor,
	RenderGraphResourceId,
} from "../../../rendergraph/types";

export const WEBGPU_FRAME_GRAPH_NODE_KINDS = [
	"frame-setup",
	"opaque-external",
	"shadow",
	"paged-shadow-page-mark",
	"paged-shadow-page-allocate",
	"paged-shadow-page-table-copy",
	"paged-shadow-depth",
	"paged-shadow-feedback",
	"planar-reflection-capture",
	"planar-reflection-composite",
	"opaque-scene",
	"deferred-decal",
	"deferred-lighting",
	"hiz-build",
	"occlusion-test",
	"transparent-forward",
	"oit-prepare",
	"oit-clear",
	"oit-mesh-accumulate",
	"oit-particle-accumulate",
	"oit-resolve",
	"transmission",
	"particle-alpha-forward",
	"particle-additive",
	"post-process-pass",
	"presentation",
] as const;

export type WebGPUFrameGraphNodeKind =
	(typeof WEBGPU_FRAME_GRAPH_NODE_KINDS)[number];

export interface WebGPUFrameGraphPlannerState {
	readonly deferredActive: boolean;
	readonly oitActive: boolean;
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly hasFrameTargets?: boolean;
	readonly hasMSAATargets?: boolean;
	readonly needsTransmissionTargets?: boolean;
	readonly needsPlanarReflectionMask?: boolean;
	readonly needsOcclusionTest?: boolean;
	readonly needsHiZBuild?: boolean;
	readonly needsPlanarReflectionComposite?: boolean;
	readonly hasOITMeshContributors?: boolean;
	readonly hasTransmissionPackets?: boolean;
	readonly hasAlphaBillboardParticles?: boolean;
	readonly hasAdditiveBillboardParticles?: boolean;
}

export type { WebGPUFrameGraphResourceId } from "./WebGPUFrameGraphResourceCatalog";

export type WebGPUFrameGraphResourceUsage =
	| "external"
	| "render-attachment"
	| "depth-attachment"
	| "texture-binding"
	| "storage-binding"
	| "copy-src"
	| "copy-dst"
	| "present";

export interface WebGPUFrameGraphResourceRef {
	readonly id: WebGPUFrameGraphResourceId;
	readonly usage: WebGPUFrameGraphResourceUsage;
	readonly optional?: boolean;
}

export interface WebGPUFrameGraphResourceMutation {
	readonly id: WebGPUFrameGraphResourceId;
	readonly usage?: WebGPUFrameGraphResourceUsage;
	readonly optional?: boolean;
}

export interface WebGPUFrameGraphNode {
	readonly id: string;
	readonly stage: FramePassStage;
	readonly kind: WebGPUFrameGraphNodeKind;
	readonly label: string;
	readonly domain?: RenderGraphExecutionDomain;
	readonly retention?: "always" | "if-reachable";
	readonly opaque?: boolean;
	readonly dependsOn?: readonly string[];
	readonly creates?: readonly WebGPUFrameGraphResourceMutation[];
	readonly reads?: readonly WebGPUFrameGraphResourceRef[];
	readonly writes?: readonly WebGPUFrameGraphResourceRef[];
	readonly destroys?: readonly WebGPUFrameGraphResourceMutation[];
	readonly postProcess?: PostProcessSubgraphNodePayload;
}

export interface WebGPUComposedFrameGraphStage {
	readonly namespace: string;
	readonly definition: import("../../../rendergraph/types").RenderGraphDefinition<
		WebGPUFrameGraphNode,
		WebGPUFrameGraphNodeKind
	>;
	readonly inputs: Readonly<Record<string, RenderGraphResourceId>>;
}

export interface WebGPUFrameGraphStagePlan {
	readonly pass: FramePass;
	readonly nodes: readonly WebGPUFrameGraphNode[];
	readonly composition?: WebGPUComposedFrameGraphStage;
}

export interface WebGPUFrameGraphFramePlan {
	readonly resources: readonly RenderGraphResourceDescriptor[];
	readonly bindings: readonly RenderGraphPhysicalBinding[];
	readonly stages: readonly WebGPUFrameGraphStagePlan[];
	readonly exports?: readonly RenderGraphExport[];
	readonly completeness?: RenderGraphAnalysisCompleteness;
	readonly shadowDiagnostics?: readonly RenderGraphDiagnostic[];
}

export type WebGPUFrameGraphValidationMode = "throw" | "warn";

export interface WebGPUFrameGraphBarrier {
	readonly resource: WebGPUFrameGraphResourceId;
	readonly beforeNodeId: string | null;
	readonly nodeId: string;
	readonly fromUsage: WebGPUFrameGraphResourceUsage | null;
	readonly toUsage: WebGPUFrameGraphResourceUsage;
	readonly reason:
		| "read-after-write"
		| "write-after-read"
		| "write-after-write"
		| "usage-transition";
}

export interface WebGPUFrameGraphDiagnostic {
	readonly severity: "error" | "warning";
	readonly nodeId: string;
	readonly resource: WebGPUFrameGraphResourceId;
	readonly code:
		| "read-before-create"
		| "destroy-before-create"
		| "duplicate-create"
		| "physical-feedback-loop";
	readonly message: string;
}

export interface WebGPUCompiledFrameGraphStage {
	readonly pass: FramePass;
	readonly nodes: readonly WebGPUFrameGraphNode[];
	readonly barriers: readonly WebGPUFrameGraphBarrier[];
	readonly diagnostics: readonly WebGPUFrameGraphDiagnostic[];
}

export interface WebGPUCompiledFrameGraph {
	readonly graph: CompiledRenderGraph<WebGPUFrameGraphNode, WebGPUFrameGraphNodeKind>;
	readonly stages: readonly WebGPUCompiledFrameGraphStage[];
}

export interface WebGPUFrameGraphResourceDebugState {
	readonly id: WebGPUFrameGraphResourceId;
	readonly initialized: boolean;
	readonly lastNodeId: string | null;
	readonly lastAccess: "create" | "read" | "write" | "destroy" | null;
	readonly lastUsage: WebGPUFrameGraphResourceUsage | null;
}

export interface WebGPUFrameGraphTargetDebugState {
	readonly width: number;
	readonly height: number;
	readonly sampleCount: number;
	readonly texturePoolOwnerCount: number;
	readonly frameTargets: unknown;
	readonly msaaTargets: unknown;
}

export interface WebGPUFrameGraphDebugState {
	readonly active: boolean;
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly deferredActive: boolean;
	readonly oitActive: boolean;
	readonly targetWidth: number;
	readonly targetHeight: number;
	readonly texturePoolOwnerCount: number;
	readonly frameTargets: unknown;
	readonly msaaTargets: unknown;
	readonly motionHistoryWriteTarget: unknown;
	readonly pendingFrameTargetInvalidation: boolean;
	readonly pendingShaderRuntimeInvalidation: boolean;
	readonly hiZ: {
		readonly allocated: boolean;
		readonly status: "unavailable" | "pending" | "ready" | "failed";
		readonly width: number;
		readonly height: number;
		readonly mipLevelCount: number;
		readonly buildCount: number;
	};
	readonly lastPlannedNodeIds: readonly string[];
	readonly lastExecutedNodeIds: readonly string[];
	readonly compiledStages: readonly WebGPUCompiledFrameGraphStage[];
	readonly compiledGraph: CompiledRenderGraph<WebGPUFrameGraphNode, WebGPUFrameGraphNodeKind> | null;
	readonly graphResources: readonly WebGPUFrameGraphResourceDebugState[];
	readonly graphBarriers: readonly WebGPUFrameGraphBarrier[];
	readonly graphDiagnostics: readonly WebGPUFrameGraphDiagnostic[];
	readonly graphAnalysis: RenderGraphTrackerDebugState;
	readonly targetManager: WebGPUFrameGraphTargetDebugState;
	readonly commit: WebGPUFrameCommitDebugState | null;
	readonly postProcess: PostProcessGraphDebugState;
}
