import type {
	FramePass,
	FramePassStage,
} from "../../../pipeline/types";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";

export type WebGPUFrameGraphNodeKind =
	| "shadow"
	| "planar-reflection-capture"
	| "opaque-scene"
	| "deferred-decal"
	| "deferred-lighting"
	| "transparent-scene"
	| "oit-transparent"
	| "particles"
	| "oit-particles";

export interface WebGPUFrameGraphPlannerState {
	readonly deferredActive: boolean;
	readonly oitActive: boolean;
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly hasFrameTargets?: boolean;
	readonly hasMSAATargets?: boolean;
	readonly needsPlanarReflectionMask?: boolean;
}

export type WebGPUFrameGraphResourceId = string;

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
	readonly creates?: readonly WebGPUFrameGraphResourceMutation[];
	readonly reads?: readonly WebGPUFrameGraphResourceRef[];
	readonly writes?: readonly WebGPUFrameGraphResourceRef[];
	readonly destroys?: readonly WebGPUFrameGraphResourceMutation[];
}

export interface WebGPUFrameGraphStagePlan {
	readonly pass: FramePass;
	readonly nodes: readonly WebGPUFrameGraphNode[];
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
		| "duplicate-create";
	readonly message: string;
}

export interface WebGPUCompiledFrameGraphStage {
	readonly pass: FramePass;
	readonly nodes: readonly WebGPUFrameGraphNode[];
	readonly barriers: readonly WebGPUFrameGraphBarrier[];
	readonly diagnostics: readonly WebGPUFrameGraphDiagnostic[];
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
	readonly msaaSampleCount: number;
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
	readonly lastPlannedNodeIds: readonly string[];
	readonly lastExecutedNodeIds: readonly string[];
	readonly compiledStages: readonly WebGPUCompiledFrameGraphStage[];
	readonly graphResources: readonly WebGPUFrameGraphResourceDebugState[];
	readonly graphBarriers: readonly WebGPUFrameGraphBarrier[];
	readonly graphDiagnostics: readonly WebGPUFrameGraphDiagnostic[];
	readonly targetManager: WebGPUFrameGraphTargetDebugState;
}
