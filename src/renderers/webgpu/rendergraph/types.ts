import type {
	FramePass,
	FramePassStage,
} from "../../../pipeline/types";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";

export type WebGPUFrameGraphNodeKind =
	| "shadow"
	| "planar-reflection-capture"
	| "opaque-scene"
	| "transparent-scene"
	| "oit-transparent"
	| "particles"
	| "oit-particles";

export interface WebGPUFrameGraphPlannerState {
	readonly deferredActive: boolean;
	readonly oitActive: boolean;
	readonly sceneTargetMode: WebGPUSceneTargetMode;
}

export interface WebGPUFrameGraphNode {
	readonly id: string;
	readonly stage: FramePassStage;
	readonly kind: WebGPUFrameGraphNodeKind;
	readonly label: string;
}

export interface WebGPUFrameGraphStagePlan {
	readonly pass: FramePass;
	readonly nodes: readonly WebGPUFrameGraphNode[];
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
}
