import type { WebGPUFrameCommitDebugState } from "./WebGPUFrameCommitter";
import type { WebGPUFrameSessionState } from "./WebGPUFrameSession";
import type { WebGPUFrameTargetManagerDebugState } from "./WebGPUFrameTargetManager";
import type {
	WebGPUCompiledFrameGraph,
	WebGPUCompiledFrameGraphStage,
} from "./types";

/** @internal Optional immutable diagnostics sink used by WebGPU runtime tests. */
export interface WebGPUFrameSessionTransitionSnapshot {
	readonly previous: WebGPUFrameSessionState | null;
	readonly next: WebGPUFrameSessionState | null;
}

export interface WebGPUFrameDiagnosticsObserver {
	onSessionTransition?(
		snapshot: Readonly<WebGPUFrameSessionTransitionSnapshot>,
	): void;
	onTargetsConfigured?(state: Readonly<WebGPUFrameTargetManagerDebugState>): void;
	onGraphCompiled?(
		graph: WebGPUCompiledFrameGraph | null,
		stages: readonly WebGPUCompiledFrameGraphStage[],
	): void;
	onNodeExecuted?(nodeId: string): void;
	onCommitSettled?(state: Readonly<WebGPUFrameCommitDebugState>): void;
}
