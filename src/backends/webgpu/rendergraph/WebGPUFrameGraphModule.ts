import type { FrameContext, FramePass } from "../../../pipeline/types";
import type { PostProcessColorDomain } from "../../../postprocess/PostProcessPass";
import type { RenderGraphResourceDescriptor } from "../../../rendergraph/types";
import type { FramePreparationRequirements } from "../../../pipeline/FrameRequirements";
import type { WebGPUFrameNodeExecutor } from "./WebGPUFrameNodeExecutorRegistry";
import type { WebGPUFrameExecutionContext } from "./WebGPUFrameExecutionContext";
import type { WebGPUFrameGraphCompiler } from "./WebGPUFrameGraphCompiler";
import type {
	WebGPUCommittingFrameSession,
	WebGPURecordingFrameSession,
} from "./WebGPUFrameSession";
import type {
	WebGPUFrameMessageHandler,
	WebGPUFrameMessageInput,
	WebGPUFrameMessageDescriptor,
	WebGPUFrameMessageReader,
} from "./WebGPUFrameMessage";
import type {
	WebGPUComposedFrameGraphStage,
	WebGPUCompiledFrameGraphStage,
	WebGPUFrameGraphNode,
	WebGPUFrameGraphNodeKind,
	WebGPUFrameResourceAllocationSnapshot,
	WebGPUFrameGraphResourceId,
} from "./types";

/** @internal Input shared with frame modules after analysis is sealed. */
export interface WebGPUFrameModulePlanningInput {
	readonly pass: FramePass;
	readonly context: FrameContext;
	readonly state: WebGPUFrameResourceAllocationSnapshot;
	readonly messages: WebGPUFrameMessageReader;
	readonly finalization?: boolean;
	readonly finalColorResource?: WebGPUFrameGraphResourceId;
}

export const WEBGPU_FRAME_STAGE_LANES = [
	"setup",
	"geometry",
	"lighting",
	"composite",
	"visibility",
	"transparent",
	"postprocess",
	"present",
] as const;

export type WebGPUFrameStageLane = (typeof WEBGPU_FRAME_STAGE_LANES)[number];

/** @internal Graph fragment from one backend-private module. */
export interface WebGPUFrameGraphContribution {
	readonly lane: WebGPUFrameStageLane;
	readonly before?: readonly string[];
	readonly after?: readonly string[];
	readonly exclusive?: boolean;
	readonly nodes?: readonly WebGPUFrameGraphNode[];
	readonly composition?: WebGPUComposedFrameGraphStage;
	readonly imports?: readonly RenderGraphResourceDescriptor[];
	readonly finalOutput?: {
		readonly resource: WebGPUFrameGraphResourceId;
		readonly colorDomain: PostProcessColorDomain;
	};
}

/**
 * Backend-private unit of WebGPU frame analysis, planning, execution, and
 * feature-local lifecycle.
 *
 * @internal Registered by the WebGPU runtime composition root. Applications
 * must use `Renderer.renderFrame()` instead.
 */
export interface WebGPUFrameGraphModule {
	readonly id: string;
	readonly messageHandlers?: readonly WebGPUFrameMessageHandler[];
	readonly planningInputs?: readonly WebGPUFrameMessageInput[];
	readonly executors: Readonly<
		Partial<Record<WebGPUFrameGraphNodeKind, WebGPUFrameNodeExecutor>>
	>;
	planStage?(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[];
	beginFrame?(context: FrameContext): void;
	syncFrame?(context: FrameContext): void;
	createAnalysisSeeds?(context: FrameContext): readonly {
		readonly descriptor: WebGPUFrameMessageDescriptor<unknown>;
		readonly value: unknown;
	}[];
	sealFrame?(context: FrameContext): FramePreparationRequirements | null;
	executeComposedStage?(
		stage: WebGPUCompiledFrameGraphStage | undefined,
		compiler: Pick<WebGPUFrameGraphCompiler, "recordSkippedNode">,
		recordExecutedNode: (nodeId: string) => void,
	): Promise<boolean>;
	activateFrame?(context: WebGPUFrameExecutionContext): void;
	closeFrame?(): void;
	finalizeRecording?(session: WebGPURecordingFrameSession): void | Promise<void>;
	afterSubmit?(session: WebGPUCommittingFrameSession): void | Promise<void>;
	commitFrameState?(): void;
	abortFrameState?(error?: unknown): void;
	invalidateFrameResources?(): void;
	onDisplayOutputChanged?(): void;
	onShaderRuntimeChanged?(): void;
	destroy(): void;
}
