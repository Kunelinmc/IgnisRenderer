import type { FrameContext, FramePass } from "../../../pipeline/types";
import type { PostProcessColorDomain } from "../../../postprocess/PostProcessPass";
import type { RenderGraphResourceDescriptor } from "../../../rendergraph/types";
import type { WebGPUFrameNodeExecutor } from "./WebGPUFrameNodeExecutorRegistry";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import type {
	WebGPUFrameMessageHandler,
	WebGPUFrameMessageInput,
	WebGPUFrameMessageReader,
} from "./WebGPUFrameMessage";
import type {
	WebGPUComposedFrameGraphStage,
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
	finalizeRecording?(session: WebGPUFrameSession): void | Promise<void>;
	afterSubmit?(session: WebGPUFrameSession): void | Promise<void>;
	commitFrameState?(): void;
	abortFrameState?(error?: unknown): void;
	invalidateFrameResources?(): void;
	onDisplayOutputChanged?(): void;
	onShaderRuntimeChanged?(): void;
	destroy(): void;
}
