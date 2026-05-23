import type { PostProcessIncrementalMetadata } from "../pipeline/incremental";
import type {
	FrameContext,
	FramePassStage,
} from "../pipeline/types";
import type {
	PostProcessCapabilities,
	ResolvedPostProcessState,
} from "../pipeline/PostProcessController";
import type { PostProcessPlacement } from "./ordering";

export type PostProcessBackendKind = "software" | "webgpu" | "webgl" | (string & {});

export type LogicalGBufferSemantic =
	| "color"
	| "depth"
	| "normal"
	| "motion"
	| "world-position"
	| "albedo"
	| "material"
	| "emissive"
	| "occlusion";

export type LogicalGBufferHandle =
	| {
			readonly backend: "software";
			readonly data: ArrayBufferView | null;
			readonly stride?: number;
	  }
	| {
			readonly backend: "webgpu";
			readonly texture: unknown;
	  }
	| {
			readonly backend: "webgl";
			readonly texture: WebGLTexture | null;
	  }
	| {
			readonly backend: string;
			readonly resource: unknown;
	  };

export interface LogicalGBufferChannel {
	readonly semantic: LogicalGBufferSemantic;
	readonly handle: LogicalGBufferHandle;
	readonly width: number;
	readonly height: number;
	readonly format?: string;
	readonly encoding?: string;
}

export interface LogicalGBufferBridge {
	readonly width: number;
	readonly height: number;
	readonly normalSpace: "world" | "view";
	readonly depthEncoding: "linear-view-z" | "ndc" | "hardware";
	readonly motionEncoding?: "ndc-delta";
	readonly channels: Partial<Record<LogicalGBufferSemantic, LogicalGBufferChannel>>;
	readonly worldPosition: {
		readonly source: "derived";
		readonly available: boolean;
	};
}

export interface PostProcessPassRequirements {
	readonly gBuffer?: readonly LogicalGBufferSemantic[];
	readonly history?: readonly string[];
}

export interface PostProcessHistoryDescriptor {
	readonly id: string;
	readonly widthScale?: number;
	readonly heightScale?: number;
	readonly format?: string;
	readonly usage?: readonly string[];
}

export interface PostProcessResourceDescriptor {
	readonly id: string;
	readonly width: number;
	readonly height: number;
	readonly format: string;
	readonly usage: readonly string[];
}

export interface PostProcessResourceHandle {
	readonly id: string;
	readonly backend: PostProcessBackendKind;
	readonly width: number;
	readonly height: number;
	readonly format: string;
	readonly resource: unknown;
}

export interface PostProcessHistorySlot {
	readonly id: string;
	readonly read: PostProcessResourceHandle;
	readonly write: PostProcessResourceHandle;
	readonly valid: boolean;
}

export type PostProcessHistorySlots = Record<string, PostProcessHistorySlot>;

export interface PostProcessPassImplementation {
	readonly id?: string;
}

export interface PostProcessPassDescriptor<TOptions = unknown> {
	/**
	 * Unique logical pass id used by resolved post-process state and backend
	 * executors.
	 */
	readonly id: string;
	/**
	 * Stable insertion bucket for custom pass scheduling. Built-in passes use
	 * their fixed built-in order regardless of this value.
	 */
	readonly placement?: PostProcessPlacement;
	/**
	 * Fine-grained ordering offset inside the selected custom placement bucket.
	 * This is not a dependency declaration.
	 */
	readonly order?: number;
	readonly requirements?: PostProcessPassRequirements;
	readonly history?: readonly PostProcessHistoryDescriptor[];
	readonly incremental?: PostProcessIncrementalMetadata;
	readonly isEnabled?: (state: ResolvedPostProcessState) => boolean;
	readonly implementations: Partial<
		Record<PostProcessBackendKind, PostProcessPassImplementation>
	>;
	readonly defaultOptions?: TOptions;
}

export interface PostProcessFrameRequest {
	readonly frameContext: FrameContext;
	readonly postProcess: ResolvedPostProcessState;
	readonly gBuffer: LogicalGBufferBridge;
	readonly histories: PostProcessHistorySlots;
}

export interface PostProcessPassRequest extends PostProcessFrameRequest {
	readonly pass: PostProcessPassDescriptor;
	readonly passId: string;
	readonly implementation: PostProcessPassImplementation;
	readonly options: unknown;
	readonly startPassId: string | null;
}

export interface PostProcessPassResult {
	readonly ran?: boolean;
	readonly historyUpdated?: boolean;
	readonly updatedHistoryIds?: readonly string[];
}

export interface PostProcessFrameEndRequest extends PostProcessFrameRequest {
	readonly executedPassIds: readonly string[];
}

export interface IPostProcessExecutor {
	readonly backend: PostProcessBackendKind;
	readonly capabilities: PostProcessCapabilities;
	createResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle;
	destroyResource(handle: PostProcessResourceHandle): void;
	beginFrame?(request: PostProcessFrameRequest): void | Promise<void>;
	executePass(
		passId: string,
		request: PostProcessPassRequest
	): PostProcessPassResult | Promise<PostProcessPassResult>;
	endFrame?(request: PostProcessFrameEndRequest): void | Promise<void>;
}

export interface PostProcessBackendSupport {
	/**
	 * Declares the logical post-process passes supported by the backend.
	 */
	readonly postProcessCapabilities: PostProcessCapabilities;
	/**
	 * Executes logical post-process passes resolved by `PostProcessPipeline`.
	 */
	readonly postProcessExecutor: IPostProcessExecutor;
	/**
	 * Creates the logical G-buffer view consumed by cross-backend passes.
	 *
	 * @param context Frame context containing backend attachments for the
	 * current render.
	 * @returns A backend-specific bridge mapped to logical G-buffer semantics.
	 * @remarks Implementations must not register graph passes or mutate the
	 * public post-process registry. Resource ownership remains with the backend.
	 */
	createPostProcessGBufferBridge(context: FrameContext): LogicalGBufferBridge;
}

export interface PostProcessPipelineExecuteRequest {
	readonly frameContext: FrameContext;
	readonly executor: IPostProcessExecutor;
	readonly gBuffer: LogicalGBufferBridge;
	readonly startPassId?: string | null;
	readonly warn?: (key: string, message: string) => void;
}

export interface PostProcessPipelineExecuteResult {
	readonly executedPassIds: readonly string[];
	readonly firstStage: FramePassStage | null;
	readonly startPassId: string | null;
}
