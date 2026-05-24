import type { PostProcessIncrementalMetadata } from "../pipeline/incremental";
import type {
	FrameContext,
	FramePassStage,
} from "../pipeline/types";
import type {
	PostProcessCapabilities,
	PostProcessPass,
	PostProcessPassRegistrySnapshot,
	PostProcessPassResolveRequest,
	PostProcessPassWarmupRequest,
} from "./PostProcessPass";
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

export interface PostProcessHistoryResolveRequest {
	readonly frameContext: FrameContext;
	readonly postProcess: PostProcessPassRegistrySnapshot;
	readonly backend: PostProcessBackendKind;
	readonly gBuffer: LogicalGBufferBridge;
	readonly width: number;
	readonly height: number;
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

export interface PostProcessPassImplementation<
	TContext = unknown,
	TOptions = unknown,
> {
	readonly id?: string;
	/**
	 * Executes the logical pass through pass-owned backend implementation logic.
	 *
	 * @param request Current pass request produced by `PostProcessPipeline`.
	 * @param context Backend-provided low-level execution helpers.
	 * @returns Pass execution result used for scheduling and history tracking.
	 * @sideEffects May mutate backend render targets or post-process histories.
	 */
	execute?(
		request: PostProcessPassRequest<TOptions>,
		context: TContext
	): PostProcessPassResult | Promise<PostProcessPassResult>;
	/**
	 * Prepares backend resources needed by this implementation.
	 *
	 * @param context Backend-provided low-level execution helpers.
	 * @returns Nothing.
	 * @sideEffects May allocate backend-owned pipelines, programs, or buffers.
	 */
	warmup?(
		context: TContext,
		request?: PostProcessPassWarmupRequest<TOptions>
	): void | Promise<void>;
	/**
	 * Invalidates cached backend bindings for this implementation.
	 *
	 * @returns Nothing.
	 * @sideEffects Drops implementation-owned binding caches.
	 */
	invalidate?(): void;
	/**
	 * Destroys resources owned by this implementation.
	 *
	 * @returns Nothing.
	 * @sideEffects Releases implementation-owned backend resources.
	 */
	destroy?(): void;
}

export interface PostProcessFrameRequest {
	readonly frameContext: FrameContext;
	readonly postProcess: PostProcessPassRegistrySnapshot;
	readonly gBuffer: LogicalGBufferBridge;
	readonly histories: PostProcessHistorySlots;
}

export interface PostProcessPassRequest<TOptions = unknown>
	extends PostProcessFrameRequest {
	readonly pass: PostProcessPass<unknown, TOptions>;
	readonly passId: string;
	readonly options: TOptions;
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
	createResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle;
	destroyResource(handle: PostProcessResourceHandle): void;
	beginFrame?(request: PostProcessFrameRequest): void | Promise<void>;
	/**
	 * Provides backend-specific low-level helpers to pass-owned implementations.
	 *
	 * @param passId Logical pass id being executed.
	 * @param request Current pass request.
	 * @returns Backend context object consumed by the pass implementation.
	 * @sideEffects May synchronize backend history handles into frame targets.
	 */
	getPassExecutionContext?(
		passId: string,
		request: PostProcessPassRequest
	): unknown;
	getPassWarmupContext?(
		passId: string,
		request: PostProcessPassWarmupRequest
	): unknown;
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
