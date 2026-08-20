import type {
	FrameContext,
} from "../pipeline/types";
import type { FramePreparationRequirements } from "../pipeline/FrameRequirements";
import type { RenderGraphAccess, RenderGraphUsage } from "../rendergraph/types";
import type { RenderBackendType } from "../backends/IRenderBackend";
import type {
	PostProcessPass,
	PostProcessPassRegistrySnapshot,
	PostProcessPassResolveRequest,
	PostProcessPassWarmupRequest,
} from "./PostProcessPass";

/**
 * @deprecated Use `RenderBackendType` from `backends/IRenderBackend`.
 */
export type PostProcessBackendKind = RenderBackendType;

export type LogicalGBufferSemantic =
	| "color"
	| "depth"
	| "normal"
	| "motion"
	| "world-position"
	| "albedo"
	| "roughness"
	| "metallic"
	| "specular"
	| "transmission"
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
			readonly backend: RenderBackendType;
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
	/** Coordinate space of vectors stored in the logical `normal` channel. */
	readonly normalSpace: "world" | "view";
	readonly depthEncoding: "linear-view-z" | "ndc" | "hardware";
	readonly motionEncoding?: "ndc-delta";
	readonly channels: Partial<Record<LogicalGBufferSemantic, LogicalGBufferChannel>>;
	readonly worldPosition: {
		readonly source: "derived";
		readonly available: boolean;
	};
}

export interface PostProcessBaseResourceDescriptor {
	readonly id: string;
	readonly format?: string;
	readonly usage?: readonly string[];
}

export type PostProcessResourceMipMode = "single" | "full-chain";

export interface PostProcessScaledResourceDescriptor
	extends PostProcessBaseResourceDescriptor {
	readonly widthScale?: number;
	readonly heightScale?: number;
}

export interface PostProcessHistoryDescriptor
	extends PostProcessScaledResourceDescriptor {}

export interface PostProcessTransientDescriptor
	extends PostProcessScaledResourceDescriptor {
	readonly mipMode?: PostProcessResourceMipMode;
}

export interface PostProcessHistoryResolveRequest {
	readonly frameContext: FrameContext;
	readonly postProcess: PostProcessPassRegistrySnapshot;
	readonly backend: RenderBackendType;
	readonly gBuffer: LogicalGBufferBridge;
	readonly width: number;
	readonly height: number;
}

export interface PostProcessResourceDescriptor
	extends PostProcessBaseResourceDescriptor {
	readonly width: number;
	readonly height: number;
	readonly format: string;
	readonly usage: readonly string[];
	readonly mipMode?: PostProcessResourceMipMode;
}

export interface PostProcessResourceHandle {
	readonly id: string;
	readonly backend: RenderBackendType;
	readonly width: number;
	readonly height: number;
	readonly format: string;
	readonly mipMode?: PostProcessResourceMipMode;
	readonly resource: unknown;
}

export interface PostProcessHistorySlot {
	readonly id: string;
	readonly read: PostProcessResourceHandle;
	readonly write: PostProcessResourceHandle;
	readonly valid: boolean;
}

export type PostProcessHistorySlots = Record<string, PostProcessHistorySlot>;

export interface PostProcessTransientSlot {
	readonly id: string;
	readonly handle: PostProcessResourceHandle;
}

export type PostProcessTransientSlots = Record<string, PostProcessTransientSlot>;

/** Logical color behavior declared by one backend implementation. */
export interface PostProcessColorDeclaration {
	readonly access: "none" | "read" | "read-write";
	readonly output: "preserve" | "new-version";
}

/** One logical resource use declared by a post-process implementation. */
export interface PostProcessExecutionResourceUse {
	readonly access: RenderGraphAccess;
	readonly usage: RenderGraphUsage;
	readonly optional?: boolean;
}

export interface PostProcessGBufferDeclaration
	extends PostProcessExecutionResourceUse {
	readonly semantic: LogicalGBufferSemantic;
}

export interface PostProcessHistoryDeclaration {
	readonly descriptor: PostProcessHistoryDescriptor;
	readonly read: readonly PostProcessExecutionResourceUse[];
	readonly write: readonly PostProcessExecutionResourceUse[];
}

export interface PostProcessTransientDeclaration {
	readonly descriptor: PostProcessTransientDescriptor;
	readonly uses: readonly PostProcessExecutionResourceUse[];
}

export interface PostProcessSharedResourceDeclaration
	extends PostProcessExecutionResourceUse {
	readonly id: string;
}

/** Complete execution contract returned once by one backend implementation. */
export interface PostProcessExecutionDeclaration {
	readonly color: PostProcessColorDeclaration;
	readonly gBuffer?: readonly PostProcessGBufferDeclaration[];
	readonly histories?: readonly PostProcessHistoryDeclaration[];
	readonly transients?: readonly PostProcessTransientDeclaration[];
	readonly shared?: readonly PostProcessSharedResourceDeclaration[];
	/** Generic pre-scene effects required before this implementation executes. */
	readonly frameRequirements?: FramePreparationRequirements;
}

export interface PostProcessNativeHistorySlot<TResource> {
	readonly read: TResource | null;
	readonly write: TResource | null;
	readonly valid: boolean;
}

/** Fixed declaration-checked resource view supplied by one backend. */
export interface PostProcessResourceAccessor<TResource = unknown> {
	readonly color: {
		readonly input: TResource | null;
		readonly output: TResource | null;
	};
	getGBuffer(semantic: LogicalGBufferSemantic): TResource | null;
	getHistory(id: string): PostProcessNativeHistorySlot<TResource>;
	getTransient(id: string): TResource | null;
	getShared(id: string): TResource | null;
	copyGBufferToHistory(
		semantic: LogicalGBufferSemantic,
		historyId: string
	): void | Promise<void>;
}

export interface PostProcessPassImplementation<
	TContext = unknown,
	TOptions = unknown,
> {
	readonly id?: string;
	readonly warmupHints?: readonly string[];
	/**
	 * Returns the complete resource contract for this backend and frame.
	 * @internal Owned by the post-process planner.
	 * @param request Resolved options and frame information.
	 * @returns Immutable logical resource declaration.
	 * @sideEffects None.
	 */
	describeExecution(
		request: PostProcessPassResolveRequest<TOptions>
	): PostProcessExecutionDeclaration;
	/**
	 * Executes the logical pass through pass-owned backend implementation logic.
	 *
	 * @param request Current pass request produced by backend post-process runtime.
	 * @param context Backend-provided low-level execution helpers.
	 * @returns Pass execution result used for scheduling and history tracking.
	 * @sideEffects May mutate backend render targets or post-process histories.
	 */
	execute(
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
	readonly transients: PostProcessTransientSlots;
}

export interface PostProcessPassRequest<TOptions = unknown>
	extends PostProcessFrameRequest {
	readonly pass: PostProcessPass<unknown, TOptions>;
	readonly passId: string;
	readonly implementation:
		| PostProcessPassImplementation<unknown, TOptions>
		| null;
	readonly options: TOptions;
	readonly startPassId: string | null;
	readonly declaration: PostProcessExecutionDeclaration;
}

/**
 * Backend context request for a pass-owned post-process implementation.
 *
 * Backend post-process runtime creates this only when the selected backend
 * implementation exposes `execute()`. Backend executors must use `pass` and
 * `implementation` from this request as the contract source instead of
 * reclassifying passes from string ids alone.
 */
export interface PostProcessPassExecutionContextRequest<TOptions = unknown>
	extends PostProcessPassRequest<TOptions> {
	readonly implementation: PostProcessPassImplementation<unknown, TOptions>;
}

export interface PostProcessPassResult {
	readonly ran: boolean;
	readonly preservesOutsideDirtyTiles?: boolean;
	readonly updatedHistoryIds?: readonly string[];
}

/** @internal Backend-owned outcome of automatic assigned-color commit. */
export interface PostProcessPassCompletion {
	readonly committed?: boolean;
	readonly physicalId?: string;
}

/** @internal Backend-private binding transaction for one logical post-process graph. */
export interface PostProcessGraphFrameBinding {
	beginPass?(request: PostProcessPassRequest): void | Promise<void>;
	completePass?(
		request: PostProcessPassRequest,
		result: PostProcessPassResult
	): PostProcessPassCompletion | void | Promise<PostProcessPassCompletion | void>;
	endFrame?(resolvedOutputColor: string): void | Promise<void>;
	abortFrame?(error?: unknown): void | Promise<void>;
}

export interface PostProcessFrameEndRequest extends PostProcessFrameRequest {
	readonly executedPassIds: readonly string[];
}

export interface PostProcessFrameAbortRequest extends PostProcessFrameRequest {
	readonly executedPassIds: readonly string[];
	readonly error?: unknown;
}

export interface IPostProcessExecutor {
	readonly backend: RenderBackendType;
	/** @internal Physical coordinate space used by the backend normal channel. */
	readonly gBufferNormalSpace: LogicalGBufferBridge["normalSpace"];
	/**
	 * Creates the logical G-buffer view consumed by cross-backend passes.
	 *
	 * @param context Frame context containing backend attachments for the
	 * current render.
	 * @returns A backend-specific bridge mapped to logical G-buffer semantics.
	 * @sideEffects None. Resource ownership remains with the backend.
	 */
	createGBufferBridge(context: FrameContext): LogicalGBufferBridge;
	/** @internal Reports whether a backend-shared graph resource is ready. */
	isGraphResourceAvailable?(resourceId: string): boolean;
	/** @internal Opens a backend-private logical-to-physical binding transaction. */
	createGraphBinding?(request: PostProcessFrameRequest): PostProcessGraphFrameBinding;
	createResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle;
	destroyResource(handle: PostProcessResourceHandle): void;
	/**
	 * Invalidates backend binding caches that may retain destroyed frame resources.
	 *
	 * @returns Nothing.
	 * @sideEffects Drops backend-owned bind groups or equivalent cached bindings.
	 */
	invalidateResourceBindings?(): void;
	beginFrame?(request: PostProcessFrameRequest): void | Promise<void>;
	/**
	 * Provides backend-specific low-level helpers to pass-owned implementations.
	 *
	 * @param request Current pass-owned implementation context request.
	 * @returns Backend context object consumed by the pass implementation.
	 * @sideEffects May create a per-pass backend context and reset pending
	 * backend pass output state.
	 */
	createPassExecutionContext?(
		request: PostProcessPassExecutionContextRequest
	): unknown;
	/**
	 * Applies backend-owned side effects recorded while one logical pass ran.
	 *
	 * @param request Pass request that just completed.
	 * @param result Result returned by the pass implementation or executor.
	 * @returns Nothing.
	 * @constraints Implementations must ignore unrecognized or skipped pass
	 * results and must validate any backend-owned resources before attaching
	 * them to frame state.
	 * @sideEffects May publish validated pass outputs into backend frame state.
	 */
	completePass?(
		request: PostProcessPassRequest,
		result: PostProcessPassResult
	): PostProcessPassCompletion | void | Promise<PostProcessPassCompletion | void>;
	endFrame?(request: PostProcessFrameEndRequest): void | Promise<void>;
	/**
	 * Aborts backend post-process frame state after a failed pass or failed
	 * renderer frame.
	 *
	 * @param request Current post-process frame state and original error.
	 * @returns Nothing.
	 * @constraints Implementations must tolerate repeated calls and calls after
	 * `endFrame`.
	 * @sideEffects Releases per-frame backend state without presenting, copying,
	 * or committing temporal history.
	 */
	abortFrame?(request: PostProcessFrameAbortRequest): void | Promise<void>;
}
