import type {
	FrameContext,
} from "../pipeline/types";
import type {
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
	readonly backend: PostProcessBackendKind;
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
	readonly backend: PostProcessBackendKind;
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

export interface PostProcessPassImplementationMetadata<TContextMetadata = unknown> {
	/**
	 * Backend-specific context declaration consumed by
	 * `IPostProcessExecutor.getPassExecutionContext`.
	 *
	 * @remarks Backends that do not understand this metadata must ignore it.
	 * @sideEffects None.
	 */
	readonly context?: TContextMetadata;
	/**
	 * Backend-specific warmup hints owned by this implementation.
	 *
	 * @remarks Backends may use these ids to pre-warm internal runtime passes.
	 * @sideEffects None.
	 */
	readonly warmupHints?: readonly string[];
}

export interface PostProcessPassImplementation<
	TContext = unknown,
	TOptions = unknown,
	TContextMetadata = unknown,
> {
	readonly id?: string;
	/**
	 * Optional backend-specific implementation metadata.
	 *
	 * @remarks This is declarative data used by backend executors to prepare
	 * implementation-owned contexts and warmup work. Implementations must not
	 * rely on unsupported backends interpreting this value.
	 * @sideEffects None.
	 */
	readonly metadata?: PostProcessPassImplementationMetadata<TContextMetadata>;
	/**
	 * Executes the logical pass through pass-owned backend implementation logic.
	 *
	 * @param request Current pass request produced by backend post-process runtime.
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
	readonly ran?: boolean;
	readonly preservesOutsideDirtyTiles?: boolean;
	readonly historyUpdated?: boolean;
	readonly updatedHistoryIds?: readonly string[];
}

export interface PostProcessFrameEndRequest extends PostProcessFrameRequest {
	readonly executedPassIds: readonly string[];
}

export interface PostProcessFrameAbortRequest extends PostProcessFrameRequest {
	readonly executedPassIds: readonly string[];
	readonly error?: unknown;
}

export interface IPostProcessExecutor {
	readonly backend: PostProcessBackendKind;
	/**
	 * Creates the logical G-buffer view consumed by cross-backend passes.
	 *
	 * @param context Frame context containing backend attachments for the
	 * current render.
	 * @returns A backend-specific bridge mapped to logical G-buffer semantics.
	 * @sideEffects None. Resource ownership remains with the backend.
	 */
	createGBufferBridge(context: FrameContext): LogicalGBufferBridge;
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
	getPassExecutionContext?(
		request: PostProcessPassExecutionContextRequest
	): unknown;
	executePass(
		passId: string,
		request: PostProcessPassRequest
	): PostProcessPassResult | Promise<PostProcessPassResult>;
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
	): void | Promise<void>;
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
