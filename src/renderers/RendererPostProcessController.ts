import type { FrameContext } from "../pipeline/types";
import {
	hasPostProcessExecutionPasses,
	PostProcessPipeline,
	resolvePostProcessBackendAdapter,
	type PostProcessBackendKind,
	type PostProcessPassRegistry,
	type PostProcessPassRegistrySnapshot,
	type ResolvedPostProcessPass,
} from "../postprocess";
import type {
	IRenderBackend,
	RendererBackendResourceEvent,
	RenderBackendType,
} from "./IRenderBackend";

export type RendererPostProcessWarn = (
	key: string,
	message: string
) => void;

export interface RendererPostProcessControllerOptions {
	readonly backend: IRenderBackend;
	readonly postProcess: PostProcessPassRegistry;
	readonly pipeline: PostProcessPipeline;
	readonly warn: RendererPostProcessWarn;
}

/**
 * Coordinates renderer-owned post-process state with backend adapters.
 */
export class RendererPostProcessController {
	private readonly _backend: IRenderBackend;
	private readonly _postProcess: PostProcessPassRegistry;
	private readonly _pipeline: PostProcessPipeline;
	private readonly _warn: RendererPostProcessWarn;
	private _missingAdapterWarned = false;

	public constructor(options: RendererPostProcessControllerOptions) {
		this._backend = options.backend;
		this._postProcess = options.postProcess;
		this._pipeline = options.pipeline;
		this._warn = options.warn;
	}

	/**
	 * Resolves the executable logical post-process order for warmup planning.
	 *
	 * @param postProcess Per-frame post-process snapshot.
	 * @param frameContext Optional frame context used by frame-conditional passes.
	 * @returns Enabled executable passes, or an empty list when no adapter exists.
	 * @sideEffects Emits one diagnostic when enabled post-process work has no
	 * backend adapter.
	 */
	public getExecutionOrder(
		postProcess: PostProcessPassRegistrySnapshot,
		frameContext?: FrameContext
	): ResolvedPostProcessPass[] {
		const adapter = resolvePostProcessBackendAdapter(this._backend);
		if (!adapter) {
			this._warnIfMissingAdapterHasWork(postProcess, frameContext);
			return [];
		}
		return this._pipeline.getExecutionOrder(
			postProcess,
			adapter,
			this._warn,
			frameContext
		);
	}

	/**
	 * Executes the renderer-owned logical post-process stage for one frame.
	 *
	 * @param context Current frame context.
	 * @returns Nothing.
	 * @sideEffects Dispatches backend post-process work through the registered
	 * adapter and may allocate or update post-process history resources.
	 */
	public async execute(context: FrameContext): Promise<void> {
		if (
			context.incremental.enabled &&
			!context.incremental.forceFullFrame &&
			context.incremental.dirtyRects.length === 0
		) {
			return;
		}

		const adapter = resolvePostProcessBackendAdapter(this._backend);
		if (!adapter) {
			this._warnIfMissingAdapterHasWork(context.postProcess, context);
			return;
		}
		if (
			!hasPostProcessExecutionPasses(context.postProcess, {
				backend: adapter.backend,
				frameContext: context,
			})
		) {
			return;
		}
		await this._pipeline.execute({
			frameContext: context,
			executor: adapter,
			gBuffer: adapter.createGBufferBridge(context),
			historyFinalization: "manual",
			warn: this._warn,
		});
	}

	/**
	 * Commits pending post-process temporal history after a successful frame.
	 *
	 * @returns Nothing.
	 * @sideEffects Swaps updated history handles and clears pending frame state.
	 */
	public commitFrame(): void {
		this._pipeline.commitFrame();
	}

	/**
	 * Aborts pending post-process work after a failed frame.
	 *
	 * @param error Optional original frame error.
	 * @returns Nothing.
	 * @sideEffects Clears pending history updates and asks the adapter executor
	 * to abort active backend post-process state.
	 */
	public async abortFrame(error?: unknown): Promise<void> {
		try {
			await this._pipeline.abortFrame(error);
		} catch (abortError) {
			this._warn(
				"renderer-postprocess-abort-failed",
				`Failed to abort post-process frame state: ${String(abortError)}`
			);
		}
	}

	/**
	 * Applies a backend resource lifetime event to post-process state.
	 *
	 * @param event Backend resource event.
	 * @returns Nothing.
	 * @sideEffects Invalidates or destroys renderer-owned post-process state.
	 */
	public handleBackendResourceEvent(
		event: RendererBackendResourceEvent
	): void {
		if (event.resource !== "postprocess") {
			return;
		}
		const backend = this._resolveEventBackend(event.backend);
		if (event.action === "destroy") {
			this.destroyBackendResources(backend);
			return;
		}
		this.invalidateBackendResources(backend);
	}

	/**
	 * Destroys post-process resources for one backend kind.
	 *
	 * @param backend Backend kind whose resources must be released.
	 * @returns Nothing.
	 * @sideEffects Destroys pipeline histories, transients, and pass resources.
	 */
	public destroyBackendResources(backend: PostProcessBackendKind): void {
		const adapter = resolvePostProcessBackendAdapter(this._backend);
		if (adapter && adapter.backend === backend) {
			this._pipeline.destroy(adapter);
		}
		this._postProcess.destroyPasses(backend);
	}

	/**
	 * Invalidates pass-owned resources for one backend kind.
	 *
	 * @param backend Backend kind whose pass resources are frame-target stale.
	 * @returns Nothing.
	 * @sideEffects Invalidates pass-owned backend resources without unregistering.
	 */
	public invalidateBackendResources(backend: PostProcessBackendKind): void {
		this._postProcess.invalidatePasses(backend);
	}

	private _warnIfMissingAdapterHasWork(
		postProcess: PostProcessPassRegistrySnapshot,
		frameContext?: FrameContext
	): void {
		if (this._missingAdapterWarned) {
			return;
		}
		if (
			!hasPostProcessExecutionPasses(postProcess, {
				backend: this._backend.type as PostProcessBackendKind,
				frameContext,
			})
		) {
			return;
		}
		this._missingAdapterWarned = true;
		const key = `${this._backend.type}-postprocess-adapter-missing`;
		this._warn(
			key,
			`Backend "${this._backend.type}" has enabled post-process passes but no registered post-process adapter; skipping post-process stage`
		);
	}

	private _resolveEventBackend(
		backend: RenderBackendType | undefined
	): PostProcessBackendKind {
		const adapter = resolvePostProcessBackendAdapter(this._backend);
		return (backend ?? adapter?.backend ?? this._backend.type) as
			PostProcessBackendKind;
	}
}
