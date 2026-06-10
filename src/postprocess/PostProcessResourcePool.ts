import { PostProcessHistoryManager } from "./PostProcessHistoryManager";
import { PostProcessTransientManager } from "./PostProcessTransientManager";
import type {
	IPostProcessExecutor,
	PostProcessHistorySlots,
	PostProcessTransientSlots,
} from "./types";
import type { CompiledPostProcessGraph } from "./PostProcessGraphCompiler";

export interface PostProcessResourcePoolPrepareRequest {
	readonly executor: IPostProcessExecutor;
	readonly graph: CompiledPostProcessGraph;
	readonly reset: boolean;
}

export interface PostProcessResourcePoolPrepareResult {
	readonly histories: PostProcessHistorySlots;
	readonly transients: PostProcessTransientSlots;
	readonly transientsChanged: boolean;
}

/**
 * Owns backend concrete post-process history and transient resources.
 */
export class PostProcessResourcePool {
	private _history = new PostProcessHistoryManager();
	private _transients = new PostProcessTransientManager();

	/**
	 * Ensures all resources required by a compiled post-process graph exist.
	 *
	 * @internal Owned by backend post-process runtimes.
	 * @param request Executor, graph, and reset state for the frame.
	 * @returns Prepared history and transient slots for pass execution.
	 * @sideEffects Allocates, destroys, or invalidates backend resources.
	 */
	public prepare(
		request: PostProcessResourcePoolPrepareRequest
	): PostProcessResourcePoolPrepareResult {
		const graph = request.graph;
		const histories = this._history.prepare({
			executor: request.executor,
			descriptors: graph.historyDescriptors,
			width: graph.width,
			height: graph.height,
			reset: request.reset,
			signature: graph.signature,
		});
		const transientResult = this._transients.prepare({
			executor: request.executor,
			descriptors: graph.transientDescriptors,
			width: graph.width,
			height: graph.height,
		});
		return {
			histories,
			transients: transientResult.slots,
			transientsChanged: transientResult.changed,
		};
	}

	/**
	 * Marks multiple history slots as written by the active frame.
	 *
	 * @internal Owned by backend post-process runtimes.
	 * @param ids History ids reported by executed logical passes.
	 * @returns Nothing.
	 * @sideEffects Records pending history swaps for `commitFrame()`.
	 */
	public markUpdatedMany(ids: readonly string[]): void {
		this._history.markUpdatedMany(ids);
	}

	/**
	 * Commits pending history writes after a successful backend frame.
	 *
	 * @internal Owned by backend post-process runtimes.
	 * @returns Nothing.
	 * @sideEffects Swaps marked history read/write resources.
	 */
	public commitFrame(): void {
		this._history.endFrame();
	}

	/**
	 * Aborts pending history writes for a failed backend frame.
	 *
	 * @internal Owned by backend post-process runtimes.
	 * @returns Nothing.
	 * @sideEffects Clears pending writes without invalidating valid histories.
	 */
	public abortFrame(): void {
		this._history.abortFrame();
	}

	/**
	 * Destroys all resources owned by the pool.
	 *
	 * @internal Owned by backend post-process runtimes.
	 * @param executor Backend executor that owns concrete resources.
	 * @returns Nothing.
	 * @sideEffects Calls `executor.destroyResource()` for active handles.
	 */
	public destroy(executor: IPostProcessExecutor): void {
		this.abortFrame();
		this._history.destroy(executor);
		this._transients.destroy(executor);
	}

	/**
	 * Invalidates resources sized by the active frame target.
	 *
	 * @internal Owned by backend resize and lifecycle paths.
	 * @param executor Backend executor that owns concrete resources.
	 * @returns Nothing.
	 * @sideEffects Destroys active histories and transients.
	 */
	public invalidateFrameSized(executor: IPostProcessExecutor): void {
		this.destroy(executor);
	}
}
