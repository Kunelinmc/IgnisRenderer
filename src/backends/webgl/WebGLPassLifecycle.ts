import type { FrameContext } from "../../pipeline/types";

/**
 * @internal WebGL pass lifecycle contract for backend-owned render passes.
 */
export interface WebGLRenderPass<TOptions = void> {
	/**
	 * Executes pass-owned WebGL work for the current frame.
	 *
	 * @internal WebGL frame execution hook.
	 *
	 * @param context Active renderer frame context.
	 * @param options Optional pass-specific render options.
	 * @returns Nothing.
	 * @sideEffects May mutate WebGL state and pass-owned resources.
	 */
	render(context: FrameContext, options?: TOptions): void;

	/**
	 * Invalidates pass-owned resources that can be recreated lazily.
	 *
	 * @internal WebGL resource lifecycle hook.
	 *
	 * @returns Nothing.
	 * @sideEffects May release pass-owned WebGL resources.
	 */
	invalidate?(): void;

	/**
	 * Releases all pass-owned WebGL resources.
	 *
	 * @internal WebGL resource lifecycle hook.
	 *
	 * @returns Nothing.
	 * @sideEffects Deletes pass-owned WebGL resources.
	 */
	destroy(): void;
}
