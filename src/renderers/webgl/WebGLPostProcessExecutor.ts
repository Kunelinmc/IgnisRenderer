import type { FrameContext } from "../../pipeline/types";
import type {
	IPostProcessExecutor,
	LogicalGBufferBridge,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import type { WebGLFrameExecutor } from "./WebGLFrameExecutor";

export interface WebGLPostProcessExecutorHost {
	/**
	 * Resolves the active WebGL frame executor.
	 *
	 * @returns Current frame executor, or `null` before initialization/after loss.
	 * @sideEffects None.
	 */
	getFrameExecutor(): WebGLFrameExecutor | null;
}

/**
 * Supplies WebGL post-process resources, G-buffer metadata, and pass helpers.
 */
export class WebGLPostProcessExecutor implements IPostProcessExecutor {
	/**
	 * Backend kind used for pass implementation resolution.
	 */
	public readonly backend = "webgl";
	private readonly _host: WebGLPostProcessExecutorHost;

	public constructor(host: WebGLPostProcessExecutorHost) {
		this._host = host;
	}

	/**
	 * Allocates a WebGL post-process texture resource.
	 *
	 * @param desc Resource descriptor from the backend post-process resource pool.
	 * @returns Resource handle wrapping a WebGL texture.
	 * @sideEffects Allocates texture storage through the active frame executor.
	 */
	public createResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle {
		const executor = this._requireFrameExecutor(
			"create post-process resource"
		);
		return executor.createPostProcessResource(desc);
	}

	/**
	 * Releases a WebGL post-process texture resource.
	 *
	 * @param handle Resource handle previously returned by `createResource`.
	 * @returns Nothing.
	 * @sideEffects Destroys the backend texture when the frame executor exists.
	 */
	public destroyResource(handle: PostProcessResourceHandle): void {
		this._host.getFrameExecutor()?.destroyPostProcessResource(handle);
	}

	/**
	 * Creates a logical G-buffer bridge for the current WebGL frame.
	 *
	 * @param context Current renderer frame context.
	 * @returns Logical bridge wrapping active WebGL frame targets.
	 * @sideEffects None.
	 */
	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return (
			this._host.getFrameExecutor()?.createGBufferBridge(context) ??
			this._createFallbackGBufferBridge(context)
		);
	}

	/**
	 * Provides WebGL helper objects for pass-owned implementations.
	 *
	 * @param request Pass-owned implementation context request.
	 * @returns Context object expected by the selected WebGL implementation.
	 * @sideEffects None.
	 */
	public getPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
		return this._host.getFrameExecutor()?.getPassExecutionContext(request);
	}

	/**
	 * Executes one fallback logical WebGL post-process pass.
	 *
	 * @param passId Logical pass id.
	 * @param request Current pass request.
	 * @returns Execution result for pipeline history tracking.
	 * @sideEffects May run backend-owned fullscreen work through the frame executor.
	 */
	public executePass(
		passId: string,
		request: PostProcessPassRequest
	): PostProcessPassResult {
		return (
			this._host.getFrameExecutor()?.executePostProcessPass(passId, request) ??
			{ ran: false }
		);
	}

	private _requireFrameExecutor(operation: string): WebGLFrameExecutor {
		const executor = this._host.getFrameExecutor();
		if (!executor) {
			throw new Error(
				`WebGL frame executor is not initialized; cannot ${operation}.`
			);
		}
		return executor;
	}

	private _createFallbackGBufferBridge(
		context: FrameContext
	): LogicalGBufferBridge {
		return {
			width: Math.max(1, context.attachments.width),
			height: Math.max(1, context.attachments.height),
			normalSpace: "world",
			depthEncoding: "hardware",
			channels: {},
			worldPosition: {
				source: "derived",
				available: false,
			},
		};
	}
}
