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
import type { WebGPUFrameExecutor } from "./WebGPUFrameExecutor";

export interface WebGPUPostProcessExecutorHost {
	/**
	 * Resolves the active WebGPU frame executor.
	 *
	 * @returns Current frame executor, or `null` before initialization/after loss.
	 * @sideEffects None.
	 */
	getFrameExecutor(): WebGPUFrameExecutor | null;
	/**
	 * Throws when the WebGPU device cannot service a post-process operation.
	 *
	 * @param operation Human-readable operation used in diagnostics.
	 * @returns Nothing.
	 * @sideEffects May throw when the backend device is lost or unavailable.
	 */
	assertDeviceOperational(operation: string): void;
}

/**
 * Supplies WebGPU post-process resources, G-buffer metadata, and pass helpers.
 */
export class WebGPUPostProcessExecutor implements IPostProcessExecutor {
	/**
	 * Backend kind used for pass implementation resolution.
	 */
	public readonly backend = "webgpu";
	private readonly _host: WebGPUPostProcessExecutorHost;

	public constructor(host: WebGPUPostProcessExecutorHost) {
		this._host = host;
	}

	/**
	 * Allocates a WebGPU post-process texture resource.
	 *
	 * @param desc Resource descriptor from the backend post-process resource pool.
	 * @returns Resource handle wrapping a backend texture.
	 * @sideEffects Allocates a GPU texture through the active frame executor.
	 */
	public createResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle {
		this._host.assertDeviceOperational("create post-process resource");
		const executor = this._requireFrameExecutor(
			"create post-process resource"
		);
		return executor.createPostProcessResource(desc);
	}

	/**
	 * Releases a WebGPU post-process texture resource.
	 *
	 * @param handle Resource handle previously returned by `createResource`.
	 * @returns Nothing.
	 * @sideEffects Destroys the backend texture when the frame executor exists.
	 */
	public destroyResource(handle: PostProcessResourceHandle): void {
		this._host.getFrameExecutor()?.destroyPostProcessResource(handle);
	}

	/**
	 * Invalidates cached bind groups that may reference stale post-process inputs.
	 *
	 * @returns Nothing.
	 * @sideEffects Drops WebGPU post-process binding caches.
	 */
	public invalidateResourceBindings(): void {
		this._host.getFrameExecutor()?.invalidatePostProcessBindings();
	}

	/**
	 * Creates a logical G-buffer bridge for the current WebGPU frame.
	 *
	 * @param context Current renderer frame context.
	 * @returns Logical bridge wrapping active WebGPU frame targets.
	 * @sideEffects None.
	 */
	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return (
			this._host.getFrameExecutor()?.createGBufferBridge(context) ??
			this._createFallbackGBufferBridge(context)
		);
	}

	/**
	 * Provides WebGPU helper objects for pass-owned implementations.
	 *
	 * @param request Pass-owned implementation context request.
	 * @returns Context object expected by the selected WebGPU implementation.
	 * @sideEffects May reset pending pass output state in the frame executor.
	 */
	public getPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
		return this._host.getFrameExecutor()?.getPassExecutionContext(request);
	}

	/**
	 * Executes one fallback logical WebGPU post-process pass.
	 *
	 * @param passId Logical pass id.
	 * @param request Current pass request.
	 * @returns Execution result for pipeline history tracking.
	 * @sideEffects None for WebGPU pass-owned implementations.
	 */
	public executePass(
		passId: string,
		request: PostProcessPassRequest
	): PostProcessPassResult {
		void passId;
		void request;
		return { ran: false };
	}

	/**
	 * Applies backend-owned side effects recorded by a completed logical pass.
	 *
	 * @param request Logical pass request that just completed.
	 * @param result Pass execution result.
	 * @returns Nothing.
	 * @sideEffects May publish a validated color target into WebGPU frame state.
	 */
	public completePass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult
	): void {
		this._host.getFrameExecutor()?.completePostProcessPass(request, result);
	}

	private _requireFrameExecutor(operation: string): WebGPUFrameExecutor {
		const executor = this._host.getFrameExecutor();
		if (!executor) {
			throw new Error(
				`WebGPU frame executor is not initialized; cannot ${operation}.`
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
