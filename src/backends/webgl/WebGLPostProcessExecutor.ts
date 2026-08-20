import type { FrameContext } from "../../pipeline/types";
import type {
	IPostProcessExecutor,
	LogicalGBufferBridge,
	PostProcessGBufferBridgeOptions,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessPassCompletion,
	PostProcessGraphFrameBinding,
	PostProcessFrameRequest,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import { createSyntheticLogicalGBufferBridge } from "../../postprocess/GBufferBridge";

import { WEBGL_POST_PROCESS_GBUFFER_METADATA } from "./WebGLPostProcessContracts";

export interface WebGLPostProcessDeviceServices {
	createPostProcessResource(
		desc: PostProcessResourceDescriptor,
	): PostProcessResourceHandle;
	destroyPostProcessResource(handle: PostProcessResourceHandle): void;
	createGBufferBridge(context: FrameContext): LogicalGBufferBridge;
	createPassExecutionContext(request: PostProcessPassExecutionContextRequest): unknown;
	beginPostProcessFrame(): void;
	endPostProcessFrame(): void;
	abortPostProcessFrame(): void;
	completePostProcessPass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult,
	): PostProcessPassCompletion;
}

export interface WebGLPostProcessExecutorHost {
	/**
	 * Resolves the active context-scoped WebGL device services.
	 *
	 * @returns Current services, or `null` before initialization/after loss.
	 * @sideEffects None.
	 */
	getDeviceServices(): WebGLPostProcessDeviceServices | null;
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

	constructor(host: WebGLPostProcessExecutorHost) {
		this._host = host;
	}

	/**
	 * Allocates a WebGL post-process texture resource.
	 *
	 * @param desc Resource descriptor from the backend post-process resource pool.
	 * @returns Resource handle wrapping a WebGL texture.
	 * @sideEffects Allocates texture storage through the active device services.
	 */
	public createResource(desc: PostProcessResourceDescriptor): PostProcessResourceHandle {
		const services = this._requireDeviceServices("create post-process resource");
		return services.createPostProcessResource(desc);
	}

	/**
	 * Releases a WebGL post-process texture resource.
	 *
	 * @param handle Resource handle previously returned by `createResource`.
	 * @returns Nothing.
	 * @sideEffects Destroys the backend texture when device services exist.
	 */
	public destroyResource(handle: PostProcessResourceHandle): void {
		this._host.getDeviceServices()?.destroyPostProcessResource(handle);
	}

	/**
	 * Creates a logical G-buffer bridge for the current WebGL frame.
	 *
	 * @param context Current renderer frame context.
	 * @param options Selects physical targets or synthetic metadata.
	 * @returns Logical bridge wrapping active WebGL frame targets.
	 * @sideEffects Synthetic mode does not require device services.
	 */
	public createGBufferBridge(
		context: FrameContext,
		options: PostProcessGBufferBridgeOptions = {},
	): LogicalGBufferBridge {
		if (options.resourceMode === "synthetic") {
			return createSyntheticLogicalGBufferBridge(context, {
				backend: this.backend,
				...WEBGL_POST_PROCESS_GBUFFER_METADATA,
			});
		}
		return (
			this._host.getDeviceServices()?.createGBufferBridge(context) ??
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
	public createPassExecutionContext(request: PostProcessPassExecutionContextRequest): unknown {
		return this._host.getDeviceServices()?.createPassExecutionContext(request);
	}

	/** @internal Opens the WebGL controlled-publication transaction. */
	public beginFrame(): void {
		this._host.getDeviceServices()?.beginPostProcessFrame();
	}

	/** @internal Creates the WebGL logical-to-physical publication transaction. */
	public createGraphBinding(_request: PostProcessFrameRequest): PostProcessGraphFrameBinding {
		const services = this._host.getDeviceServices();
		services?.beginPostProcessFrame();
		return {
			completePass: (request, result) =>
				services?.completePostProcessPass(request, result) ?? {},
			endFrame: () => services?.endPostProcessFrame(),
			abortFrame: () => services?.abortPostProcessFrame(),
		};
	}

	/** @internal Closes the WebGL controlled-publication transaction. */
	public endFrame(): void {
		this._host.getDeviceServices()?.endPostProcessFrame();
	}

	/** @internal Applies backend-owned effects after a pass result is known. */
	public completePass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult,
	): PostProcessPassCompletion {
		return this._host.getDeviceServices()?.completePostProcessPass(request, result) ?? {};
	}

	private _requireDeviceServices(operation: string): WebGLPostProcessDeviceServices {
		const services = this._host.getDeviceServices();
		if (!services) {
			throw new Error(`WebGL device services are not initialized; cannot ${operation}.`);
		}
		return services;
	}

	private _createFallbackGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return {
			width: Math.max(1, context.attachments.width),
			height: Math.max(1, context.attachments.height),
			...WEBGL_POST_PROCESS_GBUFFER_METADATA,
			channels: {},
			worldPosition: {
				source: "derived",
				available: false,
			},
		};
	}
}
