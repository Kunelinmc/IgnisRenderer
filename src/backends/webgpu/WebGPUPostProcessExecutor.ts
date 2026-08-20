import type { FrameContext } from "../../pipeline/types";
import type {
	IPostProcessExecutor,
	LogicalGBufferBridge,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessPassCompletion,
	PostProcessGraphFrameBinding,
	PostProcessFrameRequest,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import { tryGetTextureFormatInfo } from "../TextureFormatInfo";
import {
	TextureFormat,
	TextureUsage,
	type IRenderTexture,
} from "../types";
import type { WebGPUFrameHost } from "./rendergraph/WebGPUFrameHost";

export interface WebGPUPostProcessSessionPort {
	createGBufferBridge(context: FrameContext): LogicalGBufferBridge;
	createPassExecutionContext(request: PostProcessPassExecutionContextRequest): unknown;
	completePass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult,
	): PostProcessPassCompletion;
	isGraphResourceAvailable(resourceId: string): boolean;
	invalidateResourceBindings(): void;
}

/** Supplies backend resources and delegates frame-only work through a session port. */
export class WebGPUPostProcessExecutor implements IPostProcessExecutor {
	public readonly backend = "webgpu";
	public readonly gBufferNormalSpace = "view";
	private _sessionPort: WebGPUPostProcessSessionPort | null = null;

	constructor(private readonly _host: WebGPUFrameHost) {}

	public bindSession(port: WebGPUPostProcessSessionPort): void {
		if (this._sessionPort) {
			throw new Error("WebGPU post-process executor already has an active session port.");
		}
		this._sessionPort = port;
	}

	public unbindSession(port?: WebGPUPostProcessSessionPort): void {
		if (port && this._sessionPort !== port) return;
		this._sessionPort = null;
	}

	public createResource(desc: PostProcessResourceDescriptor): PostProcessResourceHandle {
		this._host.assertDeviceOperational("create post-process resource");
		const requestedFormat =
			tryGetTextureFormatInfo(desc.format)?.format ?? TextureFormat.RGBA16Float;
		const texture = this._host.createTexture({
			width: desc.width,
			height: desc.height,
			format: requestedFormat,
			mipLevelCount:
				desc.mipMode === "full-chain"
					? Math.floor(Math.log2(Math.max(desc.width, desc.height))) + 1
					: undefined,
			usage:
				TextureUsage.TextureBinding |
				TextureUsage.StorageBinding |
				TextureUsage.RenderAttachment |
				TextureUsage.CopyDst |
				TextureUsage.CopySrc,
			label: `WebGPUPostHistory_${desc.id}`,
		});
		return {
			id: desc.id,
			backend: "webgpu",
			width: desc.width,
			height: desc.height,
			format: texture.format ?? requestedFormat,
			mipMode: desc.mipMode ?? "single",
			resource: texture,
		};
	}

	public destroyResource(handle: PostProcessResourceHandle): void {
		(handle.resource as IRenderTexture | null)?.destroy?.();
	}

	public invalidateResourceBindings(): void {
		this._sessionPort?.invalidateResourceBindings();
	}

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return (
			this._sessionPort?.createGBufferBridge(context) ??
			this._createFallbackGBufferBridge(context)
		);
	}

	public isGraphResourceAvailable(resourceId: string): boolean {
		return this._sessionPort?.isGraphResourceAvailable(resourceId) ?? true;
	}

	/** @internal Creates a session-scoped controlled publication transaction. */
	public createGraphBinding(_request: PostProcessFrameRequest): PostProcessGraphFrameBinding {
		const session = this._requireSession("create post-process graph binding");
		return {
			completePass: (request, result) => session.completePass(request, result),
		};
	}

	public createPassExecutionContext(request: PostProcessPassExecutionContextRequest): unknown {
		return this._requireSession("create post-process pass context").createPassExecutionContext(
			request,
		);
	}

	public completePass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult,
	): PostProcessPassCompletion {
		return this._requireSession("complete post-process passes").completePass(request, result);
	}

	private _requireSession(operation: string): WebGPUPostProcessSessionPort {
		if (!this._sessionPort) {
			throw new Error(`WebGPU post-process session is not active; cannot ${operation}.`);
		}
		return this._sessionPort;
	}

	private _createFallbackGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return {
			width: Math.max(1, context.attachments.width),
			height: Math.max(1, context.attachments.height),
			normalSpace: "view",
			depthEncoding: "hardware",
			channels: {},
			worldPosition: { source: "derived", available: false },
		};
	}
}
