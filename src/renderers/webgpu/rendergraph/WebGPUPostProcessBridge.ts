import type { FrameContext } from "../../../pipeline/types";
import type {
	LogicalGBufferBridge,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../../postprocess";
import { Logger } from "../../../foundation/Logger";
import type { ICommandEncoder } from "../../ICommandEncoder";
import {
	TextureFormat,
	TextureUsage,
	type IRenderTexture,
} from "../../types";
import { tryGetTextureFormatInfo } from "../../TextureFormatInfo";
import type { WebGPUBackend } from "../../WebGPUBackend";
import type { WebGPUPreparedFrameResources } from "../WebGPURenderResources";
import {
	isWebGPUPostProcessContextMetadata,
	type WebGPUFrameTargets,
	type WebGPUPostProcessContextMetadata,
	type WebGPUPostProcessFrameTargets,
} from "../WebGPUPostProcessContracts";
import { WebGPUPostProcessRuntime } from "../WebGPUPostProcessRuntime";

export interface WebGPUPostProcessBridgeCallbacks {
	getEncoder(): ICommandEncoder | null;
	getFrameTargets(): WebGPUFrameTargets | null;
	requireFrameResources(): WebGPUPreparedFrameResources;
	presentToCanvas(source: IRenderTexture): Promise<void>;
	warmupPresent(): Promise<void>;
	setMotionHistoryWriteTarget(texture: IRenderTexture | null): void;
}

/**
 * Packs WebGPU-specific post-process helpers and validates published targets.
 */
export class WebGPUPostProcessBridge {
	private readonly _backend: WebGPUBackend;
	private readonly _runtime: WebGPUPostProcessRuntime;
	private readonly _callbacks: WebGPUPostProcessBridgeCallbacks;
	private _pendingColorTarget: IRenderTexture | null = null;

	public constructor(
		backend: WebGPUBackend,
		runtime: WebGPUPostProcessRuntime,
		callbacks: WebGPUPostProcessBridgeCallbacks
	) {
		this._backend = backend;
		this._runtime = runtime;
		this._callbacks = callbacks;
	}

	public createResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle {
		const requestedFormat =
			tryGetTextureFormatInfo(desc.format)?.format ?? TextureFormat.RGBA16Float;
		const texture = this._backend.createTexture({
			width: desc.width,
			height: desc.height,
			format: requestedFormat,
			mipLevelCount:
				desc.mipMode === "full-chain" ?
					Math.floor(Math.log2(Math.max(desc.width, desc.height))) + 1
				:	undefined,
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

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		const targets = this._callbacks.getFrameTargets();
		const width = Math.max(1, context.attachments.width);
		const height = Math.max(1, context.attachments.height);
		const channels: LogicalGBufferBridge["channels"] = {};
		if (targets) {
			channels.color = {
				semantic: "color",
				handle: { backend: "webgpu", texture: targets.sceneColor },
				width,
				height,
				format: TextureFormat.RGBA16Float,
			};
			if (targets.gMotionDepth) {
				channels.depth = {
					semantic: "depth",
					handle: { backend: "webgpu", texture: targets.gMotionDepth },
					width,
					height,
					format: TextureFormat.RGBA16Float,
					encoding: "motion-depth.z",
				};
				channels.motion = {
					semantic: "motion",
					handle: { backend: "webgpu", texture: targets.gMotionDepth },
					width,
					height,
					format: TextureFormat.RGBA16Float,
					encoding: "motion-depth.xy",
				};
			}
			if (targets.gNormalRoughMetal) {
				channels.normal = {
					semantic: "normal",
					handle: {
						backend: "webgpu",
						texture: targets.gNormalRoughMetal,
					},
					width,
					height,
					format: TextureFormat.RGBA16Float,
					encoding: "encoded-world-normal",
				};
			}
			if (targets.gAlbedoAlpha) {
				channels.albedo = {
					semantic: "albedo",
					handle: { backend: "webgpu", texture: targets.gAlbedoAlpha },
					width,
					height,
					format: TextureFormat.RGBA8Unorm,
					encoding: "linear-rgb-alpha",
				};
			}
			if (targets.gTransmissionSurface0) {
				channels.transmission = {
					semantic: "transmission",
					handle: {
						backend: "webgpu",
						texture: targets.gTransmissionSurface0,
					},
					width,
					height,
					format: TextureFormat.RGBA16Float,
					encoding: "normal-depth-transmission",
				};
			}
		}
		return {
			width,
			height,
			normalSpace: "world",
			depthEncoding: "linear-view-z",
			motionEncoding: targets?.gMotionDepth ? "ndc-delta" : undefined,
			channels,
			worldPosition: {
				source: "derived",
				available: !!targets?.gMotionDepth,
			},
		};
	}

	public getPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
		if (!this._callbacks.getEncoder() || !this._callbacks.getFrameTargets()) {
			return undefined;
		}
		const metadata = request.implementation.metadata?.context;
		if (!isWebGPUPostProcessContextMetadata(metadata)) {
			return undefined;
		}
		this._pendingColorTarget = null;
		return this._createContext(metadata, request, "execute");
	}

	public getPassWarmupExecutionContext(
		metadata: WebGPUPostProcessContextMetadata
	): unknown {
		return this._createContext(metadata, null, "warmup");
	}

	public completePass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult
	): void {
		const colorTarget = this._pendingColorTarget;
		this._pendingColorTarget = null;
		const targets = this._callbacks.getFrameTargets();
		if (result.ran === false || !colorTarget || !targets) {
			return;
		}
		if (!this._isOwnedColorTarget(colorTarget, targets)) {
			Logger.warn(
				`[webgpu-postprocess-color-target-unowned] ` +
					`Post-process pass "${request.passId}" published a color target ` +
					"that is not owned by the active WebGPU frame; ignoring it.",
				{
					scope: "WebGPUFrameExecutor",
					onceKey: `webgpu-postprocess-color-target-unowned:${request.passId}`,
				}
			);
			return;
		}
		targets.sceneColor = colorTarget;
	}

	public clearPendingFrameState(): void {
		this._pendingColorTarget = null;
	}

	private _createContext(
		metadata: WebGPUPostProcessContextMetadata,
		request: PostProcessPassRequest | null,
		mode: "execute" | "warmup"
	): Record<string, unknown> | undefined {
		if (
			mode === "execute" &&
			(!this._callbacks.getEncoder() || !this._callbacks.getFrameTargets())
		) {
			return undefined;
		}
		if (metadata.kind === "present") {
			return {
				targets: this._createFrameTargetsView(),
				presentToCanvas: (source: IRenderTexture) =>
					this._callbacks.presentToCanvas(source),
				warmupPresent: () => this._callbacks.warmupPresent(),
			};
		}

		const context: Record<string, unknown> = {
			encoder: this._callbacks.getEncoder() ?? undefined,
			targets: this._createFrameTargetsView(),
			shared: this._runtime.sharedContext,
		};
		if (metadata.publishColorTarget && mode === "execute") {
			context.publishColorTarget = (texture: IRenderTexture): void => {
				this._pendingColorTarget = texture;
			};
		}
		if (metadata.frameBinding && mode === "execute") {
			context.frameBinding =
				this._callbacks.requireFrameResources().frameBinding;
		}
		if (metadata.lightingState && mode === "execute") {
			context.lightingState =
				this._callbacks.requireFrameResources().lightingState;
		}
		if (request && mode === "execute") {
			for (const binding of metadata.histories ?? []) {
				context[binding.property] = this._getHistoryTexture(
					request,
					binding.historyId,
					binding.side
				);
			}
			for (const binding of metadata.transients ?? []) {
				context[binding.property] = this._getTransientTexture(
					request,
					binding.transientId
				);
			}
			const motionCopy = metadata.motionHistoryCopy;
			if (motionCopy) {
				const method = motionCopy.method ?? "writeMotionHistoryFromCurrent";
				context[method] = (): void => {
					this._callbacks.setMotionHistoryWriteTarget(
						(context[motionCopy.writeProperty] as IRenderTexture | null) ??
							null
					);
				};
			}
		}
		return context;
	}

	private _createFrameTargetsView(): WebGPUPostProcessFrameTargets | undefined {
		const targets = this._callbacks.getFrameTargets();
		if (!targets) {
			return undefined;
		}
		return Object.freeze({ ...targets });
	}

	private _isOwnedColorTarget(
		texture: IRenderTexture,
		targets: WebGPUFrameTargets
	): boolean {
		return (
			texture === targets.sceneColorMain ||
			texture === targets.postPing ||
			texture === targets.postPong
		);
	}

	private _getHistoryTexture(
		request: PostProcessPassRequest,
		id: string,
		side: "read" | "write"
	): IRenderTexture | null {
		const slot = request.histories[id]?.[side];
		return (slot?.resource as IRenderTexture | null) ?? null;
	}

	private _getTransientTexture(
		request: PostProcessPassRequest,
		id: string
	): IRenderTexture | null {
		const slot = request.transients?.[id];
		return (slot?.handle.resource as IRenderTexture | null) ?? null;
	}
}
