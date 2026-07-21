import type { FrameContext } from "../../../pipeline/types";
import type {
	LogicalGBufferBridge,
	LogicalGBufferSemantic,
	PostProcessExecutionDeclaration,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessPassCompletion,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../../postprocess";
import { createPostProcessResourceAccessor } from "../../../postprocess/PostProcessResourceAccessor";
import { WEBGPU_HIZ_SHARED_RESOURCE } from "../../../postprocess/executionDeclarations";
import { Logger } from "../../../foundation/Logger";
import type { ICommandEncoder } from "../../ICommandEncoder";
import {
	TextureFormat,
	TextureUsage,
	type IRenderTexture,
} from "../../types";
import { tryGetTextureFormatInfo } from "../../TextureFormatInfo";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import type { WebGPUPreparedFrameResources } from "../WebGPUResourceContracts";
import type {
	WebGPUFrameTargets,
	WebGPUPostProcessFrameTargets,
} from "../WebGPUPostProcessContracts";
import { WebGPUPostProcessRuntime } from "../WebGPUPostProcessRuntime";

export interface WebGPUPostProcessBridgeCallbacks {
	getEncoder(): ICommandEncoder | null;
	getFrameTargets(): WebGPUFrameTargets | null;
	isHiZReady(): boolean;
	requireFrameResources(): WebGPUPreparedFrameResources;
	presentToCanvas(source: IRenderTexture): Promise<void>;
	warmupPresent(): Promise<void>;
	setMotionHistoryWriteTarget(texture: IRenderTexture | null): void;
}

/**
 * Packs the fixed WebGPU post-process context and commits assigned targets.
 */
export class WebGPUPostProcessBridge {
	private readonly _host: WebGPUFrameHost;
	private readonly _runtime: WebGPUPostProcessRuntime;
	private readonly _callbacks: WebGPUPostProcessBridgeCallbacks;
	private _expectedColorTarget: IRenderTexture | null = null;
	private readonly _physicalIds = new WeakMap<object, string>();
	private _nextPhysicalId = 1;

	constructor(
		host: WebGPUFrameHost,
		runtime: WebGPUPostProcessRuntime,
		callbacks: WebGPUPostProcessBridgeCallbacks,
	) {
		this._host = host;
		this._runtime = runtime;
		this._callbacks = callbacks;
	}

	public createResource(desc: PostProcessResourceDescriptor): PostProcessResourceHandle {
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
				const normalRoughMetalHandle = {
					backend: "webgpu" as const,
					texture: targets.gNormalRoughMetal,
				};
				channels.normal = {
					semantic: "normal",
					handle: normalRoughMetalHandle,
					width,
					height,
					format: TextureFormat.RGBA16Float,
					encoding: "encoded-world-normal",
				};
				channels.roughness = {
					semantic: "roughness",
					handle: normalRoughMetalHandle,
					width,
					height,
					format: TextureFormat.RGBA16Float,
					encoding: "normal-roughness-metallic.z",
				};
				channels.metallic = {
					semantic: "metallic",
					handle: normalRoughMetalHandle,
					width,
					height,
					format: TextureFormat.RGBA16Float,
					encoding: "normal-roughness-metallic.w",
				};
			}
			if (targets.gSpecular) {
				channels.specular = {
					semantic: "specular",
					handle: { backend: "webgpu", texture: targets.gSpecular },
					width,
					height,
					format: TextureFormat.RGBA16Float,
					encoding: "specular-color-factor.rgba",
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

	public createPassExecutionContext(request: PostProcessPassExecutionContextRequest): unknown {
		if (!this._callbacks.getEncoder() || !this._callbacks.getFrameTargets()) {
			throw new Error(
				`Post-process pass "${request.passId}" cannot create its required ` +
					"WebGPU execution context.",
			);
		}
		const targets = this._callbacks.getFrameTargets();
		this._expectedColorTarget = request.declaration.color.output === "new-version" && targets ?
			(targets.sceneColor === targets.postPong ? targets.postPing : targets.postPong) : null;
		if (
			request.declaration.color.output === "new-version" &&
			!this._expectedColorTarget
		) {
			throw new Error(
				`Post-process pass "${request.passId}" cannot create its required ` +
					"WebGPU color output binding.",
			);
		}
		if (targets && this._expectedColorTarget === targets.sceneColor) {
			throw new Error("WebGPU post-process graph selected one texture for sampled input and storage output.");
		}
		return this._createContext(request);
	}

	public getPassWarmupExecutionContext(
		passId: string,
		declaration: PostProcessExecutionDeclaration,
	): unknown {
		return this._createWarmupContext(passId, declaration);
	}

	public completePass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult
	): PostProcessPassCompletion {
		const expectedColorTarget = this._expectedColorTarget;
		this._expectedColorTarget = null;
		const targets = this._callbacks.getFrameTargets();
		if (result.ran === false || request.declaration.color.output === "preserve") {
			return { committed: false };
		}
		if (!expectedColorTarget || !targets) {
			throw new Error(
				`Post-process pass "${request.passId}" has no required physical color binding.`,
			);
		}
		if (!this._isOwnedColorTarget(expectedColorTarget, targets)) {
			Logger.warn(
				`[webgpu-postprocess-color-target-unowned] ` +
					`Post-process pass "${request.passId}" received a color target ` +
					"that is not owned by the active WebGPU frame; ignoring it.",
				{
					scope: "WebGPUFrameOrchestrator",
					onceKey: `webgpu-postprocess-color-target-unowned:${request.passId}`,
				},
			);
			return { committed: false };
		}
		targets.sceneColor = expectedColorTarget;
		return { committed: true, physicalId: this._getPhysicalId(expectedColorTarget) };
	}

	public clearPendingFrameState(): void {
		this._expectedColorTarget = null;
	}

	private _createContext(
		request: PostProcessPassExecutionContextRequest,
	): Record<string, unknown> | undefined {
		const frameResources = this._callbacks.requireFrameResources();
		return Object.freeze({
			encoder: this._callbacks.getEncoder() ?? undefined,
			targets: this._createFrameTargetsView(),
			shared: this._runtime.sharedContext,
			frameBinding: frameResources.frameBinding,
			lightingState: frameResources.lightingState,
			getFrameData: <T>(key: unknown): T | undefined =>
				frameResources.featureData.get(key as never) as T | undefined,
			resources: createPostProcessResourceAccessor<IRenderTexture>({
				passId: request.passId,
				declaration: request.declaration,
				colorInput: this._callbacks.getFrameTargets()?.sceneColor ?? null,
				colorOutput: this._expectedColorTarget,
				getGBuffer: (semantic) => this._getGBufferTexture(request, semantic),
				getHistory: (id) => {
					const slot = request.histories[id];
					return slot ? {
						read: (slot.read.resource as IRenderTexture | null) ?? null,
						write: (slot.write.resource as IRenderTexture | null) ?? null,
						valid: slot.valid,
					} : null;
				},
				getTransient: (id) => this._getTransientTexture(request, id),
				getShared: (id) => this._getSharedTexture(id),
				copyGBufferToHistory: (_semantic, historyId) => {
					this._callbacks.setMotionHistoryWriteTarget(
						this._getHistoryTexture(request, historyId, "write"),
					);
				},
			}),
		});
	}

	private _createWarmupContext(
		passId: string,
		declaration: PostProcessExecutionDeclaration,
	): Record<string, unknown> {
		const unavailable = (): null => null;
		return Object.freeze({
			encoder: undefined,
			targets: undefined,
			shared: this._runtime.sharedContext,
			frameBinding: undefined,
			lightingState: undefined,
			getFrameData: (): undefined => undefined,
			resources: createPostProcessResourceAccessor<IRenderTexture>({
				passId,
				declaration,
				colorInput: null,
				colorOutput: null,
				getGBuffer: unavailable,
				getHistory: () => null,
				getTransient: unavailable,
				getShared: unavailable,
			}),
		});
	}

	private _getGBufferTexture(
		request: PostProcessPassRequest,
		semantic: LogicalGBufferSemantic,
	): IRenderTexture | null {
		const handle = request.gBuffer.channels[semantic]?.handle;
		return handle?.backend === "webgpu" && "texture" in handle ?
			(handle.texture as IRenderTexture | null) ?? null : null;
	}

	private _getSharedTexture(id: string): IRenderTexture | null {
		const targets = this._callbacks.getFrameTargets();
		if (!targets) return null;
		switch (id) {
			case WEBGPU_HIZ_SHARED_RESOURCE.id:
				return this._callbacks.isHiZReady() ? targets.hiZ ?? null : null;
			case "backend:transmission-scene-color":
				return targets.transmissionSceneColorCopy ?? null;
			case "backend:transmission-lighting":
				return targets.transmissionLighting ?? null;
			case "backend:transmission-surface-1":
				return targets.gTransmissionSurface1 ?? null;
			case "backend:transmission-surface-2":
				return targets.gTransmissionSurface2 ?? null;
			case "backend:planar-reflection-mask":
				return targets.planarReflectionMask ?? null;
			default:
				return null;
		}
	}

	private _createFrameTargetsView(): WebGPUPostProcessFrameTargets | undefined {
		const targets = this._callbacks.getFrameTargets();
		if (!targets) {
			return undefined;
		}
		return Object.freeze({
			postPing: targets.postPing,
			postPong: targets.postPong,
		});
	}

	private _isOwnedColorTarget(texture: IRenderTexture, targets: WebGPUFrameTargets): boolean {
		return (
			texture === targets.sceneColorMain ||
			texture === targets.postPing ||
			texture === targets.postPong
		);
	}

	private _getPhysicalId(texture: IRenderTexture): string {
		const object = texture as unknown as object;
		let id = this._physicalIds.get(object);
		if (!id) {
			id = `webgpu:${this._nextPhysicalId++}`;
			this._physicalIds.set(object, id);
		}
		return id;
	}

	private _getHistoryTexture(
		request: PostProcessPassRequest,
		id: string,
		side: "read" | "write",
	): IRenderTexture | null {
		const slot = request.histories[id]?.[side];
		return (slot?.resource as IRenderTexture | null) ?? null;
	}

	private _getTransientTexture(
		request: PostProcessPassRequest,
		id: string,
	): IRenderTexture | null {
		const slot = request.transients?.[id];
		return (slot?.handle.resource as IRenderTexture | null) ?? null;
	}
}
