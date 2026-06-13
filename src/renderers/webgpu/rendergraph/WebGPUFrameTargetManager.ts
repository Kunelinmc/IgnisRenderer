import type { FrameContext } from "../../../pipeline/types";
import { Logger } from "../../../foundation/Logger";
import {
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_TARGET_COUNT,
	WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT,
} from "../constants";
import type { WebGPUFrameTargets } from "../WebGPUPostProcessContracts";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import type { WebGPUBackendSession } from "../../WebGPUBackend";
import {
	TextureFormat,
	TextureUsage,
	type IRenderTexture,
} from "../../types";
import { TexturePool, type TexturePoolOptions } from "../TexturePool";

export interface WebGPUFrameMSAATargets {
	sceneColorMain: IRenderTexture;
	gAlbedoAlpha?: IRenderTexture | null;
	gNormalRoughMetal?: IRenderTexture | null;
	gEmissiveOcclusion?: IRenderTexture | null;
	gMotionDepth?: IRenderTexture | null;
	planarReflectionMask?: IRenderTexture | null;
	depth: IRenderTexture;
}

export interface WebGPUFrameTargetRequirements {
	sceneTargetMode: Exclude<WebGPUSceneTargetMode, "single">;
	needsPostProcessTargets: boolean;
	needsOITTargets: boolean;
	needsTransmissionTargets: boolean;
	needsPlanarReflectionMask: boolean;
}

export interface WebGPUFrameTargetManagerCallbacks {
	resolveMSAASampleCount(): number;
	configureDeferredLightingSupport(): void;
	frameHasDeferredLightingWork(context: FrameContext): boolean;
	getFrameContext(): FrameContext | null;
	isDeferredEnabled(): boolean;
	setDeferredEnabled(enabled: boolean): void;
}

export interface WebGPUFrameTargetManagerDebugState {
	readonly width: number;
	readonly height: number;
	readonly msaaSampleCount: number;
	readonly texturePoolOwnerCount: number;
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly needsPostProcessTargets: boolean;
	readonly needsOITTargets: boolean;
	readonly needsTransmissionTargets: boolean;
	readonly needsPlanarReflectionMask: boolean;
	readonly frameTargets: WebGPUFrameTargets | null;
	readonly msaaTargets: WebGPUFrameMSAATargets | null;
}

/**
 * Owns WebGPU frame target allocation, reuse, and pooled texture lifetime.
 */
export class WebGPUFrameTargetManager {
	private readonly _backend: WebGPUBackendSession;
	private readonly _callbacks: WebGPUFrameTargetManagerCallbacks;
	private _frameTargets: WebGPUFrameTargets | null = null;
	private _msaaTargets: WebGPUFrameMSAATargets | null = null;
	private _targetWidth = 0;
	private _targetHeight = 0;
	private _targetMSAASampleCount = 1;
	private _targetSceneTargetMode: WebGPUSceneTargetMode = "single";
	private _targetNeedsPostProcessTargets = false;
	private _targetNeedsOITTargets = false;
	private _targetNeedsTransmissionTargets = false;
	private _targetNeedsPlanarReflectionMask = false;
	private _texturePools = new Map<string, TexturePool>();
	private _texturePoolOwners = new Map<IRenderTexture, TexturePool>();

	public constructor(
		backend: WebGPUBackendSession,
		callbacks: WebGPUFrameTargetManagerCallbacks
	) {
		this._backend = backend;
		this._callbacks = callbacks;
	}

	public get frameTargets(): WebGPUFrameTargets | null {
		return this._frameTargets;
	}

	public get msaaTargets(): WebGPUFrameMSAATargets | null {
		return this._msaaTargets;
	}

	public get targetWidth(): number {
		return this._targetWidth;
	}

	public get targetHeight(): number {
		return this._targetHeight;
	}

	public get targetMSAASampleCount(): number {
		return this._targetMSAASampleCount;
	}

	public get targetSceneTargetMode(): WebGPUSceneTargetMode {
		return this._targetSceneTargetMode;
	}

	public get texturePoolOwnerCount(): number {
		return this._texturePoolOwners.size;
	}

	public getDebugState(): WebGPUFrameTargetManagerDebugState {
		return {
			width: this._targetWidth,
			height: this._targetHeight,
			msaaSampleCount: this._targetMSAASampleCount,
			texturePoolOwnerCount: this._texturePoolOwners.size,
			sceneTargetMode: this._targetSceneTargetMode,
			needsPostProcessTargets: this._targetNeedsPostProcessTargets,
			needsOITTargets: this._targetNeedsOITTargets,
			needsTransmissionTargets: this._targetNeedsTransmissionTargets,
			needsPlanarReflectionMask: this._targetNeedsPlanarReflectionMask,
			frameTargets: this._frameTargets,
			msaaTargets: this._msaaTargets,
		};
	}

	public ensureFrameTargets(
		width: number,
		height: number,
		requirementsOrDeferred: WebGPUFrameTargetRequirements | boolean
	): void {
		const requirements =
			typeof requirementsOrDeferred === "boolean" ?
				{
					sceneTargetMode: requirementsOrDeferred ? "gbuffer" : "mrt",
					needsPostProcessTargets: true,
					needsOITTargets: true,
					needsTransmissionTargets: false,
					needsPlanarReflectionMask: true,
				} satisfies WebGPUFrameTargetRequirements
			:	requirementsOrDeferred;
		const msaaSampleCount = this._callbacks.resolveMSAASampleCount();
		if (width <= 0 || height <= 0) {
			this.destroyFrameTargets();
			return;
		}

		if (
			this._frameTargets &&
			this._targetWidth === width &&
			this._targetHeight === height &&
			this._targetMSAASampleCount === msaaSampleCount &&
			this._targetSceneTargetMode === requirements.sceneTargetMode &&
			this._targetNeedsPostProcessTargets ===
				requirements.needsPostProcessTargets &&
			this._targetNeedsOITTargets === requirements.needsOITTargets &&
			this._targetNeedsTransmissionTargets ===
				requirements.needsTransmissionTargets &&
			this._targetNeedsPlanarReflectionMask ===
				requirements.needsPlanarReflectionMask
		) {
			this._frameTargets.sceneColor = this._frameTargets.sceneColorMain;
			return;
		}

		const acquiredTextures: IRenderTexture[] = [];
		let committed = false;
		const acquireTexture = (
			poolId: string,
			options: TexturePoolOptions,
			textureWidth: number,
			textureHeight: number,
			format: TextureFormat
		): IRenderTexture => {
			const texture = this._acquirePooledTexture(
				poolId,
				options,
				textureWidth,
				textureHeight,
				format
			);
			acquiredTextures.push(texture);
			return texture;
		};

		try {
			this.destroyFrameTargets();
			this._targetWidth = width;
			this._targetHeight = height;
			this._targetMSAASampleCount = msaaSampleCount;
			this._targetSceneTargetMode = requirements.sceneTargetMode;
			this._targetNeedsPostProcessTargets =
				requirements.needsPostProcessTargets;
			this._targetNeedsOITTargets = requirements.needsOITTargets;
			this._targetNeedsTransmissionTargets =
				requirements.needsTransmissionTargets;
			this._targetNeedsPlanarReflectionMask =
				requirements.needsPlanarReflectionMask;
			const needsBaseGBuffer =
				requirements.sceneTargetMode === "mrt" ||
				requirements.sceneTargetMode === "gbuffer";
			const enableDeferred = requirements.sceneTargetMode === "gbuffer";

			const sceneColorMain = acquireTexture(
				"scene-color-main",
				{
					usage:
						TextureUsage.RenderAttachment |
						TextureUsage.TextureBinding |
						TextureUsage.CopySrc |
						TextureUsage.CopyDst,
					label: "WebGPUSceneColorMain",
				},
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const rgba16StoragePool: TexturePoolOptions = {
				usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
				label: "WebGPUStorageRGBA16",
			};
			const postPing =
				requirements.needsPostProcessTargets ?
					acquireTexture(
						"rgba16-storage",
						rgba16StoragePool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const postPong =
				requirements.needsPostProcessTargets ?
					acquireTexture(
						"rgba16-storage",
						rgba16StoragePool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gAlbedoAlpha =
				needsBaseGBuffer ?
					acquireTexture(
						"gbuffer-albedo",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding |
								TextureUsage.StorageBinding |
								TextureUsage.CopySrc |
								TextureUsage.CopyDst,
							label: "WebGPUGBuffer_AlbedoAlpha",
						},
						width,
						height,
						TextureFormat.RGBA8Unorm
					)
				:	null;
			const gNormalRoughMetal =
				needsBaseGBuffer ?
					acquireTexture(
						"gbuffer-rgba16",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding |
								TextureUsage.StorageBinding |
								TextureUsage.CopySrc |
								TextureUsage.CopyDst,
							label: "WebGPUGBuffer_RGBA16",
						},
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gEmissiveOcclusion =
				needsBaseGBuffer ?
					acquireTexture(
						"gbuffer-rgba16",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding |
								TextureUsage.StorageBinding |
								TextureUsage.CopySrc |
								TextureUsage.CopyDst,
							label: "WebGPUGBuffer_RGBA16",
						},
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gMotionDepth =
				needsBaseGBuffer ?
					acquireTexture(
						"gbuffer-motion-depth",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding |
								TextureUsage.StorageBinding |
								TextureUsage.CopySrc |
								TextureUsage.CopyDst,
							label: "WebGPUGBuffer_MotionDepth",
						},
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const deferredColorPool: TexturePoolOptions = {
				usage:
					TextureUsage.RenderAttachment |
					TextureUsage.TextureBinding |
					TextureUsage.StorageBinding |
					TextureUsage.CopySrc |
					TextureUsage.CopyDst,
				label: "WebGPUGBufferDeferredRGBA16",
			};
			const deferredStoragePool: TexturePoolOptions = {
				usage:
					TextureUsage.TextureBinding |
					TextureUsage.StorageBinding |
					TextureUsage.CopySrc |
					TextureUsage.CopyDst,
				label: "WebGPUGBufferDeferredStorageRGBA16",
			};
			const gSpecular =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-color",
						deferredColorPool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gCoatSheen =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-color",
						deferredColorPool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gSheenReflectance =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-color",
						deferredColorPool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gMaterialExt0 =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-storage",
						deferredStoragePool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gMaterialExt1 =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-storage",
						deferredStoragePool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gMaterialExt2 =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-storage",
						deferredStoragePool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gMaterialExt3 =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-storage",
						deferredStoragePool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const depth = acquireTexture(
				"depth-sampleable",
				{
					usage:
						TextureUsage.RenderAttachment |
						TextureUsage.TextureBinding |
						TextureUsage.CopySrc,
					label: "WebGPUDepthSampleable",
				},
				width,
				height,
				TextureFormat.Depth32Float
			);
			const oitAccum =
				requirements.needsOITTargets ?
					acquireTexture(
						"oit-accum",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding,
							label: "WebGPUOITAccum",
						},
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const oitReveal =
				requirements.needsOITTargets ?
					acquireTexture(
						"oit-reveal",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding,
							label: "WebGPUOITReveal",
						},
						width,
						height,
						TextureFormat.R8Unorm
					)
				:	null;
			const oitSceneColorCopy =
				requirements.needsOITTargets ?
					acquireTexture(
						"oit-scene-copy",
						{
							usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
							label: "WebGPUOITSceneColorCopy",
						},
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const transmissionSceneColorCopy =
				requirements.needsTransmissionTargets ?
					acquireTexture(
						"transmission-scene-copy",
						{
							usage:
								TextureUsage.TextureBinding |
								TextureUsage.CopyDst |
								TextureUsage.CopySrc,
							label: "WebGPUTransmissionSceneColorCopy",
						},
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const transmissionLighting =
				requirements.needsTransmissionTargets ?
					acquireTexture(
						"transmission-lighting",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding |
								TextureUsage.CopySrc |
								TextureUsage.CopyDst,
							label: "WebGPUTransmissionLighting",
						},
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gTransmissionSurface0 =
				requirements.needsTransmissionTargets ?
					acquireTexture(
						"transmission-surface",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding |
								TextureUsage.CopySrc |
								TextureUsage.CopyDst,
							label: "WebGPUTransmissionSurface",
						},
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gTransmissionSurface1 =
				requirements.needsTransmissionTargets ?
					acquireTexture(
						"transmission-surface",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding |
								TextureUsage.CopySrc |
								TextureUsage.CopyDst,
							label: "WebGPUTransmissionSurface",
						},
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gTransmissionSurface2 =
				requirements.needsTransmissionTargets ?
					acquireTexture(
						"transmission-surface",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding |
								TextureUsage.CopySrc |
								TextureUsage.CopyDst,
							label: "WebGPUTransmissionSurface",
						},
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const transmissionDepth =
				requirements.needsTransmissionTargets ?
					acquireTexture(
						"transmission-depth",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding |
								TextureUsage.CopyDst,
							label: "WebGPUTransmissionDepth",
						},
						width,
						height,
						TextureFormat.Depth32Float
					)
				:	null;
			const planarReflectionMask =
				requirements.needsPlanarReflectionMask ?
					acquireTexture(
						"planar-reflection-mask",
						{
							usage:
								TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding,
							label: "WebGPUPlanarReflectionMask",
						},
						width,
						height,
						TextureFormat.R8Unorm
					)
				:	null;
			const useMSAA = msaaSampleCount > 1;
			const msaaPoolKey = `msaa-${msaaSampleCount}`;
			const msaaPoolOptions: TexturePoolOptions = {
				usage: TextureUsage.RenderAttachment,
				sampleCount: msaaSampleCount,
				label: `WebGPUMSAA_${msaaSampleCount}x`,
			};
			const nextMSAATargets: WebGPUFrameMSAATargets | null =
				useMSAA ?
					{
						sceneColorMain: acquireTexture(
							msaaPoolKey,
							msaaPoolOptions,
							width,
							height,
							TextureFormat.RGBA16Float
						),
						gAlbedoAlpha:
							needsBaseGBuffer ?
								acquireTexture(
									msaaPoolKey,
									msaaPoolOptions,
									width,
									height,
									TextureFormat.RGBA8Unorm
								)
							:	null,
						gNormalRoughMetal:
							needsBaseGBuffer ?
								acquireTexture(
									msaaPoolKey,
									msaaPoolOptions,
									width,
									height,
									TextureFormat.RGBA16Float
								)
							:	null,
						gEmissiveOcclusion:
							needsBaseGBuffer ?
								acquireTexture(
									msaaPoolKey,
									msaaPoolOptions,
									width,
									height,
									TextureFormat.RGBA16Float
								)
							:	null,
						gMotionDepth:
							needsBaseGBuffer ?
								acquireTexture(
									msaaPoolKey,
									msaaPoolOptions,
									width,
									height,
									TextureFormat.RGBA16Float
								)
							:	null,
						planarReflectionMask:
							requirements.needsPlanarReflectionMask ?
								acquireTexture(
									msaaPoolKey,
									msaaPoolOptions,
									width,
									height,
									TextureFormat.R8Unorm
								)
							:	null,
						depth: acquireTexture(
							msaaPoolKey,
							msaaPoolOptions,
							width,
							height,
							TextureFormat.Depth32Float
						),
					}
				:	null;

			this._msaaTargets = nextMSAATargets;
			this._frameTargets = {
				sceneColor: sceneColorMain,
				sceneColorMain,
				postPing,
				postPong,
				gAlbedoAlpha,
				gNormalRoughMetal,
				gEmissiveOcclusion,
				gMotionDepth,
				gSpecular,
				gCoatSheen,
				gSheenReflectance,
				gMaterialExt0,
				gMaterialExt1,
				gMaterialExt2,
				gMaterialExt3,
				depth,
				oitAccum,
				oitReveal,
				oitSceneColorCopy,
				transmissionSceneColorCopy,
				transmissionLighting,
				gTransmissionSurface0,
				gTransmissionSurface1,
				gTransmissionSurface2,
				transmissionDepth,
				planarReflectionMask,
			};
			committed = true;
		} catch (error) {
			if (!committed) {
				for (const texture of new Set(acquiredTextures)) {
					this._releasePooledTexture(texture);
				}
			}
			this.destroyFrameTargets();
			if (requirements.sceneTargetMode === "gbuffer") {
				this._callbacks.setDeferredEnabled(false);
				const key = "webgpu-deferred-runtime-fallback";
				Logger.warn(
					`[${key}] WebGPU deferred frame target allocation failed; retrying with legacy MRT forward path. ${String(error)}`,
					{ scope: "WebGPUFrameExecutor", onceKey: key }
				);
				this.ensureFrameTargets(width, height, {
					...requirements,
					sceneTargetMode:
						requirements.sceneTargetMode === "gbuffer" ?
							"mrt"
						:	requirements.sceneTargetMode,
				});
				return;
			}
			if (msaaSampleCount > 1) {
				const setter = (
					this._backend as {
						setMSAASampleCount?: (sampleCount: number) => void;
					}
				).setMSAASampleCount;
				if (typeof setter === "function") {
					setter.call(this._backend, 1);
				}
				const key = "webgpu-msaa-runtime-fallback-1x";
				Logger.warn(
					`[${key}] WebGPU ${msaaSampleCount}x MSAA target allocation failed; retrying at 1x.`,
					{ scope: "WebGPUFrameExecutor", onceKey: key }
				);
				this._callbacks.configureDeferredLightingSupport();
				const frameContext = this._callbacks.getFrameContext();
				this.ensureFrameTargets(width, height, {
					...requirements,
					sceneTargetMode:
						this._callbacks.isDeferredEnabled() &&
						frameContext &&
						this._callbacks.frameHasDeferredLightingWork(frameContext) ?
							"gbuffer"
						:	requirements.sceneTargetMode,
				});
				return;
			}
			throw error;
		}
	}

	public destroyFrameTargets(): void {
		const textures = new Set<IRenderTexture>();
		if (this._frameTargets) {
			for (const texture of Object.values(this._frameTargets)) {
				if (texture && typeof texture === "object") {
					textures.add(texture as IRenderTexture);
				}
			}
		}
		if (this._msaaTargets) {
			for (const texture of Object.values(this._msaaTargets)) {
				if (texture && typeof texture === "object") {
					textures.add(texture as IRenderTexture);
				}
			}
		}
		for (const texture of textures) {
			this._releasePooledTexture(texture);
		}
		this._frameTargets = null;
		this._msaaTargets = null;
		this._targetWidth = 0;
		this._targetHeight = 0;
		this._targetMSAASampleCount = 1;
		this._targetSceneTargetMode = "single";
		this._targetNeedsPostProcessTargets = false;
		this._targetNeedsOITTargets = false;
		this._targetNeedsTransmissionTargets = false;
		this._targetNeedsPlanarReflectionMask = false;
	}

	public destroyTexturePools(): void {
		this._texturePoolOwners.clear();
		for (const pool of this._texturePools.values()) {
			pool.destroy();
		}
		this._texturePools.clear();
	}

	private _acquirePooledTexture(
		poolId: string,
		options: TexturePoolOptions,
		width: number,
		height: number,
		format: TextureFormat
	): IRenderTexture {
		let pool = this._texturePools.get(poolId);
		if (!pool) {
			pool = new TexturePool(this._backend, options);
			this._texturePools.set(poolId, pool);
		}
		const texture = pool.acquire(width, height, format);
		this._texturePoolOwners.set(texture, pool);
		return texture;
	}

	private _releasePooledTexture(texture: IRenderTexture): void {
		const owner = this._texturePoolOwners.get(texture);
		if (!owner) {
			texture.destroy();
			return;
		}
		this._texturePoolOwners.delete(texture);
		owner.release(texture);
	}
}
