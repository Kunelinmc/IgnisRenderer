import type { Texture } from "../../core/Texture";
import {
	getTextureFormatInfo,
	TextureFormat,
} from "../../core/TextureFormat";
import {
	AddressMode,
	FilterMode,
	TextureUsage,
	type IRenderTexture,
	type ISampler,
} from "../types";
import { Logger } from "../../foundation/Logger";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUResourceManager } from "./WebGPUResourceManager";
import { tryGetWebGPUTextureHandle } from "./WebGPUResourceAccess";
import {
	createTextureMipUploadLevels,
	resolveWebGPUTextureUploadFormat,
	WEBGPU_TEXTURE_SLOT,
} from "./";
import {
	canGenerateWebGPUMipmaps,
	resolveWebGPUMipmapLevelCount,
	textureFilterRequiresMipmaps,
	WebGPUMipmapGenerator,
} from "./WebGPUMipmapGenerator";

interface TextureCacheEntry {
	resource: IRenderTexture;
	mipLevelCount: number;
	externalImageCopyCompatible: boolean;
	format: TextureFormat;
}

interface SamplerCacheEntry {
	sampler: ISampler;
	key: string;
}

interface TextureOwnedResources {
	texture: IRenderTexture | null;
	sampler: ISampler | null;
}

interface TextureDisposeObserver {
	registry: WebGPUTextureRegistry | null;
}

export class WebGPUTextureRegistry {
	private _backend: WebGPUDeviceResourceHost;
	private _resourceManager: WebGPUResourceManager;
	private _textureCache = new WeakMap<Texture, TextureCacheEntry>();
	private _samplerCache = new WeakMap<Texture, SamplerCacheEntry>();
	private _uploadedVersionCache = new WeakMap<Texture, number>();
	private _ownedTextures = new Set<IRenderTexture>();
	private _ownedSamplers = new Set<ISampler>();
	private _ownedResources = new WeakMap<Texture, TextureOwnedResources>();
	private _disposeUnsubscribers = new WeakMap<Texture, () => void>();
	private _mipmapGenerationPromises = new WeakMap<Texture, Promise<void>>();
	private _disposeObserver: TextureDisposeObserver = { registry: this };
	private _finalizationRegistry: FinalizationRegistry<TextureOwnedResources> | null =
		typeof FinalizationRegistry === "function" ?
			new FinalizationRegistry((resources) => {
				this._destroyOwnedResources(resources);
			})
		: null;
	private _mipmapGenerator: WebGPUMipmapGenerator | null = null;
	private _whiteTexture: IRenderTexture | null = null;
	private _neutralNormalTexture: IRenderTexture | null = null;
	private _whiteSampler: ISampler | null = null;

	constructor(
		backend: WebGPUDeviceResourceHost,
		resourceManager: WebGPUResourceManager,
	) {
		this._backend = backend;
		this._resourceManager = resourceManager;
	}

	public getTextureForSlot(
		texture: Texture | null,
		slotIndex: number
	): IRenderTexture {
		if (!texture) {
			return (
					slotIndex === WEBGPU_TEXTURE_SLOT.NORMAL ||
						slotIndex === WEBGPU_TEXTURE_SLOT.CLEARCOAT_NORMAL
				) ?
					this.getNeutralNormalTexture()
				:	this.getWhiteTexture();
		}

		let cacheEntry = this._textureCache.get(texture);
		const externalImageCopyCompatible =
			resolveExternalTextureSource(texture) !== null;
		const uploadFormat =
			externalImageCopyCompatible ?
				resolveExternalImageCopyFormat(texture)
			:	resolveWebGPUTextureUploadFormat(texture);
		if (
			!cacheEntry &&
			(!this._isTextureDimensionValid(texture.width, texture.height) ||
				(!texture.data && !externalImageCopyCompatible))
		) {
			return (
					slotIndex === WEBGPU_TEXTURE_SLOT.NORMAL ||
						slotIndex === WEBGPU_TEXTURE_SLOT.CLEARCOAT_NORMAL
				) ?
					this.getNeutralNormalTexture()
				:	this.getWhiteTexture();
		}

		const mipPolicy = resolveTextureMipPolicy(texture, uploadFormat);
		if (mipPolicy.skipReason) {
			this._warnAutoMipmapSkipped(texture, uploadFormat, mipPolicy.skipReason);
		}
		const mipLevelCount = mipPolicy.mipLevelCount;
		const shouldRecreateTexture =
			!cacheEntry ||
			cacheEntry.resource.width !== texture.width ||
			cacheEntry.resource.height !== texture.height ||
			cacheEntry.mipLevelCount !== mipLevelCount ||
			cacheEntry.format !== uploadFormat ||
			(externalImageCopyCompatible && !cacheEntry.externalImageCopyCompatible);

		if (shouldRecreateTexture) {
			if (cacheEntry?.resource) {
				this._releaseOwnedTextureResource(texture, cacheEntry.resource);
			}
			const usage =
				TextureUsage.TextureBinding |
				TextureUsage.CopyDst |
				(externalImageCopyCompatible || mipPolicy.autoGenerate ?
					TextureUsage.RenderAttachment
				:	0);

			const resource = this._backend.createTexture({
				width: texture.width,
				height: texture.height,
				format: uploadFormat,
				usage,
				mipLevelCount,
				label: `Texture_${slotIndex}_${texture.width}x${texture.height}`,
			});

			cacheEntry = {
				resource,
				mipLevelCount,
				externalImageCopyCompatible,
				format: resource.format ?? uploadFormat,
			};
			this._ownedTextures.add(resource);
			this._setOwnedTextureResource(texture, resource);
			this._textureCache.set(texture, cacheEntry);
			this._uploadedVersionCache.delete(texture);
		}

		const uploadedVersion = this._uploadedVersionCache.get(texture);
		if (uploadedVersion !== texture.version) {
			const usedExternalImageFastPath = this._tryUploadExternalSource(
				texture,
				cacheEntry.resource
			);

			if (!usedExternalImageFastPath) {
				if (
					getTextureFormatInfo(uploadFormat).isCompressed &&
					cacheEntry.format !== uploadFormat
				) {
					return cacheEntry.resource;
				}
				const uploads = createTextureMipUploadLevels(texture, cacheEntry.format);
				for (const upload of uploads) {
					const uploadData = this._toArrayBufferBackedView(upload.data);
					this._resourceManager.writeTexture(
						cacheEntry.resource,
						uploadData,
						{
							bytesPerRow: upload.bytesPerRow,
							rowsPerImage: upload.height,
							mipLevel: upload.mipLevel,
						},
						{
							width: upload.width,
							height: upload.height,
							depthOrArrayLayers: 1,
						}
					);
				}
			}
			if (mipPolicy.autoGenerate) {
				this._generateMipmaps(texture, cacheEntry);
			}
			this._uploadedVersionCache.set(texture, texture.version);
		}

		return cacheEntry.resource;
	}

	public async getTextureForSlotAsync(
		texture: Texture | null,
		slotIndex: number
	): Promise<IRenderTexture> {
		const resource = this.getTextureForSlot(texture, slotIndex);
		if (texture) {
			await this._mipmapGenerationPromises.get(texture);
		}
		return resource;
	}

	/**
	 * Registers a Texture -> IRenderTexture mapping without taking ownership of
	 * the provided GPU resource. Intended for externally-produced GPU textures.
	 */
	public registerExternalTexture(
		texture: Texture,
		resource: IRenderTexture,
		uploadedVersion: number = texture.version,
		mipLevelCount: number = 1
	): void {
		const cached = this._textureCache.get(texture);
		if (
			cached &&
			cached.resource !== resource &&
			this._ownedTextures.has(cached.resource)
		) {
			const resources = this._ownedResources.get(texture);
			this._releaseOwnedTextureResource(texture, cached.resource, resources);
			if (resources) {
				this._clearOwnershipIfEmpty(texture, resources);
			}
		}
		this._textureCache.set(texture, {
			resource,
			mipLevelCount: Math.max(1, Math.floor(mipLevelCount)),
			externalImageCopyCompatible: false,
			format: resolveRegisteredTextureFormat(resource, texture),
		});
		this._uploadedVersionCache.set(texture, uploadedVersion);
	}

	/**
	 * Releases registry-owned GPU resources associated with one CPU texture.
	 * Externally owned texture resources are removed from the cache but are not
	 * destroyed.
	 *
	 * @internal Owned by the WebGPU texture lifecycle. Applications should call
	 * `Texture.dispose()`.
	 */
	public releaseTexture(texture: Texture): void {
		const resources = this._ownedResources.get(texture);
		if (resources) {
			if (resources.texture) {
				this._releaseOwnedTextureResource(texture, resources.texture, resources);
			}
			if (resources.sampler) {
				this._releaseOwnedSamplerResource(resources.sampler, resources);
			}
			this._clearOwnership(texture, resources);
		}

		this._textureCache.delete(texture);
		this._samplerCache.delete(texture);
		this._uploadedVersionCache.delete(texture);
		this._mipmapGenerationPromises.delete(texture);
	}

	/**
	 * Removes a previously registered external mapping.
	 * This does not destroy externally-owned resources.
	 */
	public unregisterExternalTexture(texture: Texture): void {
		const cached = this._textureCache.get(texture);
		if (!cached) {
			return;
		}
		if (!this._ownedTextures.has(cached.resource)) {
			this._textureCache.delete(texture);
		}
		this._uploadedVersionCache.delete(texture);
	}

	public getSamplerForTexture(texture: Texture | null): ISampler {
		if (!texture) {
			return this.getWhiteSampler();
		}

		const key = this._getSamplerKey(texture);
		const cached = this._samplerCache.get(texture);
		if (cached && cached.key === key) {
			return cached.sampler;
		}

		if (cached) {
			this._releaseOwnedSamplerResource(
				cached.sampler,
				this._ownedResources.get(texture),
			);
		}

		const sampler = this._backend.createSampler({
			addressModeU: this._mapWrapMode(texture.wrapS),
			addressModeV: this._mapWrapMode(texture.wrapT),
			magFilter: this._mapFilterMode(texture.magFilter),
			minFilter: this._mapFilterMode(texture.minFilter),
			mipmapFilter: this._mapFilterMode(texture.minFilter),
			label: `Sampler_${texture.width}x${texture.height}`,
		});
		this._samplerCache.set(texture, {
			sampler,
			key,
		});
		this._ownedSamplers.add(sampler);
		this._setOwnedSamplerResource(texture, sampler);
		return sampler;
	}

	public getWhiteTexture(): IRenderTexture {
		if (!this._whiteTexture) {
			this._whiteTexture = this._backend.createTexture({
				width: 1,
				height: 1,
				format: TextureFormat.RGBA8Unorm,
				usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
				label: "WebGPUWhiteTexture",
			});
			const data = new Uint8Array(256).fill(255);
			this._resourceManager.writeTexture(
				this._whiteTexture,
				new Uint8Array(data),
				{ bytesPerRow: 256, rowsPerImage: 1 },
				{ width: 1, height: 1, depthOrArrayLayers: 1 }
			);
			this._ownedTextures.add(this._whiteTexture);
		}

		return this._whiteTexture;
	}

	public getNeutralNormalTexture(): IRenderTexture {
		if (!this._neutralNormalTexture) {
			this._neutralNormalTexture = this._backend.createTexture({
				width: 1,
				height: 1,
				format: TextureFormat.RGBA8Unorm,
				usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
				label: "WebGPUNeutralNormalTexture",
			});
			const data = new Uint8Array(256);
			data[0] = 128;
			data[1] = 128;
			data[2] = 255;
			data[3] = 255;
			this._resourceManager.writeTexture(
				this._neutralNormalTexture,
				new Uint8Array(data),
				{ bytesPerRow: 256, rowsPerImage: 1 },
				{ width: 1, height: 1, depthOrArrayLayers: 1 }
			);
			this._ownedTextures.add(this._neutralNormalTexture);
		}

		return this._neutralNormalTexture;
	}

	public getWhiteSampler(): ISampler {
		if (!this._whiteSampler) {
			this._whiteSampler = this._backend.createSampler({
				addressModeU: AddressMode.Repeat,
				addressModeV: AddressMode.Repeat,
				magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear,
				mipmapFilter: FilterMode.Linear,
				label: "WebGPUWhiteSampler",
			});
			this._ownedSamplers.add(this._whiteSampler);
		}

		return this._whiteSampler;
	}

	public destroy(): void {
		this._disposeObserver.registry = null;
		for (const texture of this._ownedTextures) {
			texture.destroy();
		}
		this._ownedTextures.clear();
		for (const sampler of this._ownedSamplers) {
			this._destroySampler(sampler);
		}
		this._ownedSamplers.clear();
		this._textureCache = new WeakMap<Texture, TextureCacheEntry>();
		this._samplerCache = new WeakMap<Texture, SamplerCacheEntry>();
		this._uploadedVersionCache = new WeakMap<Texture, number>();
		this._ownedResources = new WeakMap<Texture, TextureOwnedResources>();
		this._disposeUnsubscribers = new WeakMap<Texture, () => void>();
		this._mipmapGenerationPromises = new WeakMap<Texture, Promise<void>>();
		this._finalizationRegistry = null;
		this._mipmapGenerator?.destroy();
		this._mipmapGenerator = null;
		this._whiteTexture = null;
		this._neutralNormalTexture = null;
		this._whiteSampler = null;
	}

	private _mapWrapMode(value?: string): AddressMode {
		switch (value) {
			case "Clamp":
				return AddressMode.ClampToEdge;
			case "MirroredRepeat":
				return AddressMode.MirrorRepeat;
			default:
				return AddressMode.Repeat;
		}
	}

	private _mapFilterMode(value?: string): FilterMode {
		return value === "Nearest" || value === "NearestMipmapNearest" ?
				FilterMode.Nearest
			:	FilterMode.Linear;
	}

	private _getSamplerKey(texture: Texture): string {
		const addressModeU = this._mapWrapMode(texture.wrapS);
		const addressModeV = this._mapWrapMode(texture.wrapT);
		const magFilter = this._mapFilterMode(texture.magFilter);
		const minFilter = this._mapFilterMode(texture.minFilter);
		const mipmapFilter = this._mapFilterMode(texture.minFilter);
		return [
			addressModeU,
			addressModeV,
			magFilter,
			minFilter,
			mipmapFilter,
		].join("|");
	}

	private _toArrayBufferBackedView(data: Uint8Array): Uint8Array<ArrayBuffer> {
		if (data.buffer instanceof ArrayBuffer) {
			return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		}
		return new Uint8Array(data);
	}

	private _generateMipmaps(texture: Texture, cacheEntry: TextureCacheEntry): void {
		if (!this._mipmapGenerator) {
			this._mipmapGenerator = new WebGPUMipmapGenerator(this._backend);
		}
		const generation = this._mipmapGenerator.generate(
			cacheEntry.resource,
			cacheEntry.format,
			cacheEntry.mipLevelCount
		).then(() => undefined).catch((error) => {
			const label = texture.label ?? "unnamed";
			Logger.warn(
				`WebGPU mipmap generation failed for texture "${label}": ` +
					String(error),
				{
					scope: "WebGPUTextureRegistry",
					onceKey:
					`webgpu-mipmap-generation-failed-${texture.label ?? ""}-` +
						`${texture.width}x${texture.height}-${cacheEntry.format}`,
				}
			);
		}).finally(() => {
			if (this._mipmapGenerationPromises.get(texture) === generation) {
				this._mipmapGenerationPromises.delete(texture);
			}
		});
		this._mipmapGenerationPromises.set(texture, generation);
	}

	private _warnAutoMipmapSkipped(
		texture: Texture,
		format: TextureFormat,
		reason: string
	): void {
		const label = texture.label ?? "unnamed";
		Logger.warn(
			`WebGPU automatic mipmap generation skipped for texture "${label}": ` +
				reason,
			{
				scope: "WebGPUTextureRegistry",
				onceKey:
					`webgpu-mipmap-skipped-${texture.label ?? ""}-` +
					`${texture.width}x${texture.height}-${format}-${reason}`,
			}
		);
	}

	private _isTextureDimensionValid(
		width: number | undefined,
		height: number | undefined
	): boolean {
		return (
			typeof width === "number" &&
			typeof height === "number" &&
			Number.isFinite(width) &&
			Number.isFinite(height) &&
			width > 0 &&
			height > 0
		);
	}

	private _tryUploadExternalSource(
		texture: Texture,
		target: IRenderTexture
	): boolean {
		if (texture.mipmaps.length > 1) {
			return false;
		}

		const queue = this._backend.queue;
		if (!queue || typeof queue.copyExternalImageToTexture !== "function") {
			return false;
		}

		const source = resolveExternalTextureSource(texture);
		if (
			!source ||
			texture.width <= 0 ||
			texture.height <= 0
		) {
			return false;
		}

		try {
			const gpuTexture = tryGetWebGPUTextureHandle(target);
			if (!gpuTexture) {
				return false;
			}
			queue.copyExternalImageToTexture(
				{
					source,
				},
				{
					texture: gpuTexture,
				},
				{
					width: texture.width,
					height: texture.height,
					depthOrArrayLayers: 1,
				}
			);
			return true;
		} catch {
			return false;
		}
	}

	private _ensureOwnedResources(texture: Texture): TextureOwnedResources {
		let resources = this._ownedResources.get(texture);
		if (resources) {
			return resources;
		}

		resources = { texture: null, sampler: null };
		this._ownedResources.set(texture, resources);
		this._finalizationRegistry?.register(texture, resources, texture);
		const observer = this._disposeObserver;
		this._disposeUnsubscribers.set(
			texture,
			texture.onDispose((disposedTexture) => {
				observer.registry?.releaseTexture(disposedTexture);
			}),
		);
		return resources;
	}

	private _setOwnedTextureResource(
		texture: Texture,
		resource: IRenderTexture,
	): void {
		const resources = this._ensureOwnedResources(texture);
		if (resources.texture && resources.texture !== resource) {
			this._releaseOwnedTextureResource(texture, resources.texture, resources);
		}
		resources.texture = resource;
		this._ownedTextures.add(resource);
	}

	private _setOwnedSamplerResource(texture: Texture, sampler: ISampler): void {
		const resources = this._ensureOwnedResources(texture);
		if (resources.sampler && resources.sampler !== sampler) {
			this._releaseOwnedSamplerResource(resources.sampler, resources);
		}
		resources.sampler = sampler;
		this._ownedSamplers.add(sampler);
	}

	private _releaseOwnedTextureResource(
		texture: Texture | null,
		resource: IRenderTexture,
		resources: TextureOwnedResources | undefined =
			texture ? this._ownedResources.get(texture) : undefined,
	): void {
		if (resources?.texture === resource) {
			resources.texture = null;
		}
		if (!this._ownedTextures.has(resource)) {
			return;
		}

		const destroy = () => {
			if (this._ownedTextures.delete(resource)) {
				resource.destroy();
			}
		};
		const pendingGeneration =
			texture ? this._mipmapGenerationPromises.get(texture) : undefined;
		if (pendingGeneration) {
			void pendingGeneration.then(destroy, destroy);
			return;
		}
		destroy();
	}

	private _releaseOwnedSamplerResource(
		sampler: ISampler,
		resources?: TextureOwnedResources,
	): void {
		if (resources?.sampler === sampler) {
			resources.sampler = null;
		}
		if (this._ownedSamplers.delete(sampler)) {
			this._destroySampler(sampler);
		}
	}

	private _destroyOwnedResources(resources: TextureOwnedResources): void {
		if (resources.texture) {
			this._releaseOwnedTextureResource(null, resources.texture, resources);
		}
		if (resources.sampler) {
			this._releaseOwnedSamplerResource(resources.sampler, resources);
		}
	}

	private _clearOwnershipIfEmpty(
		texture: Texture,
		resources: TextureOwnedResources,
	): void {
		if (resources.texture || resources.sampler) {
			return;
		}
		this._clearOwnership(texture, resources);
	}

	private _clearOwnership(
		texture: Texture,
		resources: TextureOwnedResources,
	): void {
		this._finalizationRegistry?.unregister(texture);
		if (this._ownedResources.get(texture) === resources) {
			this._ownedResources.delete(texture);
		}
		this._disposeUnsubscribers.get(texture)?.();
		this._disposeUnsubscribers.delete(texture);
	}

	private _destroySampler(sampler: ISampler): void {
		const destroyFn = (sampler as { destroy?: () => void }).destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(sampler);
		}
	}
}

interface TextureMipPolicy {
	mipLevelCount: number;
	autoGenerate: boolean;
	skipReason: string | null;
}

function resolveTextureMipPolicy(
	texture: Texture,
	format: TextureFormat
): TextureMipPolicy {
	const explicitMipLevelCount = Math.max(
		1,
		texture.levels.length,
		texture.mipmaps.length,
		1
	);
	if (explicitMipLevelCount > 1) {
		return {
			mipLevelCount: explicitMipLevelCount,
			autoGenerate: false,
			skipReason: null,
		};
	}
	if (!textureFilterRequiresMipmaps(texture.minFilter)) {
		return {
			mipLevelCount: 1,
			autoGenerate: false,
			skipReason: null,
		};
	}

	const generatedMipLevelCount = resolveWebGPUMipmapLevelCount(
		texture.width,
		texture.height
	);
	if (generatedMipLevelCount <= 1) {
		return {
			mipLevelCount: 1,
			autoGenerate: false,
			skipReason: null,
		};
	}
	if (!canGenerateWebGPUMipmaps(format)) {
		return {
			mipLevelCount: 1,
			autoGenerate: false,
			skipReason: `format "${format}" is not filterable render-attachment color data`,
		};
	}

	return {
		mipLevelCount: generatedMipLevelCount,
		autoGenerate: true,
		skipReason: null,
	};
}

function resolveRegisteredTextureFormat(
	resource: IRenderTexture,
	texture: Texture
): TextureFormat {
	const resourceFormat =
		resource.format ??
		(resource as { desc?: { format?: TextureFormat } }).desc?.format;
	if (resourceFormat) {
		return resourceFormat;
	}
	return resolveWebGPUTextureUploadFormat(texture);
}

function resolveExternalImageCopyFormat(texture: Texture): TextureFormat {
	switch (texture.format) {
		case TextureFormat.RGBA8UnormSrgb:
			return TextureFormat.RGBA8UnormSrgb;
		case TextureFormat.RGBA8Unorm:
		default:
			return TextureFormat.RGBA8Unorm;
	}
}

function resolveExternalTextureSource(texture: Texture): TexImageSource | null {
	const source = texture.getUploadSource(0);
	return source && !ArrayBuffer.isView(source) ? source : null;
}
