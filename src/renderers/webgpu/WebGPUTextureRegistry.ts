import type { Texture } from "../../core/Texture";
import {
	AddressMode,
	FilterMode,
	TextureFormat,
	TextureUsage,
	type IRenderTexture,
	type ISampler,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import { createTextureMipUploadLevels, WEBGPU_TEXTURE_SLOT } from "./";

export class WebGPUTextureRegistry {
	private _backend: WebGPUBackend;
	private _textureCache = new WeakMap<Texture, IRenderTexture>();
	private _samplerCache = new WeakMap<Texture, ISampler>();
	private _whiteTexture: IRenderTexture | null = null;
	private _neutralNormalTexture: IRenderTexture | null = null;
	private _whiteSampler: ISampler | null = null;

	constructor(backend: WebGPUBackend) {
		this._backend = backend;
	}

	public getTextureForSlot(
		texture: Texture | null,
		slotIndex: number
	): IRenderTexture {
		if (!texture?.data || texture.width <= 0 || texture.height <= 0) {
			return (
					slotIndex === WEBGPU_TEXTURE_SLOT.NORMAL ||
						slotIndex === WEBGPU_TEXTURE_SLOT.CLEARCOAT_NORMAL
				) ?
					this.getNeutralNormalTexture()
				:	this.getWhiteTexture();
		}

		let cached = this._textureCache.get(texture);
		if (!cached) {
			const mipLevelCount = Math.max(1, texture.mipmaps.length || 1);
			cached = this._backend.createTexture({
				width: texture.width,
				height: texture.height,
				format: TextureFormat.RGBA8Unorm,
				usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
				mipLevelCount,
				label: `Texture_${slotIndex}_${texture.width}x${texture.height}`,
			});

			const uploads = createTextureMipUploadLevels(texture);
			for (const upload of uploads) {
				this._backend.writeTexture(
					cached,
					new Uint8Array(upload.data),
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
			this._textureCache.set(texture, cached);
		}

		return cached;
	}

	public getSamplerForTexture(texture: Texture | null): ISampler {
		if (!texture) {
			return this.getWhiteSampler();
		}

		let cached = this._samplerCache.get(texture);
		if (!cached) {
			cached = this._backend.createSampler({
				addressModeU: this._mapWrapMode(texture.wrapS),
				addressModeV: this._mapWrapMode(texture.wrapT),
				magFilter: this._mapFilterMode(texture.magFilter),
				minFilter: this._mapFilterMode(texture.minFilter),
				mipmapFilter: this._mapFilterMode(texture.minFilter),
				label: `Sampler_${texture.width}x${texture.height}`,
			});
			this._samplerCache.set(texture, cached);
		}

		return cached;
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
			this._backend.writeTexture(
				this._whiteTexture,
				new Uint8Array(data),
				{ bytesPerRow: 256, rowsPerImage: 1 },
				{ width: 1, height: 1, depthOrArrayLayers: 1 }
			);
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
			this._backend.writeTexture(
				this._neutralNormalTexture,
				new Uint8Array(data),
				{ bytesPerRow: 256, rowsPerImage: 1 },
				{ width: 1, height: 1, depthOrArrayLayers: 1 }
			);
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
		}

		return this._whiteSampler;
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
}
