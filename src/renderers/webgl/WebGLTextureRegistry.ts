import type { Texture } from "../../core/Texture";
import { clamp } from "../../maths/Common";
import { Logger } from "../../foundation/Logger";
import { float32ToFloat16Bits } from "../../foundation/Float16";
import { TextureFormat } from "../types";

interface TextureEntry {
	texture: WebGLTexture;
	version: number;
	width: number;
	height: number;
	isLinear: boolean;
	uploadKind: WebGLTextureUploadKind;
	actualFormat: TextureFormat;
}

export interface WebGLTextureRegistryOptions {
	/**
	 * Controls whether eligible textures are uploaded during resolve or queued
	 * for a later frame-budgeted upload.
	 */
	uploadScheduling?: "immediate" | "deferred";
	/**
	 * Maximum queued texture uploads processed by `processPendingUploads`.
	 */
	maxUploadsPerFrame?: number;
	/**
	 * Approximate byte budget for queued texture uploads processed by
	 * `processPendingUploads`.
	 */
	maxUploadBytesPerFrame?: number;
	/**
	 * Called when queued uploads remain and another frame should be scheduled.
	 */
	onUploadPending?: () => void;
}

export interface ResolvedWebGLTexture {
	texture: WebGLTexture;
	isLinear: boolean;
}

type WebGLTextureUploadKind = string;

interface WebGLTextureUploadFormat {
	kind: WebGLTextureUploadKind;
	textureFormat: TextureFormat;
	internalFormat: number;
	format: number;
	type: number;
	isFloat: boolean;
	hardwareSRGB: boolean;
	channelCount: number;
}

interface WebGLTextureResolveOptions {
	preferFloat?: boolean;
}

interface PendingTextureUpload {
	texture: Texture;
	label: string;
	targetTexture: WebGLTexture;
	uploadFormat: WebGLTextureUploadFormat;
	cacheKey: string;
	version: number;
	width: number;
	height: number;
	isLinear: boolean;
	estimatedBytes: number;
	queued: boolean;
}

interface DeferredTextureUploadRequest {
	texture: Texture;
	label: string;
	srgbDefault: boolean;
	targetTexture: WebGLTexture;
	uploadFormat: WebGLTextureUploadFormat;
	cacheKey: string;
	cached: TextureEntry | undefined;
	width: number;
	height: number;
	isLinear: boolean;
}

export const DEFAULT_DEFERRED_UPLOADS_PER_FRAME = 4;
export const DEFAULT_DEFERRED_UPLOAD_BYTES_PER_FRAME = 32 * 1024 * 1024;

export class WebGLTextureRegistry {
	private _gl: WebGL2RenderingContext;
	private _maxTextureSize: number;
	private _cache = new WeakMap<Texture, Map<string, TextureEntry>>();
	private _pendingUploadsByTexture = new WeakMap<
		Texture,
		Map<string, PendingTextureUpload>
	>();
	private _pendingUploadQueue: PendingTextureUpload[] = [];
	private _owned = new Set<WebGLTexture>();
	private _whiteTexture: WebGLTexture | null = null;
	private _neutralNormalTexture: WebGLTexture | null = null;
	private _supportsFloatLinearFiltering: boolean | null = null;
	private _uploadScheduling: "immediate" | "deferred";
	private _maxUploadsPerFrame: number;
	private _maxUploadBytesPerFrame: number;
	private _onUploadPending: (() => void) | null;

	constructor(
		gl: WebGL2RenderingContext,
		_warn?: (key: string, message: string) => void,
		options: WebGLTextureRegistryOptions = {}
	) {
		this._gl = gl;
		this._maxTextureSize = this._resolveMaxTextureSize(gl);
		this._uploadScheduling = options.uploadScheduling ?? "immediate";
		this._maxUploadsPerFrame = Math.max(
			1,
			Math.floor(
				options.maxUploadsPerFrame ?? DEFAULT_DEFERRED_UPLOADS_PER_FRAME
			)
		);
		this._maxUploadBytesPerFrame = Math.max(
			1,
			Math.floor(
				options.maxUploadBytesPerFrame ??
					DEFAULT_DEFERRED_UPLOAD_BYTES_PER_FRAME
			)
		);
		this._onUploadPending = options.onUploadPending ?? null;
	}

	/**
	 * Returns the number of queued texture uploads waiting for frame-budgeted
	 * processing. Reading this value has no side effects.
	 */
	public get pendingUploadCount(): number {
		return this._pendingUploadQueue.length;
	}

	/**
	 * Processes deferred texture uploads at the start of a WebGL frame.
	 *
	 * The method respects the configured upload count and byte budgets. If queued
	 * uploads remain after processing, `onUploadPending` is called so the renderer
	 * can schedule another frame.
	 */
	public beginFrame(): void {
		this.processPendingUploads();
	}

	public getBaseColorTexture(texture: Texture | null): ResolvedWebGLTexture {
		return this._resolveTexture(texture, "base-color", true);
	}

	public getEnvironmentTexture(texture: Texture | null): ResolvedWebGLTexture {
		return this._resolveTexture(texture, "environment", true);
	}

	public getEnvironmentSpecularTexture(
		texture: Texture | null
	): ResolvedWebGLTexture {
		return this._resolveTexture(texture, "env-specular", true, {
			preferFloat: true,
		});
	}

	public getBRDFLUTTexture(texture: Texture | null): ResolvedWebGLTexture {
		return this._resolveTexture(texture, "brdf-lut", false);
	}

	public getNeutralNormalTexture(): ResolvedWebGLTexture {
		if (!this._neutralNormalTexture) {
			const tex = this._createTexture();
			if (!tex) {
				return this.getWhiteTexture();
			}
			this._neutralNormalTexture = tex;
			this._owned.add(tex);
			const gl = this._gl;
			gl.bindTexture(gl.TEXTURE_2D, tex);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
			gl.texImage2D(
				gl.TEXTURE_2D,
				0,
				gl.RGBA,
				1,
				1,
				0,
				gl.RGBA,
				gl.UNSIGNED_BYTE,
				new Uint8Array([128, 128, 255, 255])
			);
			gl.bindTexture(gl.TEXTURE_2D, null);
		}
		return { texture: this._neutralNormalTexture, isLinear: true };
	}

	public getWhiteTexture(): ResolvedWebGLTexture {
		if (!this._whiteTexture) {
			const tex = this._createTexture();
			if (!tex) {
				throw new Error("Failed to create fallback WebGL white texture");
			}
			this._whiteTexture = tex;
			this._owned.add(tex);
			const gl = this._gl;
			gl.bindTexture(gl.TEXTURE_2D, tex);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
			gl.texImage2D(
				gl.TEXTURE_2D,
				0,
				gl.RGBA,
				1,
				1,
				0,
				gl.RGBA,
				gl.UNSIGNED_BYTE,
				new Uint8Array([255, 255, 255, 255])
			);
			gl.bindTexture(gl.TEXTURE_2D, null);
		}
		return { texture: this._whiteTexture, isLinear: true };
	}

	public destroy(): void {
		for (const texture of this._owned) {
			this._gl.deleteTexture(texture);
		}
		this._owned.clear();
		this._pendingUploadQueue = [];
		this._pendingUploadsByTexture = new WeakMap();
		this._whiteTexture = null;
		this._neutralNormalTexture = null;
	}

	private _resolveTexture(
		texture: Texture | null,
		label: string,
		srgbDefault: boolean,
		options: WebGLTextureResolveOptions = {}
	): ResolvedWebGLTexture {
		if (!texture) {
			return this.getWhiteTexture();
		}
		const width = texture.width | 0;
		const height = texture.height | 0;
		if (
			!Number.isFinite(width) ||
			!Number.isFinite(height) ||
			width <= 0 ||
			height <= 0
		) {
			const key = `webgl-texture-invalid-size-${label}`;
			Logger.warn(
				`[${key}] WebGL ${label} texture has invalid dimensions (${texture.width}x${texture.height}); using fallback`,
				{ scope: "WebGLTextureRegistry", onceKey: key }
			);
			return this.getWhiteTexture();
		}
		if (width > this._maxTextureSize || height > this._maxTextureSize) {
			const key = `webgl-texture-oversize-${label}`;
			Logger.warn(
				`[${key}] WebGL ${label} texture exceeds max texture size ${this._maxTextureSize}; using fallback`,
				{ scope: "WebGLTextureRegistry", onceKey: key }
			);
			return this.getWhiteTexture();
		}

		const isLinear =
			texture.colorSpace === "Linear" || texture.colorSpace === "HDR";
		const uploadFormat = this._resolveUploadFormat(
			texture,
			label,
			isLinear,
			options
		);
		const cacheKey = uploadFormat.kind;
		const cachedEntries = this._cache.get(texture);
		const cached = cachedEntries?.get(cacheKey);
		if (
			cached &&
			cached.version === texture.version &&
			cached.width === width &&
			cached.height === height &&
			cached.isLinear === isLinear &&
			cached.uploadKind === uploadFormat.kind &&
			cached.actualFormat === uploadFormat.textureFormat
		) {
			return {
				texture: cached.texture,
				isLinear: uploadFormat.hardwareSRGB || isLinear || !srgbDefault,
			};
		}

		const pending = this._getPendingUpload(texture, cacheKey);
		let glTexture = cached?.texture ?? pending?.targetTexture ?? null;
		if (!glTexture) {
			glTexture = this._createTexture();
			if (!glTexture) {
				const key = `webgl-texture-allocation-${label}`;
				Logger.warn(
					`[${key}] Failed to allocate WebGL texture for ${label}; using fallback`,
					{ scope: "WebGLTextureRegistry", onceKey: key }
				);
				return this.getWhiteTexture();
			}
			this._owned.add(glTexture);
		}

		if (this._shouldDeferUpload(label)) {
			return this._queueDeferredTextureUpload({
				texture,
				label,
				srgbDefault,
				targetTexture: glTexture,
				uploadFormat,
				cacheKey,
				cached,
				width,
				height,
				isLinear,
			});
		}

		const uploadOk = this._uploadTexture(
			glTexture,
			texture,
			label,
			uploadFormat
		);
		if (!uploadOk) {
			return this.getWhiteTexture();
		}
		const entries = cachedEntries ?? new Map<string, TextureEntry>();
		entries.set(cacheKey, {
			texture: glTexture,
			version: texture.version,
			width,
			height,
			isLinear,
			uploadKind: uploadFormat.kind,
			actualFormat: uploadFormat.textureFormat,
		});
		if (!cachedEntries) {
			this._cache.set(texture, entries);
		}
		return {
			texture: glTexture,
			isLinear: uploadFormat.hardwareSRGB || isLinear || !srgbDefault,
		};
	}

	/**
	 * Uploads queued WebGL textures within the configured frame budgets.
	 *
	 * Stale pending requests are discarded when the source texture version or
	 * dimensions changed before upload. Successful uploads update the registry
	 * cache, and remaining queued uploads trigger `onUploadPending`.
	 */
	public processPendingUploads(): void {
		if (this._pendingUploadQueue.length === 0) {
			return;
		}

		let uploads = 0;
		let uploadedBytes = 0;
		while (this._pendingUploadQueue.length > 0) {
			const pending = this._pendingUploadQueue[0];
			const wouldExceedUploadCount = uploads >= this._maxUploadsPerFrame;
			const wouldExceedByteBudget =
				uploadedBytes + pending.estimatedBytes >
				this._maxUploadBytesPerFrame;
			if (
				uploads > 0 &&
				(wouldExceedUploadCount || wouldExceedByteBudget)
			) {
				break;
			}

			this._pendingUploadQueue.shift();
			pending.queued = false;
			const entries = this._pendingUploadsByTexture.get(pending.texture);
			if (entries?.get(pending.cacheKey) !== pending) {
				continue;
			}
			if (
				pending.version !== pending.texture.version ||
				pending.width !== (pending.texture.width | 0) ||
				pending.height !== (pending.texture.height | 0)
			) {
				entries.delete(pending.cacheKey);
				continue;
			}

			const uploadOk = this._uploadTexture(
				pending.targetTexture,
				pending.texture,
				pending.label,
				pending.uploadFormat
			);
			entries.delete(pending.cacheKey);
			if (uploadOk) {
				this._commitTextureEntry(pending);
				uploads++;
				uploadedBytes += pending.estimatedBytes;
			}
		}

		if (this._pendingUploadQueue.length > 0) {
			this._notifyUploadPending();
		}
	}

	private _queueDeferredTextureUpload(
		request: DeferredTextureUploadRequest
	): ResolvedWebGLTexture {
		let pendingEntries = this._pendingUploadsByTexture.get(request.texture);
		if (!pendingEntries) {
			pendingEntries = new Map();
			this._pendingUploadsByTexture.set(request.texture, pendingEntries);
		}

		let pending = pendingEntries.get(request.cacheKey);
		const estimatedBytes = estimateTextureUploadBytes(
			request.texture,
			request.uploadFormat
		);
		if (!pending) {
			pending = {
				texture: request.texture,
				label: request.label,
				targetTexture: request.targetTexture,
				uploadFormat: request.uploadFormat,
				cacheKey: request.cacheKey,
				version: request.texture.version,
				width: request.width,
				height: request.height,
				isLinear: request.isLinear,
				estimatedBytes,
				queued: false,
			};
			pendingEntries.set(request.cacheKey, pending);
		} else {
			pending.label = request.label;
			pending.targetTexture = request.targetTexture;
			pending.uploadFormat = request.uploadFormat;
			pending.version = request.texture.version;
			pending.width = request.width;
			pending.height = request.height;
			pending.isLinear = request.isLinear;
			pending.estimatedBytes = estimatedBytes;
		}

		if (!pending.queued) {
			pending.queued = true;
			this._pendingUploadQueue.push(pending);
		}
		this._notifyUploadPending();

		const visibleTexture = request.cached?.texture;
		if (visibleTexture) {
			return {
				texture: visibleTexture,
				isLinear:
					request.uploadFormat.hardwareSRGB ||
					request.isLinear ||
					!request.srgbDefault,
			};
		}
		return this.getWhiteTexture();
	}

	private _commitTextureEntry(pending: PendingTextureUpload): void {
		const entries = this._cache.get(pending.texture) ?? new Map();
		entries.set(pending.cacheKey, {
			texture: pending.targetTexture,
			version: pending.version,
			width: pending.width,
			height: pending.height,
			isLinear: pending.isLinear,
			uploadKind: pending.uploadFormat.kind,
			actualFormat: pending.uploadFormat.textureFormat,
		});
		if (!this._cache.get(pending.texture)) {
			this._cache.set(pending.texture, entries);
		}
	}

	private _shouldDeferUpload(label: string): boolean {
		return this._uploadScheduling === "deferred" && label === "base-color";
	}

	private _notifyUploadPending(): void {
		this._onUploadPending?.();
	}

	private _getPendingUpload(
		texture: Texture,
		cacheKey: string
	): PendingTextureUpload | undefined {
		return this._pendingUploadsByTexture.get(texture)?.get(cacheKey);
	}

	private _uploadTexture(
		targetTexture: WebGLTexture,
		texture: Texture,
		label: string,
		uploadFormat: WebGLTextureUploadFormat
	): boolean {
		const gl = this._gl;
		const mipCount = Math.max(1, texture.mipmaps.length || 1);
		const shouldGenerateMipmaps =
			!uploadFormat.isFloat &&
			mipCount <= 1 &&
			requiresMipmaps(texture.minFilter);
		const hasMipmaps = mipCount > 1 || shouldGenerateMipmaps;
		const maxMipLevel =
			shouldGenerateMipmaps ?
				resolveMaxMipmapLevel(texture.width, texture.height)
			:	mipCount - 1;
		const supportsLinearFiltering =
			!uploadFormat.isFloat || this._canLinearlyFilterFloatTextures();

		gl.bindTexture(gl.TEXTURE_2D, targetTexture);
		gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
		gl.texParameteri(
			gl.TEXTURE_2D,
			gl.TEXTURE_WRAP_S,
			mapWrapMode(gl, texture.wrapS)
		);
		gl.texParameteri(
			gl.TEXTURE_2D,
			gl.TEXTURE_WRAP_T,
			mapWrapMode(gl, texture.wrapT)
		);
		gl.texParameteri(
			gl.TEXTURE_2D,
			gl.TEXTURE_MAG_FILTER,
			mapMagFilter(gl, texture.magFilter, supportsLinearFiltering)
		);
		gl.texParameteri(
			gl.TEXTURE_2D,
			gl.TEXTURE_MIN_FILTER,
			mapMinFilter(
				gl,
				texture.minFilter,
				hasMipmaps,
				supportsLinearFiltering
			)
		);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, maxMipLevel);
		for (let level = 0; level < mipCount; level++) {
			const width = Math.max(1, texture.width >> level);
			const height = Math.max(1, texture.height >> level);
			const source =
				texture.mipmaps[level] ??
				(level === 0 ? texture.data : null) ??
				texture.mipmaps[0] ??
				null;

			if (!source) {
				const key = `webgl-texture-empty-${label}`;
				Logger.warn(
					`[${key}] Texture ${label} has empty pixel data; using fallback`,
					{ scope: "WebGLTextureRegistry", onceKey: key }
				);
				gl.bindTexture(gl.TEXTURE_2D, null);
				return false;
			}

			const data = createUploadData(source, width, height, uploadFormat);
			gl.texImage2D(
				gl.TEXTURE_2D,
				level,
				uploadFormat.internalFormat,
				width,
				height,
				0,
				uploadFormat.format,
				uploadFormat.type,
				data
			);
		}

		if (shouldGenerateMipmaps) {
			gl.generateMipmap(gl.TEXTURE_2D);
		}

		gl.bindTexture(gl.TEXTURE_2D, null);
		return true;
	}

	private _resolveUploadFormat(
		texture: Texture,
		label: string,
		isLinear: boolean,
		options: WebGLTextureResolveOptions
	): WebGLTextureUploadFormat {
		const gl = this._gl;
		const requestedFormat = texture.format ?? TextureFormat.RGBA8Unorm;
		if (texture.formatExplicit) {
			const requestedUpload = this._resolveRequestedUploadFormat(
				requestedFormat,
				label
			);
			if (requestedUpload) {
				return requestedUpload;
			}
		}
		if (options.preferFloat && isLinear && hasFloat32PixelData(texture)) {
			const floatFormat = this._resolveFloatUploadFormat(label);
			if (floatFormat) {
				return floatFormat;
			}
		}
		return {
			kind: "rgba8",
			textureFormat: TextureFormat.RGBA8Unorm,
			internalFormat: gl.RGBA,
			format: gl.RGBA,
			type: gl.UNSIGNED_BYTE,
			isFloat: false,
			hardwareSRGB: false,
			channelCount: 4,
		};
	}

	private _resolveRequestedUploadFormat(
		requestedFormat: TextureFormat,
		label: string
	): WebGLTextureUploadFormat | null {
		const gl = this._gl as WebGL2RenderingContext & Record<string, number>;
		switch (requestedFormat) {
			case TextureFormat.R8Unorm:
				return this._createSizedWebGLUploadFormat(
					requestedFormat,
					"r8",
					gl.R8,
					gl.RED,
					gl.UNSIGNED_BYTE,
					1,
					label
				);
			case TextureFormat.RG8Unorm:
				return this._createSizedWebGLUploadFormat(
					requestedFormat,
					"rg8",
					gl.RG8,
					gl.RG,
					gl.UNSIGNED_BYTE,
					2,
					label
				);
			case TextureFormat.RGBA8Unorm:
				return {
					kind: "rgba8",
					textureFormat: TextureFormat.RGBA8Unorm,
					internalFormat: gl.RGBA,
					format: gl.RGBA,
					type: gl.UNSIGNED_BYTE,
					isFloat: false,
					hardwareSRGB: false,
					channelCount: 4,
				};
			case TextureFormat.RGBA8UnormSrgb:
				if (typeof gl.SRGB8_ALPHA8 === "number") {
					return {
						kind: "rgba8-srgb",
						textureFormat: TextureFormat.RGBA8UnormSrgb,
						internalFormat: gl.SRGB8_ALPHA8,
						format: gl.RGBA,
						type: gl.UNSIGNED_BYTE,
						isFloat: false,
						hardwareSRGB: true,
						channelCount: 4,
					};
				}
				this._warnFormatFallback(
					label,
					requestedFormat,
					TextureFormat.RGBA8Unorm
				);
				return null;
			case TextureFormat.R16Float:
				return this._createFloatWebGLUploadFormat(
					requestedFormat,
					"r16f",
					gl.R16F,
					gl.RED,
					1,
					label
				);
			case TextureFormat.RG16Float:
				return this._createFloatWebGLUploadFormat(
					requestedFormat,
					"rg16f",
					gl.RG16F,
					gl.RG,
					2,
					label
				);
			case TextureFormat.RGBA16Float:
				return this._resolveFloatUploadFormat(label);
			case TextureFormat.R32Float:
				return this._createFloat32WebGLUploadFormat(
					requestedFormat,
					"r32f",
					gl.R32F,
					gl.RED,
					1,
					label
				);
			case TextureFormat.RG32Float:
				return this._createFloat32WebGLUploadFormat(
					requestedFormat,
					"rg32f",
					gl.RG32F,
					gl.RG,
					2,
					label
				);
			case TextureFormat.RGBA32Float:
				return this._createFloat32WebGLUploadFormat(
					requestedFormat,
					"rgba32f",
					gl.RGBA32F,
					gl.RGBA,
					4,
					label
				);
			default:
				this._warnFormatFallback(
					label,
					requestedFormat,
					TextureFormat.RGBA8Unorm
				);
				return null;
		}
	}

	private _resolveFloatUploadFormat(
		label: string
	): WebGLTextureUploadFormat | null {
		const gl = this._gl as WebGL2RenderingContext & {
			RGBA16F?: number;
			RGBA32F?: number;
			HALF_FLOAT?: number;
			FLOAT?: number;
		};
		if (
			typeof gl.RGBA16F === "number" &&
			typeof gl.HALF_FLOAT === "number"
		) {
			return {
				kind: "rgba16f",
				textureFormat: TextureFormat.RGBA16Float,
				internalFormat: gl.RGBA16F,
				format: gl.RGBA,
				type: gl.HALF_FLOAT,
				isFloat: true,
				hardwareSRGB: false,
				channelCount: 4,
			};
		}
		if (
			typeof gl.RGBA32F === "number" &&
			typeof gl.FLOAT === "number"
		) {
			return {
				kind: "rgba32f",
				textureFormat: TextureFormat.RGBA32Float,
				internalFormat: gl.RGBA32F,
				format: gl.RGBA,
				type: gl.FLOAT,
				isFloat: true,
				hardwareSRGB: false,
				channelCount: 4,
			};
		}

		const key = `webgl-texture-float-unsupported-${label}`;
		Logger.warn(
			`[${key}] WebGL ${label} texture has Float32 pixel data, but neither RGBA16F/HALF_FLOAT nor RGBA32F/FLOAT upload is available; falling back to RGBA8`,
			{ scope: "WebGLTextureRegistry", onceKey: key }
		);
		return null;
	}

	private _createSizedWebGLUploadFormat(
		textureFormat: TextureFormat,
		kind: string,
		internalFormat: number | undefined,
		format: number | undefined,
		type: number,
		channelCount: number,
		label: string
	): WebGLTextureUploadFormat | null {
		if (typeof internalFormat !== "number" || typeof format !== "number") {
			this._warnFormatFallback(label, textureFormat, TextureFormat.RGBA8Unorm);
			return null;
		}
		return {
			kind,
			textureFormat,
			internalFormat,
			format,
			type,
			isFloat: false,
			hardwareSRGB: false,
			channelCount,
		};
	}

	private _createFloatWebGLUploadFormat(
		textureFormat: TextureFormat,
		kind: string,
		internalFormat: number | undefined,
		format: number | undefined,
		channelCount: number,
		label: string
	): WebGLTextureUploadFormat | null {
		const gl = this._gl as WebGL2RenderingContext & { HALF_FLOAT?: number };
		if (
			typeof internalFormat !== "number" ||
			typeof format !== "number" ||
			typeof gl.HALF_FLOAT !== "number"
		) {
			this._warnFormatFallback(label, textureFormat, TextureFormat.RGBA16Float);
			return this._resolveFloatUploadFormat(label);
		}
		return {
			kind,
			textureFormat,
			internalFormat,
			format,
			type: gl.HALF_FLOAT,
			isFloat: true,
			hardwareSRGB: false,
			channelCount,
		};
	}

	private _createFloat32WebGLUploadFormat(
		textureFormat: TextureFormat,
		kind: string,
		internalFormat: number | undefined,
		format: number | undefined,
		channelCount: number,
		label: string
	): WebGLTextureUploadFormat | null {
		const gl = this._gl as WebGL2RenderingContext & { FLOAT?: number };
		if (
			typeof internalFormat !== "number" ||
			typeof format !== "number" ||
			typeof gl.FLOAT !== "number"
		) {
			this._warnFormatFallback(label, textureFormat, TextureFormat.RGBA16Float);
			return this._resolveFloatUploadFormat(label);
		}
		return {
			kind,
			textureFormat,
			internalFormat,
			format,
			type: gl.FLOAT,
			isFloat: true,
			hardwareSRGB: false,
			channelCount,
		};
	}

	private _warnFormatFallback(
		label: string,
		requestedFormat: TextureFormat,
		actualFormat: TextureFormat
	): void {
		const key = `webgl-texture-format-fallback-${label}-${requestedFormat}-${actualFormat}`;
		Logger.warn(
			`[${key}] WebGL ${label} texture format "${requestedFormat}" is unavailable; using "${actualFormat}"`,
			{ scope: "WebGLTextureRegistry", onceKey: key }
		);
	}

	private _canLinearlyFilterFloatTextures(): boolean {
		if (this._supportsFloatLinearFiltering !== null) {
			return this._supportsFloatLinearFiltering;
		}
		const gl = this._gl as WebGL2RenderingContext & {
			getExtension?: (name: string) => unknown;
		};
		this._supportsFloatLinearFiltering =
			typeof gl.getExtension === "function" &&
			!!(
				gl.getExtension("OES_texture_float_linear") ||
				gl.getExtension("OES_texture_half_float_linear")
			);
		return this._supportsFloatLinearFiltering;
	}

	private _createTexture(): WebGLTexture | null {
		try {
			return this._gl.createTexture();
		} catch {
			return null;
		}
	}

	private _resolveMaxTextureSize(gl: WebGL2RenderingContext): number {
		try {
			const value = gl.getParameter(gl.MAX_TEXTURE_SIZE);
			if (typeof value === "number" && Number.isFinite(value) && value > 0) {
				return value | 0;
			}
		} catch {}
		return 4096;
	}
}

function mapWrapMode(gl: WebGL2RenderingContext, value?: string): number {
	switch (value) {
		case "Clamp":
			return gl.CLAMP_TO_EDGE;
		case "MirroredRepeat":
			return gl.MIRRORED_REPEAT;
		default:
			return gl.REPEAT;
	}
}

function mapMagFilter(
	gl: WebGL2RenderingContext,
	value?: string,
	linearFilteringAllowed = true
): number {
	if (!linearFilteringAllowed) {
		return gl.NEAREST;
	}
	return value === "Nearest" ? gl.NEAREST : gl.LINEAR;
}

function mapMinFilter(
	gl: WebGL2RenderingContext,
	value: string | undefined,
	hasMipmaps: boolean,
	linearFilteringAllowed = true
): number {
	if (!linearFilteringAllowed) {
		return hasMipmaps ? gl.NEAREST_MIPMAP_NEAREST : gl.NEAREST;
	}
	switch (value) {
		case "Nearest":
			return gl.NEAREST;
		case "NearestMipmapNearest":
			return hasMipmaps ? gl.NEAREST_MIPMAP_NEAREST : gl.NEAREST;
		case "NearestMipmapLinear":
			return hasMipmaps ? gl.NEAREST_MIPMAP_LINEAR : gl.NEAREST;
		case "LinearMipmapNearest":
			return hasMipmaps ? gl.LINEAR_MIPMAP_NEAREST : gl.LINEAR;
		case "LinearMipmapLinear":
			return hasMipmaps ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR;
		case "Linear":
		default:
			return gl.LINEAR;
	}
}

function requiresMipmaps(value: string | undefined): boolean {
	switch (value) {
		case "NearestMipmapNearest":
		case "NearestMipmapLinear":
		case "LinearMipmapNearest":
		case "LinearMipmapLinear":
			return true;
		default:
			return false;
	}
}

function resolveMaxMipmapLevel(width: number, height: number): number {
	const maxDimension = Math.max(1, width | 0, height | 0);
	return Math.max(0, Math.floor(Math.log2(maxDimension)));
}

function estimateTextureUploadBytes(
	texture: Texture,
	format: WebGLTextureUploadFormat
): number {
	const mipCount = Math.max(1, texture.mipmaps.length || 1);
	const bytesPerChannel = resolveUploadBytesPerChannel(format);
	let totalBytes = 0;
	for (let level = 0; level < mipCount; level++) {
		const width = Math.max(1, texture.width >> level);
		const height = Math.max(1, texture.height >> level);
		totalBytes += width * height * format.channelCount * bytesPerChannel;
	}
	if (!format.isFloat && mipCount <= 1 && requiresMipmaps(texture.minFilter)) {
		totalBytes += Math.ceil(totalBytes / 3);
	}
	return Math.max(1, totalBytes);
}

function resolveUploadBytesPerChannel(
	format: WebGLTextureUploadFormat
): number {
	if (!format.isFloat) {
		return 1;
	}
	return format.kind.endsWith("32f") ? 4 : 2;
}

function hasFloat32PixelData(texture: Texture): boolean {
	if (texture.data instanceof Float32Array) {
		return true;
	}
	for (const mip of texture.mipmaps) {
		if (mip instanceof Float32Array) {
			return true;
		}
	}
	return false;
}

function createUploadData(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	width: number,
	height: number,
	format: WebGLTextureUploadFormat
): Uint8Array | Uint16Array | Float32Array {
	if (format.isFloat) {
		return format.kind.endsWith("32f") ?
				toFloat32Data(source, width, height, format.channelCount)
			:	toFloat16Data(source, width, height, format.channelCount);
	}
	return toUint8Data(source, width, height, format.channelCount);
}

function toUint8Data(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	width: number,
	height: number,
	channelCount: number
): Uint8Array {
	const expectedLength = Math.max(1, width * height * channelCount);
	if (source instanceof Uint8Array && !(source instanceof Uint8ClampedArray)) {
		if (source.length === expectedLength) {
			return source;
		}
		if (source.length < expectedLength) {
			const resized = new Uint8Array(expectedLength);
			resized.set(source.subarray(0, expectedLength));
			return resized;
		}
	}

	const data = new Uint8Array(expectedLength);
	const pixelCount = Math.max(1, width * height);
	const sourceChannels = inferUploadSourceChannels(source, pixelCount, channelCount);
	for (let pixel = 0; pixel < pixelCount; pixel++) {
		for (let channel = 0; channel < channelCount; channel++) {
			const srcIndex = pixel * sourceChannels + Math.min(channel, sourceChannels - 1);
			data[pixel * channelCount + channel] = toUint8UploadValue(
				source,
				source[srcIndex] ?? defaultUploadChannelValue(channel)
			);
		}
	}
	return data;
}

function toFloat32Data(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	width: number,
	height: number,
	channelCount: number
): Float32Array {
	const expectedLength = Math.max(1, width * height * channelCount);
	if (source instanceof Float32Array && source.length === expectedLength) {
		return source;
	}
	const data = new Float32Array(expectedLength);
	const pixelCount = Math.max(1, width * height);
	const sourceChannels = inferUploadSourceChannels(source, pixelCount, channelCount);
	for (let pixel = 0; pixel < pixelCount; pixel++) {
		for (let channel = 0; channel < channelCount; channel++) {
			const srcIndex = pixel * sourceChannels + Math.min(channel, sourceChannels - 1);
			data[pixel * channelCount + channel] = toFloatUploadValue(
				source,
				srcIndex,
				channel
			);
		}
	}
	return data;
}

function toFloat16Data(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	width: number,
	height: number,
	channelCount: number
): Uint16Array {
	const expectedLength = Math.max(1, width * height * channelCount);
	const data = new Uint16Array(expectedLength);
	const pixelCount = Math.max(1, width * height);
	const sourceChannels = inferUploadSourceChannels(source, pixelCount, channelCount);
	for (let pixel = 0; pixel < pixelCount; pixel++) {
		for (let channel = 0; channel < channelCount; channel++) {
			const srcIndex = pixel * sourceChannels + Math.min(channel, sourceChannels - 1);
			data[pixel * channelCount + channel] = float32ToFloat16Bits(
				toFloatUploadValue(source, srcIndex, channel)
			);
		}
	}
	return data;
}

function inferUploadSourceChannels(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	pixelCount: number,
	targetChannels: number
): number {
	if (source.length >= pixelCount * 4) {
		return 4;
	}
	if (source.length >= pixelCount * targetChannels) {
		return targetChannels;
	}
	if (source.length >= pixelCount * 2) {
		return 2;
	}
	return 1;
}

function defaultUploadChannelValue(channel: number): number {
	return channel === 3 ? 1 : 0;
}

function toUint8UploadValue(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	value: number
): number {
	if (source instanceof Float32Array) {
		return clamp(Math.round(value * 255), 0, 255);
	}
	if (Number.isInteger(value) && value >= 0 && value <= 255) {
		return value;
	}
	return clamp(Math.round(value * 255), 0, 255);
}

function toFloatUploadValue(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	index: number,
	channel: number
): number {
	const value = source[index] ?? defaultUploadChannelValue(channel);
	if (source instanceof Float32Array) {
		return value;
	}
	return value / 255;
}

function toRGBA8Data(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	width: number,
	height: number
): Uint8Array {
	return toUint8Data(source, width, height, 4);
}

function toRGBA32FData(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	width: number,
	height: number
): Float32Array {
	return toFloat32Data(source, width, height, 4);
}

function toRGBA16FData(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	width: number,
	height: number
): Uint16Array {
	return toFloat16Data(source, width, height, 4);
}

function legacyToRGBA8Data(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	width: number,
	height: number
): Uint8Array {
	const expectedLength = Math.max(1, width * height * 4);
	if (source instanceof Uint8Array && !(source instanceof Uint8ClampedArray)) {
		if (source.length === expectedLength) {
			return source;
		}
		const resized = new Uint8Array(expectedLength);
		resized.set(source.subarray(0, expectedLength));
		return resized;
	}

	if (source instanceof Uint8ClampedArray) {
		const data = new Uint8Array(expectedLength);
		data.set(source.subarray(0, expectedLength));
		return data;
	}

	const data = new Uint8Array(expectedLength);
	for (let i = 0; i < expectedLength; i++) {
		data[i] = clamp(Math.round((source[i] ?? 0) * 255), 0, 255);
	}
	return data;
}
