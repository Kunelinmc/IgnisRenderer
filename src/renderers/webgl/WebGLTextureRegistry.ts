import type { Texture } from "../../core/Texture";
import { clamp } from "../../maths/Common";
import { Logger } from "../../foundation/Logger";
import { float32ToFloat16Bits } from "../../foundation/Float16";

interface TextureEntry {
	texture: WebGLTexture;
	version: number;
	width: number;
	height: number;
	isLinear: boolean;
	uploadKind: WebGLTextureUploadKind;
}

export interface ResolvedWebGLTexture {
	texture: WebGLTexture;
	isLinear: boolean;
}

type WebGLTextureUploadKind = "rgba8" | "rgba16f" | "rgba32f";

interface WebGLTextureUploadFormat {
	kind: WebGLTextureUploadKind;
	internalFormat: number;
	format: number;
	type: number;
	isFloat: boolean;
}

interface WebGLTextureResolveOptions {
	preferFloat?: boolean;
}

export class WebGLTextureRegistry {
	private _gl: WebGL2RenderingContext;
	private _maxTextureSize: number;
	private _cache = new WeakMap<Texture, Map<string, TextureEntry>>();
	private _owned = new Set<WebGLTexture>();
	private _whiteTexture: WebGLTexture | null = null;
	private _neutralNormalTexture: WebGLTexture | null = null;
	private _supportsFloatLinearFiltering: boolean | null = null;

	constructor(
		gl: WebGL2RenderingContext,
		_warn?: (key: string, message: string) => void
	) {
		this._gl = gl;
		this._maxTextureSize = this._resolveMaxTextureSize(gl);
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
			cached.uploadKind === uploadFormat.kind
		) {
			return {
				texture: cached.texture,
				isLinear: isLinear || !srgbDefault,
			};
		}

		let glTexture = cached?.texture ?? null;
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
		});
		if (!cachedEntries) {
			this._cache.set(texture, entries);
		}
		return {
			texture: glTexture,
			isLinear: isLinear || !srgbDefault,
		};
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
		if (options.preferFloat && isLinear && hasFloat32PixelData(texture)) {
			const floatFormat = this._resolveFloatUploadFormat(label);
			if (floatFormat) {
				return floatFormat;
			}
		}
		return {
			kind: "rgba8",
			internalFormat: gl.RGBA,
			format: gl.RGBA,
			type: gl.UNSIGNED_BYTE,
			isFloat: false,
		};
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
				internalFormat: gl.RGBA16F,
				format: gl.RGBA,
				type: gl.HALF_FLOAT,
				isFloat: true,
			};
		}
		if (
			typeof gl.RGBA32F === "number" &&
			typeof gl.FLOAT === "number"
		) {
			return {
				kind: "rgba32f",
				internalFormat: gl.RGBA32F,
				format: gl.RGBA,
				type: gl.FLOAT,
				isFloat: true,
			};
		}

		const key = `webgl-texture-float-unsupported-${label}`;
		Logger.warn(
			`[${key}] WebGL ${label} texture has Float32 pixel data, but neither RGBA16F/HALF_FLOAT nor RGBA32F/FLOAT upload is available; falling back to RGBA8`,
			{ scope: "WebGLTextureRegistry", onceKey: key }
		);
		return null;
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
	switch (format.kind) {
		case "rgba16f":
			return toRGBA16FData(source, width, height);
		case "rgba32f":
			return toRGBA32FData(source, width, height);
		case "rgba8":
		default:
			return toRGBA8Data(source, width, height);
	}
}

function toRGBA8Data(
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

function toRGBA32FData(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	width: number,
	height: number
): Float32Array {
	const expectedLength = Math.max(1, width * height * 4);
	if (source instanceof Float32Array) {
		if (source.length === expectedLength) {
			return source;
		}
		const resized = new Float32Array(expectedLength);
		resized.set(source.subarray(0, expectedLength));
		return resized;
	}

	const data = new Float32Array(expectedLength);
	for (let i = 0; i < expectedLength; i++) {
		data[i] = (source[i] ?? 0) / 255;
	}
	return data;
}

function toRGBA16FData(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	width: number,
	height: number
): Uint16Array {
	const expectedLength = Math.max(1, width * height * 4);
	const data = new Uint16Array(expectedLength);
	if (source instanceof Float32Array) {
		const limit = Math.min(source.length, expectedLength);
		for (let i = 0; i < limit; i++) {
			data[i] = float32ToFloat16Bits(source[i] ?? 0);
		}
		return data;
	}

	for (let i = 0; i < expectedLength; i++) {
		data[i] = float32ToFloat16Bits((source[i] ?? 0) / 255);
	}
	return data;
}
