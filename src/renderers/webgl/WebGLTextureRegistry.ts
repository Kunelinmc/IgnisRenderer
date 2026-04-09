import type { Texture } from "../../core/Texture";
import { clamp } from "../../maths/Common";

interface TextureEntry {
	texture: WebGLTexture;
	version: number;
	width: number;
	height: number;
	isLinear: boolean;
}

export interface ResolvedWebGLTexture {
	texture: WebGLTexture;
	isLinear: boolean;
}

type WarnFn = (key: string, message: string) => void;

export class WebGLTextureRegistry {
	private _gl: WebGL2RenderingContext;
	private _logWarning: WarnFn;
	private _maxTextureSize: number;
	private _cache = new WeakMap<Texture, TextureEntry>();
	private _owned = new Set<WebGLTexture>();
	private _whiteTexture: WebGLTexture | null = null;
	private _neutralNormalTexture: WebGLTexture | null = null;

	constructor(gl: WebGL2RenderingContext, warn: WarnFn) {
		this._gl = gl;
		this._logWarning = warn;
		this._maxTextureSize = this._resolveMaxTextureSize(gl);
	}

	public getBaseColorTexture(texture: Texture | null): ResolvedWebGLTexture {
		return this._resolveTexture(texture, "base-color", true);
	}

	public getSkyboxTexture(texture: Texture | null): ResolvedWebGLTexture {
		return this._resolveTexture(texture, "skybox", true);
	}

	public getEnvironmentSpecularTexture(
		texture: Texture | null
	): ResolvedWebGLTexture {
		return this._resolveTexture(texture, "env-specular", true);
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
		srgbDefault: boolean
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
			this._logWarning(
				`webgl-texture-invalid-size-${label}`,
				`WebGL ${label} texture has invalid dimensions (${texture.width}x${texture.height}); using fallback`
			);
			return this.getWhiteTexture();
		}
		if (width > this._maxTextureSize || height > this._maxTextureSize) {
			this._logWarning(
				`webgl-texture-oversize-${label}`,
				`WebGL ${label} texture exceeds max texture size ${this._maxTextureSize}; using fallback`
			);
			return this.getWhiteTexture();
		}

		const isLinear =
			texture.colorSpace === "Linear" || texture.colorSpace === "HDR";
		const cached = this._cache.get(texture);
		if (
			cached &&
			cached.version === texture.version &&
			cached.width === width &&
			cached.height === height &&
			cached.isLinear === isLinear
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
				this._logWarning(
					`webgl-texture-allocation-${label}`,
					`Failed to allocate WebGL texture for ${label}; using fallback`
				);
				return this.getWhiteTexture();
			}
			this._owned.add(glTexture);
		}

		const uploadOk = this._uploadTexture(glTexture, texture, label);
		if (!uploadOk) {
			return this.getWhiteTexture();
		}
		this._cache.set(texture, {
			texture: glTexture,
			version: texture.version,
			width,
			height,
			isLinear,
		});
		return {
			texture: glTexture,
			isLinear: isLinear || !srgbDefault,
		};
	}

	private _uploadTexture(
		targetTexture: WebGLTexture,
		texture: Texture,
		label: string
	): boolean {
		const gl = this._gl;
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
			mapMagFilter(gl, texture.magFilter)
		);
		gl.texParameteri(
			gl.TEXTURE_2D,
			gl.TEXTURE_MIN_FILTER,
			mapMinFilter(gl, texture.minFilter, texture.mipmaps.length > 1)
		);

		const mipCount = Math.max(1, texture.mipmaps.length || 1);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, mipCount - 1);
		for (let level = 0; level < mipCount; level++) {
			const width = Math.max(1, texture.width >> level);
			const height = Math.max(1, texture.height >> level);
			const source =
				texture.mipmaps[level] ??
				(level === 0 ? texture.data : null) ??
				texture.mipmaps[0] ??
				null;

			if (!source) {
				this._logWarning(
					`webgl-texture-empty-${label}`,
					`Texture ${label} has empty pixel data; using fallback`
				);
				gl.bindTexture(gl.TEXTURE_2D, null);
				return false;
			}

			const data = toRGBA8Data(source, width, height);
			gl.texImage2D(
				gl.TEXTURE_2D,
				level,
				gl.RGBA,
				width,
				height,
				0,
				gl.RGBA,
				gl.UNSIGNED_BYTE,
				data
			);
		}

		if (
			texture.mipmaps.length <= 1 &&
			(texture.minFilter === "NearestMipmapNearest" ||
				texture.minFilter === "Linear")
		) {
			gl.generateMipmap(gl.TEXTURE_2D);
		}

		gl.bindTexture(gl.TEXTURE_2D, null);
		return true;
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

function mapMagFilter(gl: WebGL2RenderingContext, value?: string): number {
	return value === "Nearest" ? gl.NEAREST : gl.LINEAR;
}

function mapMinFilter(
	gl: WebGL2RenderingContext,
	value: string | undefined,
	hasMipmaps: boolean
): number {
	if (value === "Nearest") {
		return hasMipmaps ? gl.NEAREST_MIPMAP_NEAREST : gl.NEAREST;
	}
	if (value === "NearestMipmapNearest") {
		return hasMipmaps ? gl.NEAREST_MIPMAP_NEAREST : gl.NEAREST;
	}
	return hasMipmaps ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR;
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
