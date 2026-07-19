import type { IVector2 } from "../maths/types";
import type { RGBA } from "../foundation/Color";
import { clamp } from "../maths/Common";
import { TextureFormat } from "../backends/types";

export type TextureFilter =
	| "Nearest"
	| "Linear"
	| "NearestMipmapNearest"
	| "NearestMipmapLinear"
	| "LinearMipmapNearest"
	| "LinearMipmapLinear";

export type TextureWrap = "Repeat" | "Clamp" | "MirroredRepeat";

/**
 * Describes the color space of texture data.
 * - `"sRGB"`: Standard sRGB-encoded data (typical for 8-bit images loaded via canvas/browser).
 * - `"Linear"`: Linear color space data (e.g. normal maps, metallic-roughness maps that store non-color data).
 * - `"HDR"`: High dynamic range linear data (e.g. .hdr environment maps with Float32Array values in [0, ∞)).
 */
export type TextureColorSpace = "sRGB" | "Linear" | "HDR";

export type TextureData =
	| Uint8Array
	| Uint8ClampedArray
	| Float32Array;

export interface TextureMipLevel {
	data: TextureData | null;
	width: number;
	height: number;
	depthOrArrayLayers?: number;
	bytesPerRow?: number;
	rowsPerImage?: number;
}

export interface TextureDescriptor {
	data?: TextureData | null;
	width?: number;
	height?: number;
	format?: TextureFormat;
	colorSpace?: TextureColorSpace;
	levels?: TextureMipLevel[];
	label?: string;
	usageHint?: "color" | "data" | "normal" | "depth" | "compressed";
}

/**
 * Texture class to store image data and metadata for UV mapping.
 */
export class Texture {
	private static _dynamicTextures = new Set<Texture>();

	data: TextureData | null;
	width: number;
	height: number;
	format: TextureFormat;
	formatExplicit: boolean;
	wrapS: TextureWrap;
	wrapT: TextureWrap;
	minFilter: TextureFilter;
	magFilter: TextureFilter;
	offset: IVector2;
	repeat: IVector2;
	rotation: number;
	/**
	 * The color space of this texture's data.
	 * Used by samplers and lighting to decide whether gamma decode is needed.
	 */
	colorSpace: TextureColorSpace;

	/**
	 * Mipmaps for the texture, used for pre-filtered environment maps (roughness levels).
	 * mipmaps[0] is the base texture (same as this.data).
	 */
	mipmaps: TextureData[];
	levels: TextureMipLevel[];
	version: number;
	label?: string;
	usageHint?: TextureDescriptor["usageHint"];
	private _isDynamicTexture: boolean;
	private _isLoadErrorFallback: boolean;

	constructor(descriptor?: TextureDescriptor);
	constructor(
		data?: TextureData | null,
		width?: number,
		height?: number,
		colorSpace?: TextureColorSpace
	);
	constructor(
		dataOrDescriptor: TextureData | TextureDescriptor | null = null,
		width: number = 0,
		height: number = 0,
		colorSpace: TextureColorSpace = "sRGB"
	) {
		const descriptor =
			isTextureDescriptor(dataOrDescriptor) ?
				dataOrDescriptor
			:	{
					data: dataOrDescriptor,
					width,
					height,
					colorSpace,
				};
		const resolvedColorSpace = descriptor.colorSpace ?? "sRGB";
		const resolvedData = descriptor.data ?? descriptor.levels?.[0]?.data ?? null;
		const resolvedWidth = Math.max(
			0,
			Math.floor(descriptor.width ?? descriptor.levels?.[0]?.width ?? width ?? 0)
		);
		const resolvedHeight = Math.max(
			0,
			Math.floor(descriptor.height ?? descriptor.levels?.[0]?.height ?? height ?? 0)
		);

		this.data = resolvedData;
		this.width = resolvedWidth;
		this.height = resolvedHeight;
		this.format =
			descriptor.format ??
			inferDefaultTextureFormat(resolvedData, resolvedColorSpace);
		this.formatExplicit = !!descriptor.format;
		this.wrapS = "Repeat";
		this.wrapT = "Repeat";
		this.minFilter = "Linear";
		this.magFilter = "Linear";
		this.offset = { x: 0, y: 0 };
		this.repeat = { x: 1, y: 1 };
		this.rotation = 0;
		this.colorSpace = resolvedColorSpace;
		this.levels = normalizeTextureLevels(
			descriptor.levels,
			resolvedData,
			resolvedWidth,
			resolvedHeight
		);
		this.mipmaps = this.levels
			.map((level) => level.data)
			.filter((levelData): levelData is TextureData => !!levelData);
		this.version = 0;
		this.label = descriptor.label;
		this.usageHint = descriptor.usageHint;
		this._isDynamicTexture = false;
		this._isLoadErrorFallback = false;
	}

	public clone(): Texture {
		const cloned = new Texture({
			data: this.data,
			width: this.width,
			height: this.height,
			format: this.format,
			colorSpace: this.colorSpace,
			levels: this.levels.map((level) => ({ ...level })),
			label: this.label,
			usageHint: this.usageHint,
		});
		cloned.wrapS = this.wrapS;
		cloned.wrapT = this.wrapT;
		cloned.minFilter = this.minFilter;
		cloned.magFilter = this.magFilter;
		cloned.offset = { ...this.offset };
		cloned.repeat = { ...this.repeat };
		cloned.rotation = this.rotation;
		cloned._isLoadErrorFallback = this._isLoadErrorFallback;
		return cloned;
	}

	/**
	 * Replaces all mip levels and updates the legacy `data`/`mipmaps` mirrors.
	 */
	public setMipLevels(levels: TextureMipLevel[]): void {
		this.levels = normalizeTextureLevels(levels, null, this.width, this.height);
		this.data = this.levels[0]?.data ?? null;
		this.width = this.levels[0]?.width ?? this.width;
		this.height = this.levels[0]?.height ?? this.height;
		this.mipmaps = this.levels
			.map((level) => level.data)
			.filter((levelData): levelData is TextureData => !!levelData);
		this.markNeedsUpdate();
	}

	/**
	 * Returns the effective mip-level descriptor, including legacy mutations.
	 */
	public getMipLevelDescriptor(level: number): TextureMipLevel | null {
		const mipLevel = Math.max(0, Math.floor(level));
		const explicitLevel = this.levels[mipLevel] ?? null;
		const legacyData =
			this.mipmaps[mipLevel] ??
			(mipLevel === 0 ? this.data : null) ??
			this.mipmaps[0] ??
			null;
		if (explicitLevel && explicitLevel.data === legacyData) {
			return explicitLevel;
		}
		if (!legacyData && !explicitLevel) {
			return null;
		}
		return {
			data: legacyData,
			width: Math.max(1, this.width >> mipLevel),
			height: Math.max(1, this.height >> mipLevel),
			depthOrArrayLayers: explicitLevel?.depthOrArrayLayers,
			bytesPerRow: explicitLevel?.bytesPerRow,
			rowsPerImage: explicitLevel?.rowsPerImage,
		};
	}

	/**
	 * Marks this texture as a loader fallback generated from a load failure.
	 * Renderers can use this to avoid presenting diagnostic colors as environmentes.
	 */
	public markAsLoadErrorFallback(): void {
		this._isLoadErrorFallback = true;
	}

	/**
	 * Clears the loader-fallback marker.
	 */
	public clearLoadErrorFallback(): void {
		this._isLoadErrorFallback = false;
	}

	/**
	 * True when this texture is a diagnostic fallback created after load failure.
	 */
	public get isLoadErrorFallback(): boolean {
		return this._isLoadErrorFallback;
	}

	/**
	 * Marks this texture as changed so render backends can re-upload it.
	 */
	public markNeedsUpdate(): void {
		this.version++;
	}

	/**
	 * Per-frame update hook for animated textures (e.g. video).
	 * Returns true when texture content changed this frame.
	 */
	public update(_timeMs: number = 0): boolean {
		return false;
	}

	/**
	 * Releases dynamic-texture bookkeeping.
	 */
	public dispose(): void {
		if (!this._isDynamicTexture) return;
		Texture._dynamicTextures.delete(this);
		this._isDynamicTexture = false;
	}

	protected _registerAsDynamicTexture(): void {
		if (this._isDynamicTexture) return;
		this._isDynamicTexture = true;
		Texture._dynamicTextures.add(this);
	}

	public static updateDynamicTextures(timeMs: number = 0): boolean {
		let updated = false;
		for (const texture of Texture._dynamicTextures) {
			if (texture.update(timeMs)) {
				updated = true;
			}
		}
		return updated;
	}

	/**
	 * Samples the texture at given UV coordinates.
	 */
	public sample(u: number, v: number): RGBA {
		return this.sampleLevel(u, v, 0);
	}

	/**
	 * Samples the texture at given UV coordinates and mipmap level.
	 */
	public sampleLevel(u: number, v: number, level: number = 0): RGBA {
		if (this.mipmaps.length === 0) return { r: 255, g: 255, b: 255, a: 255 };

		const maxLevel = this.mipmaps.length - 1;
		const l = Math.max(0, Math.min(maxLevel, level));

		const lWidth = Math.max(1, this.width >> Math.floor(l));
		const lHeight = Math.max(1, this.height >> Math.floor(l));
		const levelDescriptor = this.getMipLevelDescriptor(Math.floor(l));
		const data = levelDescriptor?.data ?? null;

		if (!data) return { r: 255, g: 255, b: 255, a: 255 };

		let uu = u * this.repeat.x;
		let vv = v * this.repeat.y;

		if (this.rotation !== 0) {
			const c = Math.cos(this.rotation);
			const s = Math.sin(this.rotation);
			const ru = uu * c - vv * s;
			const rv = uu * s + vv * c;
			uu = ru;
			vv = rv;
		}

		uu += this.offset.x;
		vv += this.offset.y;

		// Handle wrapping
		if (this.wrapS === "Repeat") {
			uu = uu - Math.floor(uu);
		} else if (this.wrapS === "MirroredRepeat") {
			const iter = Math.floor(uu);
			uu = uu - iter;
			if (Math.abs(iter) % 2 === 1) uu = 1.0 - uu;
		} else {
			uu = clamp(uu);
		}

		if (this.wrapT === "Repeat") {
			vv = vv - Math.floor(vv);
		} else if (this.wrapT === "MirroredRepeat") {
			const iter = Math.floor(vv);
			vv = vv - iter;
			if (Math.abs(iter) % 2 === 1) vv = 1.0 - vv;
		} else {
			vv = clamp(vv);
		}

		let x = Math.floor(uu * lWidth);
		let y = Math.floor(vv * lHeight);

		// Clamp to valid range
		if (x >= lWidth) x = lWidth - 1;
		if (y >= lHeight) y = lHeight - 1;

		const idx = (y * lWidth + x) << 2;

		if (this.colorSpace === "HDR") {
			return {
				r: Math.max(0, Math.min(255, data[idx] * 255)),
				g: Math.max(0, Math.min(255, data[idx + 1] * 255)),
				b: Math.max(0, Math.min(255, data[idx + 2] * 255)),
				a: 255,
			};
		}

		return {
			r: data[idx],
			g: data[idx + 1],
			b: data[idx + 2],
			a: data[idx + 3],
		};
	}
}

function isTextureDescriptor(value: unknown): value is TextureDescriptor {
	return (
		!!value &&
		typeof value === "object" &&
		!ArrayBuffer.isView(value) &&
		("data" in value ||
			"width" in value ||
			"height" in value ||
			"colorSpace" in value ||
			"levels" in value ||
			"format" in value ||
			"usageHint" in value ||
			"label" in value)
	);
}

function inferDefaultTextureFormat(
	data: TextureData | null,
	colorSpace: TextureColorSpace
): TextureFormat {
	if (data instanceof Float32Array || colorSpace === "HDR") {
		return TextureFormat.RGBA16Float;
	}
	return TextureFormat.RGBA8Unorm;
}

function normalizeTextureLevels(
	levels: TextureMipLevel[] | undefined,
	data: TextureData | null,
	width: number,
	height: number
): TextureMipLevel[] {
	if (levels && levels.length > 0) {
		return levels.map((level, index) => ({
			data: level.data ?? null,
			width: Math.max(1, Math.floor(level.width ?? width >> index)),
			height: Math.max(1, Math.floor(level.height ?? height >> index)),
			depthOrArrayLayers: Math.max(
				1,
				Math.floor(level.depthOrArrayLayers ?? 1)
			),
			bytesPerRow: level.bytesPerRow,
			rowsPerImage: level.rowsPerImage,
		}));
	}
	if (!data) {
		return [];
	}
	return [
		{
			data,
			width: Math.max(1, Math.floor(width)),
			height: Math.max(1, Math.floor(height)),
			depthOrArrayLayers: 1,
		},
	];
}
