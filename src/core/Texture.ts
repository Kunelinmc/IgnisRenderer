import type { IVector2 } from "../maths/types";
import type { RGBA } from "../foundation/Color";
import { clamp } from "../maths/Common";

export type TextureFilter =
	| "Nearest"
	| "Linear"
	| "NearestMipmapNearest"
	| string;
export type TextureWrap = "Repeat" | "Clamp" | "MirroredRepeat";

/**
 * Describes the color space of texture data.
 * - `"sRGB"`: Standard sRGB-encoded data (typical for 8-bit images loaded via canvas/browser).
 * - `"Linear"`: Linear color space data (e.g. normal maps, metallic-roughness maps that store non-color data).
 * - `"HDR"`: High dynamic range linear data (e.g. .hdr environment maps with Float32Array values in [0, ∞)).
 */
export type TextureColorSpace = "sRGB" | "Linear" | "HDR";

/**
 * Texture class to store image data and metadata for UV mapping.
 */
export class Texture {
	private static _dynamicTextures = new Set<Texture>();

	data: Uint8ClampedArray | Float32Array | Uint8Array | null;
	width: number;
	height: number;
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
	mipmaps: (Uint8ClampedArray | Float32Array | Uint8Array)[];
	version: number;
	private _isDynamicTexture: boolean;

	constructor(
		data: Uint8ClampedArray | Float32Array | Uint8Array | null = null,
		width: number = 0,
		height: number = 0,
		colorSpace: TextureColorSpace = "sRGB"
	) {
		this.data = data;
		this.width = width;
		this.height = height;
		this.wrapS = "Repeat";
		this.wrapT = "Repeat";
		this.minFilter = "Linear";
		this.magFilter = "Linear";
		this.offset = { x: 0, y: 0 };
		this.repeat = { x: 1, y: 1 };
		this.rotation = 0;
		this.colorSpace = colorSpace;
		this.mipmaps = data ? [data] : [];
		this.version = 0;
		this._isDynamicTexture = false;
	}

	public clone(): Texture {
		const cloned = new Texture(
			this.data,
			this.width,
			this.height,
			this.colorSpace
		);
		cloned.wrapS = this.wrapS;
		cloned.wrapT = this.wrapT;
		cloned.minFilter = this.minFilter;
		cloned.magFilter = this.magFilter;
		cloned.offset = { ...this.offset };
		cloned.repeat = { ...this.repeat };
		cloned.rotation = this.rotation;
		return cloned;
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
		const data = this.mipmaps[Math.floor(l)];

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
