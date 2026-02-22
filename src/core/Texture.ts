import type { IVector2 } from "../maths/types";
import type { RGBA } from "../utils/Color";

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
	 * Samples the texture at given UV coordinates.
	 */
	public sample(u: number, v: number): RGBA {
		if (!this.data) return { r: 255, g: 255, b: 255, a: 255 };

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
			uu = Math.max(0, Math.min(1, uu));
		}

		if (this.wrapT === "Repeat") {
			vv = vv - Math.floor(vv);
		} else if (this.wrapT === "MirroredRepeat") {
			const iter = Math.floor(vv);
			vv = vv - iter;
			if (Math.abs(iter) % 2 === 1) vv = 1.0 - vv;
		} else {
			vv = Math.max(0, Math.min(1, vv));
		}

		let x = Math.floor(uu * this.width);
		let y = Math.floor(vv * this.height);

		// Clamp to valid range
		if (x >= this.width) x = this.width - 1;
		if (y >= this.height) y = this.height - 1;

		const idx = (y * this.width + x) << 2;

		if (this.colorSpace === "HDR") {
			// HDR textures store linear floating-point data; scale to [0-255]
			return {
				r: Math.max(0, Math.min(255, this.data[idx] * 255)),
				g: Math.max(0, Math.min(255, this.data[idx + 1] * 255)),
				b: Math.max(0, Math.min(255, this.data[idx + 2] * 255)),
				a: 255,
			};
		}

		return {
			r: this.data[idx],
			g: this.data[idx + 1],
			b: this.data[idx + 2],
			a: this.data[idx + 3],
		};
	}
}
