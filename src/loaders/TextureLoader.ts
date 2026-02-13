import { Texture } from "../core/Texture";
import { Loader } from "./Loader";

/**
 * TextureLoader handles loading images from various formats into Texture objects.
 */
export class TextureLoader extends Loader {
	private _cache: Map<string, Texture>;

	constructor() {
		super();
		this._cache = new Map();
	}

	/**
	 * Loads a texture from a URL.
	 */
	public async load(url: string): Promise<Texture> {
		if (this._cache.has(url)) {
			const texture = this._cache.get(url)!;
			this.emit("load", texture);
			return texture;
		}

		try {
			const buffer = await this._fetchWithProgress(url);
			const blob = new Blob([buffer]);
			const blobUrl = URL.createObjectURL(blob);
			const texture = await this._loadImage(blobUrl);
			URL.revokeObjectURL(blobUrl);

			this._cache.set(url, texture);
			this.emit("load", texture);
			return texture;
		} catch (error) {
			this.emit("error", error);
			console.error(`TextureLoader: Failed to load ${url}`, error);
			// Return a magenta solid color texture on error
			return this.createSolidColorTexture(255, 0, 255);
		}
	}

	/**
	 * Loads image and converts to Texture.
	 */
	private _loadImage(url: string): Promise<Texture> {
		return new Promise((resolve, reject) => {
			const img = new Image();
			img.crossOrigin = "anonymous";
			img.onload = () => {
				const canvas = document.createElement("canvas");
				canvas.width = img.width;
				canvas.height = img.height;
				const ctx = canvas.getContext("2d");
				if (!ctx) {
					reject(new Error("Failed to get 2D context"));
					return;
				}
				ctx.drawImage(img, 0, 0);

				const imageData = ctx.getImageData(0, 0, img.width, img.height);
				const texture = new Texture(imageData.data, img.width, img.height);
				resolve(texture);
			};
			img.onerror = () => reject(new Error(`Failed to load image at ${url}`));
			img.src = url;
		});
	}

	/**
	 * Creates a simple solid color texture.
	 * @param r 0-255
	 * @param g 0-255
	 * @param b 0-255
	 * @param a 0-255
	 */
	public createSolidColorTexture(
		r: number,
		g: number,
		b: number,
		a: number = 255
	): Texture {
		const data = new Uint8ClampedArray([r, g, b, a]);
		return new Texture(data, 1, 1);
	}

	/**
	 * Creates a texture from a Blob or File.
	 */
	public async loadFromBlob(blob: Blob | File): Promise<Texture> {
		const url = URL.createObjectURL(blob);
		const texture = await this.load(url);
		URL.revokeObjectURL(url);
		return texture;
	}
}
