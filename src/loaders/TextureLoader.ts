import { Texture } from "../core/Texture";
import { Logger } from "../foundation/Logger";
import { Loader } from "./Loader";

/**
 * TextureLoader handles loading images from various formats into Texture objects.
 */
export class TextureLoader extends Loader {
	constructor() {
		super();
	}

	/**
	 * Loads a texture from a URL.
	 */
	public async load(url: string): Promise<Texture> {
		try {
			const texture = await this._loadCached(`texture:${url}`, async () => {
				const buffer = await this._fetchWithProgress(url);
				const blob = new Blob([buffer]);
				const blobUrl = URL.createObjectURL(blob);
				try {
					return await this._loadImage(blobUrl);
				} finally {
					URL.revokeObjectURL(blobUrl);
				}
			});
			this.emit("load", texture);
			return texture;
		} catch (error) {
			this.emit("error", error);
			Logger.error([`TextureLoader: Failed to load ${url}`, error], {
				scope: "TextureLoader",
			});
			return this._createLoadErrorFallbackTexture();
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
		try {
			const texture = await this._loadImage(url);
			this.emit("load", texture);
			return texture;
		} catch (error) {
			this.emit("error", error);
			Logger.error(["TextureLoader: Failed to load Blob/File", error], {
				scope: "TextureLoader",
			});
			return this._createLoadErrorFallbackTexture();
		} finally {
			URL.revokeObjectURL(url);
		}
	}

	private _createLoadErrorFallbackTexture(): Texture {
		// Keep magenta for diagnostics, but tag it so renderer code can avoid
		// using it as a skybox/environment by mistake.
		const texture = this.createSolidColorTexture(255, 0, 255);
		texture.markAsLoadErrorFallback();
		return texture;
	}
}
