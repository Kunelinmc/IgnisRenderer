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
				return this._loadBlob(blob);
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
				try {
					resolve(this._createTextureFromImageSource(img, img.width, img.height));
				} catch (error) {
					reject(error);
				}
			};
			img.onerror = () => reject(new Error(`Failed to load image at ${url}`));
			img.src = url;
		});
	}

	private async _loadBlob(blob: Blob): Promise<Texture> {
		const createImageBitmapFn = (globalThis as {
			createImageBitmap?: (image: Blob) => Promise<ImageBitmap>;
		}).createImageBitmap;
		if (typeof createImageBitmapFn === "function") {
			try {
				const bitmap = await createImageBitmapFn(blob);
				try {
					return this._createTextureFromImageSource(
						bitmap,
						bitmap.width,
						bitmap.height
					);
				} finally {
					bitmap.close();
				}
			} catch {
				// Fall back to HTMLImageElement for formats/browser paths that
				// createImageBitmap cannot decode.
			}
		}

		const blobUrl = URL.createObjectURL(blob);
		try {
			return await this._loadImage(blobUrl);
		} finally {
			URL.revokeObjectURL(blobUrl);
		}
	}

	private _createTextureFromImageSource(
		source: CanvasImageSource,
		width: number,
		height: number
	): Texture {
		const canvas = document.createElement("canvas");
		canvas.width = width;
		canvas.height = height;
		const ctx = canvas.getContext("2d", { willReadFrequently: true });
		if (!ctx) {
			throw new Error("Failed to get 2D context");
		}
		ctx.drawImage(source, 0, 0);

		const imageData = ctx.getImageData(0, 0, width, height);
		return new Texture(imageData.data, width, height);
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
		try {
			const texture = await this._loadBlob(blob);
			this.emit("load", texture);
			return texture;
		} catch (error) {
			this.emit("error", error);
			Logger.error(["TextureLoader: Failed to load Blob/File", error], {
				scope: "TextureLoader",
			});
			return this._createLoadErrorFallbackTexture();
		}
	}

	private _createLoadErrorFallbackTexture(): Texture {
		// Keep magenta for diagnostics, but tag it so renderer code can avoid
		// using it as a environment/environment by mistake.
		const texture = this.createSolidColorTexture(255, 0, 255);
		texture.markAsLoadErrorFallback();
		return texture;
	}
}
