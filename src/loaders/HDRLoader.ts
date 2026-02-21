import { Texture } from "../core/Texture";
import { Loader } from "./Loader";

/**
 * HDRLoader loads Radiance HDR (.hdr / .rgbe) environment maps into a Texture.
 * The resulting Texture will have a Float32Array for its data containing linear RGB floating point values.
 */
export class HDRLoader extends Loader {
	private _cache: Map<string, Texture>;

	constructor() {
		super();
		this._cache = new Map();
	}

	/**
	 * Loads an HDR texture from a URL.
	 */
	public async load(url: string): Promise<Texture> {
		if (this._cache.has(url)) {
			const texture = this._cache.get(url)!;
			this.emit("load", texture);
			return texture;
		}

		try {
			const buffer = await this._fetchWithProgress(url);
			const texture = this.parse(buffer);

			this._cache.set(url, texture);
			this.emit("load", texture);
			return texture;
		} catch (error) {
			this.emit("error", error);
			console.error(`HDRLoader: Failed to load ${url}`, error);
			// Return a simple 1x1 black HDR texture on error
			return new Texture(new Float32Array([0, 0, 0, 1]), 1, 1, "HDR");
		}
	}

	/**
	 * Parses Radiance RGBE format bytes into a Float32 Texture.
	 */
	public parse(buffer: ArrayBuffer): Texture {
		const data = new Uint8Array(buffer);
		let offset = 0;

		// 1. Read Header
		let header = "";
		let matchFound = false;
		while (offset < data.length) {
			const char = String.fromCharCode(data[offset++]);
			header += char;
			if (header.endsWith("\n\n")) {
				matchFound = true;
				break;
			}
		}

		if (!matchFound) {
			throw new Error("HDRLoader: File is not a valid HDR image (no header).");
		}

		if (!header.startsWith("#?RADIANCE") && !header.startsWith("#?RGBE")) {
			throw new Error("HDRLoader: Invalid HDR signature.");
		}

		// 2. Read Resolution string (e.g. "-Y 1024 +X 2048")
		let resStr = "";
		while (offset < data.length) {
			const char = String.fromCharCode(data[offset++]);
			resStr += char;
			if (char === "\n") break;
		}

		const resMatch = resStr.match(/-Y\s+(\d+)\s+\+X\s+(\d+)/);
		if (!resMatch) {
			throw new Error(
				"HDRLoader: Unsupported HDR resolution format or orientation."
			);
		}

		const height = parseInt(resMatch[1], 10);
		const width = parseInt(resMatch[2], 10);

		// 3. Read Pixels
		const numPixels = width * height;
		const floatData = new Float32Array(numPixels * 4);
		let pixelOffset = 0;

		const scanlineBuffer = new Uint8Array(width * 4);

		for (let y = 0; y < height; y++) {
			if (offset + 4 > data.length) break;

			const r = data[offset++];
			const g = data[offset++];
			const b = data[offset++];
			const e = data[offset++];

			const isNewRLE = r === 2 && g === 2 && (b << 8) + e === width;

			if (isNewRLE) {
				// New RLE format per scanline: 4 channels are stored sequentially
				for (let channel = 0; channel < 4; channel++) {
					let ptr = channel * width;
					const ptrEnd = (channel + 1) * width;
					while (ptr < ptrEnd && offset < data.length) {
						let count = data[offset++];
						if (count > 128) {
							// Run length
							count -= 128;
							const val = data[offset++];
							for (let i = 0; i < count; i++) {
								scanlineBuffer[ptr++] = val;
							}
						} else {
							// Uncompressed run
							for (let i = 0; i < count; i++) {
								scanlineBuffer[ptr++] = data[offset++];
							}
						}
					}
				}

				// Convert scanline from RGBE to Float
				for (let x = 0; x < width; x++) {
					const rc = scanlineBuffer[x];
					const gc = scanlineBuffer[x + width];
					const bc = scanlineBuffer[x + width * 2];
					const ec = scanlineBuffer[x + width * 3];

					if (ec === 0) {
						floatData[pixelOffset++] = 0;
						floatData[pixelOffset++] = 0;
						floatData[pixelOffset++] = 0;
						floatData[pixelOffset++] = 1;
					} else {
						const f = Math.pow(2.0, ec - 128 - 8);
						floatData[pixelOffset++] = rc * f;
						floatData[pixelOffset++] = gc * f;
						floatData[pixelOffset++] = bc * f;
						floatData[pixelOffset++] = 1;
					}
				}
			} else {
				// Old flat/RLE format (fallback, simplified)
				// For a correct parser, this should handle old RLE, but typically not needed for environment maps
				offset -= 4; // rewind
				for (let x = 0; x < width; x++) {
					const rc = data[offset++];
					const gc = data[offset++];
					const bc = data[offset++];
					const ec = data[offset++];

					if (ec === 0) {
						floatData[pixelOffset++] = 0;
						floatData[pixelOffset++] = 0;
						floatData[pixelOffset++] = 0;
						floatData[pixelOffset++] = 1;
					} else {
						const f = Math.pow(2.0, ec - 128 - 8);
						floatData[pixelOffset++] = rc * f;
						floatData[pixelOffset++] = gc * f;
						floatData[pixelOffset++] = bc * f;
						floatData[pixelOffset++] = 1;
					}
				}
			}
		}

		return new Texture(floatData, width, height, "HDR");
	}
}
