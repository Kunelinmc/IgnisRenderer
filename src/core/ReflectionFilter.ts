import type { Renderer } from "./Renderer";
import type { Plane } from "../maths/Plane";

export interface ReflectionFilterContext {
	renderer: Renderer;
	plane: Plane;
	imageData: ImageData;
	width: number;
	height: number;
}

export abstract class ReflectionFilter<P = unknown> {
	/**
	 * Apply the filter to the reflection image data.
	 * @param context - The rendering context for the filter.
	 * @param params - Filter specific parameters.
	 */
	abstract apply(context: ReflectionFilterContext, params: P): void;
}

export class BlurFilter extends ReflectionFilter<number> {
	private _temp: Uint8ClampedArray | null = null;

	apply(context: ReflectionFilterContext, radius: number): void {
		radius = radius | 0;
		if (radius <= 0) return;

		const { width, height, imageData } = context;
		const pixels = imageData.data;
		const len = pixels.length;

		// Reuse temp buffer
		let temp = this._temp;
		if (!temp || temp.length !== len) {
			temp = new Uint8ClampedArray(len);
			this._temp = temp;
		}

		const w = width;
		const h = height;

		// ---- Horizontal pass: pixels -> temp ----
		for (let y = 0; y < h; y++) {
			const rowStart = (y * w) << 2;

			let rSum = 0,
				gSum = 0,
				bSum = 0;
			let count = 0;

			// Prime the window for x=0 (clamped)
			for (let dx = -radius; dx <= radius; dx++) {
				const nx =
					dx < 0 ? 0
					: dx >= w ? w - 1
					: dx;
				const idx = rowStart + (nx << 2);
				rSum += pixels[idx];
				gSum += pixels[idx + 1];
				bSum += pixels[idx + 2];
				count++;
			}

			for (let x = 0; x < w; x++) {
				const outIdx = rowStart + (x << 2);

				temp[outIdx] = (rSum / count) | 0;
				temp[outIdx + 1] = (gSum / count) | 0;
				temp[outIdx + 2] = (bSum / count) | 0;
				temp[outIdx + 3] = pixels[outIdx + 3]; // keep alpha as-is (same as your current behavior)

				// Slide window: remove left, add right (clamped)
				const leftX = x - radius;
				const rightX = x + radius + 1;

				const clLeft =
					leftX < 0 ? 0
					: leftX >= w ? w - 1
					: leftX;
				const clRight =
					rightX < 0 ? 0
					: rightX >= w ? w - 1
					: rightX;

				const leftIdx = rowStart + (clLeft << 2);
				const rightIdx = rowStart + (clRight << 2);

				rSum += pixels[rightIdx] - pixels[leftIdx];
				gSum += pixels[rightIdx + 1] - pixels[leftIdx + 1];
				bSum += pixels[rightIdx + 2] - pixels[leftIdx + 2];
				// count stays constant because we clamp (window always full)
			}
		}

		// ---- Vertical pass: temp -> pixels ----
		for (let x = 0; x < w; x++) {
			let rSum = 0,
				gSum = 0,
				bSum = 0;
			let count = 0;

			// Prime the window for y=0 (clamped)
			for (let dy = -radius; dy <= radius; dy++) {
				const ny =
					dy < 0 ? 0
					: dy >= h ? h - 1
					: dy;
				const idx = (ny * w + x) << 2;
				rSum += temp[idx];
				gSum += temp[idx + 1];
				bSum += temp[idx + 2];
				count++;
			}

			for (let y = 0; y < h; y++) {
				const outIdx = (y * w + x) << 2;

				pixels[outIdx] = (rSum / count) | 0;
				pixels[outIdx + 1] = (gSum / count) | 0;
				pixels[outIdx + 2] = (bSum / count) | 0;
				pixels[outIdx + 3] = temp[outIdx + 3]; // preserve alpha from horizontal pass

				const topY = y - radius;
				const bottomY = y + radius + 1;

				const clTop =
					topY < 0 ? 0
					: topY >= h ? h - 1
					: topY;
				const clBottom =
					bottomY < 0 ? 0
					: bottomY >= h ? h - 1
					: bottomY;

				const topIdx = (clTop * w + x) << 2;
				const bottomIdx = (clBottom * w + x) << 2;

				rSum += temp[bottomIdx] - temp[topIdx];
				gSum += temp[bottomIdx + 1] - temp[topIdx + 1];
				bSum += temp[bottomIdx + 2] - temp[topIdx + 2];
			}
		}
	}
}

export class RippleFilter extends ReflectionFilter<number> {
	private _temp: Uint8ClampedArray | null = null;
	private _dxRow: Float32Array | null = null;
	private _dyCol: Float32Array | null = null;

	apply(context: ReflectionFilterContext, intensity: number): void {
		if (intensity <= 0) return;

		const { width, height, imageData, renderer } = context;
		const pixels = imageData.data;
		const len = pixels.length;

		// Reuse temp buffer
		let temp = this._temp;
		if (!temp || temp.length !== len) {
			temp = new Uint8ClampedArray(len);
			this._temp = temp;
		}

		// Reuse displacement caches
		let dxRow = this._dxRow;
		if (!dxRow || dxRow.length !== height) {
			dxRow = new Float32Array(height);
			this._dxRow = dxRow;
		}

		let dyCol = this._dyCol;
		if (!dyCol || dyCol.length !== width) {
			dyCol = new Float32Array(width);
			this._dyCol = dyCol;
		}

		const time = renderer.lastTime * 0.005;

		// Precompute dx for each y (depends only on y + time)
		for (let y = 0; y < height; y++) {
			dxRow[y] = Math.sin(y * 0.1 + time) * intensity;
		}

		// Precompute dy for each x (depends only on x + time)
		for (let x = 0; x < width; x++) {
			dyCol[x] = Math.cos(x * 0.1 + time) * intensity;
		}

		// Sample with clamping
		for (let y = 0; y < height; y++) {
			const dyBase = y; // for readability
			const dx = dxRow[y];

			for (let x = 0; x < width; x++) {
				const dy = dyCol[x];

				let sx = (x + dx) | 0; // floor for positive; can be off by 1 for negative
				let sy = (dyBase + dy) | 0;

				// If dx/dy can go negative, use Math.floor to match your original exactly:
				// let sx = Math.floor(x + dx);
				// let sy = Math.floor(y + dy);

				if (sx < 0) sx = 0;
				else if (sx >= width) sx = width - 1;

				if (sy < 0) sy = 0;
				else if (sy >= height) sy = height - 1;

				const sourceIdx = (sy * width + sx) << 2;
				const destIdx = (y * width + x) << 2;

				temp[destIdx] = pixels[sourceIdx];
				temp[destIdx + 1] = pixels[sourceIdx + 1];
				temp[destIdx + 2] = pixels[sourceIdx + 2];
				temp[destIdx + 3] = pixels[sourceIdx + 3];
			}
		}

		pixels.set(temp);
	}
}
