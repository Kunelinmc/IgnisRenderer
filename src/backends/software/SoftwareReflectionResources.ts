export interface SoftwarePlanarReflectionBuffer {
	readonly imageData: ImageData;
	readonly width: number;
	readonly height: number;
}

/** @internal Owns Software planar-reflection targets and reusable image storage. */
export class SoftwareReflectionResources {
	private readonly _imageDataPool = new Map<string, ImageData[]>();
	public readonly buffers = new Map<string, SoftwarePlanarReflectionBuffer>();

	public ensure(
		key: string,
		width: number,
		height: number,
	): SoftwarePlanarReflectionBuffer {
		let buffer = this.buffers.get(key);
		if (buffer && (buffer.width !== width || buffer.height !== height)) {
			this._release(buffer);
			buffer = undefined;
		}
		if (!buffer) {
			buffer = {
				imageData: this._acquire(width, height) ?? new ImageData(width, height),
				width,
				height,
			};
			this.buffers.set(key, buffer);
		}
		return buffer;
	}

	public trim(activeKeys: ReadonlySet<string>): void {
		for (const [key, buffer] of this.buffers) {
			if (activeKeys.has(key)) continue;
			this._release(buffer);
			this.buffers.delete(key);
		}
	}

	public clear(): void {
		for (const buffer of this.buffers.values()) this._release(buffer);
		this.buffers.clear();
	}

	public destroy(): void {
		this.clear();
		this._imageDataPool.clear();
	}

	private _acquire(width: number, height: number): ImageData | null {
		const pool = this._imageDataPool.get(`${width},${height}`);
		return pool && pool.length > 0 ? pool.pop()! : null;
	}

	private _release(buffer: SoftwarePlanarReflectionBuffer): void {
		const key = `${buffer.width},${buffer.height}`;
		let pool = this._imageDataPool.get(key);
		if (!pool) {
			pool = [];
			this._imageDataPool.set(key, pool);
		}
		pool.push(buffer.imageData);
	}
}
