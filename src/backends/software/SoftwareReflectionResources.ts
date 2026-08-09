export interface SoftwarePlanarReflectionBuffer {
	readonly color: Float32Array;
	readonly width: number;
	readonly height: number;
}

/** @internal Owns reusable RGBA32F Software planar-reflection targets. */
export class SoftwareReflectionResources {
	private readonly _colorPool = new Map<string, Float32Array[]>();
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
				color: this._acquire(width, height) ?? new Float32Array(width * height * 4),
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
		this._colorPool.clear();
	}

	private _acquire(width: number, height: number): Float32Array | null {
		const pool = this._colorPool.get(`${width},${height}`);
		return pool && pool.length > 0 ? pool.pop()! : null;
	}

	private _release(buffer: SoftwarePlanarReflectionBuffer): void {
		const key = `${buffer.width},${buffer.height}`;
		let pool = this._colorPool.get(key);
		if (!pool) {
			pool = [];
			this._colorPool.set(key, pool);
		}
		pool.push(buffer.color);
	}
}
