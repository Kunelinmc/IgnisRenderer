import type { WebGPUBackendSession } from "../WebGPUBackend";
import type { IRenderTexture, TextureFormat, TextureUsage } from "../types";

export interface TexturePoolOptions {
	usage: TextureUsage;
	sampleCount?: number;
	label?: string;
	mipLevelCount?: number | ((width: number, height: number) => number);
	maxPerBucket?: number;
}

export class TexturePool {
	private _backend: WebGPUBackendSession;
	private _usage: TextureUsage;
	private _sampleCount: number;
	private _label: string;
	private _mipLevelCount:
		| number
		| ((width: number, height: number) => number)
		| null;
	private _maxPerBucket: number;
	private _available = new Map<string, IRenderTexture[]>();
	private _owned = new Set<IRenderTexture>();
	private _textureBucket = new Map<IRenderTexture, string>();

	constructor(backend: WebGPUBackendSession, options: TexturePoolOptions) {
		this._backend = backend;
		this._usage = options.usage;
		this._sampleCount = Math.max(1, Math.floor(options.sampleCount ?? 1));
		this._label = options.label ?? "WebGPUTexturePool";
		this._mipLevelCount = options.mipLevelCount ?? null;
		this._maxPerBucket = Math.max(1, Math.floor(options.maxPerBucket ?? 2));
	}

	public acquire(
		width: number,
		height: number,
		format: TextureFormat
	): IRenderTexture {
		const resolvedWidth = this._resolvePositiveInteger(width);
		const resolvedHeight = this._resolvePositiveInteger(height);
		const key = this._bucketKey(resolvedWidth, resolvedHeight, format);
		const bucket = this._available.get(key);
		if (bucket && bucket.length > 0) {
			return bucket.pop()!;
		}

		const mipLevelCount = this._resolveMipLevelCount(
			resolvedWidth,
			resolvedHeight
		);
		const texture = this._backend.createTexture({
			width: resolvedWidth,
			height: resolvedHeight,
			format,
			usage: this._usage,
			sampleCount: this._sampleCount,
			mipLevelCount,
			label: `${this._label}_${resolvedWidth}x${resolvedHeight}_${format}`,
		});
		this._owned.add(texture);
		this._textureBucket.set(texture, key);
		return texture;
	}

	public release(texture: IRenderTexture): void {
		if (!this._owned.has(texture)) {
			texture.destroy();
			return;
		}
		const key = this._textureBucket.get(texture);
		if (!key) {
			this._owned.delete(texture);
			texture.destroy();
			return;
		}
		let bucket = this._available.get(key);
		if (!bucket) {
			bucket = [];
			this._available.set(key, bucket);
		}
		if (bucket.includes(texture)) {
			return;
		}
		bucket.push(texture);
		if (bucket.length > this._maxPerBucket) {
			const stale = bucket.shift();
			if (stale) {
				stale.destroy();
				this._owned.delete(stale);
				this._textureBucket.delete(stale);
			}
		}
	}

	public destroy(): void {
		for (const texture of this._owned) {
			texture.destroy();
		}
		this._owned.clear();
		this._textureBucket.clear();
		this._available.clear();
	}

	private _resolveMipLevelCount(
		width: number,
		height: number
	): number | undefined {
		if (typeof this._mipLevelCount === "function") {
			return Math.max(1, Math.floor(this._mipLevelCount(width, height)));
		}
		if (typeof this._mipLevelCount === "number") {
			return Math.max(1, Math.floor(this._mipLevelCount));
		}
		return undefined;
	}

	private _bucketKey(
		width: number,
		height: number,
		format: TextureFormat
	): string {
		return `${width}x${height}:${format}`;
	}

	private _resolvePositiveInteger(value: number): number {
		if (!Number.isFinite(value)) {
			return 1;
		}
		return Math.max(1, Math.floor(value));
	}
}
