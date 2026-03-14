import { EventEmitter } from "../core/EventEmitter";

export interface LoadStartEvent {
	url: string;
}
export interface ProgressEvent {
	loaded: number;
	total: number;
	url?: string;
}
export interface ParseStartEvent {}
export interface ParseProgressEvent {
	current: number;
	total: number;
	message: string;
}
export interface LoaderEvents {
	loadstart: [LoadStartEvent];
	progress: [ProgressEvent];
	parsestart: [];
	parseprogress: [ParseProgressEvent];
	load: [any];
	error: [any];
	[key: string]: any[];
}

/**
 * Base Loader class that provides event emission capabilities.
 * Emits:
 * - 'loadstart': When loading begins
 * - 'progress': { loaded, total } during network load
 * - 'parsestart': When parsing begins
 * - 'parseprogress': { current, total, message } during parsing
 * - 'load': When loading and parsing is complete
 * - 'error': When an error occurs
/**
 * Base Loader class that provides event emission capabilities.
 */
export class Loader<
	E extends LoaderEvents = LoaderEvents,
> extends EventEmitter<E> {
	private static _resourceCache = new Map<string, unknown>();
	private static _pendingResourceCache = new Map<string, Promise<unknown>>();

	constructor() {
		super();
	}
	/**
	 * Internal helper to report network progress.
	 * @protected
	 */
	protected async _fetchWithProgress(url: string): Promise<ArrayBuffer> {
		(this as any).emit("loadstart", { url });
		const response = await fetch(url);
		if (!response.ok) {
			const error = new Error(
				`Failed to load: ${response.statusText} (${url})`
			);
			(this as any).emit("error", error);
			throw error;
		}
		const contentLength = response.headers.get("content-length");
		const total = contentLength ? parseInt(contentLength, 10) : NaN;
		if (isNaN(total) || !response.body) {
			// Fallback if no content-length or body stream not available
			const buffer = await response.arrayBuffer();
			(this as any).emit("progress", {
				loaded: buffer.byteLength,
				total: buffer.byteLength,
			});
			return buffer;
		}
		const reader = response.body.getReader();
		let loaded = 0;
		const chunks: Uint8Array[] = [];
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			if (value) {
				chunks.push(value);
				loaded += value.length;
				(this as any).emit("progress", { loaded, total, url });
			}
		}
		const buffer = new Uint8Array(loaded);
		let offset = 0;
		for (const chunk of chunks) {
			buffer.set(chunk, offset);
			offset += chunk.length;
		}
		return buffer.buffer;
	}

	protected async _loadCached<T>(
		cacheKey: string,
		loader: () => Promise<T>
	): Promise<T> {
		const scopedKey = this._createScopedCacheKey(cacheKey);
		const cached = Loader._resourceCache.get(scopedKey);
		if (cached !== undefined) {
			return cached as T;
		}

		const pending = Loader._pendingResourceCache.get(scopedKey);
		if (pending) {
			return (await pending) as T;
		}

		const loadingPromise = (async () => {
			try {
				const loaded = await loader();
				Loader._resourceCache.set(scopedKey, loaded);
				return loaded;
			} finally {
				Loader._pendingResourceCache.delete(scopedKey);
			}
		})();

		Loader._pendingResourceCache.set(scopedKey, loadingPromise);
		return (await loadingPromise) as T;
	}

	protected _getCached<T>(cacheKey: string): T | undefined {
		const scopedKey = this._createScopedCacheKey(cacheKey);
		const cached = Loader._resourceCache.get(scopedKey);
		return cached as T | undefined;
	}

	protected _setCached(cacheKey: string, value: unknown): void {
		const scopedKey = this._createScopedCacheKey(cacheKey);
		Loader._resourceCache.set(scopedKey, value);
	}

	protected _deleteCached(cacheKey: string): void {
		const scopedKey = this._createScopedCacheKey(cacheKey);
		Loader._resourceCache.delete(scopedKey);
		Loader._pendingResourceCache.delete(scopedKey);
	}

	protected _createScopedCacheKey(cacheKey: string): string {
		return `${this.constructor.name}:${cacheKey}`;
	}

	public static clearSharedCache(): void {
		Loader._resourceCache.clear();
		Loader._pendingResourceCache.clear();
	}
}
