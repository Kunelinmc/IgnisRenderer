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
}
