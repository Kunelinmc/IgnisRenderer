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
/**
 * Base Loader class that provides event emission capabilities.
 * Emits:
 * - 'loadstart': When loading begins
 * - 'progress': { loaded, total } during network load
 * - 'parsestart': When parsing begins
 * - 'parseprogress': { current, total, message } during parsing
 * - 'load': When loading and parsing is complete
 * - 'error': When an error occurs
 */
export class Loader extends EventEmitter {
	constructor() {
		super();
	}
	/**
	 * Internal helper to report network progress.
	 * @protected
	 */
	protected async _fetchWithProgress(url: string): Promise<ArrayBuffer> {
		this.emit("loadstart", { url } as LoadStartEvent);
		const response = await fetch(url);
		if (!response.ok) {
			const error = new Error(
				`Failed to load: ${response.statusText} (${url})`
			);
			this.emit("error", error);
			throw error;
		}
		const contentLength = response.headers.get("content-length");
		const total = contentLength ? parseInt(contentLength, 10) : NaN;
		if (isNaN(total) || !response.body) {
			// Fallback if no content-length or body stream not available
			const buffer = await response.arrayBuffer();
			this.emit("progress", {
				loaded: buffer.byteLength,
				total: buffer.byteLength,
			} as ProgressEvent);
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
				this.emit("progress", { loaded, total, url } as ProgressEvent);
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
