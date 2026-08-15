import type { ICommandEncoder } from "../../ICommandEncoder";

import {
	WebGPUFrameCommitter,
	type WebGPUFrameCommitDebugState,
} from "./WebGPUFrameCommitter";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";

/**
 * Owns the mutable command-recording and submission state for one WebGPU frame.
 *
 * @internal Owned by a recording or committing `WebGPUFrameSession`.
 */
export class WebGPUFrameCommandStream {
	private _encoder: ICommandEncoder | null;
	private readonly _committer: WebGPUFrameCommitter;

	public constructor(private readonly _host: WebGPUFrameHost) {
		this._encoder = _host.createCommandEncoder();
		this._committer = new WebGPUFrameCommitter(_host);
	}

	public get encoder(): ICommandEncoder | null {
		return this._encoder;
	}

	public requireEncoder(): ICommandEncoder {
		if (!this._encoder) {
			throw new Error("WebGPU frame command stream has no active encoder.");
		}
		return this._encoder;
	}

	public enqueueCurrent(label: string): void {
		const encoder = this.requireEncoder();
		this._encoder = null;
		this._committer.enqueueEncoder(label, encoder);
	}

	public enqueueEncoder(label: string, encoder: ICommandEncoder): void {
		this._committer.enqueueEncoder(label, encoder);
	}

	public resume(): ICommandEncoder {
		if (this._encoder) {
			throw new Error("WebGPU frame command stream already has an active encoder.");
		}
		this._encoder = this._host.createCommandEncoder();
		return this._encoder;
	}

	public async commit(
		label: string,
		postSubmit?: () => void | Promise<void>,
	): Promise<void> {
		this.enqueueCurrent(label);
		await this._committer.commit(postSubmit);
	}

	public abort(): void {
		this._encoder = null;
		this._committer.abort();
	}

	public getDebugState(): WebGPUFrameCommitDebugState {
		return this._committer.getDebugState();
	}
}
