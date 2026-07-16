import { WebGPUFramePartialSubmitError } from "../../../foundation/Error";
import type { ICommandBuffer, ICommandEncoder } from "../../ICommandEncoder";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";

interface PendingFrameCommand {
	readonly label: string;
	readonly commandBuffer: ICommandBuffer;
}

export interface WebGPUFrameCommitDebugState {
	readonly state: "recording" | "submitting" | "completed" | "aborted" | "failed";
	readonly queuedLabels: readonly string[];
	readonly submittedLabels: readonly string[];
}

/** Retains and submits one frame's command buffers in deterministic order. */
export class WebGPUFrameCommitter {
	private readonly _commands: PendingFrameCommand[] = [];
	private _state: WebGPUFrameCommitDebugState["state"] = "recording";
	private _submittedLabels: string[] = [];

	public constructor(private readonly _host: WebGPUFrameHost) {}

	public enqueueEncoder(label: string, encoder: ICommandEncoder): void {
		this.enqueue(label, encoder.finish());
	}

	public enqueue(label: string, commandBuffer: ICommandBuffer): void {
		if (this._state !== "recording") {
			throw new Error(`WebGPU frame committer cannot enqueue in state "${this._state}".`);
		}
		this._commands.push({ label, commandBuffer });
	}

	public async commit(postSubmit?: () => void | Promise<void>): Promise<void> {
		if (this._state !== "recording") {
			throw new Error(`WebGPU frame committer cannot commit from state "${this._state}".`);
		}
		this._state = "submitting";
		const labels = this._commands.map((entry) => entry.label);
		try {
			for (const entry of this._commands) {
				this._host.submit([entry.commandBuffer]);
				this._submittedLabels.push(entry.label);
			}
			await postSubmit?.();
			this._state = "completed";
		} catch (cause) {
			this._state = "failed";
			if (this._submittedLabels.length <= 0) {
				throw cause;
			}
			throw new WebGPUFramePartialSubmitError({
				cause,
				phase:
					this._submittedLabels.length < this._commands.length
						? "submit"
						: "post-submit",
				submittedLabels: this._submittedLabels,
				pendingLabels: labels.slice(this._submittedLabels.length),
				totalCount: this._commands.length,
			});
		}
	}

	public abort(): void {
		if (this._state === "recording") {
			this._commands.length = 0;
			this._state = "aborted";
		}
	}

	public getDebugState(): WebGPUFrameCommitDebugState {
		return {
			state: this._state,
			queuedLabels: this._commands.map((entry) => entry.label),
			submittedLabels: this._submittedLabels.slice(),
		};
	}
}
