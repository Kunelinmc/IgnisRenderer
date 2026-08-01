import type { FrameContext } from "../../pipeline/types";
import type { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import type { IParticleSimulator } from "../../simulation/particles/IParticleSimulator";

import type { WebGPUFrameServiceOwner } from "./WebGPUFrameServiceOwner";
import type {
	WebGPUPostProcessExecutor,
	WebGPUPostProcessSessionPort,
} from "./WebGPUPostProcessExecutor";
import type { WebGPUFrameOrchestrator } from "./rendergraph/WebGPUFrameOrchestrator";

export type WebGPUFrameTransactionState =
	| "recording"
	| "submitting"
	| "committing"
	| "committed"
	| "aborting"
	| "aborted"
	| "invalidated";

export interface WebGPUFrameTransactionServices {
	readonly orchestrator: WebGPUFrameOrchestrator;
	readonly resources: WebGPUFrameServiceOwner;
	readonly particleSimulator: IParticleSimulator | null;
	readonly postProcessRuntime: BackendPostProcessRuntime | null;
	readonly postProcessExecutor: WebGPUPostProcessExecutor | null;
	reportCleanupError(scope: string, error: unknown): void;
}

/**
 * Coordinates one WebGPU backend frame across GPU submission and logical state.
 *
 * @internal Owned by `WebGPUBackend`; applications must use `Renderer`.
 */
export class WebGPUFrameTransaction {
	public readonly context: FrameContext;
	private readonly _services: WebGPUFrameTransactionServices;
	private _state: WebGPUFrameTransactionState = "recording";
	private _started = false;
	private _particleFrameActive = false;
	private _postProcessSessionPort: WebGPUPostProcessSessionPort | null = null;
	private _postProcessCommitted = false;
	private _temporalCommitted = false;
	private _frameStateCommitted = false;

	constructor(context: FrameContext, services: WebGPUFrameTransactionServices) {
		this.context = context;
		this._services = services;
	}

	public get state(): WebGPUFrameTransactionState {
		return this._state;
	}

	public get isOpen(): boolean {
		return !(
			this._state === "committed" ||
			this._state === "aborted" ||
			this._state === "invalidated"
		);
	}

	public begin(): void {
		if (this._started || this._state !== "recording") {
			throw new Error(`WebGPU frame transaction cannot begin from state "${this._state}".`);
		}
		this._started = true;
		try {
			this._services.particleSimulator?.beginFrame(this.context);
			this._particleFrameActive = this._services.particleSimulator !== null;
			this._services.resources.beginFrameResourceLifecycle();
			const port = this._services.orchestrator.createPostProcessSessionPort();
			if (port) {
				this._services.postProcessExecutor?.bindSession(port);
			}
			this._postProcessSessionPort = port;
			this._services.orchestrator.beginFrame(this.context);
		} catch (error) {
			this._abortSynchronous(error);
			throw error;
		}
	}

	public assertRecordingContext(context: FrameContext): void {
		if (this._state !== "recording") {
			throw new Error(
				`WebGPU frame transaction cannot execute passes in state "${this._state}".`,
			);
		}
		if (context !== this.context) {
			throw new Error(
				"WebGPU frame pass context must match the context passed to beginFrame().",
			);
		}
	}

	public async commit(): Promise<void> {
		if (this._state !== "recording") {
			throw new Error(`WebGPU frame transaction cannot commit from state "${this._state}".`);
		}
		this._state = "submitting";
		try {
			await this._services.orchestrator.endFrame(async () => {
				this._state = "committing";
				this._commitParticleFrame();
				this._services.postProcessRuntime?.commitFrame();
				this._postProcessCommitted = true;
				this._services.resources.commitTemporalFrame();
				this._temporalCommitted = true;
				this._services.orchestrator.commitFrameState();
				this._frameStateCommitted = true;
			});
			this._state = "committed";
		} catch (error) {
			await this._abortInternal(error);
			throw error;
		} finally {
			this._unbindPostProcessSession();
		}
	}

	public async abort(error?: unknown): Promise<void> {
		if (!this.isOpen) {
			return;
		}
		await this._abortInternal(error);
	}

	/** Invalidates the transaction synchronously during device teardown. */
	public invalidate(error?: unknown): void {
		if (!this.isOpen) {
			return;
		}
		this._state = "invalidated";
		this._runCleanup("frame recording invalidation", () =>
			this._services.orchestrator.abortRecording(error),
		);
		if (!this._frameStateCommitted) {
			this._runCleanup("frame state invalidation", () =>
				this._services.orchestrator.abortFrameState(error),
			);
		}
		if (!this._temporalCommitted) {
			this._runCleanup("temporal frame invalidation", () =>
				this._services.resources.abortTemporalFrame(),
			);
		}
		this._endParticleFrame();
		this._unbindPostProcessSession();
	}

	private async _abortInternal(error?: unknown): Promise<void> {
		if (!this.isOpen) {
			return;
		}
		this._state = "aborting";
		if (!this._postProcessCommitted) {
			try {
				await this._services.postProcessRuntime?.abortFrame(error);
			} catch (cleanupError) {
				this._services.reportCleanupError("post-process frame abort", cleanupError);
			}
		}
		this._runCleanup("frame recording abort", () =>
			this._services.orchestrator.abortRecording(error),
		);
		if (!this._frameStateCommitted) {
			this._runCleanup("frame state abort", () =>
				this._services.orchestrator.abortFrameState(error),
			);
		}
		if (!this._temporalCommitted) {
			this._runCleanup("temporal frame abort", () =>
				this._services.resources.abortTemporalFrame(),
			);
		}
		this._endParticleFrame();
		this._unbindPostProcessSession();
		this._state = "aborted";
	}

	private _abortSynchronous(error: unknown): void {
		this._state = "aborting";
		this._runCleanup("frame recording abort", () =>
			this._services.orchestrator.abortRecording(error),
		);
		this._runCleanup("frame state abort", () =>
			this._services.orchestrator.abortFrameState(error),
		);
		this._runCleanup("temporal frame abort", () =>
			this._services.resources.abortTemporalFrame(),
		);
		this._endParticleFrame();
		this._unbindPostProcessSession();
		this._state = "aborted";
	}

	private _endParticleFrame(): void {
		if (!this._particleFrameActive) {
			return;
		}
		this._particleFrameActive = false;
		this._runCleanup("particle frame cleanup", () =>
			this._services.particleSimulator?.endFrame(),
		);
	}

	private _commitParticleFrame(): void {
		if (!this._particleFrameActive) {
			return;
		}
		this._particleFrameActive = false;
		this._services.particleSimulator?.endFrame();
	}

	private _unbindPostProcessSession(): void {
		const port = this._postProcessSessionPort;
		this._postProcessSessionPort = null;
		this._runCleanup("post-process session unbind", () =>
			this._services.postProcessExecutor?.unbindSession(port ?? undefined),
		);
	}

	private _runCleanup(scope: string, operation: () => void): void {
		try {
			operation();
		} catch (error) {
			this._services.reportCleanupError(scope, error);
		}
	}
}
