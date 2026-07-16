import type { FrameContext } from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import type { IRenderTexture } from "../../types";
import type { WebGPUPreparedFrameResources } from "../WebGPURenderResources";
import type { WebGPUFrameConfiguration } from "./WebGPUFrameConfigurationResolver";
import type { WebGPUDeferredOpaqueFrameState } from "./WebGPUScenePassRecorder";

export type WebGPUFrameSessionState = "recording" | "committing" | "skipped";

export type WebGPUFrameHiZStatus = "unavailable" | "pending" | "ready" | "failed";

interface WebGPURecordingFrameSessionOptions {
	readonly context: FrameContext;
	readonly configuration: WebGPUFrameConfiguration;
	readonly encoder: ICommandEncoder;
	readonly hiZStatus: WebGPUFrameHiZStatus;
}

/**
 * Owns mutable state for one WebGPU frame lifecycle.
 *
 * @internal Owned by the WebGPU frame orchestrator. Applications should use
 * `Renderer.renderFrame()` instead.
 */
export class WebGPUFrameSession {
	public readonly context: FrameContext;
	public readonly configuration: WebGPUFrameConfiguration | null;
	public state: WebGPUFrameSessionState;
	public encoder: ICommandEncoder | null;
	public resources: WebGPUPreparedFrameResources | null = null;
	public presented = false;
	public motionHistoryWriteTarget: IRenderTexture | null = null;
	public deferredOpaqueFrameState: WebGPUDeferredOpaqueFrameState | null = null;
	public hiZStatus: WebGPUFrameHiZStatus;
	public hiZBuildCount = 0;

	private constructor(
		context: FrameContext,
		configuration: WebGPUFrameConfiguration | null,
		state: WebGPUFrameSessionState,
		encoder: ICommandEncoder | null,
		hiZStatus: WebGPUFrameHiZStatus,
	) {
		this.context = context;
		this.configuration = configuration;
		this.state = state;
		this.encoder = encoder;
		this.hiZStatus = hiZStatus;
	}

	public static createRecording(
		options: WebGPURecordingFrameSessionOptions,
	): WebGPUFrameSession {
		return new WebGPUFrameSession(
			options.context,
			options.configuration,
			"recording",
			options.encoder,
			options.hiZStatus,
		);
	}

	public static createSkipped(context: FrameContext): WebGPUFrameSession {
		return new WebGPUFrameSession(
			context,
			null,
			"skipped",
			null,
			"unavailable",
		);
	}

	public assertContext(context: FrameContext): void {
		if (context !== this.context) {
			throw new Error(
				"WebGPU frame pass context must match the context passed to beginFrame().",
			);
		}
	}

	public beginCommit(): void {
		if (this.state !== "recording") {
			throw new Error(
				`WebGPU frame session cannot commit from state "${this.state}".`,
			);
		}
		this.state = "committing";
	}
}
