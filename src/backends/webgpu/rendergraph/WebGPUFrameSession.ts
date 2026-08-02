import type { FrameContext } from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import type { IRenderTexture } from "../../types";
import type { WebGPUPreparedFrameResources } from "../WebGPUResourceContracts";
import type { WebGPUFrameConfiguration } from "./WebGPUFrameConfigurationResolver";
import type { WebGPUDeferredOpaqueFrameState } from "./WebGPUScenePassRecorder";
import type { WebGPUFrameFeatureAnalysis } from "./WebGPUFrameFeatureAnalyzer";
import type { WebGPUFrameCommitter } from "./WebGPUFrameCommitter";

export type WebGPUFrameSessionState =
	| "preparing"
	| "recording"
	| "committing"
	| "skipped";

export type WebGPUFrameHiZStatus = "unavailable" | "pending" | "ready" | "failed";

export type WebGPUTransparencyMode =
	| "legacy"
	| "oit"
	| "legacy-runtime-fallback";

interface WebGPURecordingFrameSessionOptions {
	readonly context: FrameContext;
	readonly configuration: WebGPUFrameConfiguration;
	readonly encoder: ICommandEncoder;
	readonly hiZStatus: WebGPUFrameHiZStatus;
	readonly analysis: WebGPUFrameFeatureAnalysis;
	readonly committer: WebGPUFrameCommitter;
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
	public readonly analysis: WebGPUFrameFeatureAnalysis | null;
	public readonly committer: WebGPUFrameCommitter | null;
	public transparencyMode: WebGPUTransparencyMode;

	private constructor(
		context: FrameContext,
		configuration: WebGPUFrameConfiguration | null,
		state: WebGPUFrameSessionState,
		encoder: ICommandEncoder | null,
		hiZStatus: WebGPUFrameHiZStatus,
		analysis: WebGPUFrameFeatureAnalysis | null,
		committer: WebGPUFrameCommitter | null,
	) {
		this.context = context;
		this.configuration = configuration;
		this.state = state;
		this.encoder = encoder;
		this.hiZStatus = hiZStatus;
		this.analysis = analysis;
		this.committer = committer;
		this.transparencyMode = configuration?.transparencyMode ?? "legacy";
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
			options.analysis,
			options.committer,
		);
	}

	public static createPreparing(context: FrameContext): WebGPUFrameSession {
		return new WebGPUFrameSession(
			context,
			null,
			"preparing",
			null,
			"unavailable",
			null,
			null,
		);
	}

	public static createSkipped(context: FrameContext): WebGPUFrameSession {
		return new WebGPUFrameSession(
			context,
			null,
			"skipped",
			null,
			"unavailable",
			null,
			null,
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
