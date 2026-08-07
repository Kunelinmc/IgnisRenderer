import type { FrameContext } from "../../../pipeline/types";
import type { PreparedFramePacketSet } from "../../../pipeline/FramePacketContributorRegistry";
import type { ICommandEncoder } from "../../ICommandEncoder";
import type { IRenderTexture } from "../../types";
import type { WebGPUPreparedFrameResources } from "../WebGPUResourceContracts";
import type { WebGPUFrameConfiguration } from "./WebGPUFrameConfigurationResolver";
import type { WebGPUDeferredOpaqueFrameState } from "./WebGPUScenePassRecorder";
import type { WebGPUFrameCommitter } from "./WebGPUFrameCommitter";
import { WebGPUFrameModuleStateStore } from "./WebGPUFrameGraphModule";

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
	readonly framePackets: PreparedFramePacketSet;
	readonly committer: WebGPUFrameCommitter;
	readonly moduleState: WebGPUFrameModuleStateStore;
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
	public readonly framePackets: PreparedFramePacketSet | null;
	public readonly committer: WebGPUFrameCommitter | null;
	public readonly moduleState: WebGPUFrameModuleStateStore;
	public transparencyMode: WebGPUTransparencyMode;

	private constructor(
		context: FrameContext,
		configuration: WebGPUFrameConfiguration | null,
		state: WebGPUFrameSessionState,
		encoder: ICommandEncoder | null,
		hiZStatus: WebGPUFrameHiZStatus,
		framePackets: PreparedFramePacketSet | null,
		committer: WebGPUFrameCommitter | null,
		moduleState: WebGPUFrameModuleStateStore,
	) {
		this.context = context;
		this.configuration = configuration;
		this.state = state;
		this.encoder = encoder;
		this.hiZStatus = hiZStatus;
		this.framePackets = framePackets;
		this.committer = committer;
		this.moduleState = moduleState;
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
			options.framePackets,
			options.committer,
			options.moduleState,
		);
	}

	public static createPreparing(context: FrameContext): WebGPUFrameSession {
		const moduleState = new WebGPUFrameModuleStateStore();
		moduleState.seal();
		return new WebGPUFrameSession(
			context,
			null,
			"preparing",
			null,
			"unavailable",
			null,
			null,
			moduleState,
		);
	}

	public static createSkipped(context: FrameContext): WebGPUFrameSession {
		const moduleState = new WebGPUFrameModuleStateStore();
		moduleState.seal();
		return new WebGPUFrameSession(
			context,
			null,
			"skipped",
			null,
			"unavailable",
			null,
			null,
			moduleState,
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
