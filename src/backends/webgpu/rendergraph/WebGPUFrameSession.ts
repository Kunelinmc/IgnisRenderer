import type { FrameContext } from "../../../pipeline/types";
import type { PreparedFramePacketSet } from "../../../pipeline/FramePacketContributorRegistry";
import type { WebGPUPreparedFrameResources } from "../WebGPUResourceContracts";
import type { WebGPUFrameCommandStream } from "./WebGPUFrameCommandStream";
import type { WebGPUFrameConfiguration } from "./WebGPUFrameConfiguration";
import { WebGPUFrameMessageSnapshot } from "./WebGPUFrameMessage";
import type { WebGPUFrameTargetView } from "./WebGPUFrameTargetManager";
import type { WebGPUFrameDirtyRectOperations } from "./WebGPUFrameExecutionContext";

export type WebGPUFrameSessionState =
	| "preparing"
	| "recording"
	| "committing"
	| "skipped";

interface WebGPUFrameSessionBase {
	readonly context: FrameContext;
	readonly state: WebGPUFrameSessionState;
}

export interface WebGPUPreparingFrameSession extends WebGPUFrameSessionBase {
	readonly state: "preparing";
}

export interface WebGPUSkippedFrameSession extends WebGPUFrameSessionBase {
	readonly state: "skipped";
	readonly messages: WebGPUFrameMessageSnapshot;
}

export interface WebGPURecordingFrameSession extends WebGPUFrameSessionBase {
	readonly state: "recording";
	readonly configuration: WebGPUFrameConfiguration;
	readonly resources: WebGPUPreparedFrameResources;
	readonly framePackets: PreparedFramePacketSet;
	readonly messages: WebGPUFrameMessageSnapshot;
	readonly targets: WebGPUFrameTargetView;
	readonly commands: WebGPUFrameCommandStream;
	readonly earlyZPrepassEnabled: boolean;
	readonly dirtyRects: WebGPUFrameDirtyRectOperations;
}

export interface WebGPUCommittingFrameSession extends WebGPUFrameSessionBase {
	readonly state: "committing";
	readonly configuration: WebGPUFrameConfiguration;
	readonly resources: WebGPUPreparedFrameResources;
	readonly framePackets: PreparedFramePacketSet;
	readonly messages: WebGPUFrameMessageSnapshot;
	readonly targets: WebGPUFrameTargetView;
	readonly commands: WebGPUFrameCommandStream;
	readonly earlyZPrepassEnabled: boolean;
	readonly dirtyRects: WebGPUFrameDirtyRectOperations;
}

export type WebGPUExecutableFrameSession =
	| WebGPURecordingFrameSession
	| WebGPUCommittingFrameSession;

export type WebGPUFrameSession =
	| WebGPUPreparingFrameSession
	| WebGPURecordingFrameSession
	| WebGPUCommittingFrameSession
	| WebGPUSkippedFrameSession;

export interface WebGPURecordingFrameSessionOptions {
	readonly context: FrameContext;
	readonly configuration: WebGPUFrameConfiguration;
	readonly resources: WebGPUPreparedFrameResources;
	readonly framePackets: PreparedFramePacketSet;
	readonly messages: WebGPUFrameMessageSnapshot;
	readonly targets: WebGPUFrameTargetView;
	readonly commands: WebGPUFrameCommandStream;
	readonly earlyZPrepassEnabled: boolean;
	readonly dirtyRects: WebGPUFrameDirtyRectOperations;
}

export const WebGPUFrameSession = {
	createPreparing(context: FrameContext): WebGPUPreparingFrameSession {
		return { state: "preparing", context };
	},
	createSkipped(context: FrameContext): WebGPUSkippedFrameSession {
		return {
			state: "skipped",
			context,
			messages: new WebGPUFrameMessageSnapshot(),
		};
	},
	createRecording(
		options: WebGPURecordingFrameSessionOptions,
	): WebGPURecordingFrameSession {
		return {
			state: "recording",
			context: options.context,
			configuration: options.configuration,
			resources: options.resources,
			framePackets: options.framePackets,
			messages: options.messages,
			targets: options.targets,
			commands: options.commands,
			earlyZPrepassEnabled: options.earlyZPrepassEnabled,
			dirtyRects: options.dirtyRects,
		};
	},
	beginCommit(session: WebGPURecordingFrameSession): WebGPUCommittingFrameSession {
		return { ...session, state: "committing" };
	},
	assertContext(session: WebGPUFrameSession, context: FrameContext): void {
		if (context !== session.context) {
			throw new Error(
				"WebGPU frame pass context must match the context passed to beginFrame().",
			);
		}
	},
};
