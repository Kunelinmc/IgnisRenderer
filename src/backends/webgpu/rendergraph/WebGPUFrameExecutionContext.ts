import type { PreparedFramePacketSet } from "../../../pipeline/FramePackets";
import type { DirtyRect } from "../../../pipeline/incremental";
import type { DrawPacket, FrameContext } from "../../../pipeline/types";

import type { WebGPUPreparedFrameResources } from "../WebGPUResourceContracts";
import type { WebGPUFrameCommandStream } from "./WebGPUFrameCommandStream";
import type { WebGPUFrameConfiguration } from "./WebGPUFrameConfiguration";
import type { WebGPUFrameMessageSnapshot } from "./WebGPUFrameMessage";
import type { WebGPUFrameTargetView } from "./WebGPUFrameTargetManager";

/** @internal Frame-local dirty-region operations shared by narrow recorder ports. */
export interface WebGPUFrameDirtyRectOperations {
	isIncrementalPartial(context: FrameContext | null): boolean;
	resolveDirtyRects(
		context: FrameContext | null,
		width: number,
		height: number,
	): DirtyRect[];
	selectPacketsForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: DirtyRect,
	): DrawPacket[];
	selectTransparentSubsetForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: DirtyRect,
	): DrawPacket[];
}

/**
 * Complete immutable input for one WebGPU recording transaction.
 *
 * @internal Owned by the WebGPU frame orchestrator and passed by identity to
 * frame modules. Leaf recorders consume narrower `Pick` ports.
 */
export interface WebGPUFrameExecutionContext {
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
