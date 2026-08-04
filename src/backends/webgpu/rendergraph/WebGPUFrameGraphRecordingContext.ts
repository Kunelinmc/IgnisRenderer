import type {
	DirtyRect,
} from "../../../pipeline/incremental";
import type {
	DrawPacket,
	FrameContext,
} from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import type { WebGPUPreparedFrameResources } from "../WebGPUResourceContracts";
import type { WebGPUFrameTargets } from "../WebGPUPostProcessContracts";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import type { WebGPUFrameMSAATargets } from "./WebGPUFrameTargetManager";

/**
 * Shared per-frame access surface for WebGPU render graph recorders.
 */
export interface WebGPUFrameGraphRecordingContext {
	getEncoder(): ICommandEncoder | null;
	getFrameTargets(): WebGPUFrameTargets | null;
	getMSAATargets(): WebGPUFrameMSAATargets | null;
	getTargetWidth(): number;
	getTargetHeight(): number;
	getSampleCount(): number;
	getSceneTargetMode(): WebGPUSceneTargetMode;
	isMRTEnabled(): boolean;
	isEarlyZPrepassEnabled(): boolean;
	requireFrameResources(): WebGPUPreparedFrameResources;
	isIncrementalPartial(context: FrameContext | null): boolean;
	resolveDirtyRects(
		context: FrameContext | null,
		width: number,
		height: number
	): DirtyRect[];
	selectPacketsForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: DirtyRect
	): DrawPacket[];
	selectTransparentSubsetForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: DirtyRect
	): DrawPacket[];
}
