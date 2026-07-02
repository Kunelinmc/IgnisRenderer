import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import type { Rasterizer } from "../Rasterizer";
import {
	setSoftwarePlanarReflectionRuntime,
	SoftwarePlanarReflectionRuntime,
} from "../SoftwarePlanarReflectionRuntime";
import type { SoftwarePassLike, SoftwareSurfaceCompositePass } from "./types";

export class SoftwareReflectionPass
	implements SoftwarePassLike, SoftwareSurfaceCompositePass
{
	private _runtime: SoftwarePlanarReflectionRuntime;

	constructor(rasterizer: Rasterizer) {
		this._runtime = new SoftwarePlanarReflectionRuntime(rasterizer);
	}

	public render(context: FrameContext): void {
		this._runtime.render(context);
		setSoftwarePlanarReflectionRuntime(context.transient, this._runtime);
	}

	public composite(context: FrameContext, packets: DrawPacket[]): void {
		this._runtime.composite(context, packets);
	}
}
