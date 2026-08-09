import type { DrawPacket } from "../../../pipeline/types";
import type { Rasterizer } from "../Rasterizer";
import { SoftwarePlanarReflectionRuntime } from "../SoftwarePlanarReflectionRuntime";
import type { SoftwarePassLike, SoftwareSurfaceCompositePass } from "./types";
import type { SoftwarePassContext } from "../SoftwareFrameServices";
import type { SoftwareReflectionResources } from "../SoftwareReflectionResources";

export class SoftwareReflectionPass
	implements SoftwarePassLike, SoftwareSurfaceCompositePass
{
	private _runtime: SoftwarePlanarReflectionRuntime;

	constructor(rasterizer: Rasterizer, resources?: SoftwareReflectionResources) {
		this._runtime = new SoftwarePlanarReflectionRuntime(rasterizer, resources);
	}

	public render(context: SoftwarePassContext): void {
		this._runtime.render(context);
	}

	public composite(context: SoftwarePassContext, packets: DrawPacket[]): void {
		this._runtime.composite(context, packets);
	}

	public get runtime(): SoftwarePlanarReflectionRuntime {
		return this._runtime;
	}
}
