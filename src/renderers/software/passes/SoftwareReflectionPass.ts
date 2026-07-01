import type { FrameContext } from "../../../pipeline/types";
import type { Rasterizer } from "../Rasterizer";
import {
	setSoftwarePlanarReflectionRuntime,
	SoftwarePlanarReflectionRuntime,
} from "../SoftwarePlanarReflectionRuntime";
import type { SoftwarePassLike } from "./types";

export class SoftwareReflectionPass implements SoftwarePassLike {
	private _runtime: SoftwarePlanarReflectionRuntime;

	constructor(rasterizer: Rasterizer) {
		this._runtime = new SoftwarePlanarReflectionRuntime(rasterizer);
	}

	public render(context: FrameContext): void {
		this._runtime.render(context);
		setSoftwarePlanarReflectionRuntime(context.transient, this._runtime);
	}
}
