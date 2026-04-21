import type { FrameContext } from "../../../pipeline/types";
import type { Rasterizer } from "../Rasterizer";
import { ReflectionRenderer } from "../ReflectionRenderer";
import type { SoftwarePassLike } from "./types";

export class SoftwareReflectionPass implements SoftwarePassLike {
	private _reflectionRenderer: ReflectionRenderer;

	constructor(rasterizer: Rasterizer) {
		this._reflectionRenderer = new ReflectionRenderer(rasterizer);
	}

	public render(context: FrameContext): void {
		this._reflectionRenderer.render(context);
	}
}
