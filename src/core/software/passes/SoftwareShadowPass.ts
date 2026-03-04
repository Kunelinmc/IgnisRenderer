import type { PreparedScene, ResolvedFeatureState } from "../../pipeline/types";
import type { Renderer } from "../../Renderer";

export class SoftwareShadowPass {
	private _renderer: Renderer;

	constructor(renderer: Renderer) {
		this._renderer = renderer;
	}

	public render(frame: PreparedScene, features: ResolvedFeatureState): void {
		(this._renderer as any)._shadowRenderer?.render(frame, features);
	}
}
