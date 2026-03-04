import type { PreparedScene } from "../../pipeline/types";
import type { Renderer } from "../../Renderer";

export class SoftwareReflectionPass {
	private _renderer: Renderer;

	constructor(renderer: Renderer) {
		this._renderer = renderer;
	}

	public render(_frame: PreparedScene): void {
		this._renderer.reflectionRenderer.render();
	}
}
