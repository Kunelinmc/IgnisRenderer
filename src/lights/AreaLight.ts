import type { IVector3 } from "../maths/types";
import {
	Light,
	LightType,
	type LightParams,
} from "./Light";

export interface AreaLightParams extends LightParams {
	position?: IVector3;
	width?: number;
	height?: number;
	range?: number;
}

export class AreaLight extends Light<LightType.RectArea> {
	public width: number;
	public height: number;
	public range: number;

	constructor(params: AreaLightParams = {}) {
		super(LightType.RectArea, params);
		if (params.position) {
			this.position.copy(params.position);
		}
		this.width = params.width ?? 100;
		this.height = params.height ?? 100;
		this.range = params.range ?? 1000;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.width = this.width;
		target.height = this.height;
		target.range = this.range;
	}
}
