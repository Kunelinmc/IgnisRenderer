import type { IVector3 } from "../maths/types";
import {
	Light,
	LightType,
	type LightParams,
} from "./Light";

export interface PointLightParams extends LightParams {
	position?: IVector3;
	range?: number;
}

export class PointLight extends Light<LightType.Point> {
	public position: IVector3;
	public range: number;

	constructor(params: PointLightParams = {}) {
		super(LightType.Point, params);
		this.position = params.position ?? { x: 0, y: 0, z: 0 };
		this.range = params.range ?? 1000;
	}
}
