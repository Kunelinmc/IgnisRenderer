import {
	Light,
	LightType,
	type LightParams,
	type LightContribution,
} from "./Light";
import type { IVector3 } from "../maths/types";

export class AmbientLight extends Light<LightType.Ambient> {
	constructor(params: LightParams = {}) {
		super(LightType.Ambient, params);
	}

	public computeContribution(_point: IVector3): LightContribution {
		return {
			type: "ambient",
			color: {
				r: this.color.r * this.intensity,
				g: this.color.g * this.intensity,
				b: this.color.b * this.intensity,
			},
		};
	}
}
