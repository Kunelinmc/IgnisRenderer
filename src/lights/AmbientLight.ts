import {
	Light,
	LightType,
	type LightParams,
	type LightContribution,
	type SurfacePoint,
} from "./Light";

export class AmbientLight extends Light<LightType.Ambient> {
	constructor(params: LightParams = {}) {
		super(LightType.Ambient, params);
	}

	public computeContribution(_surface: SurfacePoint): LightContribution {
		return {
			type: "ambient",
			color: this.color,
			intensity: this.intensity,
		};
	}
}
