import { Light, LightType, type LightParams } from "./Light";

export class AmbientLight extends Light<LightType.Ambient> {
	constructor(params: LightParams = {}) {
		super(LightType.Ambient, params);
	}
}
