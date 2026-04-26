import type { IVector3 } from "../maths/types";
import { Light, LightType, type LightParams } from "./Light";

export interface PointLightParams extends LightParams {
	position?: IVector3;
	range?: number;
}

export class PointLight extends Light<LightType.Point> {
	public range: number;

	constructor(params: PointLightParams = {}) {
		super(LightType.Point, params);
		if (params.position) {
			this.position.copy(params.position);
		}
		this.range = params.range ?? 1000;
		this.shadow = params.shadow ?? {
			strategy: "single-map",
			size: 1024,
		};
	}

	/**
	 * Resolves the light origin in world space.
	 */
	public getWorldLightPosition(out?: IVector3): IVector3 {
		return this.getWorldPosition(out);
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.range = this.range;
	}
}
