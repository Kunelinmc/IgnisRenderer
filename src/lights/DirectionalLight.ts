import type { IVector3 } from "../maths/types";
import {
	Light,
	LightType,
	type LightParams,
} from "./Light";
import type { ShadowConfig } from "./ShadowMapping";

export interface DirectionalLightParams extends LightParams {
	direction?: IVector3;
	shadow?: ShadowConfig;
}

export class DirectionalLight extends Light<LightType.Directional> {
	public direction: IVector3;

	constructor(params: DirectionalLightParams = {}) {
		super(LightType.Directional, params);
		this.direction = params.direction ?? { x: 0, y: -1, z: 0 };
		this.shadow = params.shadow ?? {
			strategy: "single-map",
			size: 1024,
		};
		this.castShadow = params.castShadow ?? true;
	}

	/**
	 * Resolves the light direction in world space.
	 */
	public getWorldLightDirection(out?: IVector3): IVector3 {
		return this.getWorldDirection(this.direction, out);
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.direction = {
			x: this.direction.x,
			y: this.direction.y,
			z: this.direction.z,
		};
	}
}
