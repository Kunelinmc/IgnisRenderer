import type { RGB } from "../foundation/Color";
import type { IVector3 } from "../maths/types";
import {
	Light,
	LightType,
	type LightParams,
} from "./Light";

export interface DirectionalLightParams extends LightParams {
	color?: RGB;
	intensity?: number;
	direction?: IVector3;
}

export class DirectionalLight extends Light<LightType.Directional> {
	/**
	 * Directional light color in sRGB 0..255 channel values.
	 */
	public color: RGB;

	/**
	 * Scalar strength applied to directional light contribution.
	 */
	public intensity: number;

	public direction: IVector3;

	constructor(params: DirectionalLightParams = {}) {
		super(LightType.Directional, params);
		this.color = params.color ?? { r: 255, g: 255, b: 255 };
		this.intensity = params.intensity ?? 1;
		this.direction = params.direction ?? { x: 0, y: -1, z: 0 };
	}

	/**
	 * Resolves the light direction in world space.
	 */
	public getWorldLightDirection(out?: IVector3): IVector3 {
		return this.getWorldDirection(this.direction, out);
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.color = {
			r: this.color.r,
			g: this.color.g,
			b: this.color.b,
		};
		target.intensity = this.intensity;
		target.direction = {
			x: this.direction.x,
			y: this.direction.y,
			z: this.direction.z,
		};
	}
}
