import type { RGB } from "../foundation/Color";
import { Light, LightType, type LightParams } from "./Light";

export interface AmbientLightParams extends LightParams {
	color?: RGB;
	intensity?: number;
}

export class AmbientLight extends Light<LightType.Ambient> {
	/**
	 * Ambient light color in sRGB 0..255 channel values.
	 */
	public color: RGB;

	/**
	 * Scalar strength applied to ambient light contribution.
	 */
	public intensity: number;

	constructor(params: AmbientLightParams = {}) {
		super(LightType.Ambient, params);
		this.color = params.color ?? { r: 255, g: 255, b: 255 };
		this.intensity = params.intensity ?? 1;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.color = {
			r: this.color.r,
			g: this.color.g,
			b: this.color.b,
		};
		target.intensity = this.intensity;
	}
}
