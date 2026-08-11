import type { RGB } from "../foundation/Color";
import type { IVector3 } from "../maths/types";
import { Light, LightType, type LightParams } from "./Light";

export interface PointLightParams extends LightParams {
	color?: RGB;
	intensity?: number;
	position?: IVector3;
	range?: number;
}

export class PointLight extends Light<LightType.Point> {
	/**
	 * Point light color in sRGB 0..255 channel values.
	 */
	public color: RGB;

	/**
	 * Candela-equivalent RGB luminous intensity before inverse-square attenuation.
	 */
	public intensity: number;

	/** Finite influence range in world-space meters. */
	public range: number;

	constructor(params: PointLightParams = {}) {
		super(LightType.Point, params);
		this.color = params.color ?? { r: 255, g: 255, b: 255 };
		this.intensity = params.intensity ?? 1;
		if (params.position) {
			this.position.copy(params.position);
		}
		this.range = params.range ?? 1000;
	}

	/**
	 * Resolves the light origin in world space.
	 */
	public getWorldLightPosition(out?: IVector3): IVector3 {
		return this.getWorldPosition(out);
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.color = {
			r: this.color.r,
			g: this.color.g,
			b: this.color.b,
		};
		target.intensity = this.intensity;
		target.range = this.range;
	}
}
