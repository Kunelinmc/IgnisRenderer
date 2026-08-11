import type { RGB } from "../foundation/Color";
import type { IVector3 } from "../maths/types";
import {
	Light,
	LightType,
	type LightParams,
} from "./Light";

export interface AreaLightParams extends LightParams {
	color?: RGB;
	intensity?: number;
	position?: IVector3;
	width?: number;
	height?: number;
	range?: number;
}

export class AreaLight extends Light<LightType.RectArea> {
	/**
	 * Area light color in sRGB 0..255 channel values.
	 */
	public color: RGB;

	/**
	 * Emitted-radiance-equivalent RGB intensity.
	 */
	public intensity: number;

	/** Emitter width in world-space meters. */
	public width: number;
	/** Emitter height in world-space meters. */
	public height: number;
	/** Finite influence range in world-space meters. */
	public range: number;

	constructor(params: AreaLightParams = {}) {
		super(LightType.RectArea, params);
		this.color = params.color ?? { r: 255, g: 255, b: 255 };
		this.intensity = params.intensity ?? 1;
		if (params.position) {
			this.position.copy(params.position);
		}
		this.width = params.width ?? 100;
		this.height = params.height ?? 100;
		this.range = params.range ?? 1000;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.color = {
			r: this.color.r,
			g: this.color.g,
			b: this.color.b,
		};
		target.intensity = this.intensity;
		target.width = this.width;
		target.height = this.height;
		target.range = this.range;
	}
}
