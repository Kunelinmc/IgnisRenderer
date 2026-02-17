import { BaseShader } from "./BaseShader";
import { clamp } from "../maths/Common";
import type { RGB } from "../utils/Color";
import type { FragmentInput, SurfaceProperties } from "./types";

export class UnlitShader extends BaseShader<SurfaceProperties> {
	public shade(input: FragmentInput): RGB | null {
		const surface = this._evaluator.evaluate(input.u, input.v, this._face);
		if (!surface) return null;
		this._lastOpacity = surface.opacity;

		const res = this._cachedColor;
		if (this._context.enableGamma) {
			const gamma = this._context.gamma;
			res.r = clamp(Math.pow(surface.albedo.r / 255, gamma) * 255, 0, 255);
			res.g = clamp(Math.pow(surface.albedo.g / 255, gamma) * 255, 0, 255);
			res.b = clamp(Math.pow(surface.albedo.b / 255, gamma) * 255, 0, 255);
		} else {
			res.r = surface.albedo.r;
			res.g = surface.albedo.g;
			res.b = surface.albedo.b;
		}
		return res;
	}
}
