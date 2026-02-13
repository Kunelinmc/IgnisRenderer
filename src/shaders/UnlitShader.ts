import { BaseShader } from "./BaseShader";
import type { RGB } from "../utils/Color";
import type { FragmentInput, SurfaceProperties } from "./types";

export class UnlitShader extends BaseShader<SurfaceProperties> {
	public shade(input: FragmentInput): RGB | null {
		const surface = this._evaluator.evaluate(input.u, input.v, this._face);
		if (!surface) return null;
		this._lastOpacity = surface.opacity;

		const res = this._cachedColor;
		res.r = surface.albedo.r;
		res.g = surface.albedo.g;
		res.b = surface.albedo.b;
		return res;
	}
}
