import { BaseShader } from "./BaseShader";
import { clamp, sRGBToLinear } from "../maths/Common";
import type { RGB } from "../utils/Color";
import type { FragmentInput, SurfaceProperties } from "./types";

export class UnlitShader extends BaseShader<SurfaceProperties> {
	public shade(input: FragmentInput): RGB | null {
		const surface = this._evaluator.evaluate(input, this._face);
		if (!surface) return null;
		this._lastOpacity = surface.opacity;

		const res = this._cachedColor;
		// Shader output stays in linear space; sRGB encode happens in post-process.
		res.r = clamp(sRGBToLinear(surface.albedo.r / 255) * 255, 0, 255);
		res.g = clamp(sRGBToLinear(surface.albedo.g / 255) * 255, 0, 255);
		res.b = clamp(sRGBToLinear(surface.albedo.b / 255) * 255, 0, 255);
		return res;
	}
}
