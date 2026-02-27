import { BaseShader } from "./BaseShader";
import { clamp, sRGBToLinear } from "../maths/Common";
import type { RGB } from "../utils/Color";
import type { FragmentInput, FragmentOutput, SurfaceProperties } from "./types";

export class UnlitShader extends BaseShader<SurfaceProperties> {
	public shade(input: FragmentInput): FragmentOutput | null {
		const surface = this._evaluator.evaluate(input, this._face);
		if (!surface) return null;
		this._lastOpacity = surface.opacity;

		const res = this._cachedColor;
		// Shader output stays in linear space; sRGB encode happens in post-process.
		res.r = clamp(surface.albedo.r, 0, 255);
		res.g = clamp(surface.albedo.g, 0, 255);
		res.b = clamp(surface.albedo.b, 0, 255);
		return this._cachedOutput;
	}
}
