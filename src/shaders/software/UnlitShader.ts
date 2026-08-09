import { BaseShader } from "./BaseShader";
import type { FragmentInput, FragmentOutput, SurfaceProperties } from "./types";

export class UnlitShader extends BaseShader<SurfaceProperties> {
	public shade(input: FragmentInput): FragmentOutput | null {
		const surface = this._evaluateSurface(input);
		if (!surface) return null;
		this._lastOpacity = surface.opacity;

		const res = this._cachedColor;
		// Shader output stays in linear space; sRGB encode happens in post-process.
		res.r = Math.max(0, surface.albedo.r / 255);
		res.g = Math.max(0, surface.albedo.g / 255);
		res.b = Math.max(0, surface.albedo.b / 255);
		return this._cachedOutput;
	}
}
