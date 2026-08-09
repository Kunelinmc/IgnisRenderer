import { LitShader } from "./LitShader";
import type { RGB } from "../../foundation/Color";
import type {
	FragmentInput,
	FragmentOutput,
	PhongSurfaceProperties,
} from "./types";

/**
 * @deprecated Gouraud shading is being phased out in favor of pixel-rate shading.
 * Use LitShader with BlinnPhongStrategy for better quality.
 */
export class GouraudLitShader extends LitShader<PhongSurfaceProperties> {
	public shade(input: FragmentInput): FragmentOutput | null {
		const surface = this._evaluateSurface(input) as
			| PhongSurfaceProperties
			| null;
		if (!surface) return null;

		const res = this._cachedColor;

		const la = input.lightAmbient || { r: 0, g: 0, b: 0 };
		const ld = input.lightDiffuse || { r: 0, g: 0, b: 0 };
		const ls = input.lightSpecular || { r: 0, g: 0, b: 0 };

		const { albedo, specular } = surface;

		const dr = (albedo.r * (la.r + ld.r)) / 255 + (ls.r * specular.r) / 255;
		const dg = (albedo.g * (la.g + ld.g)) / 255 + (ls.g * specular.g) / 255;
		const db = (albedo.b * (la.b + ld.b)) / 255 + (ls.b * specular.b) / 255;

		res.r = Math.max(0, dr / 255);
		res.g = Math.max(0, dg / 255);
		res.b = Math.max(0, db / 255);

		return this._cachedOutput;
	}
}
