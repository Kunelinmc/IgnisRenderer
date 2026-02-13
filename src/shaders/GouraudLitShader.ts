import { LitShader } from "./LitShader";
import type { RGB } from "../utils/Color";
import type { FragmentInput, PhongSurfaceProperties } from "./types";

/**
 * @deprecated Gouraud shading is being phased out in favor of pixel-rate shading.
 * Use LitShader with BlinnPhongStrategy for better quality.
 */
export class GouraudLitShader extends LitShader<PhongSurfaceProperties> {
	public shade(input: FragmentInput): RGB | null {
		const surface = this._evaluator.evaluate(
			input.u,
			input.v,
			this._face
		) as PhongSurfaceProperties | null;
		if (!surface) return null;

		const res = this._cachedColor;

		const lar = input.lar ?? 0;
		const lag = input.lag ?? 0;
		const lab = input.lab ?? 0;

		const ldr = input.ldr ?? 0;
		const ldg = input.ldg ?? 0;
		const ldb = input.ldb ?? 0;

		const lsr = input.lsr ?? 0;
		const lsg = input.lsg ?? 0;
		const lsb = input.lsb ?? 0;

		const { albedo, specular } = surface;

		const dr = (albedo.r * (lar + ldr)) / 255 + (lsr * specular.r) / 255;
		const dg = (albedo.g * (lag + ldg)) / 255 + (lsg * specular.g) / 255;
		const db = (albedo.b * (lab + ldb)) / 255 + (lsb * specular.b) / 255;

		// Clamp to 0..255 (byte RGB)
		res.r =
			dr < 0 ? 0
			: dr > 255 ? 255
			: dr;
		res.g =
			dg < 0 ? 0
			: dg > 255 ? 255
			: dg;
		res.b =
			db < 0 ? 0
			: db > 255 ? 255
			: db;

		return res;
	}
}
