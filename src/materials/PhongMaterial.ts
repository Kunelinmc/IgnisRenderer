import type { RGB } from "../foundation/Color";
import { Material, type MaterialParams, ShadingModel } from "./Material";

export interface PhongMaterialParams extends MaterialParams {
	/** sRGB-encoded diffuse reflectance in 0..255 channel values. */
	diffuse?: RGB;
	/** sRGB-encoded Fresnel F0 in 0..255 channel values. */
	specular?: RGB;
	/** sRGB-encoded indirect diffuse reflectance in 0..255 channel values. */
	ambient?: RGB;
	shininess?: number;
}

export class PhongMaterial extends Material {
	/** sRGB-encoded diffuse reflectance. */
	public diffuse: RGB;
	/** sRGB-encoded Fresnel F0. */
	public specular: RGB;
	/** sRGB-encoded indirect diffuse reflectance. */
	public ambient: RGB;
	public shininess: number;

	constructor(params: PhongMaterialParams = {}) {
		super({ ...params, shading: ShadingModel.Phong });
		this.type = "Phong";
		this.diffuse = params.diffuse || { r: 255, g: 255, b: 255 };
		this.specular = params.specular || { r: 56, g: 56, b: 56 };
		this.ambient = params.ambient || {
			r: this.diffuse.r,
			g: this.diffuse.g,
			b: this.diffuse.b,
		};
		this.shininess = Math.max(params.shininess ?? 32, 0);
	}
}
