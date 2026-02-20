import type { RGB } from "../utils/Color";
import { Material, type MaterialParams } from "./Material";

export interface PhongMaterialParams extends MaterialParams {
	diffuse?: RGB;
	specular?: RGB;
	ambient?: RGB;
	shininess?: number;
}

export class PhongMaterial extends Material {
	public diffuse: RGB;
	public specular: RGB;
	public ambient: RGB;
	public shininess: number;

	constructor(params: PhongMaterialParams = {}) {
		super({ ...params, shading: "Phong" });
		this.type = "Phong";
		this.diffuse = params.diffuse || { r: 255, g: 255, b: 255 };
		this.specular = params.specular || { r: 255, g: 255, b: 255 };
		this.ambient = params.ambient || {
			r: this.diffuse.r,
			g: this.diffuse.g,
			b: this.diffuse.b,
		};
		this.shininess = params.shininess || 32;
	}
}
