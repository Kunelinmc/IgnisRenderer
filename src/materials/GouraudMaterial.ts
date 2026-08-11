import type { RGB } from "../foundation/Color";
import { Material, ShadingModel } from "./Material";
import type { PhongMaterialParams } from "./PhongMaterial";

export class GouraudMaterial extends Material {
	public diffuse: RGB;
	public specular: RGB;
	public ambient: RGB;
	public shininess: number;

	constructor(params: PhongMaterialParams = {}) {
		super({ ...params, shading: ShadingModel.Gouraud });
		this.type = "Gouraud";
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
