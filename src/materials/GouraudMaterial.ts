import type { RGB } from "../utils/Color";
import { Material } from "./Material";
import type { PhongMaterialParams } from "./PhongMaterial";

export class GouraudMaterial extends Material {
	public diffuse: RGB;
	public specular: RGB;
	public ambient: RGB;
	public shininess: number;

	constructor(params: PhongMaterialParams = {}) {
		super({ ...params, shading: "Gouraud" });
		this.type = "Gouraud";
		this.diffuse = params.diffuse || { r: 255, g: 255, b: 255 };
		this.specular = params.specular || { r: 255, g: 255, b: 255 };
		this.ambient = params.ambient || { r: 0, g: 0, b: 0 };
		this.shininess = params.shininess || 32;
	}
}
