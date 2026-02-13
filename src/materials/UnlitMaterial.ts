import type { RGB } from "../utils/Color";
import { Material } from "./Material";
import type { BasicMaterialParams } from "./BasicMaterial";

export class UnlitMaterial extends Material {
	public diffuse: RGB;

	constructor(params: BasicMaterialParams = {}) {
		super({ ...params, shading: "Unlit" });
		this.type = "Unlit";
		this.diffuse = params.diffuse || { r: 255, g: 255, b: 255 };
	}
}
