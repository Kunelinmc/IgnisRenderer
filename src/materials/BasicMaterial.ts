import type { RGB } from "../utils/Color";
import { Material, type MaterialParams } from "./Material";

export interface BasicMaterialParams extends MaterialParams {
	diffuse?: RGB;
}

export class BasicMaterial extends Material {
	public diffuse: RGB;

	constructor(params: BasicMaterialParams = {}) {
		super({ ...params, shading: "Flat" });
		this.type = "Basic";
		this.diffuse = params.diffuse || { r: 255, g: 255, b: 255 };
	}
}
