import type { RGB } from "../utils/Color";
import { Material, type MaterialParams, type TextureLike } from "./Material";

export interface PBRMaterialParams extends MaterialParams {
	albedo?: RGB;
	roughness?: number;
	metalness?: number;
	emissive?: RGB;
	emissiveIntensity?: number;
	f0?: RGB;
	metallicRoughnessMap?: TextureLike;
	normalMap?: TextureLike;
	emissiveMap?: TextureLike;
	occlusionMap?: TextureLike;
}

export class PBRMaterial extends Material {
	public albedo: RGB;
	public roughness: number;
	public metalness: number;
	public emissive: RGB;
	public emissiveIntensity: number;
	public f0: RGB;
	public metallicRoughnessMap: TextureLike;
	public normalMap: TextureLike;
	public emissiveMap: TextureLike;
	public occlusionMap: TextureLike;

	constructor(params: PBRMaterialParams = {}) {
		super({ ...params, shading: "PBR" });
		this.type = "PBR";
		this.albedo = params.albedo || { r: 255, g: 255, b: 255 };
		this.roughness = params.roughness !== undefined ? params.roughness : 0.5;
		this.metalness = params.metalness !== undefined ? params.metalness : 0.0;
		this.emissive = params.emissive || { r: 0, g: 0, b: 0 };
		this.emissiveIntensity = params.emissiveIntensity ?? 1.0;
		this.f0 = params.f0 || { r: 0.04 * 255, g: 0.04 * 255, b: 0.04 * 255 };

		this.metallicRoughnessMap = params.metallicRoughnessMap || null;
		this.normalMap = params.normalMap || null;
		this.emissiveMap = params.emissiveMap || null;
		this.occlusionMap = params.occlusionMap || null;
	}
}
