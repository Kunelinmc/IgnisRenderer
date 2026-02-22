import type { RGB } from "../utils/Color";
import { Material, type MaterialParams, type TextureLike } from "./Material";
import { clamp, sRGBToLinear } from "../maths/Common";

export interface PBRMaterialParams extends MaterialParams {
	albedo?: RGB;
	roughness?: number;
	metalness?: number;
	emissive?: RGB;
	emissiveIntensity?: number;
	/**
	 * @deprecated Use reflectance instead.
	 */
	f0?: RGB;
	/**
	 * The specular reflectance of the material.
	 * Default is 0.5 (corresponds to 0.04 F0).
	 */
	reflectance?: number;
	/** Alias for map */
	albedoMap?: TextureLike;
	metallicRoughnessMap?: TextureLike;
	normalMap?: TextureLike;
	emissiveMap?: TextureLike;
	occlusionMap?: TextureLike;
	clearcoat?: number;
	clearcoatRoughness?: number;
}

export class PBRMaterial extends Material {
	public albedo: RGB;
	public roughness: number;
	public metalness: number;
	public emissive: RGB;
	public emissiveIntensity: number;
	public reflectance: number;
	public metallicRoughnessMap: TextureLike;
	public normalMap: TextureLike;
	public emissiveMap: TextureLike;
	public occlusionMap: TextureLike;
	public clearcoat: number;
	public clearcoatRoughness: number;

	/**
	 * Converts legacy sRGB F0 color to a physical reflectance value.
	 * Clamp minimum to 0.04 (typical dielectric) to prevent non-physical dark spots.
	 */
	private static _reflectanceFromLegacyF0(f0: RGB): number {
		const r = sRGBToLinear(clamp(f0.r / 255, 0, 1));
		const g = sRGBToLinear(clamp(f0.g / 255, 0, 1));
		const b = sRGBToLinear(clamp(f0.b / 255, 0, 1));
		// Luminance based on Rec. 709
		const f0Linear = clamp(0.2126 * r + 0.7152 * g + 0.0722 * b, 0.04, 1);
		return clamp(Math.sqrt(f0Linear / 0.16), 0, 1);
	}

	constructor(params: PBRMaterialParams = {}) {
		super({ ...params, shading: "PBR" });
		this.type = "PBR";

		// Map albedoMap alias if present
		this.map = params.albedoMap || params.map || null;

		this.albedo = params.albedo || { r: 255, g: 255, b: 255 };
		this.roughness = clamp(params.roughness ?? 0.5, 0, 1);
		this.metalness = clamp(params.metalness ?? 0.0, 0, 1);
		this.emissive = params.emissive || { r: 0, g: 0, b: 0 };
		this.emissiveIntensity = params.emissiveIntensity ?? 1.0;

		const reflectance =
			params.reflectance !== undefined ? params.reflectance
			: params.f0 ? PBRMaterial._reflectanceFromLegacyF0(params.f0)
			: 0.5;
		this.reflectance = clamp(reflectance, 0, 1);

		this.metallicRoughnessMap = params.metallicRoughnessMap || null;
		this.normalMap = params.normalMap || null;
		this.emissiveMap = params.emissiveMap || null;
		this.occlusionMap = params.occlusionMap || null;

		this.clearcoat = clamp(params.clearcoat ?? 0.0, 0, 1);
		// Default clearcoatRoughness to 0.01 to avoid infinite specular spikes and aliasing
		this.clearcoatRoughness = clamp(params.clearcoatRoughness ?? 0.01, 0, 1);

		// In PBR, we use reflectance/metalness. Base Material's reflectivity is disabled here.
		this.reflectivity = 0;
	}
}
