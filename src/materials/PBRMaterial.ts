import type { RGB } from "../foundation/Color";
import {
	Material,
	type MaterialParams,
	type TextureLike,
	ShadingModel,
} from "./Material";
import { clamp, sRGBToLinear } from "../maths/Common";

export enum UVChannel {
	UV0 = 0,
	UV1 = 1,
	UV2 = 2,
}

function normalizeUVChannel(uv: number | undefined): UVChannel {
	if (uv === UVChannel.UV1) return UVChannel.UV1;
	if (uv === UVChannel.UV2) return UVChannel.UV2;
	return UVChannel.UV0;
}

export interface PBRMaterialParams extends MaterialParams {
	/**
	 * Linear albedo factor stored in 0..255 units.
	 * Color textures apply their own color-space decode before modulation.
	 */
	albedo?: RGB;
	roughness?: number;
	metalness?: number;
	/**
	 * Linear emissive factor stored in 0..255 units.
	 */
	emissive?: RGB;
	emissiveIntensity?: number;
	ior?: number;
	specularFactor?: number;
	/**
	 * Linear specular color factor stored in 0..255 units.
	 */
	specularColor?: RGB;
	specularMap?: TextureLike;
	specularColorMap?: TextureLike;
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
	normalScale?: number;
	emissiveMap?: TextureLike;
	occlusionMap?: TextureLike;
	occlusionStrength?: number;
	albedoMapUV?: UVChannel;
	metallicRoughnessMapUV?: UVChannel;
	normalMapUV?: UVChannel;
	emissiveMapUV?: UVChannel;
	occlusionMapUV?: UVChannel;
	specularMapUV?: UVChannel;
	specularColorMapUV?: UVChannel;
	clearcoat?: number;
	clearcoatMap?: TextureLike;
	clearcoatRoughness?: number;
	clearcoatRoughnessMap?: TextureLike;
	clearcoatNormalMap?: TextureLike;
	clearcoatNormalScale?: number;
	clearcoatMapUV?: UVChannel;
	clearcoatRoughnessMapUV?: UVChannel;
	clearcoatNormalMapUV?: UVChannel;
	/**
	 * Linear sheen color factor stored in 0..255 units.
	 */
	sheenColorFactor?: RGB;
	sheenColorMap?: TextureLike;
	sheenRoughnessFactor?: number;
	sheenRoughnessMap?: TextureLike;
	sheenColorMapUV?: UVChannel;
	sheenRoughnessMapUV?: UVChannel;
	transmissionFactor?: number;
	transmissionMap?: TextureLike;
	transmissionMapUV?: UVChannel;
	thicknessFactor?: number;
	thicknessMap?: TextureLike;
	thicknessMapUV?: UVChannel;
	attenuationDistance?: number;
	/**
	 * Linear volume attenuation color stored in 0..255 units.
	 */
	attenuationColor?: RGB;
}

export class PBRMaterial extends Material {
	public albedo: RGB;
	public roughness: number;
	public metalness: number;
	public emissive: RGB;
	public emissiveIntensity: number;
	private _ior?: number;
	public specularFactor: number;
	public specularColor: RGB;
	public reflectance: number;
	public metallicRoughnessMap: TextureLike;
	public normalMap: TextureLike;
	public normalScale: number;
	public emissiveMap: TextureLike;
	public occlusionMap: TextureLike;
	public occlusionStrength: number;
	public albedoMapUV: UVChannel;
	public metallicRoughnessMapUV: UVChannel;
	public normalMapUV: UVChannel;
	public emissiveMapUV: UVChannel;
	public occlusionMapUV: UVChannel;
	public specularMap: TextureLike;
	public specularColorMap: TextureLike;
	public specularMapUV: UVChannel;
	public specularColorMapUV: UVChannel;
	public clearcoat: number;
	public clearcoatMap: TextureLike;
	public clearcoatRoughness: number;
	public clearcoatRoughnessMap: TextureLike;
	public clearcoatNormalMap: TextureLike;
	public clearcoatNormalScale: number;
	public clearcoatMapUV: UVChannel;
	public clearcoatRoughnessMapUV: UVChannel;
	public clearcoatNormalMapUV: UVChannel;

	public sheenColorFactor: RGB;
	public sheenColorMap: TextureLike;
	public sheenRoughnessFactor: number;
	public sheenRoughnessMap: TextureLike;
	public sheenColorMapUV: UVChannel;
	public sheenRoughnessMapUV: UVChannel;

	public transmissionFactor: number;
	public transmissionMap: TextureLike;
	public transmissionMapUV: UVChannel;

	public thicknessFactor: number;
	public thicknessMap: TextureLike;
	public thicknessMapUV: UVChannel;
	public attenuationDistance: number;
	public attenuationColor: RGB;

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

	private static _reflectanceFromIor(ior: number): number {
		const safeIor = Math.max(1.0, ior);
		const f0 = Math.pow((safeIor - 1) / (safeIor + 1), 2);
		return clamp(Math.sqrt(f0 / 0.16), 0, 1);
	}

	public get ior(): number | undefined {
		return this._ior;
	}

	public set ior(value: number | undefined) {
		if (value === undefined) {
			this._ior = undefined;
			return;
		}
		const safeIor = Math.max(1.0, value);
		this._ior = safeIor;
		this.reflectance = PBRMaterial._reflectanceFromIor(safeIor);
	}

	constructor(params: PBRMaterialParams = {}) {
		super({ ...params, shading: ShadingModel.PBR });
		this.type = "PBR";

		// Map albedoMap alias if present
		this.map = params.albedoMap || params.map || null;

		this.albedo = params.albedo || { r: 255, g: 255, b: 255 };
		this.roughness = clamp(params.roughness ?? 0.5, 0, 1);
		this.metalness = clamp(params.metalness ?? 0.0, 0, 1);
		this.emissive = params.emissive || { r: 0, g: 0, b: 0 };
		this.emissiveIntensity = params.emissiveIntensity ?? 1.0;

		this.specularFactor = params.specularFactor ?? 1.0;
		this.specularColor = params.specularColor || { r: 255, g: 255, b: 255 };

		let reflectance = 0.5;
		if (params.ior !== undefined) {
			reflectance = PBRMaterial._reflectanceFromIor(params.ior);
		} else if (params.reflectance !== undefined) {
			reflectance = params.reflectance;
		} else if (params.f0) {
			reflectance = PBRMaterial._reflectanceFromLegacyF0(params.f0);
		}

		this.reflectance = clamp(reflectance, 0, 1);
		this._ior = undefined;
		this.ior = params.ior;

		this.metallicRoughnessMap = params.metallicRoughnessMap || null;
		this.normalMap = params.normalMap || null;
		this.normalScale = params.normalScale ?? 1.0;
		this.emissiveMap = params.emissiveMap || null;
		this.occlusionMap = params.occlusionMap || null;
		this.occlusionStrength = params.occlusionStrength ?? 1.0;
		this.albedoMapUV = normalizeUVChannel(params.albedoMapUV);
		this.metallicRoughnessMapUV = normalizeUVChannel(
			params.metallicRoughnessMapUV
		);
		this.normalMapUV = normalizeUVChannel(params.normalMapUV);
		this.emissiveMapUV = normalizeUVChannel(params.emissiveMapUV);
		this.occlusionMapUV = normalizeUVChannel(params.occlusionMapUV);
		this.specularMap = params.specularMap || null;
		this.specularMapUV = normalizeUVChannel(params.specularMapUV);
		this.specularColorMap = params.specularColorMap || null;
		this.specularColorMapUV = normalizeUVChannel(params.specularColorMapUV);

		this.clearcoat = clamp(params.clearcoat ?? 0.0, 0, 1);
		this.clearcoatMap = params.clearcoatMap || null;
		this.clearcoatMapUV = normalizeUVChannel(params.clearcoatMapUV);
		// Default clearcoatRoughness to 0.01 to avoid infinite specular spikes and aliasing
		this.clearcoatRoughness = clamp(params.clearcoatRoughness ?? 0.01, 0, 1);
		this.clearcoatRoughnessMap = params.clearcoatRoughnessMap || null;
		this.clearcoatRoughnessMapUV = normalizeUVChannel(
			params.clearcoatRoughnessMapUV
		);
		this.clearcoatNormalMap = params.clearcoatNormalMap || null;
		this.clearcoatNormalMapUV = normalizeUVChannel(params.clearcoatNormalMapUV);
		this.clearcoatNormalScale = params.clearcoatNormalScale ?? 1.0;

		this.sheenColorFactor = params.sheenColorFactor || { r: 0, g: 0, b: 0 };
		this.sheenColorMap = params.sheenColorMap || null;
		this.sheenRoughnessFactor = clamp(params.sheenRoughnessFactor ?? 0.0, 0, 1);
		this.sheenRoughnessMap = params.sheenRoughnessMap || null;
		this.sheenColorMapUV = normalizeUVChannel(params.sheenColorMapUV);
		this.sheenRoughnessMapUV = normalizeUVChannel(params.sheenRoughnessMapUV);

		this.transmissionFactor = clamp(params.transmissionFactor ?? 0.0, 0, 1);
		this.transmissionMap = params.transmissionMap || null;
		this.transmissionMapUV = normalizeUVChannel(params.transmissionMapUV);

		this.thicknessFactor = Math.max(params.thicknessFactor ?? 0.0, 0);
		this.thicknessMap = params.thicknessMap || null;
		this.thicknessMapUV = normalizeUVChannel(params.thicknessMapUV);
		this.attenuationDistance = params.attenuationDistance ?? Infinity;
		this.attenuationColor = params.attenuationColor || {
			r: 255,
			g: 255,
			b: 255,
		};

		// In PBR, we use reflectance/metalness. Base Material's reflectivity is disabled here.
		this.reflectivity = 0;
	}
}
