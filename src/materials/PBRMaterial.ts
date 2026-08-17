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
	UV3 = 3,
}

/**
 * Stable, append-only feature bits exposed by {@link PBRMaterial.featureMask}.
 */
export enum PBRMaterialFeature {
	BASE_COLOR_MAP = 1 << 0,
	METALLIC_ROUGHNESS_MAP = 1 << 1,
	NORMAL_MAP = 1 << 2,
	OCCLUSION_MAP = 1 << 3,
	SPECULAR = 1 << 4,
	CLEARCOAT = 1 << 5,
	SHEEN = 1 << 6,
	IRIDESCENCE = 1 << 7,
	ANISOTROPY = 1 << 8,
	TRANSMISSION = 1 << 9,
}

/**
 * Stable, append-only texture-presence bits exposed by
 * {@link PBRMaterial.textureMask}.
 */
export enum PBRMaterialTextureFeature {
	BASE_COLOR_MAP = 1 << 0,
	METALLIC_ROUGHNESS_MAP = 1 << 1,
	NORMAL_MAP = 1 << 2,
	EMISSIVE_MAP = 1 << 3,
	OCCLUSION_MAP = 1 << 4,
	SPECULAR_MAP = 1 << 5,
	SPECULAR_COLOR_MAP = 1 << 6,
	CLEARCOAT_MAP = 1 << 7,
	CLEARCOAT_ROUGHNESS_MAP = 1 << 8,
	CLEARCOAT_NORMAL_MAP = 1 << 9,
	SHEEN_COLOR_MAP = 1 << 10,
	SHEEN_ROUGHNESS_MAP = 1 << 11,
	TRANSMISSION_MAP = 1 << 12,
	THICKNESS_MAP = 1 << 13,
	IRIDESCENCE_MAP = 1 << 14,
	IRIDESCENCE_THICKNESS_MAP = 1 << 15,
	ANISOTROPY_MAP = 1 << 16,
}

const PBR_FEATURE_EPSILON = 0.000001;

function normalizeUVChannel(uv: number | undefined): UVChannel {
	if (typeof uv === "number" && Number.isFinite(uv)) {
		const channel = Math.floor(uv);
		if (channel >= UVChannel.UV3) {
			return UVChannel.UV3;
		}
		if (channel >= UVChannel.UV2) {
			return UVChannel.UV2;
		}
		if (channel >= UVChannel.UV1) {
			return UVChannel.UV1;
		}
	}
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
	/** KHR_materials_ior index of refraction. */
	ior?: number;
	/** KHR_materials_specular specular factor in range 0..1. */
	specularFactor?: number;
	/**
	 * KHR_materials_specular linear specular color factor stored in 0..255 units.
	 */
	specularColor?: RGB;
	/** KHR_materials_specular specular factor texture. */
	specularMap?: TextureLike;
	/** KHR_materials_specular specular color texture (RGB). */
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
	/** KHR_materials_clearcoat clearcoat factor in range 0..1. */
	clearcoat?: number;
	/** KHR_materials_clearcoat clearcoat factor texture (red channel). */
	clearcoatMap?: TextureLike;
	/** KHR_materials_clearcoat clearcoat roughness in range 0..1. */
	clearcoatRoughness?: number;
	/** KHR_materials_clearcoat clearcoat roughness texture (green channel). */
	clearcoatRoughnessMap?: TextureLike;
	/** KHR_materials_clearcoat clearcoat normal map texture. */
	clearcoatNormalMap?: TextureLike;
	/** KHR_materials_clearcoat clearcoat normal scale. */
	clearcoatNormalScale?: number;
	/** UV set used by clearcoatMap. */
	clearcoatMapUV?: UVChannel;
	/** UV set used by clearcoatRoughnessMap. */
	clearcoatRoughnessMapUV?: UVChannel;
	/** UV set used by clearcoatNormalMap. */
	clearcoatNormalMapUV?: UVChannel;
	/**
	 * KHR_materials_sheen linear sheen color factor stored in 0..255 units.
	 */
	sheenColorFactor?: RGB;
	/** KHR_materials_sheen sheen color texture (RGB). */
	sheenColorMap?: TextureLike;
	/** KHR_materials_sheen sheen roughness in range 0..1. */
	sheenRoughnessFactor?: number;
	/** KHR_materials_sheen sheen roughness texture (alpha channel). */
	sheenRoughnessMap?: TextureLike;
	/** UV set used by sheenColorMap. */
	sheenColorMapUV?: UVChannel;
	/** UV set used by sheenRoughnessMap. */
	sheenRoughnessMapUV?: UVChannel;
	/** KHR_materials_transmission transmission factor in range 0..1. */
	transmissionFactor?: number;
	/** KHR_materials_transmission transmission texture (red channel). */
	transmissionMap?: TextureLike;
	/** UV set used by transmissionMap. */
	transmissionMapUV?: UVChannel;
	/** KHR_materials_iridescence iridescence factor in range 0..1. */
	iridescenceFactor?: number;
	/** KHR_materials_iridescence iridescence intensity texture (red channel). */
	iridescenceMap?: TextureLike;
	/** UV set used by iridescenceMap. */
	iridescenceMapUV?: UVChannel;
	/** KHR_materials_iridescence index of refraction. */
	iridescenceIor?: number;
	/** KHR_materials_iridescence minimum film thickness in nanometers. */
	iridescenceThicknessMinimum?: number;
	/** KHR_materials_iridescence maximum film thickness in nanometers. */
	iridescenceThicknessMaximum?: number;
	/** KHR_materials_iridescence thickness texture (green channel). */
	iridescenceThicknessMap?: TextureLike;
	/** UV set used by iridescenceThicknessMap. */
	iridescenceThicknessMapUV?: UVChannel;
	/**
	 * KHR_materials_anisotropy strength in the range 0..1.
	 * `anisotropyMap` multiplies this value by its blue channel.
	 */
	anisotropyStrength?: number;
	/**
	 * KHR_materials_anisotropy rotation in radians, counter-clockwise in
	 * tangent/bitangent space.
	 */
	anisotropyRotation?: number;
	/**
	 * KHR_materials_anisotropy direction/strength texture.
	 * Red/green encode direction and blue encodes strength.
	 */
	anisotropyMap?: TextureLike;
	/** UV set used by `anisotropyMap`. */
	anisotropyMapUV?: UVChannel;
	/** KHR_materials_volume thickness factor in world-space meters. */
	thicknessFactor?: number;
	/** KHR_materials_volume thickness texture. */
	thicknessMap?: TextureLike;
	/** UV set used by thicknessMap. */
	thicknessMapUV?: UVChannel;
	/** KHR_materials_volume attenuation distance in world-space meters. */
	attenuationDistance?: number;
	/**
	 * KHR_materials_volume linear attenuation color stored in 0..255 units.
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
	/** KHR_materials_specular specular factor texture. */
	public specularMap: TextureLike;
	/** KHR_materials_specular specular color texture (RGB). */
	public specularColorMap: TextureLike;
	/** UV set used by specularMap. */
	public specularMapUV: UVChannel;
	/** UV set used by specularColorMap. */
	public specularColorMapUV: UVChannel;
	/** KHR_materials_clearcoat clearcoat factor in range 0..1. */
	public clearcoat: number;
	/** KHR_materials_clearcoat clearcoat factor texture (red channel). */
	public clearcoatMap: TextureLike;
	/** KHR_materials_clearcoat clearcoat roughness in range 0..1. */
	public clearcoatRoughness: number;
	/** KHR_materials_clearcoat clearcoat roughness texture (green channel). */
	public clearcoatRoughnessMap: TextureLike;
	/** KHR_materials_clearcoat clearcoat normal map texture. */
	public clearcoatNormalMap: TextureLike;
	/** KHR_materials_clearcoat clearcoat normal scale. */
	public clearcoatNormalScale: number;
	/** UV set used by clearcoatMap. */
	public clearcoatMapUV: UVChannel;
	/** UV set used by clearcoatRoughnessMap. */
	public clearcoatRoughnessMapUV: UVChannel;
	/** UV set used by clearcoatNormalMap. */
	public clearcoatNormalMapUV: UVChannel;

	/** KHR_materials_sheen linear sheen color factor stored in 0..255 units. */
	public sheenColorFactor: RGB;
	/** KHR_materials_sheen sheen color texture (RGB). */
	public sheenColorMap: TextureLike;
	/** KHR_materials_sheen sheen roughness in range 0..1. */
	public sheenRoughnessFactor: number;
	/** KHR_materials_sheen sheen roughness texture (alpha channel). */
	public sheenRoughnessMap: TextureLike;
	/** UV set used by sheenColorMap. */
	public sheenColorMapUV: UVChannel;
	/** UV set used by sheenRoughnessMap. */
	public sheenRoughnessMapUV: UVChannel;

	/** KHR_materials_transmission transmission factor in range 0..1. */
	public transmissionFactor: number;
	/** KHR_materials_transmission transmission texture (red channel). */
	public transmissionMap: TextureLike;
	/** UV set used by transmissionMap. */
	public transmissionMapUV: UVChannel;

	/** KHR_materials_iridescence iridescence factor in range 0..1. */
	public iridescenceFactor: number;
	/** KHR_materials_iridescence iridescence intensity texture (red channel). */
	public iridescenceMap: TextureLike;
	/** UV set used by iridescenceMap. */
	public iridescenceMapUV: UVChannel;
	/** KHR_materials_iridescence index of refraction. */
	public iridescenceIor: number;
	/** KHR_materials_iridescence minimum film thickness in nanometers. */
	public iridescenceThicknessMinimum: number;
	/** KHR_materials_iridescence maximum film thickness in nanometers. */
	public iridescenceThicknessMaximum: number;
	/** KHR_materials_iridescence thickness texture (green channel). */
	public iridescenceThicknessMap: TextureLike;
	/** UV set used by iridescenceThicknessMap. */
	public iridescenceThicknessMapUV: UVChannel;

	/** KHR_materials_anisotropy strength in the range 0..1. */
	public anisotropyStrength: number;
	/** KHR_materials_anisotropy rotation in radians. */
	public anisotropyRotation: number;
	/** KHR_materials_anisotropy direction/strength texture. */
	public anisotropyMap: TextureLike;
	/** UV set used by `anisotropyMap`. */
	public anisotropyMapUV: UVChannel;

	/** KHR_materials_volume thickness factor in world-space meters. */
	public thicknessFactor: number;
	/** KHR_materials_volume thickness texture. */
	public thicknessMap: TextureLike;
	/** UV set used by thicknessMap. */
	public thicknessMapUV: UVChannel;
	/** KHR_materials_volume attenuation distance in world-space meters. */
	public attenuationDistance: number;
	/** KHR_materials_volume linear attenuation color stored in 0..255 units. */
	public attenuationColor: RGB;

	/**
	 * Returns the current resolved PBR feature mask.
	 *
	 * The value is computed on access so direct factor or texture mutations are
	 * reflected without an explicit update call.
	 */
	public get featureMask(): number {
		const textureMask = this.textureMask;
		let mask = 0;
		if (textureMask & PBRMaterialTextureFeature.BASE_COLOR_MAP) {
			mask |= PBRMaterialFeature.BASE_COLOR_MAP;
		}
		if (textureMask & PBRMaterialTextureFeature.METALLIC_ROUGHNESS_MAP) {
			mask |= PBRMaterialFeature.METALLIC_ROUGHNESS_MAP;
		}
		if (textureMask & PBRMaterialTextureFeature.NORMAL_MAP) {
			mask |= PBRMaterialFeature.NORMAL_MAP;
		}
		if (textureMask & PBRMaterialTextureFeature.OCCLUSION_MAP) {
			mask |= PBRMaterialFeature.OCCLUSION_MAP;
		}

		const specularFactor = clamp(this.specularFactor, 0, 1);
		const specularR = clamp(this.specularColor.r, 0, 255) / 255;
		const specularG = clamp(this.specularColor.g, 0, 255) / 255;
		const specularB = clamp(this.specularColor.b, 0, 255) / 255;
		const hasSpecularCustomization =
			Math.abs(specularFactor - 1) > PBR_FEATURE_EPSILON ||
			Math.abs(specularR - 1) > PBR_FEATURE_EPSILON ||
			Math.abs(specularG - 1) > PBR_FEATURE_EPSILON ||
			Math.abs(specularB - 1) > PBR_FEATURE_EPSILON ||
			(
				specularFactor > PBR_FEATURE_EPSILON &&
				!!(textureMask & PBRMaterialTextureFeature.SPECULAR_MAP)
			) ||
			(
				specularFactor > PBR_FEATURE_EPSILON &&
				Math.max(specularR, specularG, specularB) > PBR_FEATURE_EPSILON &&
				!!(textureMask & PBRMaterialTextureFeature.SPECULAR_COLOR_MAP)
			);
		if (hasSpecularCustomization) {
			mask |= PBRMaterialFeature.SPECULAR;
		}

		if (clamp(this.clearcoat, 0, 1) > PBR_FEATURE_EPSILON) {
			mask |= PBRMaterialFeature.CLEARCOAT;
		}
		if (
			Math.max(
				clamp(this.sheenColorFactor.r, 0, 255),
				clamp(this.sheenColorFactor.g, 0, 255),
				clamp(this.sheenColorFactor.b, 0, 255)
			) / 255 > PBR_FEATURE_EPSILON
		) {
			mask |= PBRMaterialFeature.SHEEN;
		}
		if (clamp(this.iridescenceFactor, 0, 1) > PBR_FEATURE_EPSILON) {
			mask |= PBRMaterialFeature.IRIDESCENCE;
		}
		if (clamp(this.anisotropyStrength, 0, 1) > PBR_FEATURE_EPSILON) {
			mask |= PBRMaterialFeature.ANISOTROPY;
		}
		if (clamp(this.transmissionFactor, 0, 1) > PBR_FEATURE_EPSILON) {
			mask |= PBRMaterialFeature.TRANSMISSION;
		}
		return mask >>> 0;
	}

	/**
	 * Returns the current PBR texture-presence mask.
	 *
	 * Presence is independent of whether the texture's parent lobe is active.
	 */
	public get textureMask(): number {
		let mask = 0;
		if (this.map) mask |= PBRMaterialTextureFeature.BASE_COLOR_MAP;
		if (this.metallicRoughnessMap) {
			mask |= PBRMaterialTextureFeature.METALLIC_ROUGHNESS_MAP;
		}
		if (this.normalMap) mask |= PBRMaterialTextureFeature.NORMAL_MAP;
		if (this.emissiveMap) mask |= PBRMaterialTextureFeature.EMISSIVE_MAP;
		if (this.occlusionMap) mask |= PBRMaterialTextureFeature.OCCLUSION_MAP;
		if (this.specularMap) mask |= PBRMaterialTextureFeature.SPECULAR_MAP;
		if (this.specularColorMap) {
			mask |= PBRMaterialTextureFeature.SPECULAR_COLOR_MAP;
		}
		if (this.clearcoatMap) mask |= PBRMaterialTextureFeature.CLEARCOAT_MAP;
		if (this.clearcoatRoughnessMap) {
			mask |= PBRMaterialTextureFeature.CLEARCOAT_ROUGHNESS_MAP;
		}
		if (this.clearcoatNormalMap) {
			mask |= PBRMaterialTextureFeature.CLEARCOAT_NORMAL_MAP;
		}
		if (this.sheenColorMap) {
			mask |= PBRMaterialTextureFeature.SHEEN_COLOR_MAP;
		}
		if (this.sheenRoughnessMap) {
			mask |= PBRMaterialTextureFeature.SHEEN_ROUGHNESS_MAP;
		}
		if (this.transmissionMap) {
			mask |= PBRMaterialTextureFeature.TRANSMISSION_MAP;
		}
		if (this.thicknessMap) mask |= PBRMaterialTextureFeature.THICKNESS_MAP;
		if (this.iridescenceMap) {
			mask |= PBRMaterialTextureFeature.IRIDESCENCE_MAP;
		}
		if (this.iridescenceThicknessMap) {
			mask |= PBRMaterialTextureFeature.IRIDESCENCE_THICKNESS_MAP;
		}
		if (this.anisotropyMap) {
			mask |= PBRMaterialTextureFeature.ANISOTROPY_MAP;
		}
		return mask >>> 0;
	}

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

		this.iridescenceFactor = clamp(params.iridescenceFactor ?? 0.0, 0, 1);
		this.iridescenceMap = params.iridescenceMap || null;
		this.iridescenceMapUV = normalizeUVChannel(params.iridescenceMapUV);
		this.iridescenceIor = Math.max(params.iridescenceIor ?? 1.3, 1.0);
		this.iridescenceThicknessMinimum = Math.max(
			params.iridescenceThicknessMinimum ?? 100.0,
			0
		);
		this.iridescenceThicknessMaximum = Math.max(
			params.iridescenceThicknessMaximum ?? 400.0,
			0
		);
		this.iridescenceThicknessMap = params.iridescenceThicknessMap || null;
		this.iridescenceThicknessMapUV = normalizeUVChannel(
			params.iridescenceThicknessMapUV
		);

		this.anisotropyStrength = clamp(params.anisotropyStrength ?? 0.0, 0, 1);
		this.anisotropyRotation =
			Number.isFinite(params.anisotropyRotation) ?
				params.anisotropyRotation!
			:	0.0;
		this.anisotropyMap = params.anisotropyMap || null;
		this.anisotropyMapUV = normalizeUVChannel(params.anisotropyMapUV);

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
