import type { Vec3Tuple } from "../../maths/Vector3";
import type { Vec4Tuple } from "../../maths/Vector4";
import type { RGB } from "../../foundation/Color";
import { clamp, sRGBToLinear } from "../../maths/Common";
import {
	AlphaMode,
	ShadingModel,
	type Material,
	type TextureLike,
} from "../../materials/Material";
import {
	PBRMaterial,
	PBRMaterialFeature,
	PBRMaterialTextureFeature,
} from "../../materials/PBRMaterial";
import { getMaterialTransmissionFactor } from "../../materials/transparency";
import { resolveTextureUVTransform } from "./WebGLMaterialUniformResolver";

export type WebGLMaterialShadingFamily = "pbr" | "phong" | "flat" | "unlit";

export interface WebGLMaterialTextureState {
	readonly texture: TextureLike;
	readonly uv: 0 | 1 | 2 | 3;
	readonly transformA: Readonly<Vec4Tuple>;
	readonly transformB: Readonly<Vec4Tuple>;
}

export interface WebGLMaterialCommonState {
	readonly baseColor: Readonly<Vec4Tuple>;
	readonly emissive: Readonly<Vec4Tuple>;
	readonly alpha: Readonly<Vec4Tuple>;
	readonly renderParams: Readonly<Vec4Tuple>;
	readonly baseMap: WebGLMaterialTextureState;
	readonly emissiveMap: WebGLMaterialTextureState;
}

export const WEBGL_PBR_TEXTURE_SLOT_NAMES = [
	"metallicRoughnessMap",
	"specularMap",
	"specularColorMap",
	"clearcoatMap",
	"clearcoatRoughnessMap",
	"clearcoatNormalMap",
	"sheenColorMap",
	"sheenRoughnessMap",
	"transmissionMap",
	"thicknessMap",
	"normalMap",
	"occlusionMap",
	"iridescenceMap",
	"iridescenceThicknessMap",
	"anisotropyMap",
] as const;

export type WebGLPBRTextureSlotName = typeof WEBGL_PBR_TEXTURE_SLOT_NAMES[number];

export interface WebGLPBRMaterialState {
	readonly pbr: Readonly<Vec4Tuple>;
	readonly specular: Readonly<Vec4Tuple>;
	readonly transmissionVolume: Readonly<Vec4Tuple>;
	readonly clearcoat: Readonly<Vec4Tuple>;
	readonly sheen: Readonly<Vec4Tuple>;
	readonly iridescence: Readonly<Vec4Tuple>;
	readonly attenuationColor: Readonly<Vec4Tuple>;
	readonly anisotropy: Readonly<Vec4Tuple>;
	readonly scales: Readonly<Vec4Tuple>;
	readonly featureMask: number;
	readonly textureMask: number;
	readonly textures: Readonly<Record<WebGLPBRTextureSlotName, WebGLMaterialTextureState>>;
}

export interface WebGLPhongMaterialState {
	readonly specular: Readonly<Vec4Tuple>;
	readonly phong: Readonly<Vec4Tuple>;
	readonly phongAmbient: Readonly<Vec4Tuple>;
}

interface WebGLResolvedMaterialStateBase {
	readonly common: WebGLMaterialCommonState;
}

export type WebGLResolvedMaterialState =
	| (WebGLResolvedMaterialStateBase & {
			readonly shadingFamily: "pbr";
			readonly lighting: WebGLPBRMaterialState;
	  })
	| (WebGLResolvedMaterialStateBase & {
			readonly shadingFamily: "phong" | "flat";
			readonly lighting: WebGLPhongMaterialState;
	  })
	| (WebGLResolvedMaterialStateBase & {
			readonly shadingFamily: "unlit";
			readonly lighting: null;
	  });

type WebGLPBRMaterial = PBRMaterial & {
	readonly specularColorFactor?: RGB;
};

type WebGLLegacyMaterial = Material & {
	readonly diffuse?: RGB;
	readonly specular?: RGB;
	readonly ambient?: RGB;
	readonly emissive?: RGB;
	readonly emissiveIntensity?: number;
	readonly shininess?: number;
};

/** @internal Resolves immutable built-in WebGL material authoring state. */
export function resolveWebGLMaterialState(material: Material): WebGLResolvedMaterialState {
	const shadingFamily = resolveWebGLMaterialShadingFamily(material);
	const isPBR = shadingFamily === "pbr";
	const common = resolveCommonState(material, isPBR);
	if (shadingFamily === "pbr") {
		return {
			shadingFamily,
			common,
			lighting: resolvePBRState(material as WebGLPBRMaterial),
		};
	}
	if (shadingFamily === "phong" || shadingFamily === "flat") {
		return {
			shadingFamily,
			common,
			lighting: resolvePhongState(material as WebGLLegacyMaterial),
		};
	}
	return { shadingFamily, common, lighting: null };
}

export function resolveWebGLMaterialShadingFamily(
	material: Material,
): WebGLMaterialShadingFamily {
	if (material.shading === ShadingModel.PBR || material.type === "PBR") return "pbr";
	if (material.shading === ShadingModel.Unlit) return "unlit";
	if (material.shading === ShadingModel.Flat) return "flat";
	return "phong";
}

function resolveCommonState(material: Material, isPBR: boolean): WebGLMaterialCommonState {
	let baseColor: Vec3Tuple = [1, 1, 1];
	let emissive: Vec3Tuple = [0, 0, 0];
	let emissiveIntensity = 1;
	let baseMapUV: 0 | 1 | 2 | 3 = 0;
	let emissiveMap: TextureLike = null;
	let emissiveMapUV: 0 | 1 | 2 | 3 = 0;
	if (isPBR) {
		const pbr = material as WebGLPBRMaterial;
		baseColor = normalizedColor(pbr.albedo ?? { r: 255, g: 255, b: 255 });
		emissive = normalizedColor(pbr.emissive ?? { r: 0, g: 0, b: 0 });
		emissiveIntensity = clamp(pbr.emissiveIntensity ?? 1, 0, 64);
		baseMapUV = resolveUVSet(pbr.albedoMapUV);
		emissiveMap = pbr.emissiveMap ?? null;
		emissiveMapUV = resolveUVSet(pbr.emissiveMapUV);
	} else {
		const legacy = material as WebGLLegacyMaterial;
		baseColor = linearLegacyColor(legacy.diffuse ?? { r: 255, g: 255, b: 255 });
		if (legacy.emissive) {
			emissive = linearLegacyColor(legacy.emissive);
			emissiveIntensity = clamp(legacy.emissiveIntensity ?? 1, 0, 64);
		}
	}
	const opacity = clamp(material.opacity ?? 1, 0, 1);
	const alphaModeMask = material.alphaMode === AlphaMode.Mask ? 1 : 0;
	return {
		baseColor: [baseColor[0], baseColor[1], baseColor[2], opacity],
		emissive: [
			emissive[0] * emissiveIntensity,
			emissive[1] * emissiveIntensity,
			emissive[2] * emissiveIntensity,
			1,
		],
		alpha: [
			clamp(material.alphaCutoff ?? 0.5, 0, 1),
			alphaModeMask,
			material.alphaMode === AlphaMode.Blend ||
			getMaterialTransmissionFactor(material) > 0 ? 1 : 0,
			0,
		],
		renderParams: [material.doubleSided || material.cullMode === "none" ? 1 : 0, 0, 0, 0],
		baseMap: resolveTextureState(material.map ?? null, baseMapUV),
		emissiveMap: resolveTextureState(emissiveMap, emissiveMapUV),
	};
}

function resolvePBRState(material: WebGLPBRMaterial): WebGLPBRMaterialState {
	const specularColor = normalizedColor(
		material.specularColor ?? material.specularColorFactor ?? { r: 255, g: 255, b: 255 },
	);
	const sheenColor = normalizedColor(
		material.sheenColorFactor ?? { r: 0, g: 0, b: 0 },
	);
	const attenuationColor = normalizedColor(
		material.attenuationColor ?? { r: 255, g: 255, b: 255 },
	);
	const anisotropyRotation = Number.isFinite(material.anisotropyRotation) ?
		material.anisotropyRotation : 0;
	const textures = {
		metallicRoughnessMap: resolveTextureState(material.metallicRoughnessMap, material.metallicRoughnessMapUV),
		specularMap: resolveTextureState(material.specularMap, material.specularMapUV),
		specularColorMap: resolveTextureState(material.specularColorMap, material.specularColorMapUV),
		clearcoatMap: resolveTextureState(material.clearcoatMap, material.clearcoatMapUV),
		clearcoatRoughnessMap: resolveTextureState(material.clearcoatRoughnessMap, material.clearcoatRoughnessMapUV),
		clearcoatNormalMap: resolveTextureState(material.clearcoatNormalMap, material.clearcoatNormalMapUV),
		sheenColorMap: resolveTextureState(material.sheenColorMap, material.sheenColorMapUV),
		sheenRoughnessMap: resolveTextureState(material.sheenRoughnessMap, material.sheenRoughnessMapUV),
		transmissionMap: resolveTextureState(material.transmissionMap, material.transmissionMapUV),
		thicknessMap: resolveTextureState(material.thicknessMap, material.thicknessMapUV),
		normalMap: resolveTextureState(material.normalMap, material.normalMapUV),
		occlusionMap: resolveTextureState(material.occlusionMap, material.occlusionMapUV),
		iridescenceMap: resolveTextureState(material.iridescenceMap, material.iridescenceMapUV),
		iridescenceThicknessMap: resolveTextureState(
			material.iridescenceThicknessMap,
			material.iridescenceThicknessMapUV,
		),
		anisotropyMap: resolveTextureState(material.anisotropyMap, material.anisotropyMapUV),
	} satisfies Record<WebGLPBRTextureSlotName, WebGLMaterialTextureState>;
	return {
		pbr: [
			clamp(material.roughness ?? 0.5, 0.04, 1),
			clamp(material.metalness ?? 0, 0, 1),
			clamp(material.reflectance ?? 0.5, 0, 1),
			getMaterialTransmissionFactor(material),
		],
		specular: [specularColor[0], specularColor[1], specularColor[2], clamp(material.specularFactor ?? 1, 0, 1)],
		transmissionVolume: [
			Math.max(1, material.ior ?? 1.5),
			Math.max(0, material.thicknessFactor ?? 0),
			Number.isFinite(material.attenuationDistance) ? Math.max(material.attenuationDistance, 0) : -1,
			0,
		],
		clearcoat: [
			clamp(material.clearcoat ?? 0, 0, 1),
			clamp(material.clearcoatRoughness ?? 0.01, 0.01, 1),
			Math.max(0, material.clearcoatNormalScale ?? 1),
			0,
		],
		sheen: [sheenColor[0], sheenColor[1], sheenColor[2], clamp(material.sheenRoughnessFactor ?? 0, 0, 1)],
		iridescence: [
			clamp(material.iridescenceFactor ?? 0, 0, 1),
			Math.max(1, material.iridescenceIor ?? 1.3),
			Math.max(material.iridescenceThicknessMinimum ?? 100, 0),
			Math.max(material.iridescenceThicknessMaximum ?? 400, 0),
		],
		attenuationColor: [attenuationColor[0], attenuationColor[1], attenuationColor[2], 1],
		anisotropy: [
			clamp(material.anisotropyStrength ?? 0, 0, 1),
			Math.cos(anisotropyRotation),
			Math.sin(anisotropyRotation),
			0,
		],
		scales: [
			Math.max(0, material.normalScale ?? 1),
			clamp(material.occlusionStrength ?? 1, 0, 1),
			0,
			0,
		],
		featureMask: material instanceof PBRMaterial ? material.featureMask : resolveCompatibilityFeatureMask(material),
		textureMask: material instanceof PBRMaterial ? material.textureMask : resolveCompatibilityTextureMask(material),
		textures,
	};
}

function resolvePhongState(material: WebGLLegacyMaterial): WebGLPhongMaterialState {
	const specular = linearLegacyColor(material.specular ?? { r: 56, g: 56, b: 56 });
	const ambient = linearLegacyColor(material.ambient ?? { r: 0, g: 0, b: 0 });
	return {
		specular: [specular[0], specular[1], specular[2], 1],
		phong: [Math.max(0, material.shininess ?? 32), 0, 0, 0],
		phongAmbient: [ambient[0], ambient[1], ambient[2], 0],
	};
}

function resolveTextureState(
	texture: TextureLike,
	uvValue: unknown,
): WebGLMaterialTextureState {
	const transform = resolveTextureUVTransform(texture);
	const uv = resolveUVSet(uvValue);
	return {
		texture,
		uv,
		transformA: [transform.repeatX, transform.repeatY, transform.offsetX, transform.offsetY],
		transformB: [transform.cosRotation, transform.sinRotation, uv, 0],
	};
}

function resolveUVSet(value: unknown): 0 | 1 | 2 | 3 {
	if (typeof value !== "number" || !Number.isFinite(value)) return 0;
	return Math.max(0, Math.min(3, Math.floor(value))) as 0 | 1 | 2 | 3;
}

function normalizedColor(color: RGB): Vec3Tuple {
	return [
		clamp((color.r ?? 0) / 255, 0, 1),
		clamp((color.g ?? 0) / 255, 0, 1),
		clamp((color.b ?? 0) / 255, 0, 1),
	];
}

function linearLegacyColor(color: RGB): Vec3Tuple {
	const normalized = normalizedColor(color);
	return [sRGBToLinear(normalized[0]), sRGBToLinear(normalized[1]), sRGBToLinear(normalized[2])];
}

function resolveCompatibilityTextureMask(material: WebGLPBRMaterial): number {
	let mask = 0;
	const entries: readonly [TextureLike, PBRMaterialTextureFeature][] = [
		[material.map, PBRMaterialTextureFeature.BASE_COLOR_MAP],
		[material.metallicRoughnessMap, PBRMaterialTextureFeature.METALLIC_ROUGHNESS_MAP],
		[material.normalMap, PBRMaterialTextureFeature.NORMAL_MAP],
		[material.emissiveMap, PBRMaterialTextureFeature.EMISSIVE_MAP],
		[material.occlusionMap, PBRMaterialTextureFeature.OCCLUSION_MAP],
		[material.specularMap, PBRMaterialTextureFeature.SPECULAR_MAP],
		[material.specularColorMap, PBRMaterialTextureFeature.SPECULAR_COLOR_MAP],
		[material.clearcoatMap, PBRMaterialTextureFeature.CLEARCOAT_MAP],
		[material.clearcoatRoughnessMap, PBRMaterialTextureFeature.CLEARCOAT_ROUGHNESS_MAP],
		[material.clearcoatNormalMap, PBRMaterialTextureFeature.CLEARCOAT_NORMAL_MAP],
		[material.sheenColorMap, PBRMaterialTextureFeature.SHEEN_COLOR_MAP],
		[material.sheenRoughnessMap, PBRMaterialTextureFeature.SHEEN_ROUGHNESS_MAP],
		[material.transmissionMap, PBRMaterialTextureFeature.TRANSMISSION_MAP],
		[material.thicknessMap, PBRMaterialTextureFeature.THICKNESS_MAP],
		[material.iridescenceMap, PBRMaterialTextureFeature.IRIDESCENCE_MAP],
		[material.iridescenceThicknessMap, PBRMaterialTextureFeature.IRIDESCENCE_THICKNESS_MAP],
		[material.anisotropyMap, PBRMaterialTextureFeature.ANISOTROPY_MAP],
	];
	for (const [texture, bit] of entries) if (texture) mask |= bit;
	return mask;
}

function resolveCompatibilityFeatureMask(material: WebGLPBRMaterial): number {
	const textureMask = resolveCompatibilityTextureMask(material);
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
	if ((material.clearcoat ?? 0) > 0 || material.clearcoatMap) mask |= PBRMaterialFeature.CLEARCOAT;
	if (material.sheenColorMap || material.sheenRoughnessMap) mask |= PBRMaterialFeature.SHEEN;
	if (getMaterialTransmissionFactor(material) > 0 || material.transmissionMap) mask |= PBRMaterialFeature.TRANSMISSION;
	if ((material.iridescenceFactor ?? 0) > 0 || material.iridescenceMap) mask |= PBRMaterialFeature.IRIDESCENCE;
	if ((material.anisotropyStrength ?? 0) > 0 || material.anisotropyMap) mask |= PBRMaterialFeature.ANISOTROPY;
	if (material.specularMap || material.specularColorMap) mask |= PBRMaterialFeature.SPECULAR;
	return mask;
}
