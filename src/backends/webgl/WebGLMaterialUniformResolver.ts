import { sRGBToLinear, clamp } from "../../maths/Common";
import {
	AlphaMode,
	ShadingModel,
	type Material,
	type TextureLike,
} from "../../materials/Material";
import type { RGB } from "../../foundation/Color";
import type { PBRMaterial } from "../../materials/PBRMaterial";
import { getMaterialTransmissionFactor } from "../../materials/transparency";

type WebGLPBRMaterial = PBRMaterial & {
	/** Compatibility alias accepted from older material loaders. */
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

function isWebGLPBRMaterial(material: Material): material is WebGLPBRMaterial {
	return material.shading === ShadingModel.PBR || material.type === "PBR";
}

function isWebGLLegacyMaterial(material: Material): material is WebGLLegacyMaterial {
	return !isWebGLPBRMaterial(material);
}

export interface MaterialUniformState {
	shadingModel: number;
	baseColor: [number, number, number, number];
	emissive: [number, number, number];
	pbr: [number, number, number, number];
	specular: [number, number, number, number];
	transmissionVolume: [number, number, number, number];
	clearcoat: [number, number, number, number];
	sheen: [number, number, number, number];
	iridescence: [number, number, number, number];
	attenuationColor: [number, number, number, number];
	anisotropy: [number, number, number, number];
	phong: [number, number, number, number];
	phongAmbient: [number, number, number, number];
	alpha: [number, number, number, number];
	baseMap: TextureLike;
	baseMapUV: 0 | 1 | 2 | 3;
	metallicRoughnessMap: TextureLike;
	metallicRoughnessMapUV: 0 | 1 | 2 | 3;
	specularMap: TextureLike;
	specularMapUV: 0 | 1 | 2 | 3;
	specularColorMap: TextureLike;
	specularColorMapUV: 0 | 1 | 2 | 3;
	clearcoatMap: TextureLike;
	clearcoatMapUV: 0 | 1 | 2 | 3;
	clearcoatRoughnessMap: TextureLike;
	clearcoatRoughnessMapUV: 0 | 1 | 2 | 3;
	clearcoatNormalMap: TextureLike;
	clearcoatNormalMapUV: 0 | 1 | 2 | 3;
	clearcoatNormalScale: number;
	sheenColorMap: TextureLike;
	sheenColorMapUV: 0 | 1 | 2 | 3;
	sheenRoughnessMap: TextureLike;
	sheenRoughnessMapUV: 0 | 1 | 2 | 3;
	transmissionMap: TextureLike;
	transmissionMapUV: 0 | 1 | 2 | 3;
	thicknessMap: TextureLike;
	thicknessMapUV: 0 | 1 | 2 | 3;
	normalMap: TextureLike;
	normalMapUV: 0 | 1 | 2 | 3;
	normalScale: number;
	emissiveMap: TextureLike;
	emissiveMapUV: 0 | 1 | 2 | 3;
	occlusionMap: TextureLike;
	occlusionMapUV: 0 | 1 | 2 | 3;
	occlusionStrength: number;
	iridescenceMap: TextureLike;
	iridescenceMapUV: 0 | 1 | 2 | 3;
	iridescenceThicknessMap: TextureLike;
	iridescenceThicknessMapUV: 0 | 1 | 2 | 3;
	anisotropyMap: TextureLike;
	anisotropyMapUV: 0 | 1 | 2 | 3;
}

export function resolveMaterialUniforms(material: Material): MaterialUniformState {
	const isPBR =
		material.shading === ShadingModel.PBR || material.type === "PBR";
	const isUnlit = material.shading === ShadingModel.Unlit;
	const resolveUVSet = (value: unknown): 0 | 1 | 2 | 3 => {
		if (typeof value !== "number" || !Number.isFinite(value)) {
			return 0;
		}
		return Math.max(0, Math.min(3, Math.floor(value))) as 0 | 1 | 2 | 3;
	};

	let baseColor: [number, number, number] = [1, 1, 1];
	let emissive: [number, number, number] = [0, 0, 0];
	let roughness = 0.5;
	let metalness = 0;
	let reflectance = 0.5;
	let specularFactor = 1;
	let specularColor: [number, number, number] = [1, 1, 1];
	let transmission = 0;
	let clearcoat = 0;
	let clearcoatRoughness = 0.01;
	let clearcoatNormalScale = 1;
	let sheenColor: [number, number, number] = [0, 0, 0];
	let sheenRoughness = 0;
	let ior = 1.5;
	let thickness = 0;
	let iridescenceFactor = 0;
	let iridescenceIor = 1.3;
	let iridescenceThicknessMinimum = 100;
	let iridescenceThicknessMaximum = 400;
	let anisotropyStrength = 0;
	let anisotropyRotation = 0;
	let attenuationDistance = -1;
	let attenuationColor: [number, number, number] = [1, 1, 1];
	let shininess = 32;
	let phongAmbient: [number, number, number] = [0, 0, 0];
	let baseMap: TextureLike = material.map ?? null;
	let baseMapUV: 0 | 1 | 2 | 3 = 0;
	let metallicRoughnessMap: TextureLike = null;
	let metallicRoughnessMapUV: 0 | 1 | 2 | 3 = 0;
	let specularMap: TextureLike = null;
	let specularMapUV: 0 | 1 | 2 | 3 = 0;
	let specularColorMap: TextureLike = null;
	let specularColorMapUV: 0 | 1 | 2 | 3 = 0;
	let clearcoatMap: TextureLike = null;
	let clearcoatMapUV: 0 | 1 | 2 | 3 = 0;
	let clearcoatRoughnessMap: TextureLike = null;
	let clearcoatRoughnessMapUV: 0 | 1 | 2 | 3 = 0;
	let clearcoatNormalMap: TextureLike = null;
	let clearcoatNormalMapUV: 0 | 1 | 2 | 3 = 0;
	let sheenColorMap: TextureLike = null;
	let sheenColorMapUV: 0 | 1 | 2 | 3 = 0;
	let sheenRoughnessMap: TextureLike = null;
	let sheenRoughnessMapUV: 0 | 1 | 2 | 3 = 0;
	let transmissionMap: TextureLike = null;
	let transmissionMapUV: 0 | 1 | 2 | 3 = 0;
	let thicknessMap: TextureLike = null;
	let thicknessMapUV: 0 | 1 | 2 | 3 = 0;
	let normalMap: TextureLike = null;
	let normalMapUV: 0 | 1 | 2 | 3 = 0;
	let normalScale = 1;
	let emissiveMap: TextureLike = null;
	let emissiveMapUV: 0 | 1 | 2 | 3 = 0;
	let occlusionMap: TextureLike = null;
	let occlusionMapUV: 0 | 1 | 2 | 3 = 0;
	let occlusionStrength = 1;
	let iridescenceMap: TextureLike = null;
	let iridescenceMapUV: 0 | 1 | 2 | 3 = 0;
	let iridescenceThicknessMap: TextureLike = null;
	let iridescenceThicknessMapUV: 0 | 1 | 2 | 3 = 0;
	let anisotropyMap: TextureLike = null;
	let anisotropyMapUV: 0 | 1 | 2 | 3 = 0;

	if (isWebGLPBRMaterial(material)) {
		const pbr = material;
		const albedo = pbr.albedo ?? { r: 255, g: 255, b: 255 };
		baseColor = [
			clamp((albedo.r ?? 255) / 255, 0, 1),
			clamp((albedo.g ?? 255) / 255, 0, 1),
			clamp((albedo.b ?? 255) / 255, 0, 1),
		];
		const emissiveColor = pbr.emissive ?? { r: 0, g: 0, b: 0 };
		const emissiveIntensity = clamp(pbr.emissiveIntensity ?? 1, 0, 64);
		emissive = [
			clamp((emissiveColor.r ?? 0) / 255, 0, 1) * emissiveIntensity,
			clamp((emissiveColor.g ?? 0) / 255, 0, 1) * emissiveIntensity,
			clamp((emissiveColor.b ?? 0) / 255, 0, 1) * emissiveIntensity,
		];
		roughness = clamp(pbr.roughness ?? 0.5, 0.04, 1);
		metalness = clamp(pbr.metalness ?? 0, 0, 1);
		reflectance = clamp(pbr.reflectance ?? 0.5, 0, 1);
		specularFactor = clamp(pbr.specularFactor ?? 1, 0, 1);
		const specularColorFactor =
			pbr.specularColor ??
			pbr.specularColorFactor ??
			{ r: 255, g: 255, b: 255 };
		specularColor = [
			clamp((specularColorFactor.r ?? 255) / 255, 0, 1),
			clamp((specularColorFactor.g ?? 255) / 255, 0, 1),
			clamp((specularColorFactor.b ?? 255) / 255, 0, 1),
		];
		transmission = getMaterialTransmissionFactor(material);
		clearcoat = clamp(pbr.clearcoat ?? 0, 0, 1);
		clearcoatRoughness = clamp(pbr.clearcoatRoughness ?? 0.01, 0.01, 1);
		clearcoatNormalScale = Math.max(0, pbr.clearcoatNormalScale ?? 1);
		const resolvedSheenColor = pbr.sheenColorFactor ?? { r: 0, g: 0, b: 0 };
		sheenColor = [
			clamp((resolvedSheenColor.r ?? 0) / 255, 0, 1),
			clamp((resolvedSheenColor.g ?? 0) / 255, 0, 1),
			clamp((resolvedSheenColor.b ?? 0) / 255, 0, 1),
		];
		sheenRoughness = clamp(pbr.sheenRoughnessFactor ?? 0, 0, 1);
		ior = Math.max(1, pbr.ior ?? 1.5);
		thickness = Math.max(0, pbr.thicknessFactor ?? 0);
		iridescenceFactor = clamp(pbr.iridescenceFactor ?? 0, 0, 1);
		iridescenceIor = Math.max(1, pbr.iridescenceIor ?? 1.3);
		iridescenceThicknessMinimum = Math.max(
			pbr.iridescenceThicknessMinimum ?? 100,
			0
		);
		iridescenceThicknessMaximum = Math.max(
			pbr.iridescenceThicknessMaximum ?? 400,
			0
		);
		attenuationDistance =
			Number.isFinite(pbr.attenuationDistance) ?
				Math.max(pbr.attenuationDistance, 0)
			:	-1;
		const attenuation = pbr.attenuationColor ?? { r: 255, g: 255, b: 255 };
		attenuationColor = [
			clamp((attenuation.r ?? 255) / 255, 0, 1),
			clamp((attenuation.g ?? 255) / 255, 0, 1),
			clamp((attenuation.b ?? 255) / 255, 0, 1),
		];
		baseMap = pbr.map ?? baseMap;
		baseMapUV = resolveUVSet(pbr.albedoMapUV);
		metallicRoughnessMap = pbr.metallicRoughnessMap ?? null;
		metallicRoughnessMapUV = resolveUVSet(pbr.metallicRoughnessMapUV);
		specularMap = pbr.specularMap ?? null;
		specularMapUV = resolveUVSet(pbr.specularMapUV);
		specularColorMap = pbr.specularColorMap ?? null;
		specularColorMapUV = resolveUVSet(pbr.specularColorMapUV);
		clearcoatMap = pbr.clearcoatMap ?? null;
		clearcoatMapUV = resolveUVSet(pbr.clearcoatMapUV);
		clearcoatRoughnessMap = pbr.clearcoatRoughnessMap ?? null;
		clearcoatRoughnessMapUV = resolveUVSet(pbr.clearcoatRoughnessMapUV);
		clearcoatNormalMap = pbr.clearcoatNormalMap ?? null;
		clearcoatNormalMapUV = resolveUVSet(pbr.clearcoatNormalMapUV);
		sheenColorMap = pbr.sheenColorMap ?? null;
		sheenColorMapUV = resolveUVSet(pbr.sheenColorMapUV);
		sheenRoughnessMap = pbr.sheenRoughnessMap ?? null;
		sheenRoughnessMapUV = resolveUVSet(pbr.sheenRoughnessMapUV);
		transmissionMap = pbr.transmissionMap ?? null;
		transmissionMapUV = resolveUVSet(pbr.transmissionMapUV);
		thicknessMap = pbr.thicknessMap ?? null;
		thicknessMapUV = resolveUVSet(pbr.thicknessMapUV);
		normalMap = pbr.normalMap ?? null;
		normalMapUV = resolveUVSet(pbr.normalMapUV);
		normalScale = Math.max(0, pbr.normalScale ?? 1);
		emissiveMap = pbr.emissiveMap ?? null;
		emissiveMapUV = resolveUVSet(pbr.emissiveMapUV);
		occlusionMap = pbr.occlusionMap ?? null;
		occlusionMapUV = resolveUVSet(pbr.occlusionMapUV);
		occlusionStrength = clamp(pbr.occlusionStrength ?? 1, 0, 1);
		iridescenceMap = pbr.iridescenceMap ?? null;
		iridescenceMapUV = resolveUVSet(pbr.iridescenceMapUV);
		iridescenceThicknessMap = pbr.iridescenceThicknessMap ?? null;
		iridescenceThicknessMapUV = resolveUVSet(
			pbr.iridescenceThicknessMapUV
		);
		anisotropyStrength = clamp(pbr.anisotropyStrength ?? 0, 0, 1);
		anisotropyRotation =
			Number.isFinite(pbr.anisotropyRotation) ? pbr.anisotropyRotation : 0;
		anisotropyMap = pbr.anisotropyMap ?? null;
		anisotropyMapUV = resolveUVSet(pbr.anisotropyMapUV);
	} else if (isWebGLLegacyMaterial(material)) {
		const basic = material;
		const diffuse = basic.diffuse ?? { r: 255, g: 255, b: 255 };
		baseColor = [
			sRGBToLinear(clamp((diffuse.r ?? 255) / 255, 0, 1)),
			sRGBToLinear(clamp((diffuse.g ?? 255) / 255, 0, 1)),
			sRGBToLinear(clamp((diffuse.b ?? 255) / 255, 0, 1)),
		];
		const emissiveColor = basic.emissive;
		if (emissiveColor) {
			const emissiveIntensity = clamp(basic.emissiveIntensity ?? 1, 0, 64);
			emissive = [
				sRGBToLinear(clamp((emissiveColor.r ?? 0) / 255, 0, 1)) *
					emissiveIntensity,
				sRGBToLinear(clamp((emissiveColor.g ?? 0) / 255, 0, 1)) *
					emissiveIntensity,
				sRGBToLinear(clamp((emissiveColor.b ?? 0) / 255, 0, 1)) *
					emissiveIntensity,
			];
		}
		shininess = Math.max(0, basic.shininess ?? 32);
		const legacySpecular = basic.specular ?? { r: 56, g: 56, b: 56 };
		specularColor = [
			sRGBToLinear(clamp((legacySpecular.r ?? 56) / 255, 0, 1)),
			sRGBToLinear(clamp((legacySpecular.g ?? 56) / 255, 0, 1)),
			sRGBToLinear(clamp((legacySpecular.b ?? 56) / 255, 0, 1)),
		];
		const legacyAmbient = basic.ambient ?? { r: 0, g: 0, b: 0 };
		phongAmbient = [
			sRGBToLinear(clamp((legacyAmbient.r ?? 0) / 255, 0, 1)),
			sRGBToLinear(clamp((legacyAmbient.g ?? 0) / 255, 0, 1)),
			sRGBToLinear(clamp((legacyAmbient.b ?? 0) / 255, 0, 1)),
		];
	}

	const opacity = clamp(material.opacity ?? 1, 0, 1);
	const alphaCutoff = clamp(material.alphaCutoff ?? 0.5, 0, 1);
	const alphaModeMask = material.alphaMode === AlphaMode.Mask ? 1 : 0;

	return {
		shadingModel:
			isUnlit ? 2
			: isPBR ? 1
			: 0,
		baseColor: [baseColor[0], baseColor[1], baseColor[2], opacity],
		emissive,
		pbr: [roughness, metalness, reflectance, transmission],
		specular: [
			specularColor[0],
			specularColor[1],
			specularColor[2],
			specularFactor,
		],
		transmissionVolume: [ior, thickness, attenuationDistance, 0],
		clearcoat: [clearcoat, clearcoatRoughness, clearcoatNormalScale, 0],
		sheen: [sheenColor[0], sheenColor[1], sheenColor[2], sheenRoughness],
		iridescence: [
			iridescenceFactor,
			iridescenceIor,
			iridescenceThicknessMinimum,
			iridescenceThicknessMaximum,
		],
		attenuationColor: [
			attenuationColor[0],
			attenuationColor[1],
			attenuationColor[2],
			1,
		],
		anisotropy: [
			anisotropyStrength,
			Math.cos(anisotropyRotation),
			Math.sin(anisotropyRotation),
			0,
		],
		phong: [shininess, 0, 0, 0],
		phongAmbient: [phongAmbient[0], phongAmbient[1], phongAmbient[2], 0],
		alpha: [alphaCutoff, alphaModeMask, 0, 0],
		baseMap,
		baseMapUV,
		metallicRoughnessMap,
		metallicRoughnessMapUV,
		specularMap,
		specularMapUV,
		specularColorMap,
		specularColorMapUV,
		clearcoatMap,
		clearcoatMapUV,
		clearcoatRoughnessMap,
		clearcoatRoughnessMapUV,
		clearcoatNormalMap,
		clearcoatNormalMapUV,
		clearcoatNormalScale,
		sheenColorMap,
		sheenColorMapUV,
		sheenRoughnessMap,
		sheenRoughnessMapUV,
		transmissionMap,
		transmissionMapUV,
		thicknessMap,
		thicknessMapUV,
		normalMap,
		normalMapUV,
		normalScale,
		emissiveMap,
		emissiveMapUV,
		occlusionMap,
		occlusionMapUV,
		occlusionStrength,
		iridescenceMap,
		iridescenceMapUV,
		iridescenceThicknessMap,
		iridescenceThicknessMapUV,
		anisotropyMap,
		anisotropyMapUV,
	};
}

export function resolveTextureUVTransform(texture: TextureLike): {
	repeatX: number;
	repeatY: number;
	offsetX: number;
	offsetY: number;
	cosRotation: number;
	sinRotation: number;
} {
	const repeatX =
		Number.isFinite(texture?.repeat?.x) ? Math.max(0, texture.repeat.x) : 1;
	const repeatY =
		Number.isFinite(texture?.repeat?.y) ? Math.max(0, texture.repeat.y) : 1;
	const offsetX = Number.isFinite(texture?.offset?.x) ? texture.offset.x : 0;
	const offsetY = Number.isFinite(texture?.offset?.y) ? texture.offset.y : 0;
	const rotation = Number.isFinite(texture?.rotation) ? texture.rotation : 0;
	return {
		repeatX,
		repeatY,
		offsetX,
		offsetY,
		cosRotation: Math.cos(rotation),
		sinRotation: Math.sin(rotation),
	};
}
