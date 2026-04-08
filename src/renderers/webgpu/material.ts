import { clamp, sRGBToLinear } from "../../maths/Common";
import {
	type Material,
	ShadingModel,
	AlphaMode,
} from "../../materials/Material";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import type { Texture } from "../../core/Texture";

import { WEBGPU_TEXTURE_SLOT, WEBGPU_TEXTURE_SLOT_COUNT } from "./constants";
import type {
	WebGPUMaterialUniformData,
	WebGPUTextureSlotData,
	WebGPUWarning,
} from "./types";

export function createWebGPUMaterialUniformData(
	material: Material,
	isWireframe = false
): WebGPUMaterialUniformData {
	const warnings: WebGPUWarning[] = [];
	const mat = material as any;
	const shadingMode = resolveShadingMode(material);
	const isPBR = shadingMode === 1;
	const baseColor = getMaterialBaseColor(material, isPBR);
	const emissive = getMaterialEmissive(material, isPBR);
	const opacity = clamp(material.opacity ?? 1, 0, 1);
	const alphaMode = material.alphaMode ?? AlphaMode.Opaque;
	const alphaModeMask = alphaMode === AlphaMode.Mask ? 1 : 0;
	const alphaCutoff = clamp(material.alphaCutoff ?? 0.5, 0, 1);

	const roughness = clamp(mat.roughness ?? 0.5, 0.04, 1);
	const metalness = clamp(mat.metalness ?? 0, 0, 1);
	const reflectance = clamp(mat.reflectance ?? 0.5, 0, 1);
	const occlusionStrength = clamp(mat.occlusionStrength ?? 1, 0, 1);
	const normalScale = Math.max(0, mat.normalScale ?? 1);
	const clearcoat = clamp(mat.clearcoat ?? 0, 0, 1);
	const clearcoatRoughness = clamp(mat.clearcoatRoughness ?? 0.01, 0.04, 1);
	const sheenRoughness = clamp(mat.sheenRoughnessFactor ?? 0, 0, 1);
	const transmission = clamp(mat.transmissionFactor ?? 0, 0, 1);
	const ior = Math.max(1, mat.ior ?? 1.5);
	const thickness = Math.max(0, mat.thicknessFactor ?? 0);
	const attenuationDistance = Number.isFinite(mat.attenuationDistance)
		? Math.max(mat.attenuationDistance, 0)
		: -1;
	const specularFactor = clamp(mat.specularFactor ?? 1, 0, 1);
	const clearcoatNormalScale = Math.max(0, mat.clearcoatNormalScale ?? 1);

	const specularColor = getPBRLinearColor(
		mat.specularColor ?? { r: 255, g: 255, b: 255 }
	);
	const sheenColor = getPBRLinearColor(
		mat.sheenColorFactor ?? { r: 0, g: 0, b: 0 }
	);
	const attenuationColor = getPBRLinearColor(
		mat.attenuationColor ?? { r: 255, g: 255, b: 255 }
	);
	const phongAmbient = getPhongLinearColor(
		mat.ambient ?? mat.diffuse ?? { r: 255, g: 255, b: 255 }
	);
	const phongSpecular = getPhongLinearColor(
		mat.specular ?? { r: 255, g: 255, b: 255 }
	);
	const phongShininess = Math.max(mat.shininess ?? 32, 0);
	const emissiveIntensity = clamp(mat.emissiveIntensity ?? 1, 0, 64);
	const textureSlots = createMaterialTextureSlots(material);

	pushMaterialWarnings(material, warnings);

	return {
		baseColorFactor: [baseColor[0], baseColor[1], baseColor[2], opacity],
		emissiveFactor: [emissive[0], emissive[1], emissive[2], emissiveIntensity],
		surfaceParams0: [roughness, metalness, reflectance, alphaCutoff],
		surfaceParams1: [
			occlusionStrength,
			normalScale,
			clearcoat,
			clearcoatRoughness,
		],
		surfaceParams2: [sheenRoughness, transmission, ior, thickness],
		surfaceParams3: [attenuationDistance, 0, 0, 0],
		specularColorFactor: [
			specularColor[0],
			specularColor[1],
			specularColor[2],
			specularFactor,
		],
		phongAmbientShininess: [
			phongAmbient[0],
			phongAmbient[1],
			phongAmbient[2],
			phongShininess,
		],
		phongSpecularShading: [
			phongSpecular[0],
			phongSpecular[1],
			phongSpecular[2],
			shadingMode,
		],
		sheenColorClearcoatNormalScale: [
			sheenColor[0],
			sheenColor[1],
			sheenColor[2],
			clearcoatNormalScale,
		],
		attenuationColor: [
			attenuationColor[0],
			attenuationColor[1],
			attenuationColor[2],
			1,
		],
		materialFlags: [
			shadingMode,
			alphaModeMask,
			material.doubleSided ? 1 : 0,
			isWireframe ? 1 : 0,
		],
		textureSlots,
		pipelineKey: [
			material.cullMode,
			alphaModeMask
				? "mask"
				: alphaMode === AlphaMode.Blend
					? "blend"
					: "opaque",
			isWireframe ? "wireframe" : "solid",
		].join("-"),
		warnings,
	};
}

function createMaterialTextureSlots(
	material: Material
): WebGPUTextureSlotData[] {
	const mat = material as any;
	const slots = Array.from({ length: WEBGPU_TEXTURE_SLOT_COUNT }, () =>
		createTextureSlot(null, 0, false)
	);

	slots[WEBGPU_TEXTURE_SLOT.BASE_COLOR] = createTextureSlot(
		material.map ?? null,
		mat.albedoMapUV ?? 0,
		false
	);
	slots[WEBGPU_TEXTURE_SLOT.METALLIC_ROUGHNESS] = createTextureSlot(
		mat.metallicRoughnessMap ?? null,
		mat.metallicRoughnessMapUV ?? 0,
		true
	);
	slots[WEBGPU_TEXTURE_SLOT.NORMAL] = createTextureSlot(
		mat.normalMap ?? null,
		mat.normalMapUV ?? 0,
		true
	);
	slots[WEBGPU_TEXTURE_SLOT.EMISSIVE] = createTextureSlot(
		mat.emissiveMap ?? null,
		mat.emissiveMapUV ?? 0,
		false
	);
	slots[WEBGPU_TEXTURE_SLOT.OCCLUSION] = createTextureSlot(
		mat.occlusionMap ?? null,
		mat.occlusionMapUV ?? 0,
		true
	);
	slots[WEBGPU_TEXTURE_SLOT.SPECULAR] = createTextureSlot(
		mat.specularMap ?? null,
		mat.specularMapUV ?? 0,
		true
	);
	slots[WEBGPU_TEXTURE_SLOT.SPECULAR_COLOR] = createTextureSlot(
		mat.specularColorMap ?? null,
		mat.specularColorMapUV ?? 0,
		false
	);
	slots[WEBGPU_TEXTURE_SLOT.CLEARCOAT] = createTextureSlot(
		mat.clearcoatMap ?? null,
		mat.clearcoatMapUV ?? 0,
		true
	);
	slots[WEBGPU_TEXTURE_SLOT.CLEARCOAT_ROUGHNESS] = createTextureSlot(
		mat.clearcoatRoughnessMap ?? null,
		mat.clearcoatRoughnessMapUV ?? 0,
		true
	);
	slots[WEBGPU_TEXTURE_SLOT.CLEARCOAT_NORMAL] = createTextureSlot(
		mat.clearcoatNormalMap ?? null,
		mat.clearcoatNormalMapUV ?? 0,
		true
	);
	slots[WEBGPU_TEXTURE_SLOT.SHEEN_COLOR] = createTextureSlot(
		mat.sheenColorMap ?? null,
		mat.sheenColorMapUV ?? 0,
		false
	);
	slots[WEBGPU_TEXTURE_SLOT.SHEEN_ROUGHNESS] = createTextureSlot(
		mat.sheenRoughnessMap ?? null,
		mat.sheenRoughnessMapUV ?? 0,
		true
	);
	slots[WEBGPU_TEXTURE_SLOT.TRANSMISSION] = createTextureSlot(
		mat.transmissionMap ?? null,
		mat.transmissionMapUV ?? 0,
		true
	);
	slots[WEBGPU_TEXTURE_SLOT.THICKNESS] = createTextureSlot(
		mat.thicknessMap ?? null,
		mat.thicknessMapUV ?? 0,
		true
	);

	if (material instanceof ShaderMaterial) {
		const textureBindings = material.getTextureBindings();
		for (const binding of textureBindings) {
			if (binding.slot < 0 || binding.slot >= slots.length) {
				continue;
			}
			slots[binding.slot] = createTextureSlot(
				binding.texture,
				binding.uvSet,
				binding.linear,
				binding.linear
			);
		}
	}

	return slots;
}

function createTextureSlot(
	map: Texture | null | undefined,
	uvSet: number,
	fallbackLinear: boolean,
	forcedLinear: boolean | null = null
): WebGPUTextureSlotData {
	if (!map) {
		return {
			map: null,
			transformA: [0, 0, 1, 1],
			transformB: [
				0,
				uvSet > 0 ? 1 : 0,
				(forcedLinear ?? fallbackLinear) ? 1 : 0,
				0,
			],
		};
	}

	return {
		map,
		transformA: [map.offset.x, map.offset.y, map.repeat.x, map.repeat.y],
		transformB: [
			map.rotation,
			uvSet > 0 ? 1 : 0,
			forcedLinear !== null ?
				(forcedLinear ? 1 : 0)
			:	(map.colorSpace === "sRGB" ? 0 : 1),
			0,
		],
	};
}

function getMaterialBaseColor(
	material: Material,
	isPBR: boolean
): [number, number, number] {
	if (isPBR) {
		const albedo = (material as any).albedo ?? { r: 255, g: 255, b: 255 };
		return [
			clamp(albedo.r / 255, 0, 1),
			clamp(albedo.g / 255, 0, 1),
			clamp(albedo.b / 255, 0, 1),
		];
	}

	const diffuse = (material as any).diffuse ?? { r: 255, g: 255, b: 255 };
	return [
		sRGBToLinear(clamp(diffuse.r / 255, 0, 1)),
		sRGBToLinear(clamp(diffuse.g / 255, 0, 1)),
		sRGBToLinear(clamp(diffuse.b / 255, 0, 1)),
	];
}

function getMaterialEmissive(
	material: Material,
	isPBR: boolean
): [number, number, number] {
	const emissive = (material as any).emissive;
	if (!emissive) {
		return [0, 0, 0];
	}

	if (isPBR) {
		return [
			clamp(emissive.r / 255, 0, 1),
			clamp(emissive.g / 255, 0, 1),
			clamp(emissive.b / 255, 0, 1),
		];
	}

	return [
		sRGBToLinear(clamp(emissive.r / 255, 0, 1)),
		sRGBToLinear(clamp(emissive.g / 255, 0, 1)),
		sRGBToLinear(clamp(emissive.b / 255, 0, 1)),
	];
}

function getPhongLinearColor(color: {
	r: number;
	g: number;
	b: number;
}): [number, number, number] {
	return [
		sRGBToLinear(clamp(color.r / 255, 0, 1)),
		sRGBToLinear(clamp(color.g / 255, 0, 1)),
		sRGBToLinear(clamp(color.b / 255, 0, 1)),
	];
}

function getPBRLinearColor(color: {
	r: number;
	g: number;
	b: number;
}): [number, number, number] {
	return [
		clamp(color.r / 255, 0, 1),
		clamp(color.g / 255, 0, 1),
		clamp(color.b / 255, 0, 1),
	];
}

function resolveShadingMode(material: Material): number {
	switch (material.shading) {
		case ShadingModel.PBR:
			return 1;
		case ShadingModel.Unlit:
			return 2;
		case ShadingModel.Flat:
			return 3;
		default:
			return 0;
	}
}

function pushMaterialWarnings(
	material: Material,
	warnings: WebGPUWarning[]
): void {
	const warn = (feature: string, message: string, enabled: boolean) => {
		if (!enabled) return;
		warnings.push({
			key: `webgpu-material-${feature}:${material.type}:${material.name}`,
			message,
		});
	};

	warn(
		"reflectivity",
		`WebGPU backend does not support planar reflections yet; ignoring reflectivity on material ${material.name}`,
		(material.reflectivity ?? 0) > 0
	);
	warn(
		"mirror-plane",
		`WebGPU backend does not support mirrorPlane yet; rendering material ${material.name} as a regular lit surface`,
		!!material.mirrorPlane
	);
}
