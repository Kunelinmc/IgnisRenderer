import type { Vec3Tuple } from "../../maths/Vector3";
import type { Vec4Tuple } from "../../maths/Vector4";
import { clamp, sRGBToLinear } from "../../maths/Common";
import {
	type Material,
	ShadingModel,
	AlphaMode,
} from "../../materials/Material";
import {
	PBRMaterial,
	PBRMaterialFeature,
	PBRMaterialTextureFeature,
} from "../../materials/PBRMaterial";
import {
	isMaterialTransparentPass,
	materialUsesTransmission,
} from "../../materials/transparency";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import type {
	ResolvedShaderMaterialUniformBinding,
	ShaderMaterialUniformType,
} from "../../materials/ShaderMaterial";
import type { Texture } from "../../core/Texture";
import { isTextureFormatSRGB } from "../../core/TextureFormat";

import { WEBGPU_TEXTURE_SLOT, WEBGPU_TEXTURE_SLOT_COUNT } from "./constants";
import { createWebGPUShaderMaterialUniformLayout } from "./bufferLayouts";
import {
	mat4x4f32,
	scalar,
	vec,
	type BufferTypeSchema,
	type StructuredBufferLayout,
} from "./StructuredBufferLayout";
import type {
	WebGPUShaderUniformData,
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
	const shadingFamily = resolveShadingFamily(material);
	const isPBR = shadingFamily === "pbr";
	const baseColor = getMaterialBaseColor(material, isPBR);
	const emissive = getMaterialEmissive(material, isPBR);
	const opacity = clamp(material.opacity ?? 1, 0, 1);
	const alphaMode = material.alphaMode ?? AlphaMode.Opaque;
	const alphaModeMask = alphaMode === AlphaMode.Mask ? 1 : 0;
	const isTransmissive = materialUsesTransmission(material);
	const depthWrite = material.depthWrite;
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
	const iridescenceFactor = clamp(mat.iridescenceFactor ?? 0, 0, 1);
	const iridescenceIor = Math.max(1, mat.iridescenceIor ?? 1.3);
	const iridescenceThicknessMinimum = Math.max(
		mat.iridescenceThicknessMinimum ?? 100,
		0
	);
	const iridescenceThicknessMaximum = Math.max(
		mat.iridescenceThicknessMaximum ?? 400,
		0
	);
	const attenuationDistance = Number.isFinite(mat.attenuationDistance)
		? Math.max(mat.attenuationDistance, 0)
		: -1;
	const specularFactor = clamp(mat.specularFactor ?? 1, 0, 1);
	const clearcoatNormalScale = Math.max(0, mat.clearcoatNormalScale ?? 1);
	const anisotropyStrength = clamp(mat.anisotropyStrength ?? 0, 0, 1);
	const anisotropyRotation =
		Number.isFinite(mat.anisotropyRotation) ? mat.anisotropyRotation : 0;

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
	const pbrMasks: Vec4Tuple =
		material instanceof PBRMaterial ?
			[material.featureMask, material.textureMask, 0, 0]
		: isPBR && material.map ?
			[
				PBRMaterialFeature.BASE_COLOR_MAP,
				PBRMaterialTextureFeature.BASE_COLOR_MAP,
				0,
				0,
			]
		: [0, 0, 0, 0];
	const shaderUniforms = createShaderUniformData(material);

	pushMaterialWarnings(material, warnings);

	const common = {
		baseColorFactor: [baseColor[0], baseColor[1], baseColor[2], opacity],
		emissiveFactor: [emissive[0], emissive[1], emissive[2], emissiveIntensity],
		materialParams: [
			alphaCutoff,
			clamp(material.reflectivity ?? 0, 0, 1),
			alphaModeMask,
			material.doubleSided ? 1 : 0,
		],
		renderParams: [
			(isWireframe ? 1 : 0) +
				(alphaMode === AlphaMode.Blend || isTransmissive ? 2 : 0),
			0,
			0,
			0,
		],
		textureSlots,
	} satisfies WebGPUMaterialUniformData["common"];
	const base = {
		common,
		shaderUniforms,
		pipelineKey: [
			shadingFamily,
			material.cullMode,
			alphaModeMask
				? "mask"
				: alphaMode === AlphaMode.Blend
					? "blend"
					: isTransmissive ? "transmission"
					: "opaque",
			depthWrite ? "depth-write" : "depth-read",
			isWireframe ? "wireframe" : "solid",
		].join("-"),
		warnings,
	};
	if (shadingFamily === "pbr") {
		return {
			...base,
			shadingFamily,
			lighting: {
				surfaceParams0: [roughness, metalness, reflectance, 0],
				surfaceParams1: [
					occlusionStrength,
					normalScale,
					clearcoat,
					clearcoatRoughness,
				],
				surfaceParams2: [sheenRoughness, transmission, ior, thickness],
				surfaceParams3: [
					attenuationDistance,
					iridescenceFactor,
					iridescenceIor,
					iridescenceThicknessMinimum,
				],
				specularColorFactor: [
					specularColor[0],
					specularColor[1],
					specularColor[2],
					specularFactor,
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
					iridescenceThicknessMaximum,
				],
				anisotropyParams: [
					anisotropyStrength,
					Math.cos(anisotropyRotation),
					Math.sin(anisotropyRotation),
					0,
				],
				pbrMasks,
			},
		};
	}
	if (shadingFamily === "phong" || shadingFamily === "flat") {
		return {
			...base,
			shadingFamily,
			lighting: {
				ambientShininess: [
					phongAmbient[0],
					phongAmbient[1],
					phongAmbient[2],
					phongShininess,
				],
				specular: [
					phongSpecular[0],
					phongSpecular[1],
					phongSpecular[2],
					0,
				],
			},
		};
	}
	return { ...base, shadingFamily, lighting: null };
}

export function materialSupportsWebGPUDeferredLighting(
	material: Material | null | undefined
): boolean {
	if (!material) {
		return false;
	}
	if (material instanceof ShaderMaterial) {
		return material.hasWebGPUDeferredProgram();
	}
	if (material.wireframe) {
		return false;
	}
	if (!material.depthWrite) {
		return false;
	}
	if (isMaterialTransparentPass(material)) {
		return false;
	}
	if (materialUsesTransmission(material)) {
		return false;
	}
	return true;
}

/** Returns whether a deferred material needs the extended G-buffer payload. */
export function materialRequiresExtendedWebGPUGBuffer(
	material: Material | null | undefined
): boolean {
	if (!material) {
		return false;
	}
	if (material instanceof ShaderMaterial) {
		return material.hasWebGPUDeferredProgram();
	}
	const shadingFamily = resolveShadingFamily(material);
	if (shadingFamily === "phong" || shadingFamily === "flat") {
		return true;
	}
	if (shadingFamily !== "pbr") {
		return false;
	}
	const mat = material as any;
	const specularColor = mat.specularColor ?? { r: 255, g: 255, b: 255 };
	const sheenColor = mat.sheenColorFactor ?? { r: 0, g: 0, b: 0 };
	return (
		Math.abs((mat.clearcoat ?? 0)) > 1e-6 ||
		!!mat.clearcoatMap ||
		!!mat.clearcoatRoughnessMap ||
		!!mat.clearcoatNormalMap ||
		Math.abs(sheenColor.r) > 1e-6 ||
		Math.abs(sheenColor.g) > 1e-6 ||
		Math.abs(sheenColor.b) > 1e-6 ||
		!!mat.sheenColorMap ||
		!!mat.sheenRoughnessMap ||
		Math.abs((mat.iridescenceFactor ?? 0)) > 1e-6 ||
		!!mat.iridescenceMap ||
		!!mat.iridescenceThicknessMap ||
		Math.abs((mat.anisotropyStrength ?? 0)) > 1e-6 ||
		!!mat.anisotropyMap ||
		Math.abs((mat.specularFactor ?? 1) - 1) > 1e-6 ||
		Math.abs(specularColor.r - 255) > 1e-6 ||
		Math.abs(specularColor.g - 255) > 1e-6 ||
		Math.abs(specularColor.b - 255) > 1e-6 ||
		!!mat.specularMap ||
		!!mat.specularColorMap ||
		Math.abs((mat.reflectance ?? 0.5) - 0.5) > 1e-6
	);
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
	slots[WEBGPU_TEXTURE_SLOT.IRIDESCENCE] = createTextureSlot(
		mat.iridescenceMap ?? null,
		mat.iridescenceMapUV ?? 0,
		true
	);
	slots[WEBGPU_TEXTURE_SLOT.IRIDESCENCE_THICKNESS] = createTextureSlot(
		mat.iridescenceThicknessMap ?? null,
		mat.iridescenceThicknessMapUV ?? 0,
		true
	);
	slots[WEBGPU_TEXTURE_SLOT.ANISOTROPY] = createTextureSlot(
		mat.anisotropyMap ?? null,
		mat.anisotropyMapUV ?? 0,
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

function createShaderUniformData(material: Material): WebGPUShaderUniformData {
	if (!(material instanceof ShaderMaterial)) {
		return createEmptyShaderUniformData();
	}
	const bindings = material.getUniformBindings();
	if (bindings.length <= 0) {
		return createEmptyShaderUniformData(material.uniformValueRevision);
	}

	const layout = createWebGPUShaderMaterialUniformLayout(
		bindings.map((binding) => ({
			name: binding.wgslField,
			type: createShaderUniformTypeSchema(binding.type),
		}))
	);
	const writer = layout.createWriter();
	writer.expectByteLength(layout.byteSize, "ShaderMaterialUniforms");
	for (const binding of bindings) {
		writeShaderUniformValue(writer, binding);
	}
	const data = new Uint8Array(writer.toArrayBuffer()).slice();
	return {
		cacheKey: bindings
			.map((binding) =>
				[
					binding.wgslField,
					binding.type,
					binding.stage,
				].join(":")
			)
			.join("|"),
		byteLength: Math.max(16, layout.byteSize),
		valueRevision: material.uniformValueRevision,
		data,
	};
}

function createEmptyShaderUniformData(
	valueRevision = 0
): WebGPUShaderUniformData {
	return {
		cacheKey: "none",
		byteLength: 16,
		valueRevision,
		data: null,
	};
}

function createShaderUniformTypeSchema(
	type: ShaderMaterialUniformType
): BufferTypeSchema {
	switch (type) {
		case "i32":
			return scalar("i32");
		case "u32":
			return scalar("u32");
		case "vec2f":
			return vec(2, "f32");
		case "vec3f":
			return vec(3, "f32");
		case "vec4f":
			return vec(4, "f32");
		case "vec2i":
			return vec(2, "i32");
		case "vec3i":
			return vec(3, "i32");
		case "vec4i":
			return vec(4, "i32");
		case "vec2u":
			return vec(2, "u32");
		case "vec3u":
			return vec(3, "u32");
		case "vec4u":
			return vec(4, "u32");
		case "mat4x4f":
			return mat4x4f32();
		case "f32":
		default:
			return scalar("f32");
	}
}

function writeShaderUniformValue(
	writer: ReturnType<StructuredBufferLayout["createWriter"]>,
	binding: ResolvedShaderMaterialUniformBinding
): void {
	if (binding.type === "mat4x4f") {
		writer.writeMat4(
			binding.wgslField,
			binding.value as number[][]
		);
		return;
	}
	if (binding.type === "f32") {
		writer.writeF32(binding.wgslField, binding.value as number);
		return;
	}
	if (binding.type === "i32") {
		writer.writeI32(binding.wgslField, binding.value as number);
		return;
	}
	if (binding.type === "u32") {
		writer.writeU32(binding.wgslField, binding.value as number);
		return;
	}
	writer.writeVec(binding.wgslField, binding.value as readonly number[]);
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
				normalizeTextureUVSet(uvSet),
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
			normalizeTextureUVSet(uvSet),
			forcedLinear !== null ?
				(forcedLinear ? 1 : 0)
			:	resolveTextureSamplesLinear(map) ? 1 : 0,
			0,
		],
	};
}

function resolveTextureSamplesLinear(map: Texture): boolean {
	return map.colorSpace !== "sRGB" || isTextureFormatSRGB(map.format);
}

function normalizeTextureUVSet(uvSet: number): number {
	if (!Number.isFinite(uvSet)) {
		return 0;
	}
	return Math.max(0, Math.min(3, Math.floor(uvSet)));
}

function getMaterialBaseColor(
	material: Material,
	isPBR: boolean
): Vec3Tuple {
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
): Vec3Tuple {
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
}): Vec3Tuple {
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
}): Vec3Tuple {
	return [
		clamp(color.r / 255, 0, 1),
		clamp(color.g / 255, 0, 1),
		clamp(color.b / 255, 0, 1),
	];
}

function resolveShadingFamily(material: Material): WebGPUMaterialUniformData["shadingFamily"] {
	switch (material.shading) {
		case ShadingModel.PBR:
			return "pbr";
		case ShadingModel.Unlit:
			return "unlit";
		case ShadingModel.Flat:
			return "flat";
		default:
			return "phong";
	}
}

function pushMaterialWarnings(
	material: Material,
	warnings: WebGPUWarning[]
): void {
	void material;
	void warnings;
}
