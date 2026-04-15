import { sRGBToLinear, clamp } from "../../maths/Common";
import {
	AlphaMode,
	ShadingModel,
	type Material,
} from "../../materials/Material";
import { getMaterialTransmissionFactor } from "../../materials/transparency";

export interface MaterialUniformState {
	shadingModel: number;
	baseColor: [number, number, number, number];
	emissive: [number, number, number];
	pbr: [number, number, number, number];
	transmissionVolume: [number, number, number, number];
	attenuationColor: [number, number, number, number];
	phong: [number, number, number, number];
	alpha: [number, number, number, number];
	baseMap: any | null;
}

export function resolveMaterialUniforms(material: Material): MaterialUniformState {
	const isPBR =
		material.shading === ShadingModel.PBR || material.type === "PBR";
	const isUnlit = material.shading === ShadingModel.Unlit;

	let baseColor: [number, number, number] = [1, 1, 1];
	let emissive: [number, number, number] = [0, 0, 0];
	let roughness = 0.5;
	let metalness = 0;
	let reflectance = 0.5;
	let transmission = 0;
	let ior = 1.5;
	let thickness = 0;
	let attenuationDistance = -1;
	let attenuationColor: [number, number, number] = [1, 1, 1];
	let shininess = 32;
	let baseMap: any | null = material.map ?? null;

	if (isPBR) {
		const pbr = material as any;
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
		transmission = getMaterialTransmissionFactor(material);
		ior = Math.max(1, pbr.ior ?? 1.5);
		thickness = Math.max(0, pbr.thicknessFactor ?? 0);
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
	} else {
		const basic = material as any;
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
		shininess = Math.max(1, basic.shininess ?? 32);
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
		transmissionVolume: [ior, thickness, attenuationDistance, 0],
		attenuationColor: [
			attenuationColor[0],
			attenuationColor[1],
			attenuationColor[2],
			1,
		],
		phong: [shininess, 0, 0, 0],
		alpha: [alphaCutoff, alphaModeMask, 0, 0],
		baseMap,
	};
}

export function resolveTextureUVTransform(texture: any | null): {
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
