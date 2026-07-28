import {
	resolveDecalChannelBlendMode,
	type DecalBlendMode,
} from "../../decals";
import {
	blendDecalDirection,
	blendDecalScalar,
	resolveDecalCoverage,
	resolveDecalEdgeCoverage,
} from "../../decals/evaluation";
import {
	AlphaMode,
	ShadingModel,
	type Material,
	type PBRMaterial,
} from "../../materials";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import { clamp, linearToSRGB, sRGBToLinear } from "../../maths/Common";
import { Matrix4 } from "../../maths/Matrix4";
import {
	sampleSoftwareTextureMapInto,
} from "../../shaders/software/textureSampling";
import type {
	FragmentInput,
	ISurfaceModifier,
	PBRSurfaceProperties,
	SurfaceProperties,
} from "../../shaders/software/types";
import type { RGBA, RGB } from "../../foundation/Color";
import type { IVector3, Matrix3Arr } from "../../maths/types";
import type { DecalPacket } from "../../pipeline/types";

/**
 * @internal Applies prepared decals to software-evaluated fragment surfaces.
 */
export class SoftwareDecalSurfaceModifier implements ISurfaceModifier {
	private _packets: readonly DecalPacket[] = [];
	private _sample: RGBA = { r: 0, g: 0, b: 0, a: 1 };
	private _linearColor: RGB = { r: 0, g: 0, b: 0 };
	private _localPosition = { x: 0, y: 0, z: 0 };
	private _sourceDirection: IVector3 = { x: 0, y: 0, z: 1 };
	private _blendedDirection: IVector3 = { x: 0, y: 0, z: 1 };

	public get active(): boolean {
		return this._packets.length > 0;
	}

	public prepare(packets: readonly DecalPacket[] | null | undefined): void {
		this._packets = packets ?? [];
	}

	public apply(input: FragmentInput, surface: SurfaceProperties): void {
		for (const packet of this._packets) {
			this._applyPacket(packet, input, surface);
		}
	}

	private _applyPacket(
		packet: DecalPacket,
		input: FragmentInput,
		surface: SurfaceProperties
	): void {
		const inverse = packet.inverseWorldMatrix.elements;
		const world = input.world;
		const lx =
			inverse[0][0] * world.x +
			inverse[0][1] * world.y +
			inverse[0][2] * world.z +
			inverse[0][3];
		const ly =
			inverse[1][0] * world.x +
			inverse[1][1] * world.y +
			inverse[1][2] * world.z +
			inverse[1][3];
		const lz =
			inverse[2][0] * world.x +
			inverse[2][1] * world.y +
			inverse[2][2] * world.z +
			inverse[2][3];
		this._localPosition.x = lx;
		this._localPosition.y = ly;
		this._localPosition.z = lz;
		const edgeCoverage = resolveDecalEdgeCoverage(
			this._localPosition,
			packet.edgeFade
		);
		if (edgeCoverage <= 0) return;

		const material = packet.material;
		if (material instanceof ShaderMaterial) return;
		const pbr = material as PBRMaterial;
		const u = lx + 0.5;
		const v = ly + 0.5;
		const hasBaseMap = sampleSoftwareTextureMapInto(
			material.map,
			u,
			v,
			this._sample
		);
		const coverage = resolveDecalCoverage(
			packet.opacity,
			material.opacity ?? 1,
			hasBaseMap ? this._sample.a : 1,
			edgeCoverage,
			material.alphaMode === AlphaMode.Mask,
			material.alphaCutoff ?? 0.5
		);
		if (coverage <= 0) return;

		const baseFactor = resolveLinearBaseColor(material, this._linearColor);
		const colorSpace = material.map?.colorSpace ?? "sRGB";
		const texR = hasBaseMap ? decodeColor(this._sample.r, colorSpace) : 1;
		const texG = hasBaseMap ? decodeColor(this._sample.g, colorSpace) : 1;
		const texB = hasBaseMap ? decodeColor(this._sample.b, colorSpace) : 1;
		blendSurfaceRgb(
			surface.albedo,
			baseFactor.r * texR,
			baseFactor.g * texG,
			baseFactor.b * texB,
			resolveDecalChannelBlendMode(packet.channelBlendModes, "baseColor"),
			coverage,
			surface.type === "pbr"
		);

		this._applyNormal(
			packet,
			pbr.normalMap,
			pbr.normalScale ?? 1,
			u,
			v,
			surface.normal,
			"normal",
			coverage
		);

		const emissive = resolveLinearEmissive(material, this._linearColor);
		let emissiveR = emissive.r;
		let emissiveG = emissive.g;
		let emissiveB = emissive.b;
		if (sampleSoftwareTextureMapInto(pbr.emissiveMap, u, v, this._sample)) {
			const space = pbr.emissiveMap?.colorSpace ?? "sRGB";
			emissiveR *= decodeColor(this._sample.r, space);
			emissiveG *= decodeColor(this._sample.g, space);
			emissiveB *= decodeColor(this._sample.b, space);
		}
		blendSurfaceRgb(
			surface.emissive,
			emissiveR,
			emissiveG,
			emissiveB,
			resolveDecalChannelBlendMode(packet.channelBlendModes, "emissive"),
			coverage,
			surface.type === "pbr"
		);
		surface.emissiveIntensity = blendDecalScalar(
			surface.emissiveIntensity,
			pbr.emissiveIntensity ?? 1,
			resolveDecalChannelBlendMode(packet.channelBlendModes, "emissive"),
			coverage
		);

		if (surface.type === "pbr") {
			this._applyPbr(packet, pbr, u, v, coverage, surface);
		} else {
			const specular = resolveLinearSpecular(material, this._linearColor);
			if (specular) {
				blendSurfaceRgb(
					surface.specular,
					specular.r,
					specular.g,
					specular.b,
					resolveDecalChannelBlendMode(
						packet.channelBlendModes,
						"specularColor"
					),
					coverage,
					false
				);
			}
		}
	}

	private _applyPbr(
		packet: DecalPacket,
		material: PBRMaterial,
		u: number,
		v: number,
		coverage: number,
		surface: PBRSurfaceProperties
	): void {
		let roughness = material.roughness ?? 0.5;
		let metalness = material.metalness ?? 0;
		if (
			sampleSoftwareTextureMapInto(
				material.metallicRoughnessMap,
				u,
				v,
				this._sample
			)
		) {
			roughness *= this._sample.g / 255;
			metalness *= this._sample.b / 255;
		}
		surface.roughness = clamp(blendDecalScalar(
			surface.roughness,
			roughness,
			mode(packet, "roughness"),
			coverage
		), 0.04, 1);
		surface.metalness = clamp(blendDecalScalar(
			surface.metalness,
			metalness,
			mode(packet, "metalness"),
			coverage
		), 0, 1);

		let occlusion = 1;
		if (
			sampleSoftwareTextureMapInto(material.occlusionMap, u, v, this._sample)
		) {
			occlusion =
				1 + (material.occlusionStrength ?? 1) * (this._sample.r / 255 - 1);
		}
		surface.occlusion = clamp(blendDecalScalar(
			surface.occlusion,
			occlusion,
			mode(packet, "occlusion"),
			coverage
		), 0, 1);

		let specularFactor = material.specularFactor ?? 1;
		if (sampleSoftwareTextureMapInto(material.specularMap, u, v, this._sample)) {
			specularFactor *= this._sample.a;
		}
		surface.specularFactor = clamp(blendDecalScalar(
			surface.specularFactor,
			specularFactor,
			mode(packet, "specular"),
			coverage
		), 0, 1);
		const specularColor =
			resolveLinearSpecular(material, this._linearColor) ?? WHITE;
		let specularR = specularColor.r;
		let specularG = specularColor.g;
		let specularB = specularColor.b;
		if (
			sampleSoftwareTextureMapInto(
				material.specularColorMap,
				u,
				v,
				this._sample
			)
		) {
			const space = material.specularColorMap?.colorSpace ?? "sRGB";
			specularR *= decodeColor(this._sample.r, space);
			specularG *= decodeColor(this._sample.g, space);
			specularB *= decodeColor(this._sample.b, space);
		}
		blendRgb(
			surface.specularColor,
			specularR,
			specularG,
			specularB,
			mode(packet, "specularColor"),
			coverage
		);

		let clearcoat = material.clearcoat ?? 0;
		if (sampleSoftwareTextureMapInto(material.clearcoatMap, u, v, this._sample)) {
			clearcoat *= this._sample.r / 255;
		}
		surface.clearcoat = clamp(blendDecalScalar(
			surface.clearcoat,
			clearcoat,
			mode(packet, "clearcoat"),
			coverage
		), 0, 1);
		let clearcoatRoughness = material.clearcoatRoughness ?? 0.01;
		if (
			sampleSoftwareTextureMapInto(
				material.clearcoatRoughnessMap,
				u,
				v,
				this._sample
			)
		) {
			clearcoatRoughness *= this._sample.g / 255;
		}
		surface.clearcoatRoughness = clamp(blendDecalScalar(
			surface.clearcoatRoughness,
			clearcoatRoughness,
			mode(packet, "clearcoatRoughness"),
			coverage
		), 0.01, 1);
		this._applyNormal(
			packet,
			material.clearcoatNormalMap,
			material.clearcoatNormalScale ?? 1,
			u,
			v,
			surface.clearcoatNormal,
			"clearcoatNormal",
			coverage
		);

		const sheen = material.sheenColorFactor ?? BLACK;
		let sheenR = sheen.r;
		let sheenG = sheen.g;
		let sheenB = sheen.b;
		if (
			sampleSoftwareTextureMapInto(
				material.sheenColorMap,
				u,
				v,
				this._sample
			)
		) {
			const space = material.sheenColorMap?.colorSpace ?? "sRGB";
			sheenR *= decodeColor(this._sample.r, space);
			sheenG *= decodeColor(this._sample.g, space);
			sheenB *= decodeColor(this._sample.b, space);
		}
		blendRgb(
			surface.sheenColor,
			sheenR,
			sheenG,
			sheenB,
			mode(packet, "sheenColor"),
			coverage
		);
		let sheenRoughness = material.sheenRoughnessFactor ?? 0;
		if (
			sampleSoftwareTextureMapInto(
				material.sheenRoughnessMap,
				u,
				v,
				this._sample
			)
		) {
			sheenRoughness *= this._sample.a;
		}
		surface.sheenRoughness = clamp(blendDecalScalar(
			surface.sheenRoughness,
			sheenRoughness,
			mode(packet, "sheenRoughness"),
			coverage
		), 0, 1);

		surface.transmission = this._sampleScalarMap(
			surface.transmission,
			material.transmissionFactor ?? 0,
			material.transmissionMap,
			"r",
			packet,
			"transmission",
			u,
			v,
			coverage
		);
		surface.thickness = Math.max(0, this._sampleScalarMap(
			surface.thickness,
			material.thicknessFactor ?? 0,
			material.thicknessMap,
			"g",
			packet,
			"thickness",
			u,
			v,
			coverage
		));
		surface.iridescence = this._sampleScalarMap(
			surface.iridescence,
			material.iridescenceFactor ?? 0,
			material.iridescenceMap,
			"r",
			packet,
			"iridescence",
			u,
			v,
			coverage
		);
		let film = material.iridescenceThicknessMaximum ?? 400;
		if (
			sampleSoftwareTextureMapInto(
				material.iridescenceThicknessMap,
				u,
				v,
				this._sample
			)
		) {
			const minimum = material.iridescenceThicknessMinimum ?? 100;
			film = minimum + (film - minimum) * (this._sample.g / 255);
		}
		surface.iridescenceThickness = Math.max(0, blendDecalScalar(
			surface.iridescenceThickness,
			film,
			mode(packet, "iridescenceThickness"),
			coverage
		));
		this._applyAnisotropy(packet, material, u, v, coverage, surface);
	}

	private _sampleScalarMap(
		receiver: number,
		factor: number,
		mapValue: PBRMaterial["map"],
		channel: "r" | "g",
		packet: DecalPacket,
		decalChannel: "transmission" | "thickness" | "iridescence",
		u: number,
		v: number,
		coverage: number
	): number {
		let source = factor;
		if (sampleSoftwareTextureMapInto(mapValue, u, v, this._sample)) {
			source *= this._sample[channel] / 255;
		}
		return clamp(blendDecalScalar(
			receiver,
			source,
			mode(packet, decalChannel),
			coverage
		), 0, decalChannel === "thickness" ? Number.MAX_VALUE : 1);
	}

	private _applyNormal(
		packet: DecalPacket,
		mapValue: PBRMaterial["normalMap"],
		scale: number,
		u: number,
		v: number,
		receiver: IVector3,
		channel: "normal" | "clearcoatNormal",
		coverage: number
	): void {
		let nx = 0;
		let ny = 0;
		let nz = 1;
		if (sampleSoftwareTextureMapInto(mapValue, u, v, this._sample)) {
			nx = (this._sample.r / 255 * 2 - 1) * scale;
			ny = (this._sample.g / 255 * 2 - 1) * scale;
			nz = this._sample.b / 255 * 2 - 1;
		}
		transformDirection(packet.normalMatrix, nx, ny, nz, this._sourceDirection);
		blendDecalDirection(
			receiver,
			this._sourceDirection,
			mode(packet, channel),
			coverage,
			this._blendedDirection
		);
		receiver.x = this._blendedDirection.x;
		receiver.y = this._blendedDirection.y;
		receiver.z = this._blendedDirection.z;
	}

	private _applyAnisotropy(
		packet: DecalPacket,
		material: PBRMaterial,
		u: number,
		v: number,
		coverage: number,
		surface: PBRSurfaceProperties
	): void {
		let directionX = Math.cos(material.anisotropyRotation ?? 0);
		let directionY = Math.sin(material.anisotropyRotation ?? 0);
		let strength = material.anisotropyStrength ?? 0;
		if (
			sampleSoftwareTextureMapInto(
				material.anisotropyMap,
				u,
				v,
				this._sample
			)
		) {
			const mapX = this._sample.r / 255 * 2 - 1;
			const mapY = this._sample.g / 255 * 2 - 1;
			const rotatedX = mapX * directionX - mapY * directionY;
			directionY = mapX * directionY + mapY * directionX;
			directionX = rotatedX;
			strength *= this._sample.b / 255;
		}
		transformDirection(
			packet.worldMatrix,
			directionX,
			directionY,
			0,
			this._sourceDirection
		);
		blendDecalDirection(
			surface.anisotropyTangent,
			this._sourceDirection,
			mode(packet, "anisotropy"),
			coverage,
			this._blendedDirection
		);
		const normal = surface.normal;
		const dot =
			normal.x * this._blendedDirection.x +
			normal.y * this._blendedDirection.y +
			normal.z * this._blendedDirection.z;
		let tx = this._blendedDirection.x - normal.x * dot;
		let ty = this._blendedDirection.y - normal.y * dot;
		let tz = this._blendedDirection.z - normal.z * dot;
		const inverseLength = 1 / (Math.hypot(tx, ty, tz) || 1);
		tx *= inverseLength;
		ty *= inverseLength;
		tz *= inverseLength;
		surface.anisotropyTangent.x = tx;
		surface.anisotropyTangent.y = ty;
		surface.anisotropyTangent.z = tz;
		surface.anisotropyBitangent.x = normal.y * tz - normal.z * ty;
		surface.anisotropyBitangent.y = normal.z * tx - normal.x * tz;
		surface.anisotropyBitangent.z = normal.x * ty - normal.y * tx;
		surface.anisotropyStrength = clamp(blendDecalScalar(
			surface.anisotropyStrength,
			strength,
			mode(packet, "anisotropy"),
			coverage
		), 0, 1);
	}
}

const WHITE: RGB = { r: 255, g: 255, b: 255 };
const BLACK: RGB = { r: 0, g: 0, b: 0 };

function mode(
	packet: DecalPacket,
	channel: Parameters<typeof resolveDecalChannelBlendMode>[1]
): DecalBlendMode {
	return resolveDecalChannelBlendMode(packet.channelBlendModes, channel);
}

function resolveLinearBaseColor(material: Material, out: RGB): RGB {
	const color =
		(material as Material & { albedo?: RGB; diffuse?: RGB }).albedo ??
		(material as Material & { diffuse?: RGB }).diffuse ??
		WHITE;
	return resolveLinearMaterialColor(material, color, out);
}

function resolveLinearEmissive(material: Material, out: RGB): RGB {
	const color = (material as Material & { emissive?: RGB }).emissive ?? BLACK;
	return resolveLinearMaterialColor(material, color, out);
}

function resolveLinearSpecular(material: Material, out: RGB): RGB | null {
	const color =
		(material as Material & { specularColor?: RGB }).specularColor ??
		(material as Material & { specular?: RGB }).specular;
	return color ? resolveLinearMaterialColor(material, color, out) : null;
}

function resolveLinearMaterialColor(
	material: Material,
	color: RGB,
	out: RGB
): RGB {
	if (material.shading === ShadingModel.PBR) {
		out.r = color.r;
		out.g = color.g;
		out.b = color.b;
		return out;
	}
	out.r = sRGBToLinear(clamp(color.r / 255, 0, 1)) * 255;
	out.g = sRGBToLinear(clamp(color.g / 255, 0, 1)) * 255;
	out.b = sRGBToLinear(clamp(color.b / 255, 0, 1)) * 255;
	return out;
}

function decodeColor(
	value: number,
	colorSpace: "sRGB" | "Linear" | "HDR"
): number {
	const normalized = Math.max(0, value / 255);
	return colorSpace === "Linear" || colorSpace === "HDR" ?
			normalized
		:	sRGBToLinear(normalized);
}

function blendRgb(
	receiver: RGB,
	r: number,
	g: number,
	b: number,
	blendMode: DecalBlendMode,
	coverage: number
): void {
	const sourceScale = blendMode === "multiply" ? 1 / 255 : 1;
	receiver.r = blendDecalScalar(
		receiver.r,
		r * sourceScale,
		blendMode,
		coverage
	);
	receiver.g = blendDecalScalar(
		receiver.g,
		g * sourceScale,
		blendMode,
		coverage
	);
	receiver.b = blendDecalScalar(
		receiver.b,
		b * sourceScale,
		blendMode,
		coverage
	);
}

function blendSurfaceRgb(
	receiver: RGB,
	r: number,
	g: number,
	b: number,
	blendMode: DecalBlendMode,
	coverage: number,
	receiverIsLinear: boolean
): void {
	if (receiverIsLinear) {
		blendRgb(receiver, r, g, b, blendMode, coverage);
		return;
	}
	receiver.r = sRGBToLinear(clamp(receiver.r / 255, 0, 1)) * 255;
	receiver.g = sRGBToLinear(clamp(receiver.g / 255, 0, 1)) * 255;
	receiver.b = sRGBToLinear(clamp(receiver.b / 255, 0, 1)) * 255;
	blendRgb(receiver, r, g, b, blendMode, coverage);
	receiver.r = linearToSRGB(clamp(receiver.r / 255, 0, 1)) * 255;
	receiver.g = linearToSRGB(clamp(receiver.g / 255, 0, 1)) * 255;
	receiver.b = linearToSRGB(clamp(receiver.b / 255, 0, 1)) * 255;
}

function transformDirection(
	matrix: Matrix4 | Matrix3Arr,
	x: number,
	y: number,
	z: number,
	out: IVector3
): void {
	const elements = matrix instanceof Matrix4 ? matrix.elements : matrix;
	const tx = elements[0][0] * x + elements[0][1] * y + elements[0][2] * z;
	const ty = elements[1][0] * x + elements[1][1] * y + elements[1][2] * z;
	const tz = elements[2][0] * x + elements[2][1] * y + elements[2][2] * z;
	const inverseLength = 1 / (Math.hypot(tx, ty, tz) || 1);
	out.x = tx * inverseLength;
	out.y = ty * inverseLength;
	out.z = tz * inverseLength;
}
