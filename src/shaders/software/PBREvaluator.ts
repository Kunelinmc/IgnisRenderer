import { BaseEvaluator } from "./BaseEvaluator";
import {
	AlphaMode,
	type Material,
	type PBRMaterial,
	UVChannel,
} from "../../materials";
import { resolveTransmissionCompositeAlpha } from "../../materials/transparency";
import type { ProjectedFace } from "../../core/types";
import type { PBRSurfaceProperties, FragmentInput } from "./types";
import { Vector3 } from "../../maths/Vector3";
import { clamp, sRGBToLinear } from "../../maths/Common";

export class PBREvaluator extends BaseEvaluator<PBRSurfaceProperties> {
	private _mat!: PBRMaterial;
	private _cachedResult: PBRSurfaceProperties = {
		type: "pbr",
		albedo: { r: 0, g: 0, b: 0 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1.0,
		roughness: 0,
		metalness: 0,
		reflectance: 0.5,
		specularFactor: 1.0,
		specularColor: { r: 255, g: 255, b: 255 },
		occlusion: 1.0,
		clearcoat: 0.0,
		clearcoatRoughness: 0.0,
		clearcoatNormal: { x: 0, y: 0, z: 1 },
		sheenColor: { r: 0, g: 0, b: 0 },
		sheenRoughness: 0.0,
		transmission: 0.0,
		ior: 1.5,
		iridescence: 0.0,
		iridescenceIor: 1.3,
		iridescenceThickness: 400.0,
		anisotropyStrength: 0.0,
		anisotropyTangent: { x: 1, y: 0, z: 0 },
		anisotropyBitangent: { x: 0, y: 1, z: 0 },
		thickness: 0.0,
		attenuationDistance: Infinity,
		attenuationColor: { r: 255, g: 255, b: 255 },
	};

	constructor(material: Material) {
		super(material);
		this._mat = material as PBRMaterial;
	}

	public compile(material: Material): void {
		super.compile(material);
		this._mat = material as PBRMaterial;
	}

	private _applyNormalMap(
		input: FragmentInput,
		normalTex: { r: number; g: number; b: number },
		normalOut: FragmentInput["normal"],
		normalScale: number
	): void {
		let nx = normalOut.x;
		let ny = normalOut.y;
		let nz = normalOut.z;
		const invNLen = 1 / (Math.hypot(nx, ny, nz) || 1);
		nx *= invNLen;
		ny *= invNLen;
		nz *= invNLen;

		const tangentLenSq =
			input.tangent.x * input.tangent.x +
			input.tangent.y * input.tangent.y +
			input.tangent.z * input.tangent.z;
		const hasValidTangent =
			tangentLenSq > 1e-12 && Math.abs(input.tangent.w) > 1e-6;
		if (!hasValidTangent) {
			normalOut.x = nx;
			normalOut.y = ny;
			normalOut.z = nz;
			return;
		}

		const tNormX = ((normalTex.r / 255) * 2 - 1) * normalScale;
		const tNormY = ((normalTex.g / 255) * 2 - 1) * normalScale;
		const tNormZ = (normalTex.b / 255) * 2 - 1;

		// Gram-Schmidt: keep T orthogonal to N to avoid invalid TBN on skewed assets.
		const ndotT =
			nx * input.tangent.x + ny * input.tangent.y + nz * input.tangent.z;
		let tx = input.tangent.x - nx * ndotT;
		let ty = input.tangent.y - ny * ndotT;
		let tz = input.tangent.z - nz * ndotT;
		const tLen = Math.hypot(tx, ty, tz);
		if (tLen <= 1e-6) {
			normalOut.x = nx;
			normalOut.y = ny;
			normalOut.z = nz;
			return;
		}

		const invTLen = 1 / tLen;
		tx *= invTLen;
		ty *= invTLen;
		tz *= invTLen;

		const handedness = input.tangent.w < 0 ? -1 : 1;
		const bx = (ny * tz - nz * ty) * handedness;
		const by = (nz * tx - nx * tz) * handedness;
		const bz = (nx * ty - ny * tx) * handedness;

		normalOut.x = tx * tNormX + bx * tNormY + nx * tNormZ;
		normalOut.y = ty * tNormX + by * tNormY + ny * tNormZ;
		normalOut.z = tz * tNormX + bz * tNormY + nz * tNormZ;
		Vector3.normalizeInPlace(normalOut);
		if (normalOut.x * nx + normalOut.y * ny + normalOut.z * nz < 0) {
			normalOut.x *= -1;
			normalOut.y *= -1;
			normalOut.z *= -1;
		}
	}

	private _setFallbackTangent(
		nx: number,
		ny: number,
		nz: number,
		tangentOut: FragmentInput["normal"]
	): void {
		const ax = Math.abs(ny) < 0.999 ? 0 : 1;
		const ay = Math.abs(ny) < 0.999 ? 1 : 0;
		const az = 0;
		tangentOut.x = ay * nz - az * ny;
		tangentOut.y = az * nx - ax * nz;
		tangentOut.z = ax * ny - ay * nx;
		Vector3.normalizeInPlace(tangentOut);
	}

	private _resolveAnisotropyFrame(
		input: FragmentInput,
		normalIn: FragmentInput["normal"],
		directionX: number,
		directionY: number,
		tangentOut: FragmentInput["normal"],
		bitangentOut: FragmentInput["normal"]
	): void {
		let nx = normalIn.x;
		let ny = normalIn.y;
		let nz = normalIn.z;
		const invNLen = 1 / (Math.hypot(nx, ny, nz) || 1);
		nx *= invNLen;
		ny *= invNLen;
		nz *= invNLen;

		const tangentLenSq =
			input.tangent.x * input.tangent.x +
			input.tangent.y * input.tangent.y +
			input.tangent.z * input.tangent.z;
		const hasValidTangent =
			tangentLenSq > 1e-12 && Math.abs(input.tangent.w) > 1e-6;

		let tx = 0;
		let ty = 0;
		let tz = 0;
		let handedness = 1;
		if (hasValidTangent) {
			const ndotT =
				nx * input.tangent.x + ny * input.tangent.y + nz * input.tangent.z;
			tx = input.tangent.x - nx * ndotT;
			ty = input.tangent.y - ny * ndotT;
			tz = input.tangent.z - nz * ndotT;
			const tLen = Math.hypot(tx, ty, tz);
			if (tLen > 1e-6) {
				tx /= tLen;
				ty /= tLen;
				tz /= tLen;
				handedness = input.tangent.w < 0 ? -1 : 1;
			} else {
				this._setFallbackTangent(nx, ny, nz, tangentOut);
				tx = tangentOut.x;
				ty = tangentOut.y;
				tz = tangentOut.z;
			}
		} else {
			this._setFallbackTangent(nx, ny, nz, tangentOut);
			tx = tangentOut.x;
			ty = tangentOut.y;
			tz = tangentOut.z;
		}

		let bx = (ny * tz - nz * ty) * handedness;
		let by = (nz * tx - nx * tz) * handedness;
		let bz = (nx * ty - ny * tx) * handedness;
		const dirLen = Math.hypot(directionX, directionY);
		const dirX = dirLen > 1e-6 ? directionX / dirLen : 1;
		const dirY = dirLen > 1e-6 ? directionY / dirLen : 0;

		tangentOut.x = tx * dirX + bx * dirY;
		tangentOut.y = ty * dirX + by * dirY;
		tangentOut.z = tz * dirX + bz * dirY;
		Vector3.normalizeInPlace(tangentOut);

		bitangentOut.x = ny * tangentOut.z - nz * tangentOut.y;
		bitangentOut.y = nz * tangentOut.x - nx * tangentOut.z;
		bitangentOut.z = nx * tangentOut.y - ny * tangentOut.x;
		Vector3.normalizeInPlace(bitangentOut);
	}

	private _resolveUV(
		channel: UVChannel | undefined,
		input: FragmentInput
	): { u: number; v: number } {
		switch (channel) {
			case UVChannel.UV1:
				return { u: input.u2, v: input.v2 };
			case UVChannel.UV2:
				return { u: input.u3, v: input.v3 };
			case UVChannel.UV3:
				return { u: input.u4, v: input.v4 };
			default:
				return { u: input.u, v: input.v };
		}
	}

	public evaluate(
		input: FragmentInput,
		face: ProjectedFace
	): PBRSurfaceProperties | null {
		const u = input.u;
		const v = input.v;
		const mat = this._mat;
		const baseAlbedo = mat.albedo || { r: 255, g: 255, b: 255 };
		let albedo = {
			r: clamp(baseAlbedo.r, 0, 255),
			g: clamp(baseAlbedo.g, 0, 255),
			b: clamp(baseAlbedo.b, 0, 255),
		};
		let alpha = mat.opacity ?? 1;
		let roughness = mat.roughness ?? 0.5;
		let metalness = mat.metalness ?? 0.0;
		let occlusion = 1.0;

		// Select UV set for main map
		const albedoUV = this._resolveUV(mat.albedoMapUV, input);
		const tex = this._sampleTextureMap(mat.map, albedoUV.u, albedoUV.v);
		if (tex) {
			const colorSpace = mat.map?.colorSpace ?? "sRGB";
			const texLinear = {
				r:
					colorSpace === "Linear" || colorSpace === "HDR"
						? Math.max(0, tex.r / 255)
						: sRGBToLinear(Math.max(0, tex.r / 255)),
				g:
					colorSpace === "Linear" || colorSpace === "HDR"
						? Math.max(0, tex.g / 255)
						: sRGBToLinear(Math.max(0, tex.g / 255)),
				b:
					colorSpace === "Linear" || colorSpace === "HDR"
						? Math.max(0, tex.b / 255)
						: sRGBToLinear(Math.max(0, tex.b / 255)),
			};
			albedo = {
				r: albedo.r * texLinear.r,
				g: albedo.g * texLinear.g,
				b: albedo.b * texLinear.b,
			};
			alpha *= tex.a;
		}

		if (mat.alphaMode === AlphaMode.Mask && alpha < (mat.alphaCutoff ?? 0.5))
			return null;

		const mrUV = this._resolveUV(mat.metallicRoughnessMapUV, input);
		const metallicRoughnessTex = this._sampleTextureMap(
			mat.metallicRoughnessMap,
			mrUV.u,
			mrUV.v
		);
		if (metallicRoughnessTex) {
			// glTF metallicRoughness texture channels:
			// G = roughness, B = metallic
			roughness *= metallicRoughnessTex.g / 255;
			metalness *= metallicRoughnessTex.b / 255;
		}

		const baseEmissive = mat.emissive || { r: 0, g: 0, b: 0 };
		let emissive = {
			r: clamp(baseEmissive.r, 0, 255),
			g: clamp(baseEmissive.g, 0, 255),
			b: clamp(baseEmissive.b, 0, 255),
		};
		const emissiveUV = this._resolveUV(mat.emissiveMapUV, input);
		const emissiveTex = this._sampleTextureMap(
			mat.emissiveMap,
			emissiveUV.u,
			emissiveUV.v
		);
		if (emissiveTex) {
			const colorSpace = mat.emissiveMap?.colorSpace ?? "sRGB";
			const texLinear = {
				r:
					colorSpace === "Linear" || colorSpace === "HDR"
						? Math.max(0, emissiveTex.r / 255)
						: sRGBToLinear(Math.max(0, emissiveTex.r / 255)),
				g:
					colorSpace === "Linear" || colorSpace === "HDR"
						? Math.max(0, emissiveTex.g / 255)
						: sRGBToLinear(Math.max(0, emissiveTex.g / 255)),
				b:
					colorSpace === "Linear" || colorSpace === "HDR"
						? Math.max(0, emissiveTex.b / 255)
						: sRGBToLinear(Math.max(0, emissiveTex.b / 255)),
			};
			emissive = {
				r: emissive.r * texLinear.r,
				g: emissive.g * texLinear.g,
				b: emissive.b * texLinear.b,
			};
		}

		const occlusionUV = this._resolveUV(mat.occlusionMapUV, input);
		const occlusionTex = this._sampleTextureMap(
			mat.occlusionMap,
			occlusionUV.u,
			occlusionUV.v
		);
		if (occlusionTex) {
			// glTF occlusion is stored in R channel and affects indirect light.
			occlusion =
				1.0 + (mat.occlusionStrength ?? 1.0) * (occlusionTex.r / 255 - 1.0);
		}

		let specFactor = mat.specularFactor ?? 1.0;
		const specUV = this._resolveUV(mat.specularMapUV, input);
		const specTex = this._sampleTextureMap(mat.specularMap, specUV.u, specUV.v);
		if (specTex) {
			specFactor *= specTex.a;
		}

		const specColorInput = mat.specularColor || { r: 255, g: 255, b: 255 };
		let specColorLinear = {
			r: clamp(specColorInput.r, 0, 255) / 255,
			g: clamp(specColorInput.g, 0, 255) / 255,
			b: clamp(specColorInput.b, 0, 255) / 255,
		};
		const specColorUV = this._resolveUV(mat.specularColorMapUV, input);
		const specColorTex = this._sampleTextureMap(
			mat.specularColorMap,
			specColorUV.u,
			specColorUV.v
		);
		if (specColorTex) {
			const colorSpace = mat.specularColorMap?.colorSpace ?? "sRGB";
			const texLinear =
				colorSpace === "Linear" || colorSpace === "HDR"
					? {
							r: Math.max(0, specColorTex.r / 255),
							g: Math.max(0, specColorTex.g / 255),
							b: Math.max(0, specColorTex.b / 255),
						}
					: {
							r: sRGBToLinear(Math.max(0, specColorTex.r / 255)),
							g: sRGBToLinear(Math.max(0, specColorTex.g / 255)),
							b: sRGBToLinear(Math.max(0, specColorTex.b / 255)),
						};

			specColorLinear = {
				r: specColorLinear.r * texLinear.r,
				g: specColorLinear.g * texLinear.g,
				b: specColorLinear.b * texLinear.b,
			};
		}

		let sheenColorLinear = {
			r: clamp(mat.sheenColorFactor.r, 0, 255) / 255,
			g: clamp(mat.sheenColorFactor.g, 0, 255) / 255,
			b: clamp(mat.sheenColorFactor.b, 0, 255) / 255,
		};
		const sheenColorUV = this._resolveUV(mat.sheenColorMapUV, input);
		const sheenColorTex = this._sampleTextureMap(
			mat.sheenColorMap,
			sheenColorUV.u,
			sheenColorUV.v
		);
		if (sheenColorTex) {
			const colorSpace = mat.sheenColorMap?.colorSpace ?? "sRGB";
			sheenColorLinear = {
				r:
					sheenColorLinear.r *
					(colorSpace === "Linear" || colorSpace === "HDR"
						? Math.max(0, sheenColorTex.r / 255)
						: sRGBToLinear(Math.max(0, sheenColorTex.r / 255))),
				g:
					sheenColorLinear.g *
					(colorSpace === "Linear" || colorSpace === "HDR"
						? Math.max(0, sheenColorTex.g / 255)
						: sRGBToLinear(Math.max(0, sheenColorTex.g / 255))),
				b:
					sheenColorLinear.b *
					(colorSpace === "Linear" || colorSpace === "HDR"
						? Math.max(0, sheenColorTex.b / 255)
						: sRGBToLinear(Math.max(0, sheenColorTex.b / 255))),
			};
		}

		let sheenRoughness = mat.sheenRoughnessFactor;
		const sheenRoughnessUV = this._resolveUV(mat.sheenRoughnessMapUV, input);
		const sheenRoughnessTex = this._sampleTextureMap(
			mat.sheenRoughnessMap,
			sheenRoughnessUV.u,
			sheenRoughnessUV.v
		);
		if (sheenRoughnessTex) {
			sheenRoughness *= sheenRoughnessTex.a;
		}
		sheenRoughness = clamp(sheenRoughness, 0, 1);

		let clearcoat = mat.clearcoat ?? 0.0;
		const clearcoatUV = this._resolveUV(mat.clearcoatMapUV, input);
		const clearcoatTex = this._sampleTextureMap(
			mat.clearcoatMap,
			clearcoatUV.u,
			clearcoatUV.v
		);
		if (clearcoatTex) {
			clearcoat *= clearcoatTex.r / 255;
		}

		let clearcoatRoughness = mat.clearcoatRoughness ?? 0.01;
		const ccRoughnessUV = this._resolveUV(mat.clearcoatRoughnessMapUV, input);
		const ccRoughnessTex = this._sampleTextureMap(
			mat.clearcoatRoughnessMap,
			ccRoughnessUV.u,
			ccRoughnessUV.v
		);
		if (ccRoughnessTex) {
			clearcoatRoughness *= ccRoughnessTex.g / 255;
		}

		let transmission = mat.transmissionFactor;
		const transmissionUV = this._resolveUV(mat.transmissionMapUV, input);
		const transmissionTex = this._sampleTextureMap(
			mat.transmissionMap,
			transmissionUV.u,
			transmissionUV.v
		);
		if (transmissionTex) {
			transmission *= transmissionTex.r / 255;
		}

		let iridescence = mat.iridescenceFactor ?? 0.0;
		const iridescenceUV = this._resolveUV(mat.iridescenceMapUV, input);
		const iridescenceTex = this._sampleTextureMap(
			mat.iridescenceMap,
			iridescenceUV.u,
			iridescenceUV.v
		);
		if (iridescenceTex) {
			iridescence *= iridescenceTex.r / 255;
		}

		const iridescenceThicknessMinimum = Math.max(
			mat.iridescenceThicknessMinimum ?? 100.0,
			0
		);
		const iridescenceThicknessMaximum = Math.max(
			mat.iridescenceThicknessMaximum ?? 400.0,
			0
		);
		let iridescenceThickness = iridescenceThicknessMaximum;
		const iridescenceThicknessUV = this._resolveUV(
			mat.iridescenceThicknessMapUV,
			input
		);
		const iridescenceThicknessTex = this._sampleTextureMap(
			mat.iridescenceThicknessMap,
			iridescenceThicknessUV.u,
			iridescenceThicknessUV.v
		);
		if (iridescenceThicknessTex) {
			iridescenceThickness =
				iridescenceThicknessMinimum +
				(iridescenceThicknessMaximum - iridescenceThicknessMinimum) *
					(iridescenceThicknessTex.g / 255);
		}

		let anisotropyStrength = mat.anisotropyStrength ?? 0.0;
		let anisotropyDirectionX = 1.0;
		let anisotropyDirectionY = 0.0;
		const anisotropyUV = this._resolveUV(mat.anisotropyMapUV, input);
		const anisotropyTex = this._sampleTextureMap(
			mat.anisotropyMap,
			anisotropyUV.u,
			anisotropyUV.v
		);
		if (anisotropyTex) {
			anisotropyDirectionX = (anisotropyTex.r / 255) * 2.0 - 1.0;
			anisotropyDirectionY = (anisotropyTex.g / 255) * 2.0 - 1.0;
			anisotropyStrength *= anisotropyTex.b / 255;
		}
		const anisotropyDirLen = Math.hypot(
			anisotropyDirectionX,
			anisotropyDirectionY
		);
		if (anisotropyDirLen <= 1e-6) {
			anisotropyDirectionX = 1.0;
			anisotropyDirectionY = 0.0;
		} else {
			anisotropyDirectionX /= anisotropyDirLen;
			anisotropyDirectionY /= anisotropyDirLen;
		}
		const anisotropyRotation =
			Number.isFinite(mat.anisotropyRotation) ? mat.anisotropyRotation : 0.0;
		const anisotropyCos = Math.cos(anisotropyRotation);
		const anisotropySin = Math.sin(anisotropyRotation);
		const rotatedAnisotropyX =
			anisotropyDirectionX * anisotropyCos -
			anisotropyDirectionY * anisotropySin;
		const rotatedAnisotropyY =
			anisotropyDirectionX * anisotropySin +
			anisotropyDirectionY * anisotropyCos;

		let thickness = mat.thicknessFactor;
		const thicknessUV = this._resolveUV(mat.thicknessMapUV, input);
		const thicknessTex = this._sampleTextureMap(
			mat.thicknessMap,
			thicknessUV.u,
			thicknessUV.v
		);
		if (thicknessTex) {
			thickness *= thicknessTex.g / 255;
		}

		const res = this._cachedResult;
		res.albedo.r = clamp(albedo.r, 0, 255);
		res.albedo.g = clamp(albedo.g, 0, 255);
		res.albedo.b = clamp(albedo.b, 0, 255);
		res.opacity = resolveTransmissionCompositeAlpha(alpha, transmission);
		res.roughness = clamp(roughness);
		res.metalness = clamp(metalness);
		res.emissive.r = Math.max(0, emissive.r);
		res.emissive.g = Math.max(0, emissive.g);
		res.emissive.b = Math.max(0, emissive.b);
		res.reflectance = mat.reflectance ?? 0.5;
		res.specularFactor = clamp(specFactor, 0, 1);
		res.specularColor.r = Math.max(0, specColorLinear.r) * 255;
		res.specularColor.g = Math.max(0, specColorLinear.g) * 255;
		res.specularColor.b = Math.max(0, specColorLinear.b) * 255;
		res.emissiveIntensity = mat.emissiveIntensity ?? 1.0;
		res.occlusion = clamp(occlusion);
		res.clearcoat = clamp(clearcoat, 0, 1);
		res.clearcoatRoughness = clamp(clearcoatRoughness, 0, 1);
		res.sheenColor.r = Math.max(0, sheenColorLinear.r) * 255;
		res.sheenColor.g = Math.max(0, sheenColorLinear.g) * 255;
		res.sheenColor.b = Math.max(0, sheenColorLinear.b) * 255;
		res.sheenRoughness = clamp(sheenRoughness);
		res.transmission = clamp(transmission);
		res.ior = mat.ior ?? 1.5;
		res.iridescence = clamp(iridescence, 0, 1);
		res.iridescenceIor = Math.max(mat.iridescenceIor ?? 1.3, 1.0);
		res.iridescenceThickness = Math.max(0, iridescenceThickness);
		res.anisotropyStrength = clamp(anisotropyStrength, 0, 1);
		res.thickness = Math.max(0, thickness);
		res.attenuationDistance = mat.attenuationDistance;
		res.attenuationColor.r = clamp(mat.attenuationColor.r, 0, 255);
		res.attenuationColor.g = clamp(mat.attenuationColor.g, 0, 255);
		res.attenuationColor.b = clamp(mat.attenuationColor.b, 0, 255);

		const normal = res.normal;
		normal.x = input.normal.x;
		normal.y = input.normal.y;
		normal.z = input.normal.z;

		const normUV = this._resolveUV(mat.normalMapUV, input);
		const normalTex = this._sampleTextureMap(mat.normalMap, normUV.u, normUV.v);
		if (normalTex) {
			this._applyNormalMap(input, normalTex, normal, mat.normalScale ?? 1.0);
		} else {
			Vector3.normalizeInPlace(normal);
		}

		// Evaluate Clearcoat Normal mapping if applicable
		const clearcoatNormal = res.clearcoatNormal;
		clearcoatNormal.x = input.normal.x;
		clearcoatNormal.y = input.normal.y;
		clearcoatNormal.z = input.normal.z;

		const ccNormUV = this._resolveUV(mat.clearcoatNormalMapUV, input);
		const clearcoatNormalTex = this._sampleTextureMap(
			mat.clearcoatNormalMap,
			ccNormUV.u,
			ccNormUV.v
		);
		if (clearcoatNormalTex) {
			this._applyNormalMap(
				input,
				clearcoatNormalTex,
				clearcoatNormal,
				mat.clearcoatNormalScale ?? 1.0
			);
		} else {
			// Default to base normal if no clearcoat normal map is provided
			clearcoatNormal.x = normal.x;
			clearcoatNormal.y = normal.y;
			clearcoatNormal.z = normal.z;
		}
		if (Vector3.dot(clearcoatNormal, normal) < 0) {
			clearcoatNormal.x *= -1;
			clearcoatNormal.y *= -1;
			clearcoatNormal.z *= -1;
		}

		this._resolveAnisotropyFrame(
			input,
			normal,
			rotatedAnisotropyX,
			rotatedAnisotropyY,
			res.anisotropyTangent,
			res.anisotropyBitangent
		);

		return res;
	}
}
