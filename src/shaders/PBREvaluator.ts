import { BaseEvaluator } from "./BaseEvaluator";
import type { PBRMaterial, Material } from "../materials";
import type { ProjectedFace } from "../core/types";
import type { PBRSurfaceProperties, FragmentInput } from "./types";
import { Vector3 } from "../maths/Vector3";
import { clamp, sRGBToLinear } from "../maths/Common";

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
		sheenColor: { r: 0, g: 0, b: 0 },
		sheenRoughness: 0.0,
		transmission: 0.0,
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

	public evaluate(
		input: FragmentInput,
		face: ProjectedFace
	): PBRSurfaceProperties | null {
		const u = input.u;
		const v = input.v;
		const mat = this._mat;
		let albedo = mat.albedo || { r: 255, g: 255, b: 255 };
		let alpha = mat.opacity ?? 1;
		let roughness = mat.roughness ?? 0.5;
		let metalness = mat.metalness ?? 0.0;
		let occlusion = 1.0;

		// Select UV set for main map
		const albedoUV =
			mat.albedoMapUV === 1 ?
				{ u: input.u2, v: input.v2 }
			:	{ u: input.u, v: input.v };
		const tex = this._sampleTextureMap(mat.map, albedoUV.u, albedoUV.v);
		if (tex) {
			// Texture is sRGB, decode to linear before multiplying with linear factor
			const texLinear = {
				r: sRGBToLinear(tex.r / 255),
				g: sRGBToLinear(tex.g / 255),
				b: sRGBToLinear(tex.b / 255),
			};
			albedo = {
				r: albedo.r * texLinear.r,
				g: albedo.g * texLinear.g,
				b: albedo.b * texLinear.b,
			};
			alpha *= tex.a;
		}

		if (mat.alphaMode === "MASK" && alpha < (mat.alphaCutoff ?? 0.5))
			return null;

		const mrUV =
			mat.metallicRoughnessMapUV === 1 ?
				{ u: input.u2, v: input.v2 }
			:	{ u: input.u, v: input.v };
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

		let emissive = mat.emissive || { r: 0, g: 0, b: 0 };
		const emissiveUV =
			mat.emissiveMapUV === 1 ?
				{ u: input.u2, v: input.v2 }
			:	{ u: input.u, v: input.v };
		const emissiveTex = this._sampleTextureMap(
			mat.emissiveMap,
			emissiveUV.u,
			emissiveUV.v
		);
		if (emissiveTex) {
			// Emissive textures are sRGB in glTF
			const texLinear = {
				r: sRGBToLinear(emissiveTex.r / 255),
				g: sRGBToLinear(emissiveTex.g / 255),
				b: sRGBToLinear(emissiveTex.b / 255),
			};
			emissive = {
				r: emissive.r * texLinear.r,
				g: emissive.g * texLinear.g,
				b: emissive.b * texLinear.b,
			};
		}

		const occlusionUV =
			mat.occlusionMapUV === 1 ?
				{ u: input.u2, v: input.v2 }
			:	{ u: input.u, v: input.v };
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
		const specUV =
			mat.specularMapUV === 1 ?
				{ u: input.u2, v: input.v2 }
			:	{ u: input.u, v: input.v };
		const specTex = this._sampleTextureMap(mat.specularMap, specUV.u, specUV.v);
		if (specTex) {
			specFactor *= specTex.a;
		}

		const specColorInput = mat.specularColor || { r: 255, g: 255, b: 255 };
		let specColorLinear = {
			r: Math.max(0, specColorInput.r / 255),
			g: Math.max(0, specColorInput.g / 255),
			b: Math.max(0, specColorInput.b / 255),
		};
		const specColorUV =
			mat.specularColorMapUV === 1 ?
				{ u: input.u2, v: input.v2 }
			:	{ u: input.u, v: input.v };
		const specColorTex = this._sampleTextureMap(
			mat.specularColorMap,
			specColorUV.u,
			specColorUV.v
		);
		if (specColorTex) {
			const colorSpace = mat.specularColorMap?.colorSpace ?? "sRGB";
			const texLinear =
				colorSpace === "Linear" || colorSpace === "HDR" ?
					{
						r: Math.max(0, specColorTex.r / 255),
						g: Math.max(0, specColorTex.g / 255),
						b: Math.max(0, specColorTex.b / 255),
					}
				:	{
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
			r: Math.max(0, mat.sheenColorFactor.r / 255),
			g: Math.max(0, mat.sheenColorFactor.g / 255),
			b: Math.max(0, mat.sheenColorFactor.b / 255),
		};
		const sheenColorUV =
			mat.sheenColorMapUV === 1 ?
				{ u: input.u2, v: input.v2 }
			:	{ u: input.u, v: input.v };
		const sheenColorTex = this._sampleTextureMap(
			mat.sheenColorMap,
			sheenColorUV.u,
			sheenColorUV.v
		);
		if (sheenColorTex) {
			sheenColorLinear = {
				r:
					sheenColorLinear.r * sRGBToLinear(Math.max(0, sheenColorTex.r / 255)),
				g:
					sheenColorLinear.g * sRGBToLinear(Math.max(0, sheenColorTex.g / 255)),
				b:
					sheenColorLinear.b * sRGBToLinear(Math.max(0, sheenColorTex.b / 255)),
			};
		}

		let sheenRoughness = mat.sheenRoughnessFactor;
		const sheenRoughnessUV =
			mat.sheenRoughnessMapUV === 1 ?
				{ u: input.u2, v: input.v2 }
			:	{ u: input.u, v: input.v };
		const sheenRoughnessTex = this._sampleTextureMap(
			mat.sheenRoughnessMap,
			sheenRoughnessUV.u,
			sheenRoughnessUV.v
		);
		if (sheenRoughnessTex) {
			sheenRoughness *= sheenRoughnessTex.a / 255;
		}

		let transmission = mat.transmissionFactor;
		const transmissionUV =
			mat.transmissionMapUV === 1 ?
				{ u: input.u2, v: input.v2 }
			:	{ u: input.u, v: input.v };
		const transmissionTex = this._sampleTextureMap(
			mat.transmissionMap,
			transmissionUV.u,
			transmissionUV.v
		);
		if (transmissionTex) {
			transmission *= transmissionTex.r / 255;
		}

		let thickness = mat.thicknessFactor;
		const thicknessUV =
			mat.thicknessMapUV === 1 ?
				{ u: input.u2, v: input.v2 }
			:	{ u: input.u, v: input.v };
		const thicknessTex = this._sampleTextureMap(
			mat.thicknessMap,
			thicknessUV.u,
			thicknessUV.v
		);
		if (thicknessTex) {
			thickness *= thicknessTex.g / 255;
		}

		const res = this._cachedResult;
		res.albedo.r = albedo.r;
		res.albedo.g = albedo.g;
		res.albedo.b = albedo.b;
		res.opacity = alpha;
		res.roughness = Math.max(0, Math.min(1, roughness));
		res.metalness = Math.max(0, Math.min(1, metalness));
		res.emissive.r = emissive.r;
		res.emissive.g = emissive.g;
		res.emissive.b = emissive.b;
		res.reflectance = mat.reflectance ?? 0.5;
		res.specularFactor = clamp(specFactor, 0, 1);
		res.specularColor.r = Math.max(0, specColorLinear.r) * 255;
		res.specularColor.g = Math.max(0, specColorLinear.g) * 255;
		res.specularColor.b = Math.max(0, specColorLinear.b) * 255;
		res.emissiveIntensity = mat.emissiveIntensity ?? 1.0;
		res.occlusion = Math.max(0, Math.min(1, occlusion));
		res.clearcoat = mat.clearcoat ?? 0.0;
		res.clearcoatRoughness = Math.max(
			0,
			Math.min(1, mat.clearcoatRoughness ?? 0.0)
		);
		res.sheenColor.r = Math.max(0, sheenColorLinear.r) * 255;
		res.sheenColor.g = Math.max(0, sheenColorLinear.g) * 255;
		res.sheenColor.b = Math.max(0, sheenColorLinear.b) * 255;
		res.sheenRoughness = Math.max(0, Math.min(1, sheenRoughness));
		res.transmission = Math.max(0, Math.min(1, transmission));
		res.thickness = Math.max(0, thickness);
		res.attenuationDistance = mat.attenuationDistance;
		res.attenuationColor.r = mat.attenuationColor.r;
		res.attenuationColor.g = mat.attenuationColor.g;
		res.attenuationColor.b = mat.attenuationColor.b;

		const normal = res.normal;
		normal.x = input.normal.x;
		normal.y = input.normal.y;
		normal.z = input.normal.z;

		const normUV =
			mat.normalMapUV === 1 ?
				{ u: input.u2, v: input.v2 }
			:	{ u: input.u, v: input.v };
		const normalTex = this._sampleTextureMap(mat.normalMap, normUV.u, normUV.v);
		if (normalTex) {
			const N = { x: normal.x, y: normal.y, z: normal.z };
			Vector3.normalizeInPlace(N);

			const tangentLenSq =
				input.tangent.x * input.tangent.x +
				input.tangent.y * input.tangent.y +
				input.tangent.z * input.tangent.z;
			const hasValidTangent =
				tangentLenSq > 1e-12 && Math.abs(input.tangent.w) > 1e-6;

			if (hasValidTangent) {
				const normalScale = mat.normalScale ?? 1.0;
				const tNormX = ((normalTex.r / 255) * 2 - 1) * normalScale;
				const tNormY = ((normalTex.g / 255) * 2 - 1) * normalScale;
				const tNormZ = (normalTex.b / 255) * 2 - 1;

				// Gram-Schmidt: keep T orthogonal to N to avoid invalid TBN on skewed assets.
				const ndotT =
					N.x * input.tangent.x + N.y * input.tangent.y + N.z * input.tangent.z;
				let tx = input.tangent.x - N.x * ndotT;
				let ty = input.tangent.y - N.y * ndotT;
				let tz = input.tangent.z - N.z * ndotT;
				const tLen = Math.hypot(tx, ty, tz);

				if (tLen > 1e-6) {
					const invTLen = 1 / tLen;
					tx *= invTLen;
					ty *= invTLen;
					tz *= invTLen;

					const handedness = input.tangent.w < 0 ? -1 : 1;
					const bx = (N.y * tz - N.z * ty) * handedness;
					const by = (N.z * tx - N.x * tz) * handedness;
					const bz = (N.x * ty - N.y * tx) * handedness;

					normal.x = tx * tNormX + bx * tNormY + N.x * tNormZ;
					normal.y = ty * tNormX + by * tNormY + N.y * tNormZ;
					normal.z = tz * tNormX + bz * tNormY + N.z * tNormZ;
					Vector3.normalizeInPlace(normal);
				} else {
					normal.x = N.x;
					normal.y = N.y;
					normal.z = N.z;
				}
			} else {
				normal.x = N.x;
				normal.y = N.y;
				normal.z = N.z;
			}
		} else {
			Vector3.normalizeInPlace(normal);
		}

		return res;
	}
}
