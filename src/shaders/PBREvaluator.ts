import { BaseEvaluator } from "./BaseEvaluator";
import type { PBRMaterial } from "../materials";
import type { ProjectedFace } from "../core/types";
import type { PBRSurfaceProperties, FragmentInput } from "./types";
import { Vector3 } from "../maths/Vector3";

export class PBREvaluator extends BaseEvaluator<PBRSurfaceProperties> {
	private _cachedResult: PBRSurfaceProperties = {
		type: "pbr",
		albedo: { r: 0, g: 0, b: 0 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1.0,
		roughness: 0,
		metalness: 0,
		f0: { r: 0, g: 0, b: 0 },
		occlusion: 1.0,
		clearcoat: 0.0,
		clearcoatRoughness: 0.0,
	};

	public evaluate(
		input: FragmentInput,
		face: ProjectedFace
	): PBRSurfaceProperties | null {
		const u = input.u;
		const v = input.v;
		const mat = this.material as PBRMaterial;
		let albedo = mat.albedo || { r: 255, g: 255, b: 255 };
		let alpha = mat.opacity ?? 1;
		let roughness = mat.roughness ?? 0.5;
		let metalness = mat.metalness ?? 0.0;
		let occlusion = 1.0;

		const tex = this._sampleMainMap(u, v);
		if (tex) {
			albedo = {
				r: (albedo.r * tex.r) / 255,
				g: (albedo.g * tex.g) / 255,
				b: (albedo.b * tex.b) / 255,
			};
			alpha *= tex.a;
		}

		if (mat.alphaMode === "MASK" && alpha < (mat.alphaCutoff ?? 0.5))
			return null;

		const metallicRoughnessTex = this._sampleTextureMap(
			mat.metallicRoughnessMap,
			u,
			v
		);
		if (metallicRoughnessTex) {
			// glTF metallicRoughness texture channels:
			// G = roughness, B = metallic
			roughness *= metallicRoughnessTex.g / 255;
			metalness *= metallicRoughnessTex.b / 255;
		}

		let emissive = mat.emissive || { r: 0, g: 0, b: 0 };
		const emissiveTex = this._sampleTextureMap(mat.emissiveMap, u, v);
		if (emissiveTex) {
			emissive = {
				r: (emissive.r * emissiveTex.r) / 255,
				g: (emissive.g * emissiveTex.g) / 255,
				b: (emissive.b * emissiveTex.b) / 255,
			};
		}

		const occlusionTex = this._sampleTextureMap(mat.occlusionMap, u, v);
		if (occlusionTex) {
			// glTF occlusion is stored in R channel and affects indirect light.
			occlusion = occlusionTex.r / 255;
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
		const f0 = mat.f0 || { r: 10, g: 10, b: 10 };
		res.f0.r = f0.r;
		res.f0.g = f0.g;
		res.f0.b = f0.b;
		res.emissiveIntensity = mat.emissiveIntensity ?? 1.0;
		res.occlusion = Math.max(0, Math.min(1, occlusion));
		res.clearcoat = mat.clearcoat ?? 0.0;
		res.clearcoatRoughness = Math.max(
			0,
			Math.min(1, mat.clearcoatRoughness ?? 0.0)
		);

		const normal = res.normal;
		normal.x = input.normal.x;
		normal.y = input.normal.y;
		normal.z = input.normal.z;

		const normalTex = this._sampleTextureMap(mat.normalMap, u, v);
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
				const tNormX = (normalTex.r / 255) * 2 - 1;
				const tNormY = (normalTex.g / 255) * 2 - 1;
				const tNormZ = (normalTex.b / 255) * 2 - 1;

				// Gram-Schmidt: keep T orthogonal to N to avoid invalid TBN on skewed assets.
				const ndotT =
					N.x * input.tangent.x +
					N.y * input.tangent.y +
					N.z * input.tangent.z;
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
