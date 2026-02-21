import { BaseEvaluator } from "./BaseEvaluator";
import type { PBRMaterial } from "../materials";
import type { ProjectedFace } from "../core/types";
import type { PBRSurfaceProperties } from "./types";

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
		u: number,
		v: number,
		face: ProjectedFace
	): PBRSurfaceProperties | null {
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

		return res;
	}
}
