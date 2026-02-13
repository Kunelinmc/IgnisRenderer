import { BaseEvaluator } from "./BaseEvaluator";
import type { PhongMaterial } from "../materials";
import type { ProjectedFace } from "../core/types";
import type { PhongSurfaceProperties } from "./types";

export class PhongEvaluator extends BaseEvaluator<PhongSurfaceProperties> {
	private _cachedResult: PhongSurfaceProperties = {
		type: "phong",
		albedo: { r: 0, g: 0, b: 0 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		specular: { r: 0, g: 0, b: 0 },
		shininess: 0,
	};

	public evaluate(
		u: number,
		v: number,
		face: ProjectedFace
	): PhongSurfaceProperties | null {
		const mat = this.material as PhongMaterial;
		let color = mat.diffuse || { r: 255, g: 255, b: 255 };
		let alpha = mat.opacity ?? 1;

		const tex = this._sampleMainMap(u, v);
		if (tex) {
			color = {
				r: (color.r * tex.r) / 255,
				g: (color.g * tex.g) / 255,
				b: (color.b * tex.b) / 255,
			};
			alpha *= tex.a;
		}

		if (mat.alphaMode === "MASK" && alpha < (mat.alphaCutoff ?? 0.5))
			return null;

		const res = this._cachedResult;
		res.albedo.r = color.r;
		res.albedo.g = color.g;
		res.albedo.b = color.b;
		res.opacity = alpha;
		const spec = mat.specular || { r: 255, g: 255, b: 255 };
		res.specular.r = spec.r;
		res.specular.g = spec.g;
		res.specular.b = spec.b;
		res.shininess = mat.shininess || 32;

		return res;
	}
}
