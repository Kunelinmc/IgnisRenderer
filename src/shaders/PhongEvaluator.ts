import { BaseEvaluator } from "./BaseEvaluator";
import type { PhongMaterial, Material } from "../materials";
import type { ProjectedFace } from "../core/types";
import type { PhongSurfaceProperties, FragmentInput } from "./types";
import { Vector3 } from "../maths/Vector3";

export class PhongEvaluator extends BaseEvaluator<PhongSurfaceProperties> {
	private _mat!: PhongMaterial;
	private _cachedResult: PhongSurfaceProperties = {
		type: "phong",
		albedo: { r: 0, g: 0, b: 0 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1.0,
		ambient: { r: 0, g: 0, b: 0 },
		specular: { r: 0, g: 0, b: 0 },
		shininess: 0,
	};

	constructor(material: Material) {
		super(material);
		this._mat = material as PhongMaterial;
	}

	public compile(material: Material): void {
		super.compile(material);
		this._mat = material as PhongMaterial;
	}

	public evaluate(
		input: FragmentInput,
		face: ProjectedFace
	): PhongSurfaceProperties | null {
		const u = input.u;
		const v = input.v;
		const mat = this._mat;
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
		const ambient = mat.ambient || mat.diffuse || { r: 255, g: 255, b: 255 };
		res.ambient.r = ambient.r;
		res.ambient.g = ambient.g;
		res.ambient.b = ambient.b;
		const spec = mat.specular || { r: 255, g: 255, b: 255 };
		res.specular.r = spec.r;
		res.specular.g = spec.g;
		res.specular.b = spec.b;
		res.shininess = mat.shininess || 32;

		res.normal.x = input.normal.x;
		res.normal.y = input.normal.y;
		res.normal.z = input.normal.z;
		Vector3.normalizeInPlace(res.normal);

		return res;
	}
}
