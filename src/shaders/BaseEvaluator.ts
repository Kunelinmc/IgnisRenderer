import type { RGBA } from "../utils/Color";
import type { Material } from "../materials";
import type { ProjectedFace } from "../core/types";
import type {
	IMaterialEvaluator,
	SurfaceProperties,
	FragmentInput,
} from "./types";

export abstract class BaseEvaluator<
	T extends SurfaceProperties = SurfaceProperties,
> implements IMaterialEvaluator<T> {
	constructor(protected material: Material) {}

	public setMaterial(material: Material): void {
		this.material = material;
	}

	protected _sampleTextureMap(
		map: Material["map"],
		u: number,
		v: number
	): RGBA | null {
		if (!map || !map.data) return null;
		if (map.width <= 0 || map.height <= 0) return null;

		let uu = u * map.repeat.x + map.offset.x;
		let vv = v * map.repeat.y + map.offset.y;

		if (map.wrapS === "Repeat") uu = uu - Math.floor(uu);
		else if (map.wrapS === "MirroredRepeat") {
			const iter = Math.floor(uu);
			uu = uu - iter;
			if (Math.abs(iter) % 2 === 1) uu = 1.0 - uu;
		} else uu = Math.max(0, Math.min(1, uu));

		if (map.wrapT === "Repeat") vv = vv - Math.floor(vv);
		else if (map.wrapT === "MirroredRepeat") {
			const iter = Math.floor(vv);
			vv = vv - iter;
			if (Math.abs(iter) % 2 === 1) vv = 1.0 - vv;
		} else vv = Math.max(0, Math.min(1, vv));

		let tx = Math.floor(uu * map.width);
		let ty = Math.floor(vv * map.height);

		tx = Math.max(0, Math.min(map.width - 1, tx));
		ty = Math.max(0, Math.min(map.height - 1, ty));

		const idx = (ty * map.width + tx) << 2;
		if (map.colorSpace === "HDR" || map.colorSpace === "Linear") {
			// HDR / Linear textures store floating-point data; scale to [0-255]
			const r = map.data[idx] ?? 0;
			const g = map.data[idx + 1] ?? 0;
			const b = map.data[idx + 2] ?? 0;
			return {
				r: Math.max(0, Math.min(255, r * 255)),
				g: Math.max(0, Math.min(255, g * 255)),
				b: Math.max(0, Math.min(255, b * 255)),
				a: map.data[idx + 3] ?? 1,
			};
		}

		const alpha = map.data[idx + 3] ?? 255;
		return {
			r: map.data[idx],
			g: map.data[idx + 1],
			b: map.data[idx + 2],
			a: alpha / 255,
		};
	}

	protected _sampleMainMap(u: number, v: number): RGBA | null {
		return this._sampleTextureMap(this.material.map, u, v);
	}

	abstract evaluate(input: FragmentInput, face: ProjectedFace): T | null;
}
