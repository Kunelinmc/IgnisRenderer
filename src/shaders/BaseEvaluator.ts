import type { RGBA } from "../utils/Color";
import type { Material } from "../materials";
import type { ProjectedFace } from "../core/types";
import type { IMaterialEvaluator, SurfaceProperties } from "./types";

export abstract class BaseEvaluator<
	T extends SurfaceProperties = SurfaceProperties,
> implements IMaterialEvaluator<T> {
	constructor(protected material: Material) {}

	public setMaterial(material: Material): void {
		this.material = material;
	}

	protected _sampleMainMap(u: number, v: number): RGBA | null {
		const map = this.material.map;
		if (!map || !map.data) return null;

		let uu = u * map.repeat.x + map.offset.x;
		let vv = v * map.repeat.y + map.offset.y;

		if (map.wrapS === "Repeat") uu = uu - Math.floor(uu);
		else uu = Math.max(0, Math.min(1, uu));

		if (map.wrapT === "Repeat") vv = vv - Math.floor(vv);
		else vv = Math.max(0, Math.min(1, vv));

		let tx = Math.floor(uu * map.width);
		let ty = Math.floor(vv * map.height);

		tx = Math.max(0, Math.min(map.width - 1, tx));
		ty = Math.max(0, Math.min(map.height - 1, ty));

		const idx = (ty * map.width + tx) << 2;
		return {
			r: map.data[idx],
			g: map.data[idx + 1],
			b: map.data[idx + 2],
			a: map.data[idx + 3] / 255,
		};
	}

	abstract evaluate(u: number, v: number, face: ProjectedFace): T | null;
}
