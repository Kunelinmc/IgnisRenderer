import type { RGBA } from "../../foundation/Color";
import type { Material } from "../../materials";
import type { ProjectedFace } from "../../core/types";
import type {
	IMaterialEvaluator,
	SurfaceProperties,
	FragmentInput,
} from "./types";
import { clamp } from "../../maths/Common";

export abstract class BaseEvaluator<
	T extends SurfaceProperties = SurfaceProperties,
> implements IMaterialEvaluator<T> {
	constructor(protected material: Material) {}

	/**
	 * @deprecated Use compile(material) instead.
	 */
	public setMaterial(material: Material): void {
		this.compile(material);
	}

	public compile(material: Material): void {
		this.material = material;
	}

	protected _sampleTextureMap(
		map: Material["map"],
		u: number,
		v: number
	): RGBA | null {
		if (!map || !map.data) return null;
		if (map.width <= 0 || map.height <= 0) return null;

		let uu = u * map.repeat.x;
		let vv = v * map.repeat.y;

		if (map.rotation !== 0) {
			const c = Math.cos(map.rotation);
			const s = Math.sin(map.rotation);
			const ru = uu * c - vv * s;
			const rv = uu * s + vv * c;
			uu = ru;
			vv = rv;
		}

		uu += map.offset.x;
		vv += map.offset.y;

		if (map.wrapS === "Repeat") uu = uu - Math.floor(uu);
		else if (map.wrapS === "MirroredRepeat") {
			const iter = Math.floor(uu);
			uu = uu - iter;
			if (Math.abs(iter) % 2 === 1) uu = 1.0 - uu;
		} else uu = clamp(uu);

		if (map.wrapT === "Repeat") vv = vv - Math.floor(vv);
		else if (map.wrapT === "MirroredRepeat") {
			const iter = Math.floor(vv);
			vv = vv - iter;
			if (Math.abs(iter) % 2 === 1) vv = 1.0 - vv;
		} else vv = clamp(vv);

		let tx = Math.floor(uu * map.width);
		let ty = Math.floor(vv * map.height);

		tx = Math.max(0, Math.min(map.width - 1, tx));
		ty = Math.max(0, Math.min(map.height - 1, ty));

		const idx = (ty * map.width + tx) << 2;
		if (map.colorSpace === "HDR" || map.colorSpace === "Linear") {
			const isFloat = map.data instanceof Float32Array;
			const colorScale = isFloat ? 255 : 1;
			const alphaRaw = map.data[idx + 3];
			return {
				r: Math.max(0, Math.min(255, (map.data[idx] ?? 0) * colorScale)),
				g: Math.max(0, Math.min(255, (map.data[idx + 1] ?? 0) * colorScale)),
				b: Math.max(0, Math.min(255, (map.data[idx + 2] ?? 0) * colorScale)),
				a:
					alphaRaw === undefined ? 1
					: isFloat ? clamp(alphaRaw)
					: clamp(alphaRaw / 255),
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
