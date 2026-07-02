import type { RGBA } from "../../foundation/Color";
import type { Material } from "../../materials";
import type { ProjectedFace } from "../../core/types";
import type {
	IMaterialEvaluator,
	SurfaceProperties,
	FragmentInput,
} from "./types";
import { sampleSoftwareTextureMap } from "./textureSampling";

export abstract class BaseEvaluator<
	T extends SurfaceProperties = SurfaceProperties,
> implements IMaterialEvaluator<T> {
	constructor(protected material: Material) {}

	public compile(material: Material): void {
		this.material = material;
	}

	protected _sampleTextureMap(
		map: Material["map"],
		u: number,
		v: number
	): RGBA | null {
		return sampleSoftwareTextureMap(map, u, v);
	}

	protected _sampleMainMap(u: number, v: number): RGBA | null {
		return this._sampleTextureMap(this.material.map, u, v);
	}

	abstract evaluate(input: FragmentInput, face: ProjectedFace): T | null;
}
