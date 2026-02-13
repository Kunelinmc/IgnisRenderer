import { Vector3 } from "../maths/Vector3";
import { BaseShader } from "./BaseShader";
import type { IVector3 } from "../maths/types";
import type { RGB } from "../utils/Color";
import type {
	FragmentInput,
	ILightingStrategy,
	IMaterialEvaluator,
	SurfaceProperties,
} from "./types";

export class LitShader<
	T extends SurfaceProperties = SurfaceProperties,
> extends BaseShader<T> {
	private _world: IVector3 = { x: 0, y: 0, z: 0 };
	private _normal: IVector3 = { x: 0, y: 0, z: 0 };
	private _viewDir: IVector3 = { x: 0, y: 0, z: 0 };

	constructor(
		private _strategy: ILightingStrategy<T>,
		evaluator: IMaterialEvaluator<T>
	) {
		super(evaluator);
	}

	public shade(input: FragmentInput): RGB | null {
		const surface = this._evaluator.evaluate(input.u, input.v, this._face);
		if (!surface) return null;
		this._lastOpacity = surface.opacity;

		const world = this._world;
		world.x = input.worldX;
		world.y = input.worldY;
		world.z = input.worldZ;

		const normal = this._normal;
		normal.x = input.normalX;
		normal.y = input.normalY;
		normal.z = input.normalZ;
		Vector3.normalizeInPlace(normal);

		const viewDir = this._viewDir;
		viewDir.x = this._context.cameraPos.x - world.x;
		viewDir.y = this._context.cameraPos.y - world.y;
		viewDir.z = this._context.cameraPos.z - world.z;

		// Flip normal for double-sided materials when viewing from the back side
		if (this._face.material?.doubleSided) {
			if (Vector3.dot(normal, viewDir) < 0) {
				normal.x *= -1;
				normal.y *= -1;
				normal.z *= -1;
			}
		}

		const litColor = this._strategy.calculate(
			world,
			normal,
			viewDir,
			surface,
			this._context
		);

		const res = this._cachedColor;
		res.r = litColor.r;
		res.g = litColor.g;
		res.b = litColor.b;
		return res;
	}
}
