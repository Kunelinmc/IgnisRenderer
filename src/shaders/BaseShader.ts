import type { RGB } from "../utils/Color";
import type { ProjectedFace } from "../core/types";
import type {
	FragmentInput,
	IMaterialEvaluator,
	IShader,
	ShaderContext,
	SurfaceProperties,
} from "./types";

export abstract class BaseShader<
	T extends SurfaceProperties = SurfaceProperties,
> implements IShader {
	protected _face!: ProjectedFace;
	protected _context!: ShaderContext;
	protected _cachedColor: RGB = { r: 0, g: 0, b: 0 };
	protected _lastOpacity = 1;

	constructor(protected _evaluator: IMaterialEvaluator<T>) {}

	public setEvaluator(evaluator: IMaterialEvaluator<T>): void {
		this._evaluator = evaluator;
	}

	public initialize(face: ProjectedFace, context: ShaderContext): void {
		this._face = face;
		this._context = context;
	}

	public getOpacity(): number {
		return this._lastOpacity;
	}

	abstract shade(input: FragmentInput): RGB | null;
}
