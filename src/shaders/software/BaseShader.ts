import type { RGB } from "../../foundation/Color";
import type { ProjectedFace } from "../../core/types";
import type {
	FragmentInput,
	FragmentOutput,
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
	protected _cachedOutput: FragmentOutput = { color: this._cachedColor };
	protected _lastOpacity = 1;
	protected _lastSurfaceNormal: T["normal"] | null = null;

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

	public getSurfaceNormal(): T["normal"] | null {
		return this._lastSurfaceNormal;
	}

	protected _evaluateSurface(input: FragmentInput): T | null {
		const surface = this._evaluator.evaluate(input, this._face);
		if (!surface) {
			this._lastSurfaceNormal = null;
			return null;
		}
		this._context.surfaceModifier?.apply(input, surface);
		this._lastSurfaceNormal = surface.normal;
		return surface;
	}

	abstract shade(input: FragmentInput): FragmentOutput | null;
}
