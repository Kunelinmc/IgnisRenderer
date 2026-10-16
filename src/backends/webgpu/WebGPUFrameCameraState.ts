import { Matrix4 } from "../../maths/Matrix4";

class CachedCameraMatrix {
	public readonly inverse = Matrix4.identity();
	private readonly _values = new Float64Array(16);
	private _initialized = false;

	public validate(matrix: Matrix4, label: string): void {
		const elements = matrix?.elements;
		for (let row = 0; row < 4; row++) {
			for (let column = 0; column < 4; column++) {
				const value = elements?.[row]?.[column];
				if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
					throw new Error(`WebGPU camera ${label} matrix must contain finite f32 values.`);
				}
			}
		}
	}

	public update(matrix: Matrix4, label: string): void {
		const elements = matrix.elements;
		let changed = !this._initialized;
		for (let row = 0; row < 4; row++) {
			for (let column = 0; column < 4; column++) {
				changed ||= this._values[row * 4 + column] !== elements[row][column];
			}
		}
		if (!changed) return;
		// A rejected inverse must force recomputation even when the old input returns.
		this._initialized = false;
		// Matrix4 owns inversion; the output survives across frame preparations.
		if (!Matrix4.inverse(matrix, this.inverse)) {
			throw new Error(`WebGPU camera ${label} matrix is singular and cannot be inverted.`);
		}
		this.validate(this.inverse, `${label} inverse`);
		for (let row = 0; row < 4; row++) {
			for (let column = 0; column < 4; column++) {
				this._values[row * 4 + column] = elements[row][column];
			}
		}
		this._initialized = true;
	}
}

/**
 * @internal WebGPU view-scope camera validation and inverse cache.
 * Renderer submissions own the public camera API.
 */
export class WebGPUFrameCameraState {
	private readonly _view = new CachedCameraMatrix();
	private readonly _projection = new CachedCameraMatrix();
	private readonly _viewProjection = new CachedCameraMatrix();

	/** @internal WebGPU-owned inverse view; applications configure the Renderer camera. */
	public get inverseViewMatrix(): Matrix4 {
		return this._view.inverse;
	}

	/** @internal WebGPU-owned inverse projection; applications configure the Renderer camera. */
	public get inverseProjectionMatrix(): Matrix4 {
		return this._projection.inverse;
	}

	/**
	 * @internal Validates camera values before WebGPU temporal or GPU updates.
	 * Reuses inverse outputs and refreshes them after numeric matrix changes.
	 * Throws for non-finite or singular inputs and non-finite inverse results.
	 * Applications should configure the Renderer camera instead of calling this service.
	 * @param view - Unjittered world-to-view matrix.
	 * @param projection - Unjittered view-to-clip matrix.
	 * @param viewProjection - Unjittered world-to-clip matrix.
	 */
	public prepare(view: Matrix4, projection: Matrix4, viewProjection: Matrix4): void {
		this._view.validate(view, "view");
		this._projection.validate(projection, "projection");
		this._viewProjection.validate(viewProjection, "view-projection");
		this._view.update(view, "view");
		this._projection.update(projection, "projection");
		this._viewProjection.update(viewProjection, "view-projection");
	}
}
