import { LitShader } from "./LitShader";
import type { RGB } from "../utils/Color";
import type { ProjectedFace } from "../core/types";
import type {
	FragmentInput,
	FragmentOutput,
	ShaderContext,
	SurfaceProperties,
} from "./types";

export class FlatLitShader<
	T extends SurfaceProperties = SurfaceProperties,
> extends LitShader<T> {
	private _faceOutput: FragmentOutput | null = null;
	private _faceColorStorage: RGB = { r: 0, g: 0, b: 0 };
	private _faceOpacity = 1;

	public initialize(face: ProjectedFace, context: ShaderContext): void {
		super.initialize(face, context);
		const map = face.material?.map;
		if (!map) {
			const output = this.shade({
				zCam: 0,
				u: 0,
				v: 0,
				u2: 0,
				v2: 0,
				world: { x: face.center.x, y: face.center.y, z: face.center.z },
				normal: {
					x: face.normal?.x ?? 0,
					y: face.normal?.y ?? 0,
					z: face.normal?.z ?? 1,
				},
				tangent: { x: 1, y: 0, z: 0, w: 1 },
			});
			if (output) {
				this._faceColorStorage.r = output.color.r;
				this._faceColorStorage.g = output.color.g;
				this._faceColorStorage.b = output.color.b;
				this._faceOutput = { color: this._faceColorStorage };
				this._faceOpacity = this.getOpacity();
			} else {
				this._faceOutput = null;
				this._faceOpacity = 0;
			}
		} else {
			this._faceOutput = null;
			this._faceOpacity = 1;
		}
	}

	public shade(input: FragmentInput): FragmentOutput | null {
		if (this._faceOutput) {
			this._lastOpacity = this._faceOpacity;
			return this._faceOutput;
		}
		return super.shade(input);
	}
}
