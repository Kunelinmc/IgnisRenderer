import { LitShader } from "./LitShader";
import type { RGB } from "../utils/Color";
import type { ProjectedFace } from "../core/types";
import type { FragmentInput, ShaderContext, SurfaceProperties } from "./types";

export class FlatLitShader<
	T extends SurfaceProperties = SurfaceProperties,
> extends LitShader<T> {
	private _faceColor: RGB | null = null;
	private _faceColorStorage: RGB = { r: 0, g: 0, b: 0 };
	private _faceOpacity = 1;

	public initialize(face: ProjectedFace, context: ShaderContext): void {
		super.initialize(face, context);
		const map = face.material?.map;
		if (!map) {
			const color = this.shade({
				zCam: 0,
				u: 0,
				v: 0,
				worldX: face.center.x,
				worldY: face.center.y,
				worldZ: face.center.z,
				normalX: face.normal?.x ?? 0,
				normalY: face.normal?.y ?? 0,
				normalZ: face.normal?.z ?? 1,
			});
			if (color) {
				this._faceColorStorage.r = color.r;
				this._faceColorStorage.g = color.g;
				this._faceColorStorage.b = color.b;
				this._faceColor = this._faceColorStorage;
				this._faceOpacity = this.getOpacity();
			} else {
				this._faceColor = null;
				this._faceOpacity = 0;
			}
		} else {
			this._faceColor = null;
			this._faceOpacity = 1;
		}
	}

	public shade(input: FragmentInput): RGB | null {
		if (this._faceColor) {
			this._lastOpacity = this._faceOpacity;
			return this._faceColor;
		}
		return super.shade(input);
	}
}
