import { Camera } from "./Camera";
import { Matrix4 } from "../maths/Matrix4";

/**
 * An Orthographic Camera using a size-based orthographic projection.
 */
export class OrthographicCamera extends Camera {
	/** Provide the vertical size of the visible area. The horizontal size is computed via the aspect ratio. */
	public size: number = 100;
	public isOrthographicCamera = true;

	constructor(size: number = 100) {
		super();
		this.size = size;

		this.updateMatrices();
	}

	public override calculateProjectionMatrix(): Matrix4 {
		const halfHeight = this.size / 2;
		const halfWidth = halfHeight * this.aspectRatio;

		return Matrix4.ortho(
			-halfWidth,
			halfWidth,
			-halfHeight,
			halfHeight,
			this.near,
			this.far
		);
	}
}
