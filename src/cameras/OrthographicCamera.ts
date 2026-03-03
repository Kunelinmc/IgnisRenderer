import { Camera, CameraType } from "./Camera";
import { Matrix4 } from "../maths/Matrix4";

/**
 * An Orthographic Camera using a size-based orthographic projection.
 */
export class OrthographicCamera extends Camera {
	constructor(size: number = 100) {
		super();
		this.type = CameraType.Orthographic;
		this.size = size;

		this.updateMatrices();
	}
}
