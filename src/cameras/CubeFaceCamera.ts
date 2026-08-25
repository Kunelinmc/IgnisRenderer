import { Camera } from "./Camera";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";

const DIRECTIONS: readonly IVector3[] = [
	{ x: 1, y: 0, z: 0 },
	{ x: -1, y: 0, z: 0 },
	{ x: 0, y: 1, z: 0 },
	{ x: 0, y: -1, z: 0 },
	{ x: 0, y: 0, z: 1 },
	{ x: 0, y: 0, z: -1 },
];

const UP_VECTORS: readonly IVector3[] = [
	{ x: 0, y: -1, z: 0 },
	{ x: 0, y: -1, z: 0 },
	{ x: 0, y: 0, z: 1 },
	{ x: 0, y: 0, z: -1 },
	{ x: 0, y: -1, z: 0 },
	{ x: 0, y: -1, z: 0 },
];

/**
 * Camera configured for one canonical cubemap face.
 *
 * @internal Renderer-owned offscreen views use this camera. Applications should
 * normally use `Camera`, `PerspectiveCamera`, or `OrthographicCamera`.
 */
export class CubeFaceCamera extends Camera {
	public readonly faceIndex: number;

	constructor(position: IVector3, farInput: number, faceIndexInput: number) {
		const faceIndex = Number.isFinite(faceIndexInput)
			? Math.max(0, Math.min(5, Math.floor(faceIndexInput)))
			: 0;
		super({
			fov: 90,
			aspectRatio: 1,
			near: 0.1,
			far: Math.max(1, farInput),
		});
		this.faceIndex = faceIndex;
		this.position.set(position.x, position.y, position.z);

		this.updateWorldMatrix();

		const direction = DIRECTIONS[faceIndex];
		const up = UP_VECTORS[faceIndex];
		const target = {
			x: position.x + direction.x,
			y: position.y + direction.y,
			z: position.z + direction.z,
		};
		this.viewMatrix = Matrix4.lookAt(position, target, up);
		this.projectionMatrix = Matrix4.perspective(
			this.fov,
			this.aspectRatio,
			this.near,
			this.far,
		);
		this.viewProjectionMatrix = Matrix4.multiply(this.projectionMatrix, this.viewMatrix);

		this.frustum.setFromMatrix(this.viewProjectionMatrix);
	}
}
