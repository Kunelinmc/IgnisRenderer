import type { IVector3 } from "./types";
import { Plane } from "./Plane";
import { Matrix4 } from "./Matrix4";

export class Frustum {
	private _planes: Plane[];

	constructor() {
		this._planes = [
			new Plane(),
			new Plane(),
			new Plane(),
			new Plane(),
			new Plane(),
			new Plane(),
		];
	}

	public get planes(): Plane[] {
		return this._planes;
	}

	/**
	 * Sets the frustum planes from a view-projection matrix.
	 */
	public setFromMatrix(projection: Matrix4): this {
		const m = projection.elements;

		// Left
		this._planes[0]
			.set(
				m[3][0] + m[0][0],
				m[3][1] + m[0][1],
				m[3][2] + m[0][2],
				m[3][3] + m[0][3]
			)
			.normalize();

		// Right
		this._planes[1]
			.set(
				m[3][0] - m[0][0],
				m[3][1] - m[0][1],
				m[3][2] - m[0][2],
				m[3][3] - m[0][3]
			)
			.normalize();

		// Bottom
		this._planes[2]
			.set(
				m[3][0] + m[1][0],
				m[3][1] + m[1][1],
				m[3][2] + m[1][2],
				m[3][3] + m[1][3]
			)
			.normalize();

		// Top
		this._planes[3]
			.set(
				m[3][0] - m[1][0],
				m[3][1] - m[1][1],
				m[3][2] - m[1][2],
				m[3][3] - m[1][3]
			)
			.normalize();

		// Near
		this._planes[4]
			.set(
				m[3][0] + m[2][0],
				m[3][1] + m[2][1],
				m[3][2] + m[2][2],
				m[3][3] + m[2][3]
			)
			.normalize();

		// Far
		this._planes[5]
			.set(
				m[3][0] - m[2][0],
				m[3][1] - m[2][1],
				m[3][2] - m[2][2],
				m[3][3] - m[2][3]
			)
			.normalize();

		return this;
	}

	public intersectsSphere(center: IVector3, radius: number): boolean {
		for (const plane of this._planes) {
			const distance = plane.distanceToPoint(center);
			if (distance < -radius) {
				return false;
			}
		}
		return true;
	}

	public intersectsAABB(min: IVector3, max: IVector3): boolean {
		for (const plane of this._planes) {
			const px = plane.normal.x > 0 ? max.x : min.x;
			const py = plane.normal.y > 0 ? max.y : min.y;
			const pz = plane.normal.z > 0 ? max.z : min.z;

			if (plane.distanceToPoint({ x: px, y: py, z: pz }) < 0) {
				return false;
			}
		}
		return true;
	}
}
