import { Plane } from "../maths/Plane";
import { Quaternion } from "../maths/Quaternion";
import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";

export enum CameraType {
	Perspective = "perspective",
	Orthographic = "orthographic",
}

export class Camera {
	public type: CameraType = CameraType.Perspective;
	public position: Vector3;
	public quaternion: Quaternion;
	public up: Vector3;
	public fov: number;
	public aspectRatio: number;
	public near: number;
	public far: number;
	public viewMatrix: Matrix4;
	public projectionMatrix: Matrix4;
	public viewProjectionMatrix: Matrix4;
	private _frustumPlanes: Plane[];

	constructor() {
		this.position = new Vector3(0, 0, 0);
		this.quaternion = new Quaternion();

		this.up = new Vector3(0, 1, 0);

		this.fov = 60;
		this.aspectRatio = 16 / 9;
		this.near = 0.1;
		this.far = 1000;

		this.viewMatrix = Matrix4.identity();
		this.projectionMatrix = Matrix4.identity();
		this.viewProjectionMatrix = Matrix4.identity();

		this._frustumPlanes = [
			new Plane(),
			new Plane(),
			new Plane(),
			new Plane(),
			new Plane(),
			new Plane(),
		];

		this.updateMatrices();
	}

	public updateMatrices(): void {
		this.viewMatrix = this.calculateViewMatrix();
		this.projectionMatrix = this.calculateProjectionMatrix();
		this.viewProjectionMatrix = Matrix4.multiply(
			this.projectionMatrix,
			this.viewMatrix
		);
		this.extractFrustumPlanes();
	}

	public calculateViewMatrix(): Matrix4 {
		const targetDirection = this.quaternion.rotatePoint({ x: 0, y: 0, z: -1 });
		const up = this.quaternion.rotatePoint(this.up);

		return Matrix4.lookAt(
			this.position,
			new Vector3(
				this.position.x + targetDirection.x,
				this.position.y + targetDirection.y,
				this.position.z + targetDirection.z
			),
			up
		);
	}

	public setRotationFromEuler(x: number, y: number, z: number): void {
		this.quaternion = Quaternion.fromEuler(x, y, z).normalize();
		this.updateMatrices();
	}

	public rotateByQuaternion(q: Quaternion): void {
		this.quaternion = Quaternion.multiply(q, this.quaternion).normalize();
		this.updateMatrices();
	}

	public calculateProjectionMatrix(): Matrix4 {
		return Matrix4.perspective(this.fov, this.aspectRatio, this.near, this.far);
	}

	public extractFrustumPlanes(): void {
		const m = this.viewProjectionMatrix.elements;

		// 0: Left (Row 3 + Row 0)
		this._frustumPlanes[0]
			.set(
				m[3][0] + m[0][0],
				m[3][1] + m[0][1],
				m[3][2] + m[0][2],
				m[3][3] + m[0][3]
			)
			.normalize();

		// 1: Right (Row 3 - Row 0)
		this._frustumPlanes[1]
			.set(
				m[3][0] - m[0][0],
				m[3][1] - m[0][1],
				m[3][2] - m[0][2],
				m[3][3] - m[0][3]
			)
			.normalize();

		// 2: Bottom (Row 3 + Row 1)
		this._frustumPlanes[2]
			.set(
				m[3][0] + m[1][0],
				m[3][1] + m[1][1],
				m[3][2] + m[1][2],
				m[3][3] + m[1][3]
			)
			.normalize();

		// 3: Top (Row 3 - Row 1)
		this._frustumPlanes[3]
			.set(
				m[3][0] - m[1][0],
				m[3][1] - m[1][1],
				m[3][2] - m[1][2],
				m[3][3] - m[1][3]
			)
			.normalize();

		// 4: Near (Row 3 + Row 2)
		this._frustumPlanes[4]
			.set(
				m[3][0] + m[2][0],
				m[3][1] + m[2][1],
				m[3][2] + m[2][2],
				m[3][3] + m[2][3]
			)
			.normalize();

		// 5: Far (Row 3 - Row 2)
		this._frustumPlanes[5]
			.set(
				m[3][0] - m[2][0],
				m[3][1] - m[2][1],
				m[3][2] - m[2][2],
				m[3][3] - m[2][3]
			)
			.normalize();
	}

	public isPointInFrustum(point: IVector3): boolean {
		for (const plane of this._frustumPlanes) {
			if (plane.distanceToPoint(point) < 0) {
				return false;
			}
		}
		return true;
	}

	public isSphereInFrustum(center: IVector3, radius: number): boolean {
		for (const plane of this._frustumPlanes) {
			const distance = plane.distanceToPoint(center);
			if (distance < -radius) {
				return false;
			}
		}
		return true;
	}

	public isAABBInFrustum(min: IVector3, max: IVector3): boolean {
		for (const plane of this._frustumPlanes) {
			const px = plane.normal.x > 0 ? max.x : min.x;
			const py = plane.normal.y > 0 ? max.y : min.y;
			const pz = plane.normal.z > 0 ? max.z : min.z;

			if (
				plane.normal.x * px +
					plane.normal.y * py +
					plane.normal.z * pz +
					plane.constant <
				0
			) {
				return false;
			}
		}
		return true;
	}
}
