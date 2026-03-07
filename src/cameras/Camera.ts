import { Plane } from "../maths/Plane";
import { Quaternion } from "../maths/Quaternion";
import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import { Frustum } from "../maths/Frustum";
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
	private _frustum: Frustum;

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

		this._frustum = new Frustum();

		this.updateMatrices();
	}

	public get frustum(): Frustum {
		return this._frustum;
	}

	public updateMatrices(): void {
		this.viewMatrix = this.calculateViewMatrix();
		this.projectionMatrix = this.calculateProjectionMatrix();
		this.viewProjectionMatrix = Matrix4.multiply(
			this.projectionMatrix,
			this.viewMatrix
		);
		this._frustum.setFromMatrix(this.viewProjectionMatrix);
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

	public isPointInFrustum(point: IVector3): boolean {
		for (const plane of this._frustum.planes) {
			if (plane.distanceToPoint(point) < 0) {
				return false;
			}
		}
		return true;
	}

	public isSphereInFrustum(center: IVector3, radius: number): boolean {
		return this._frustum.intersectsSphere(center, radius);
	}

	public isAABBInFrustum(min: IVector3, max: IVector3): boolean {
		return this._frustum.intersectsAABB(min, max);
	}
}
