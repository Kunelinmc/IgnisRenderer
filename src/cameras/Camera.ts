import { Plane } from "../maths/Plane";
import { Quaternion } from "../maths/Quaternion";
import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";

export class Camera {
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
	private frustumPlanes: Plane[];

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

		this.frustumPlanes = [
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
		const fovRad = (this.fov * Math.PI) / 180;
		const f = 1.0 / Math.tan(fovRad / 2);
		const range = this.near - this.far;

		return new Matrix4([
			[f / this.aspectRatio, 0, 0, 0],
			[0, f, 0, 0],
			[
				0,
				0,
				(this.far + this.near) / range,
				(2 * this.far * this.near) / range,
			],
			[0, 0, -1, 0],
		]);
	}

	public extractFrustumPlanes(): void {
		const m = this.viewProjectionMatrix.elements;

		this.frustumPlanes[0]
			.set(
				m[0][3] + m[0][0],
				m[1][3] + m[1][0],
				m[2][3] + m[2][0],
				m[3][3] + m[3][0]
			)
			.normalize();

		this.frustumPlanes[1]
			.set(
				m[0][3] - m[0][0],
				m[1][3] - m[1][0],
				m[2][3] - m[2][0],
				m[3][3] - m[3][0]
			)
			.normalize();

		this.frustumPlanes[2]
			.set(
				m[0][3] + m[0][1],
				m[1][3] + m[1][1],
				m[2][3] + m[2][1],
				m[3][3] + m[3][1]
			)
			.normalize();

		this.frustumPlanes[3]
			.set(
				m[0][3] - m[0][1],
				m[1][3] - m[1][1],
				m[2][3] - m[2][1],
				m[3][3] - m[3][1]
			)
			.normalize();

		this.frustumPlanes[4]
			.set(
				m[0][3] + m[0][2],
				m[1][3] + m[1][2],
				m[2][3] + m[2][2],
				m[3][3] + m[3][2]
			)
			.normalize();

		this.frustumPlanes[5]
			.set(
				m[0][3] - m[0][2],
				m[1][3] - m[1][2],
				m[2][3] - m[2][2],
				m[3][3] - m[3][2]
			)
			.normalize();
	}

	public isPointInFrustum(point: IVector3): boolean {
		for (const plane of this.frustumPlanes) {
			if (plane.distanceToPoint(point) < 0) {
				return false;
			}
		}
		return true;
	}

	public isSphereInFrustum(center: IVector3, radius: number): boolean {
		for (const plane of this.frustumPlanes) {
			const distance = plane.distanceToPoint(center);
			if (distance < -radius) {
				return false;
			}
		}
		return true;
	}

	public isAABBInFrustum(min: IVector3, max: IVector3): boolean {
		for (const plane of this.frustumPlanes) {
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
