import { Frustum } from "../maths/Frustum";
import { Matrix4 } from "../maths/Matrix4";
import { Quaternion } from "../maths/Quaternion";
import type { IVector3 } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import { Node, type NodeParams } from "../core/Node";

export enum CameraType {
	Perspective = "perspective",
	Orthographic = "orthographic",
}

export interface CameraParams extends NodeParams {
	type?: CameraType;
	fov?: number;
	aspectRatio?: number;
	near?: number;
	far?: number;
}

const _tmpCameraPosition = { x: 0, y: 0, z: 0 };
const _tmpCameraForward = { x: 0, y: 0, z: -1 };
const _tmpCameraUp = { x: 0, y: 1, z: 0 };

export class Camera extends Node {
	public type: CameraType;
	public up: Vector3;
	public fov: number;
	public aspectRatio: number;
	public near: number;
	public far: number;
	public viewMatrix: Matrix4;
	public projectionMatrix: Matrix4;
	public viewProjectionMatrix: Matrix4;
	private _frustum: Frustum;

	constructor(params: CameraParams = {}) {
		super({
			...params,
			idPrefix: "camera",
		});
		this.type = params.type ?? CameraType.Perspective;
		this.up = new Vector3(0, 1, 0);
		this.fov = params.fov ?? 60;
		this.aspectRatio = params.aspectRatio ?? 16 / 9;
		this.near = params.near ?? 0.1;
		this.far = params.far ?? 5000;
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
		this.updateWorldMatrix(this.parent?.worldMatrix);
		this.viewMatrix = this.calculateViewMatrix();
		this.projectionMatrix = this.calculateProjectionMatrix();
		this.viewProjectionMatrix = Matrix4.multiply(
			this.projectionMatrix,
			this.viewMatrix
		);
		this._frustum.setFromMatrix(this.viewProjectionMatrix);
	}

	public calculateViewMatrix(): Matrix4 {
		const worldPosition = this.getWorldPosition(_tmpCameraPosition);
		const worldForward = this.getWorldDirection(
			{ x: 0, y: 0, z: -1 },
			_tmpCameraForward
		);
		const worldUp = this.getWorldDirection(this.up, _tmpCameraUp);

		return Matrix4.lookAt(
			worldPosition,
			new Vector3(
				worldPosition.x + worldForward.x,
				worldPosition.y + worldForward.y,
				worldPosition.z + worldForward.z
			),
			worldUp
		);
	}

	public override setRotationFromEuler(x: number, y: number, z: number): this {
		super.setRotationFromEuler(x, y, z);
		this.updateMatrices();
		return this;
	}

	public override rotateByQuaternion(q: Quaternion): this {
		super.rotateByQuaternion(q);
		this.updateMatrices();
		return this;
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

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.type = this.type;
		target.up.copy(this.up);
		target.fov = this.fov;
		target.aspectRatio = this.aspectRatio;
		target.near = this.near;
		target.far = this.far;
		target.viewMatrix = this.viewMatrix.clone();
		target.projectionMatrix = this.projectionMatrix.clone();
		target.viewProjectionMatrix = this.viewProjectionMatrix.clone();
		target._frustum.setFromMatrix(target.viewProjectionMatrix);
	}
}
