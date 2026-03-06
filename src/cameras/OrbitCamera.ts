import { Camera } from "./Camera";
import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import type { IVector3 } from "../maths/types";

export class OrbitCamera extends Camera {
	public target: Vector3;
	public distance: number;
	public theta: number;
	public phi: number;
	public minPhi: number;
	public maxPhi: number;
	public minDistance: number;
	public maxDistance: number;
	public lookSensitivity: number;
	public zoomSensitivity: number;
	public autoAdjustClipPlanes: boolean;
	public clipNearFactor: number;
	public clipNearMin: number;
	public clipFarFactor: number;
	public clipFarMin: number;

	constructor(target: IVector3 = new Vector3(0, 0, 0), distance = 400) {
		super();
		this.target = new Vector3(target.x, target.y, target.z);
		this.distance = distance;
		this.theta = 0;
		this.phi = Math.PI / 3;

		this.minPhi = 0.01;
		this.maxPhi = Math.PI - 0.01;
		this.minDistance = 10;
		this.maxDistance = 2000;

		this.lookSensitivity = 0.005;
		this.zoomSensitivity = 0.5;

		this.autoAdjustClipPlanes = true;
		this.clipNearFactor = 0.002;
		this.clipNearMin = 0.1;
		this.clipFarFactor = 3.0;
		this.clipFarMin = 1000;

		this.updatePosition();
	}

	public updatePosition(): void {
		this.phi = Math.max(this.minPhi, Math.min(this.maxPhi, this.phi));
		this.distance = Math.max(
			this.minDistance,
			Math.min(this.maxDistance, this.distance)
		);

		const x = this.distance * Math.sin(this.phi) * Math.sin(this.theta);
		const y = this.distance * Math.cos(this.phi);
		const z = this.distance * Math.sin(this.phi) * Math.cos(this.theta);

		this.position = new Vector3(
			this.target.x + x,
			this.target.y + y,
			this.target.z + z
		);

		this._updateClipPlanes();
		this.updateMatrices();
	}

	public calculateViewMatrix(): Matrix4 {
		const target = this.target || new Vector3(0, 0, 0);
		return Matrix4.lookAt(this.position, target, this.up);
	}

	public rotate(dx: number, dy: number): void {
		this.theta -= dx * this.lookSensitivity;
		this.phi -= dy * this.lookSensitivity;
		this.updatePosition();
	}

	public zoom(delta: number): void {
		this.distance += delta * this.zoomSensitivity;
		this.updatePosition();
	}

	public setTarget(newTarget: IVector3): void {
		this.target = new Vector3(newTarget.x, newTarget.y, newTarget.z);
		this.updatePosition();
	}

	private _updateClipPlanes(): void {
		if (!this.autoAdjustClipPlanes) {
			return;
		}

		const nextFar = Math.max(
			this.clipFarMin,
			this.distance * this.clipFarFactor
		);
		const nextNear = Math.max(
			this.clipNearMin,
			this.distance * this.clipNearFactor
		);

		this.near = Math.min(nextNear, nextFar - 1);
		this.far = Math.max(this.near + 1, nextFar);
	}
}
