import { Camera } from "./Camera";
import { Quaternion } from "../maths/Quaternion";
import { Vector3 } from "../maths/Vector3";

export class FPSCamera extends Camera {
	public yaw: number;
	public pitch: number;

	constructor() {
		super();
		this.yaw = 0;
		this.pitch = 0;

		this.updateRotation();
	}

	public updateRotation(): void {
		const maxPitch = Math.PI / 2;
		this.pitch = Math.max(-maxPitch, Math.min(maxPitch, this.pitch));

		const qYaw = Quaternion.fromAxisAngle({ x: 0, y: 1, z: 0 }, this.yaw);
		const qPitch = Quaternion.fromAxisAngle({ x: 1, y: 0, z: 0 }, this.pitch);

		this.quaternion = Quaternion.multiply(qYaw, qPitch).normalize();
		this.updateMatrices();
	}

	/** Rotates the camera by yaw and pitch angle deltas in radians. */
	public rotate(deltaYaw: number, deltaPitch: number): void {
		this.yaw -= deltaYaw;
		this.pitch -= deltaPitch;
		this.updateRotation();
	}

	public moveForward(distance: number): void {
		const qYaw = Quaternion.fromAxisAngle({ x: 0, y: 1, z: 0 }, this.yaw);
		const forward = qYaw.rotatePoint({ x: 0, y: 0, z: -1 });

		this.position = Vector3.add(this.position, {
			x: forward.x * distance,
			y: 0,
			z: forward.z * distance,
		});
		this.updateMatrices();
	}

	public moveRight(distance: number): void {
		const qYaw = Quaternion.fromAxisAngle({ x: 0, y: 1, z: 0 }, this.yaw);
		const right = qYaw.rotatePoint({ x: 1, y: 0, z: 0 });

		this.position = Vector3.add(this.position, {
			x: right.x * distance,
			y: 0,
			z: right.z * distance,
		});
		this.updateMatrices();
	}

	public moveUpLocal(distance: number): void {
		const up = this.quaternion.rotatePoint({ x: 0, y: 1, z: 0 });
		this.position = Vector3.add(this.position, Vector3.scale(up, distance));
		this.updateMatrices();
	}

	public moveUp(distance: number): void {
		this.position.y += distance;
		this.updateMatrices();
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.yaw = this.yaw;
		target.pitch = this.pitch;
	}
}
