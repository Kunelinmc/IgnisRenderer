import type { IVector3 } from "./types";
import { Vector3 } from "./Vector3";

export class Box3 {
	constructor(
		public min: Vector3 = new Vector3(Infinity, Infinity, Infinity),
		public max: Vector3 = new Vector3(-Infinity, -Infinity, -Infinity)
	) {}

	public set(min: IVector3, max: IVector3): this {
		this.min.copy(min);
		this.max.copy(max);
		return this;
	}

	public expandByPoint(point: IVector3): this {
		this.min.x = Math.min(this.min.x, point.x);
		this.min.y = Math.min(this.min.y, point.y);
		this.min.z = Math.min(this.min.z, point.z);
		this.max.x = Math.max(this.max.x, point.x);
		this.max.y = Math.max(this.max.y, point.y);
		this.max.z = Math.max(this.max.z, point.z);
		return this;
	}

	public containsPoint(point: IVector3): boolean {
		return (
			point.x >= this.min.x &&
			point.x <= this.max.x &&
			point.y >= this.min.y &&
			point.y <= this.max.y &&
			point.z >= this.min.z &&
			point.z <= this.max.z
		);
	}

	public intersectsBox(box: Box3): boolean {
		return (
			box.max.x >= this.min.x &&
			box.min.x <= this.max.x &&
			box.max.y >= this.min.y &&
			box.min.y <= this.max.y &&
			box.max.z >= this.min.z &&
			box.min.z <= this.max.z
		);
	}

	public clone(): Box3 {
		return new Box3(this.min.clone(), this.max.clone());
	}
}
