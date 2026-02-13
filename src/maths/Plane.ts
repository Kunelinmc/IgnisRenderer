import type { IVector3 } from "./types";

export class Plane {
	public normal: IVector3;
	public constant: number;

	constructor(normal: IVector3 = { x: 0, y: 1, z: 0 }, constant = 0) {
		this.normal = normal;
		this.constant = constant;
	}

	public set(a: number, b: number, c: number, d: number): this {
		this.normal = { x: a, y: b, z: c };
		this.constant = d;
		return this;
	}

	public normalize(): this {
		const length = Math.sqrt(
			this.normal.x * this.normal.x +
				this.normal.y * this.normal.y +
				this.normal.z * this.normal.z
		);

		if (length > 0) {
			const invLength = 1 / length;
			this.normal.x *= invLength;
			this.normal.y *= invLength;
			this.normal.z *= invLength;
			this.constant *= invLength;
		}

		return this;
	}

	public distanceToPoint(point: IVector3): number {
		return (
			this.normal.x * point.x +
			this.normal.y * point.y +
			this.normal.z * point.z +
			this.constant
		);
	}
}
