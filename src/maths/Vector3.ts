/**
 * Vector3 class and utility functions
 */

import type { IVector3 } from "./types";

export class Vector3 implements IVector3 {
	constructor(
		public x: number = 0,
		public y: number = 0,
		public z: number = 0
	) {}

	public set(x: number, y: number, z: number): this {
		this.x = x;
		this.y = y;
		this.z = z;
		return this;
	}

	public copy(v: IVector3): this {
		this.x = v.x;
		this.y = v.y;
		this.z = v.z;
		return this;
	}

	public clone(): Vector3 {
		return new Vector3(this.x, this.y, this.z);
	}

	public add(v: IVector3): this {
		this.x += v.x;
		this.y += v.y;
		this.z += v.z;
		return this;
	}

	public sub(v: IVector3): this {
		this.x -= v.x;
		this.y -= v.y;
		this.z -= v.z;
		return this;
	}

	public scale(s: number): this {
		this.x *= s;
		this.y *= s;
		this.z *= s;
		return this;
	}

	public dot(v: IVector3): number {
		return this.x * v.x + this.y * v.y + this.z * v.z;
	}

	public cross(v: IVector3): this {
		const x = this.x,
			y = this.y,
			z = this.z;
		this.x = y * v.z - z * v.y;
		this.y = z * v.x - x * v.z;
		this.z = x * v.y - y * v.x;
		return this;
	}

	public length(): number {
		return Math.hypot(this.x, this.y, this.z);
	}

	public normalize(): this {
		const len = this.length() || 1;
		return this.scale(1 / len);
	}

	// Static methods for functional style
	public static normalize(v: IVector3): Vector3 {
		return new Vector3(v.x, v.y, v.z).normalize();
	}

	public static dot(a: IVector3, b: IVector3): number {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	public static cross(a: IVector3, b: IVector3): Vector3 {
		return new Vector3(
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x
		);
	}

	public static add(a: IVector3, b: IVector3): Vector3 {
		return new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
	}

	public static sub(a: IVector3, b: IVector3): Vector3 {
		return new Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
	}

	public static scale(v: IVector3, s: number): Vector3 {
		return new Vector3(v.x * s, v.y * s, v.z * s);
	}

	public static length(v: IVector3): number {
		return Math.hypot(v.x, v.y, v.z);
	}

	public static normalizeInPlace(v: IVector3): void {
		const len = Math.hypot(v.x, v.y, v.z) || 1;
		const invLen = 1 / len;
		v.x *= invLen;
		v.y *= invLen;
		v.z *= invLen;
	}

	public static calculateNormal(vertices: IVector3[]): Vector3 {
		if (vertices.length < 3) return new Vector3(0, 0, 1);

		const v0 = vertices[0];
		let v1: IVector3, v2: IVector3;

		for (let i = 1; i < vertices.length; i++) {
			v1 = vertices[i];
			const ux = v1.x - v0.x;
			const uy = v1.y - v0.y;
			const uz = v1.z - v0.z;
			if (Math.hypot(ux, uy, uz) < 1e-8) continue;

			for (let j = i + 1; j < vertices.length; j++) {
				v2 = vertices[j];
				const vx = v2.x - v0.x;
				const vy = v2.y - v0.y;
				const vz = v2.z - v0.z;

				const nx = uy * vz - uz * vy;
				const ny = uz * vx - ux * vz;
				const nz = ux * vy - uy * vx;

				const len = Math.hypot(nx, ny, nz);
				if (len > 1e-12) {
					return new Vector3(nx / len, ny / len, nz / len);
				}
			}
		}

		return new Vector3(0, 0, 1);
	}
}
