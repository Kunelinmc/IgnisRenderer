/**
 * Vector4 class and utility functions
 */

import type { IVector4 } from "./types";

export class Vector4 implements IVector4 {
	constructor(
		public x: number = 0,
		public y: number = 0,
		public z: number = 0,
		public w: number = 0
	) {}

	public set(x: number, y: number, z: number, w: number): this {
		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;
		return this;
	}

	public copy(v: IVector4): this {
		this.x = v.x;
		this.y = v.y;
		this.z = v.z;
		this.w = v.w;
		return this;
	}

	public clone(): Vector4 {
		return new Vector4(this.x, this.y, this.z, this.w);
	}

	public add(v: IVector4): this {
		this.x += v.x;
		this.y += v.y;
		this.z += v.z;
		this.w += v.w;
		return this;
	}

	public sub(v: IVector4): this {
		this.x -= v.x;
		this.y -= v.y;
		this.z -= v.z;
		this.w -= v.w;
		return this;
	}

	public scale(s: number): this {
		this.x *= s;
		this.y *= s;
		this.z *= s;
		this.w *= s;
		return this;
	}

	public dot(v: IVector4): number {
		return this.x * v.x + this.y * v.y + this.z * v.z + this.w * v.w;
	}

	public length(): number {
		return Math.hypot(this.x, this.y, this.z, this.w);
	}

	public normalize(): this {
		const len = this.length() || 1;
		return this.scale(1 / len);
	}

	// Static methods for functional style
	public static normalize(v: IVector4): Vector4 {
		return new Vector4(v.x, v.y, v.z, v.w).normalize();
	}

	public static dot(a: IVector4, b: IVector4): number {
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}

	public static add(a: IVector4, b: IVector4): Vector4 {
		return new Vector4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
	}

	public static sub(a: IVector4, b: IVector4): Vector4 {
		return new Vector4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
	}

	public static scale(v: IVector4, s: number): Vector4 {
		return new Vector4(v.x * s, v.y * s, v.z * s, v.w * s);
	}

	public static length(v: IVector4): number {
		return Math.hypot(v.x, v.y, v.z, v.w);
	}
}
